# 📋 Plan: Habilitar Depth en Multiprocessing

**Objetivo**: Reemplazar DepthEstimator (HuggingFace) por MiDaS small (torch nativo) en workers

**Estado**: Ready to implement

---

## 🚨 Problema Actual

DepthEstimator usa `transformers.DPTForDepthEstimation` de HuggingFace:
- HuggingFace models causan **IOT instruction crash** en spawn workers
- Conflict entre tokenizers/safetensors y multiprocessing
- Workaround actual: `ARIA_SKIP_DEPTH=1` (depth deshabilitado)

---

## ✅ Solución: MiDaS Small

**MiDaS es 100% torch nativo** (sin HuggingFace):
- PyTorch hub model: `torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')`
- Proven working en sequential mode
- Compatible con multiprocessing spawn
- Fast inference (~20-30ms on RTX 2060)

---

## 📐 Plan de Implementación

### 1. Modificar `central_worker.py`

**Cambio**: Cargar MiDaS en lugar de DepthEstimator

```python
def _load_models(self) -> None:
    """Load MiDaS (torch native) instead of HuggingFace"""
    log.info("[CentralWorker] Loading MiDaS small...")
    
    # MiDaS model (torch native - no HuggingFace)
    self.midas_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
    self.midas_model.eval()
    self.midas_model.to('cuda')
    
    # MiDaS transforms
    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    self.midas_transform = midas_transforms.small_transform
    
    # YOLO
    self.yolo_processor = YoloProcessor()
    
    # CUDA streams for parallel execution
    self.depth_stream = torch.cuda.Stream()
    self.yolo_stream = torch.cuda.Stream()
    
    log.info("[CentralWorker] MiDaS + YOLO loaded with CUDA streams")
```

---

### 2. Actualizar `_process_frame()`

**Cambio**: Procesar depth con MiDaS en paralelo con YOLO

```python
def _process_frame(self, msg: dict) -> ResultMessage:
    frame = msg["frame"]
    frame_id = msg.get("frame_id", -1)
    start_time = time.perf_counter()
    
    depth_map = None
    depth_raw = None
    depth_ms = 0.0
    
    # Parallel execution: Depth + YOLO
    with torch.cuda.stream(self.depth_stream):
        depth_start = time.perf_counter()
        
        # MiDaS inference
        input_batch = self.midas_transform(frame).to('cuda')
        with torch.no_grad():
            depth_raw = self.midas_model(input_batch)
            depth_raw = torch.nn.functional.interpolate(
                depth_raw.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Normalize to 0-255
        depth_np = depth_raw.cpu().numpy()
        depth_min = depth_np.min()
        depth_max = depth_np.max()
        depth_map = ((depth_np - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        
        depth_ms = (time.perf_counter() - depth_start) * 1000
    
    with torch.cuda.stream(self.yolo_stream):
        yolo_start = time.perf_counter()
        detections = self.yolo_processor.process_frame(frame, depth_map, depth_raw)
        yolo_ms = (time.perf_counter() - yolo_start) * 1000
    
    # Synchronize both streams
    torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    return ResultMessage(
        frame_id=frame_id,
        camera="central",
        detections=detections,
        depth_map=depth_map,
        depth_raw=depth_np,  # Return numpy array
        latency_ms=latency_ms,
        profiling={
            "depth_ms": depth_ms,
            "yolo_ms": yolo_ms,
            "gpu_mem_mb": torch.cuda.memory_allocated() / 1e6,
        },
    )
```

---

### 3. Remover `ARIA_SKIP_DEPTH`

**Archivos a modificar**:
- `src/core/processing/central_worker.py`: Eliminar skip depth logic
- Tests: Ejecutar sin `ARIA_SKIP_DEPTH=1`

---

### 4. Testing

**Test 1 - Smoke test (50 frames)**:
```bash
.venv/bin/python run.py test
```

**Verificar**:
- ✅ Workers spawn sin crashes
- ✅ Depth map returned (not None)
- ✅ No IOT instruction errors
- ✅ ~20 FPS maintained

**Test 2 - Benchmark (200 frames)**:
```bash
.venv/bin/python run.py benchmark
```

**Métricas esperadas**:
- FPS: 18-22 (slight drop from 20.39 con depth overhead)
- Latency p50: 4-8ms (depth adds ~4-6ms)
- Latency p95: <15ms
- GPU memory: ~4.5GB (MiDaS small ~500MB)

---

## 📊 Expected Impact

| Métrica | Sin Depth (actual) | Con Depth (esperado) |
|---------|-------------------|---------------------|
| FPS | 20.39 | 18-22 |
| Latency p50 | 2.35ms | 4-8ms |
| Latency p95 | 3.67ms | 10-15ms |
| VRAM | ~4GB | ~4.5GB |

**Conclusión**: Depth overhead es **aceptable** (~4-6ms), dentro de target <30ms.

---

## 🚧 Riesgos y Mitigaciones

### Riesgo 1: MiDaS model download fail
**Mitigación**: torch.hub caches models en `~/.cache/torch/hub/`
```bash
# Pre-download antes de testing
python -c "import torch; torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')"
```

### Riesgo 2: FPS drop >30%
**Mitigación**: Frame skip depth (procesar cada 3-5 frames)
```python
# En config.py
DEPTH_FRAME_SKIP = 3  # Procesar depth cada 3 frames
```

### Riesgo 3: VRAM overflow (>5.5GB)
**Mitigación**: Ya usando MiDaS_small (~500MB), debería caber. Si no:
- Reducir YOLO batch size
- Usar FP16 inference

---

## 📝 Checklist

- [ ] Modificar `central_worker.py` - Load MiDaS
- [ ] Actualizar `_process_frame()` - Parallel Depth + YOLO
- [ ] Remover `ARIA_SKIP_DEPTH` environment variable
- [ ] Test smoke (50 frames) - Verify no crashes
- [ ] Benchmark (200 frames) - Measure FPS/latency
- [ ] Validar depth_map not None en results
- [ ] Verificar VRAM usage <5.5GB
- [ ] Update `FASE_2_Field_Notes_v2.ipynb` con resultados
- [ ] Commit: `feat(phase2): enable depth in multiprocessing with MiDaS`

---

## 🎯 Success Criteria

1. ✅ Workers spawn sin crashes (no IOT instruction)
2. ✅ Depth map returned y válido (not None)
3. ✅ FPS >= 18 (max 15% drop from 20.39)
4. ✅ Latency p50 < 10ms
5. ✅ 200 frames benchmark stable
6. ✅ VRAM < 5.5GB

---

**Estado**: ⏸️ Pausado - Continuar mañana
**Próximo paso**: Modificar `central_worker.py` para cargar MiDaS

---

**Notas**:
- MiDaS small es lightweight (~500MB)
- Parallel streams mantendrán latency baja
- Si FPS < 18, considerar depth frame skip

**Fecha**: 2025-11-15
**Autor**: Roberto Rojas
