# FASE 4: TensorRT Optimization - Notas de Implementación

**Branch:** `feature/fase4-tensorrt`  
**Fecha inicio:** 2025-11-17  
**Objetivo:** 40-60 FPS (vs 18 FPS actual)

---

## 📊 Estado Actual (Baseline FASE 2)

- **FPS secuencial:** ~10 FPS
- **FPS multiprocessing:** ~18 FPS (+80% mejora)
- **Latencia YOLO:** ~40-45ms
- **Latencia Depth:** ~10-11ms (ya optimizado 8x en FASE 2)
- **VRAM total:** ~5-6 GB (Central worker 5GB, SLAM worker 2.5GB)

**Bottleneck identificado:** Inferencia de modelos PyTorch

---

## 🎯 Objetivo FASE 4

**Speedup esperado con TensorRT:**
- YOLO: 40-45ms → **5-8ms** (5-8x faster)
- Depth: 10-11ms → **3-5ms** (2-3x faster)

**FPS esperado:**
- Secuencial: 30-40 FPS
- Multiprocessing: **50-60 FPS** ✅ Objetivo alcanzado

---

## 📝 Plan de Implementación

### Milestone 1: YOLO12n → TensorRT (Días 1-2)

**Paso 1.1: Export YOLO → ONNX** (30 min)
```bash
# Ultralytics tiene export built-in
python tools/export_yolo_tensorrt.py
```

**Paso 1.2: ONNX → TensorRT Engine** (30 min)
```bash
# Opción A: Ultralytics automático (recomendado)
yolo export model=yolo12n.pt format=engine half=True

# Opción B: Manual con trtexec
trtexec --onnx=yolo12n.onnx \
        --fp16 \
        --workspace=2048 \
        --saveEngine=yolo12n_fp16.trt
```

**Paso 1.3: Integrar en YoloProcessor** (2-3 horas)
- Crear `yolo_processor_trt.py`
- Cargar engine con `tensorrt` library
- Mantener API compatible: `process_frame(frame, depth_map, depth_raw)`
- Feature flag: `USE_TENSORRT_YOLO = True`

**Paso 1.4: Benchmark y validación** (1 hora)
- Comparar detecciones PyTorch vs TensorRT (accuracy)
- Medir latencia improvement
- Validar que no rompe multiprocessing

---

### Milestone 2: Depth-Anything-V2 → TensorRT (Días 3-4)

**Paso 2.1: Export Depth → ONNX** (2-3 horas)
```python
# Más complejo - modelo custom
# Necesita torch.jit.trace() o torch.onnx.export()

import torch
from src.external.depth_anything_v2.dpt import DepthAnythingV2

# Load model
model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
model.load_state_dict(torch.load('checkpoints/depth_anything_v2_vits.pth'))
model.cuda().eval()

# Dummy input
dummy_input = torch.randn(1, 3, 384, 384).cuda()

# Export
torch.onnx.export(
    model,
    dummy_input,
    "depth_anything_v2.onnx",
    opset_version=17,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)
```

**Paso 2.2: ONNX → TensorRT** (1 hora)
```bash
trtexec --onnx=depth_anything_v2.onnx \
        --fp16 \
        --workspace=4096 \
        --saveEngine=depth_anything_v2_fp16.trt \
        --minShapes=input:1x3x384x384 \
        --optShapes=input:1x3x384x384 \
        --maxShapes=input:1x3x384x384
```

**Paso 2.3: Integrar en CentralWorker** (2 horas)
- Modificar `central_worker.py`
- Reemplazar `infer_image_gpu()` con TensorRT inference
- Mantener output compatible (depth_map, depth_raw)

**Paso 2.4: Validación** (1 hora)
- Comparar depth maps (visual + MSE)
- Verificar que distancias siguen correctas
- Medir latencia improvement

---

### Milestone 3: System Integration & Testing (Día 5)

**Paso 3.1: Feature flags en config.py**
```python
# Añadir a Config
PHASE4_TENSORRT_ENABLED = True
PHASE4_TENSORRT_YOLO = True
PHASE4_TENSORRT_DEPTH = True
TENSORRT_WORKSPACE_SIZE = 2048  # MB
TENSORRT_FP16 = True
TENSORRT_INT8 = False  # Future
```

**Paso 3.2: Benchmark end-to-end**
```bash
# Secuencial con TensorRT
python src/main.py --tensorrt

# Multiprocessing con TensorRT
python run.py --tensorrt
```

**Paso 3.3: Stress testing**
- 10 minutos de ejecución continua
- Validar estabilidad de TensorRT engines
- Memory leak check (`nvidia-smi dmon`)

**Paso 3.4: Comparativa performance**
```
Métrica          | PyTorch | TensorRT | Speedup
-----------------|---------|----------|--------
YOLO latency     | 40ms    | 5-8ms    | 5-8x
Depth latency    | 10ms    | 3-5ms    | 2-3x
FPS secuencial   | 10      | 35-40    | 3.5-4x
FPS multiproc    | 18      | 50-60    | 3x
```

---

## 🔧 Archivos a Crear/Modificar

### Nuevos archivos:
```
tools/export_yolo_tensorrt.py          # Script export YOLO
tools/export_depth_tensorrt.py         # Script export Depth
src/core/vision/yolo_processor_trt.py  # YOLO con TensorRT
checkpoints/yolo12n_fp16.trt           # Engine compilado (generado)
checkpoints/depth_anything_v2_fp16.trt # Engine compilado (generado)
```

### Modificar:
```
src/utils/config.py                    # Flags FASE 4
src/core/processing/central_worker.py  # Usar TensorRT depth
src/core/processing/slam_worker.py     # Usar TensorRT YOLO
src/core/navigation/navigation_pipeline.py  # Conditional loading
```

---

## ⚠️ Riesgos y Mitigaciones

| Riesgo | Probabilidad | Mitigación |
|--------|--------------|------------|
| **ONNX export falla** (ops no soportadas) | Media | - Usar torch.jit.trace en vez de export<br>- Modificar model graph<br>- Fallback a PyTorch |
| **TensorRT accuracy drop** | Baja | - Comparar outputs layer-by-layer<br>- Usar FP32 primero, luego FP16<br>- Threshold de MSE aceptable |
| **Dynamic shapes issues** | Media | - Usar static shapes (1x3x384x384)<br>- Optimization profiles si necesario |
| **Memory issues** (TRT + PyTorch) | Baja | - Descargar modelos PyTorch después de export<br>- torch.cuda.empty_cache() |
| **Multiprocessing incompatibility** | Baja | - Cargar engines en workers (no en main)<br>- Spawn method ya configurado |

---

## 📚 Recursos Útiles

**Documentación:**
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)
- [Ultralytics TensorRT Export](https://docs.ultralytics.com/modes/export/#export-formats)

**Debugging:**
```bash
# Verificar TensorRT instalado
python -c "import tensorrt; print(tensorrt.__version__)"

# Test ONNX válido
python -c "import onnx; onnx.checker.check_model('model.onnx')"

# Benchmark con trtexec
trtexec --loadEngine=model.trt --warmUp=500 --duration=10 --iterations=100

# Profiling GPU
nsys profile -o profile.qdrep python src/main.py
```

---

## ✅ Checklist Pre-Implementación

- [x] Commit FASE 1 & 2 completado
- [x] Branch `feature/fase4-tensorrt` creada
- [ ] TensorRT instalado (`pip install tensorrt`)
- [ ] ONNX instalado (`pip install onnx onnxsim`)
- [ ] Baseline performance documentado (18 FPS multiproc)
- [ ] Espacio en disco para engines (~500MB cada uno)
- [ ] Backup de código funcional

---

## 📊 Tracking de Progreso

### Día 1: YOLO Export
- [ ] Script export creado
- [ ] ONNX generado y validado
- [ ] TensorRT engine compilado
- [ ] Benchmark engine standalone

### Día 2: YOLO Integration
- [ ] `yolo_processor_trt.py` implementado
- [ ] Tests unitarios
- [ ] Integrado en workers
- [ ] Benchmark latency improvement

### Día 3-4: Depth Export & Integration
- [ ] Export Depth → ONNX (desafío: model custom)
- [ ] TensorRT engine compilado
- [ ] Integrado en `central_worker.py`
- [ ] Validación depth maps

### Día 5: System Testing
- [ ] Benchmark end-to-end (secuencial + multiproc)
- [ ] Stress test 10 minutos
- [ ] FPS target alcanzado (50-60 FPS)
- [ ] Commit y documentación

---

## 🎯 Criterios de Éxito

**Mínimo aceptable:**
- ✅ FPS multiproc ≥ 40 (vs 18 actual)
- ✅ YOLO accuracy degradation < 2%
- ✅ Depth MSE < 5% vs PyTorch
- ✅ Sistema estable sin crashes

**Objetivo ideal:**
- ✅ FPS multiproc ≥ 50
- ✅ Accuracy degradation < 1%
- ✅ VRAM usage sin aumento significativo

**Stretch goal:**
- 🎯 FPS multiproc ≥ 60 (objetivo original)
- 🎯 INT8 quantization para mayor speedup

---

## 🚀 Próximo Paso Inmediato

**EMPEZAR:** Export YOLO12n → ONNX/TensorRT
```bash
# Crear script
touch tools/export_yolo_tensorrt.py

# Código inicial:
# 1. Load YOLO12n.pt
# 2. Export format='engine' half=True
# 3. Validar engine funciona
# 4. Benchmark vs PyTorch
```

**Tiempo estimado total FASE 4:** 5 días  
**ROI esperado:** +150% FPS (18 → 50 FPS)

---

**Última actualización:** 2025-11-17  
**Status:** 🟢 Ready to start
