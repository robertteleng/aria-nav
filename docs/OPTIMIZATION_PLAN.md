# 🚀 Plan de Optimización de Performance - RTX 2060

**Objetivo**: Alcanzar 60 FPS con máxima calidad en RTX 2060  
**Estado Actual**: 13.6 FPS  
**Gap**: 46.4 FPS (340% mejora necesaria)

---

## 📊 Análisis Actual del Cuello de Botella

### Tiempos Medidos (por frame)
- **Total latencia**: 66.75ms promedio
- **Frames normales**: ~40-45ms (YOLO + rendering)
- **Frames con depth**: ~90-100ms (+50ms depth overhead)

### Identificación de Problemas
1. **🐌 GIL de Python**: Threading no paralelo real
2. **📦 Streaming**: DDS sample lost (CPU->GPU transfer lento)
3. **🔄 Sincronización**: Todo secuencial, no hay paralelismo
4. **💾 Memoria**: Copias innecesarias CPU<->GPU

---

## 🎯 Plan de Optimización (5 Fases)

### **FASE 1: Quick Wins - GPU Optimization** ⚡
**Objetivo**: 20-25 FPS (mejora 50-80%)  
**Esfuerzo**: Bajo (2-3 horas)  
**Riesgo**: Bajo

#### Acciones:
1. **Batch Processing GPU**
   - [ ] Procesar YOLO y Depth en el mismo batch
   - [ ] Mantener frames en GPU (no bajar a CPU)
   - [ ] Usar `torch.cuda.Stream()` para operaciones paralelas
   
2. **Optimizar Transferencias CPU-GPU**
   - [ ] `pinned_memory=True` para tensores
   - [ ] Reducir `.cpu()` innecesarios
   - [ ] Cache de depth map en GPU
   
3. **Aumentar Batch Size**
   - [ ] YOLO procesar 2-3 frames por inferencia
   - [ ] Depth procesar cada 6-9 frames (vs actual 3)

**Código a modificar:**
- `src/core/vision/yolo_processor.py`
- `src/core/vision/depth_estimator.py`
- `src/core/navigation/navigation_pipeline.py`

---

### **FASE 2: Multiprocessing - Romper el GIL** 🔓
**Objetivo**: 35-40 FPS (mejora 170%)  
**Esfuerzo**: Medio (1-2 días)  
**Riesgo**: Medio (sincronización)

#### Acciones:
1. **Separar en Procesos Independientes**
   ```
   Process 1: Aria Streaming → Queue
   Process 2: YOLO Inference (GPU) → Queue
   Process 3: Depth Inference (GPU) → Queue
   Process 4: Audio + Rendering (CPU)
   ```

2. **Comunicación IPC**
   - [ ] `multiprocessing.Queue` con shared memory
   - [ ] Usar `torch.multiprocessing` para GPU sharing
   - [ ] Pipes para sincronización rápida

3. **Shared Memory para Frames**
   - [ ] `shared_memory.SharedMemory` para frames grandes
   - [ ] Evitar pickles pesados

**Nuevo diseño:**
```python
src/core/processing/
├── aria_capture_process.py    # Streaming loop
├── yolo_inference_process.py  # GPU YOLO
├── depth_inference_process.py # GPU Depth
└── coordinator_process.py     # Main orchestrator
```

---

### **FASE 3: GStreamer Pipeline - Zero-Copy** 📹
**Objetivo**: 50-55 FPS (mejora 270%)  
**Esfuerzo**: Alto (2-3 días)  
**Riesgo**: Alto (nueva infraestructura)

#### Acciones:
1. **Reemplazar DDS Streaming con GStreamer**
   - [ ] Pipeline directo Aria → GPU memory
   - [ ] Hardware decoding (NVDEC)
   - [ ] Zero-copy con `appsink`

2. **GStreamer Pipeline**
   ```bash
   aria_source ! 
   nvvidconv ! 
   video/x-raw(memory:NVMM) ! 
   appsink
   ```

3. **Integración con PyTorch**
   - [ ] CuPy para conversión NVMM → Torch tensor
   - [ ] Direct GPU upload sin CPU pass

**Dependencias nuevas:**
- `gstreamer1.0`
- `gstreamer1.0-plugins-bad`
- `python-gi`
- `cupy-cuda11x`

---

### **FASE 4: Model Optimization - TensorRT** 🔥
**Objetivo**: 60+ FPS (mejora 340%+)  
**Esfuerzo**: Alto (3-4 días)  
**Riesgo**: Medio (conversión de modelos)

#### Acciones:
1. **YOLO a TensorRT**
   - [ ] Exportar YOLO12n a ONNX
   - [ ] Convertir ONNX → TensorRT engine
   - [ ] FP16 precision (2x speed vs FP32)
   - [ ] Dynamic shapes para batch variable

2. **Depth Anything V2 a TensorRT**
   - [ ] Exportar modelo HuggingFace → ONNX
   - [ ] TensorRT engine con FP16
   - [ ] Fusión de capas (layer fusion)

3. **Batch Inferencing**
   - [ ] YOLO: batch=4 frames
   - [ ] Depth: batch=2 frames

**Performance esperado:**
- YOLO: 40-45ms → **5-8ms** (5-9x faster)
- Depth: 50ms → **10-15ms** (3-5x faster)

**Herramientas:**
- `torch2trt`
- `onnx`
- `tensorrt`

---

### **FASE 5: Advanced Optimizations** 🎓
**Objetivo**: 60+ FPS sostenido + features adicionales  
**Esfuerzo**: Medio (1-2 días)  
**Riesgo**: Bajo

#### Acciones:
1. **Frame Skipping Inteligente**
   - [ ] Skip frames basado en motion detection
   - [ ] Interpolación de detecciones entre frames
   - [ ] Predictive tracking

2. **GPU Direct RDMA** (si disponible)
   - [ ] Aria → GPU sin pasar por CPU RAM
   - [ ] Requiere hardware específico

3. **Async Everything**
   - [ ] CUDA streams para overlap
   - [ ] Async audio processing
   - [ ] Non-blocking rendering

4. **Model Quantization**
   - [ ] INT8 inference donde sea posible
   - [ ] Mixed precision (FP16/FP32)

---

## 📋 Plan de Ejecución Recomendado

### Semana 1: Fundamentos
- **Día 1-2**: FASE 1 (Quick Wins)
- **Día 3-5**: FASE 2 (Multiprocessing)

### Semana 2: Infraestructura
- **Día 1-3**: FASE 3 (GStreamer)
- **Día 4-5**: Testing + ajustes

### Semana 3: Aceleración
- **Día 1-4**: FASE 4 (TensorRT)
- **Día 5**: FASE 5 (Advanced)

---

## 🔧 Configuración Hardware Óptima (RTX 2060)

```python
# Config para RTX 2060 (6GB VRAM)
YOLO_IMAGE_SIZE = 640           # Max resolution
YOLO_BATCH_SIZE = 4             # 4 frames simultáneos
YOLO_FP16 = True                # Half precision
DEPTH_INPUT_SIZE = 384          # Alta resolución
DEPTH_BATCH_SIZE = 2            # 2 frames simultáneos
DEPTH_FP16 = True               # Half precision

# CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# Memory management
torch.cuda.empty_cache()        # Periódicamente
CUDA_VISIBLE_DEVICES = "0"      # Single GPU
```

---

## 📈 Mejoras Esperadas por Fase

| Fase | FPS | Mejora | Esfuerzo | Prioridad |
|------|-----|--------|----------|-----------|
| Actual | 13.6 | - | - | - |
| FASE 1 | 22-25 | +65% | Bajo | ⭐⭐⭐ |
| FASE 2 | 35-40 | +170% | Medio | ⭐⭐⭐ |
| FASE 3 | 50-55 | +270% | Alto | ⭐⭐ |
| FASE 4 | 60+ | +340% | Alto | ⭐⭐⭐ |
| FASE 5 | 60+ | - | Medio | ⭐ |

---

## 🚦 Empezar con FASE 1 - Quick Wins

### Implementación Inmediata:

1. **Aumentar resoluciones**
   ```python
   YOLO_IMAGE_SIZE = 640
   DEPTH_INPUT_SIZE = 384
   ```

2. **Optimizar frame skipping**
   ```python
   YOLO_FRAME_SKIP = 0      # Procesar todo
   DEPTH_FRAME_SKIP = 6     # Reducir overhead
   ```

3. **GPU pinned memory**
   ```python
   frame_tensor = torch.from_numpy(frame).cuda(non_blocking=True)
   ```

4. **CUDA streams**
   ```python
   yolo_stream = torch.cuda.Stream()
   depth_stream = torch.cuda.Stream()
   ```

---

## 📝 Tracking de Progreso

- [ ] FASE 1: Quick Wins (Target: 22-25 FPS)
- [ ] FASE 2: Multiprocessing (Target: 35-40 FPS)
- [ ] FASE 3: GStreamer (Target: 50-55 FPS)
- [ ] FASE 4: TensorRT (Target: 60+ FPS)
- [ ] FASE 5: Advanced (Target: 60+ FPS sostenido)

---

## 🎯 ¿Por dónde empezamos?

**Recomendación**: Empezar con FASE 1 para ganar momentum y validar el approach.

¿Quieres que implemente FASE 1 ahora? Puedo hacer los cambios en ~30 minutos.
