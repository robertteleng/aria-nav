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

### ✅ Semana 1: Fundamentos (COMPLETADO)
- **✅ Día 1-2**: FASE 1 (Quick Wins) - Completado
  - Optimizaciones CUDA, pinned memory, streams paralelos
  - Resoluciones aumentadas: YOLO 640px, Depth 384px
- **✅ Día 3-5**: FASE 2 (Multiprocessing) - Completado
  - Workers GPU spawn, 20.19 FPS sin depth
  - **BONUS**: Depth-Anything-V2 GPU-optimizado (80ms→10ms, 8x speedup)
  - Commit: fa642c0

### ⏳ Semana 2: Validación + Infraestructura (EN PROGRESO)
- **⏳ Día 1**: Testing con depth integrado
  - Medir FPS real con depth+YOLO paralelo
  - Benchmark 50-200 frames, stress test 10min
  - Validar latency <100ms, FPS ≥15
- **⏳ Día 2-3**: FASE 3 (GStreamer) - OPCIONAL
  - Evaluar si streaming es bottleneck
  - Solo si FPS <15 por problemas de transferencia
- **⏳ Día 4-5**: Documentación intermedia
  - Actualizar FASE_2_IMPLEMENTATION.md con depth
  - Field notes con resultados

### 🎯 Semana 3: Aceleración TensorRT (SIGUIENTE)
- **Día 1-2**: Conversión YOLO12n → TensorRT
  - Export ONNX, TensorRT engine FP16
  - Benchmark mejora esperada: 40ms→5-8ms
- **Día 3-4**: Conversión Depth-Anything-V2 → TensorRT
  - Export PyTorch→ONNX→TensorRT
  - Benchmark mejora esperada: 10ms→3-5ms
- **Día 5**: FASE 5 (Advanced) + Testing final
  - INT8 quantization si es necesario
  - Validación completa sistema

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

- [x] **FASE 1: Quick Wins** (Target: 22-25 FPS) ✅ COMPLETADO
  - [x] CUDA streams paralelos (depth + YOLO)
  - [x] Pinned memory para transfers
  - [x] Resoluciones optimizadas (640px YOLO, 384px Depth)
  - [x] Optimizaciones cuDNN/TF32
  - **Resultado**: Base sólida para multiprocessing

- [x] **FASE 2: Multiprocessing** (Target: 35-40 FPS) ✅ COMPLETADO
  - [x] Workers GPU con spawn method
  - [x] Central worker (depth+YOLO) + SLAM workers
  - [x] IPC con multiprocessing.Queue
  - [x] **BONUS**: Depth-Anything-V2 GPU-optimized (8x speedup)
  - **Resultado**: 20.19 FPS sin depth, ~15-18 FPS esperado con depth

- [ ] **FASE 3: GStreamer** (Target: 50-55 FPS) ⏸️ EVALUACIÓN
  - [ ] Análisis de bottleneck en streaming
  - [ ] Zero-copy pipeline si es necesario
  - [ ] Hardware decoding NVDEC
  - **Estado**: Evaluar después de tests con depth

- [ ] **FASE 4: TensorRT** (Target: 60+ FPS) 🎯 SIGUIENTE
  - [ ] YOLO12n → TensorRT FP16
  - [ ] Depth-Anything-V2 → TensorRT FP16
  - [ ] Batch inferencing optimizado
  - [ ] Benchmarking comparativo
  - **Aprendizaje esperado**: ONNX export, TensorRT API, FP16 optimization

- [ ] **FASE 5: Advanced** (Target: 60+ FPS sostenido) 🔮 FUTURO
  - [ ] INT8 quantization
  - [ ] Frame skipping inteligente
  - [ ] CUDA kernel fusion
  - [ ] Profiling avanzado

---

## 🎯 ¿Por dónde empezamos?

**Estado Actual**: FASE 1 y 2 completadas. Depth-Anything-V2 integrado y optimizado.

**Próximos Pasos Inmediatos:**
1. ✅ **Test completo con depth** (hoy/mañana)
2. ✅ **Stress testing 10min** (validar estabilidad)
3. 📊 **Análisis de resultados** (decidir si FASE 3 necesaria)
4. 🚀 **FASE 4: TensorRT** (próxima gran milestone)

---

## 📚 Roadmap de Aprendizaje - FASE 4 (TensorRT)

### Conceptos a Dominar

#### 1. **ONNX (Open Neural Network Exchange)**
**Qué es**: Formato intermedio para intercambiar modelos entre frameworks
```
PyTorch → ONNX → TensorRT
```

**Por qué importa**: 
- TensorRT no lee PyTorch directamente
- ONNX permite portabilidad (PyTorch, TensorFlow, etc.)
- Validación de compatibilidad de operaciones

**Aprenderás**:
- Export de modelos PyTorch a ONNX
- Dynamic shapes vs static shapes
- Debugging de exportación (ops no soportadas)
- Simplificación de grafos con `onnx-simplifier`

**Recursos**:
- [PyTorch ONNX Export](https://pytorch.org/docs/stable/onnx.html)
- [ONNX Operator Coverage](https://onnx.ai/onnx/operators/)

---

#### 2. **TensorRT Fundamentals**
**Qué es**: Motor de inference optimizado de NVIDIA

**Aprenderás**:
- **Builder**: Crear engine desde ONNX
- **Engine**: Modelo compilado optimizado para GPU específica
- **Context**: Runtime execution environment
- **Precision modes**: FP32, FP16, INT8
- **Layer fusion**: Combinar operaciones para reducir latency
- **Kernel auto-tuning**: TensorRT prueba múltiples implementaciones

**Pipeline TensorRT**:
```python
# 1. Parse ONNX
parser.parse(onnx_model)

# 2. Build con optimizaciones
config.set_flag(trt.BuilderFlag.FP16)
engine = builder.build_engine(network, config)

# 3. Inference
context = engine.create_execution_context()
context.execute_v2(bindings)
```

**Recursos**:
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/)
- [TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/)

---

#### 3. **Precision Optimization**

**FP32 → FP16 (Half Precision)**:
- **Speedup**: 2x típicamente (RTX 2060 tiene Tensor Cores FP16)
- **Trade-off**: Mínima pérdida de precisión (~0.1% accuracy)
- **Cuándo usar**: Siempre como primer paso

**FP16 → INT8 (Quantization)**:
- **Speedup**: 4x potencialmente
- **Trade-off**: Puede afectar accuracy (~1-3%)
- **Requiere**: Calibration dataset
- **Cuándo usar**: Si FP16 no alcanza FPS target

**Aprenderás**:
- Calibration para INT8
- Post-training quantization (PTQ)
- Quantization-aware training (QAT) - avanzado

---

#### 4. **Dynamic Shapes & Batching**

**Problema**: Tamaño de entrada variable
```python
# Batch size puede variar: 1, 2, 4 frames
input_shape = (batch_size, 3, 640, 640)  # Variable
```

**Solución TensorRT**:
```python
# Optimization profiles
profile.set_shape("input", 
    min=(1,3,640,640),
    opt=(2,3,640,640),  # Optimal
    max=(4,3,640,640))
```

**Aprenderás**:
- Optimization profiles
- Trade-off: flexibilidad vs performance
- Cuándo usar static shapes (mejor performance)

---

#### 5. **Debugging & Profiling**

**Tools que usarás**:
- `trtexec`: CLI para testing rápido
- `Nsight Systems`: Profiling GPU end-to-end
- `polygraphy`: Debugging TensorRT conversions
- `onnx-graphsurgeon`: Modificar grafos ONNX

**Aprenderás**:
- Identificar layers lentas
- Comparar accuracy PyTorch vs TensorRT
- Debugging de conversión fallida
- Memory profiling

---

### Plan de Implementación Detallado

#### **Milestone 1: YOLO12n → TensorRT** (2 días)

**Día 1 - Export & Build**:
```bash
# 1. Export PyTorch → ONNX
python export_yolo_onnx.py

# 2. Simplify ONNX
onnxsim yolo12n.onnx yolo12n_simplified.onnx

# 3. Build TensorRT engine
trtexec --onnx=yolo12n.onnx \
        --fp16 \
        --workspace=2048 \
        --saveEngine=yolo12n_fp16.trt
```

**Día 2 - Integration & Benchmark**:
```python
# Integrar en yolo_processor.py
class YoloProcessorTRT:
    def __init__(self):
        self.engine = load_trt_engine("yolo12n_fp16.trt")
        self.context = self.engine.create_execution_context()
    
    def process_frame(self, frame):
        # Inference con TensorRT
        pass
```

**Validación**:
- Accuracy: Compare detections PyTorch vs TRT (debe ser ~99% igual)
- Performance: Measure latency improvement
- Target: 40ms → 5-8ms

---

#### **Milestone 2: Depth-Anything-V2 → TensorRT** (2 días)

**Día 1 - Export Custom Model**:
```python
# Más complejo que YOLO (modelo custom)
# Puede requerir modificaciones al grafo

# 1. Trace model
traced = torch.jit.trace(depth_model, example_input)

# 2. Export to ONNX
torch.onnx.export(traced, ...)

# 3. Fix incompatible ops si es necesario
```

**Día 2 - Optimization & Integration**:
```python
# Integrar en central_worker.py
# Reemplazar infer_image_gpu() con TRT version
```

**Validación**:
- Depth map comparison (visual + MSE)
- Performance: 10-11ms → 3-5ms esperado

---

#### **Milestone 3: System Integration** (1 día)

**Testing Completo**:
- Benchmark end-to-end: YOLO TRT + Depth TRT
- Stress test 10min con TensorRT
- Memory profiling
- FPS target: 40-60 FPS esperado

**Rollback Plan**:
- Keep PyTorch versions como fallback
- Feature flag para switch entre PyTorch/TRT
```python
USE_TENSORRT = os.getenv("USE_TENSORRT", "true") == "true"
```

---

### Recursos de Aprendizaje

**Documentación Oficial**:
- [TensorRT Quick Start Guide](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/)
- [TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/best-practices/)
- [ONNX Tutorials](https://onnx.ai/onnx/intro/)

**Ejemplos Prácticos**:
- [TensorRT GitHub Samples](https://github.com/NVIDIA/TensorRT/tree/main/samples)
- [YOLOv8 TensorRT Example](https://github.com/triple-Mu/YOLOv8-TensorRT)
- [torch2trt Examples](https://github.com/NVIDIA-AI-IOT/torch2trt)

**Community**:
- [NVIDIA Developer Forums - TensorRT](https://forums.developer.nvidia.com/c/ai/tensorrt/)
- r/computervision subreddit
- Stack Overflow `[tensorrt]` tag

---

### Checklist Pre-TensorRT

Antes de empezar FASE 4, asegurarte de:
- [ ] PyTorch models funcionan correctamente
- [ ] Benchmarks baseline documentados (latency, accuracy)
- [ ] Test dataset preparado para validación
- [ ] TensorRT instalado y funcionando (`trtexec --help`)
- [ ] Familiarizado con ONNX export básico

---

### Expected Challenges & Solutions

**Challenge 1: Unsupported ONNX Ops**
```
Error: Unsupported operator 'CustomOp'
```
**Solution**: 
- Check ONNX operator support
- Rewrite op usando ops soportados
- Use `torch.onnx.register_custom_op_symbolic()`

**Challenge 2: Accuracy Degradation**
```
TensorRT predictions differ significantly
```
**Solution**:
- Compare intermediate outputs layer-by-layer
- Use FP32 first, then FP16
- Adjust calibration for INT8

**Challenge 3: Dynamic Shapes Issues**
```
Error: Input shape does not match
```
**Solution**:
- Use optimization profiles
- Consider static shapes for best performance

---

## 🎓 ¿Listo para FASE 4?

**Primero**: Completa testing actual (depth integrado)
**Cuando estés listo**: Seguiremos esta guía paso a paso
**Objetivo final**: Sistema completo optimizado + conocimiento profundo de inference optimization
