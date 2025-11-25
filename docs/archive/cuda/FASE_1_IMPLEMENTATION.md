# 🚀 FASE 1: Optimizaciones GPU - Implementación

**Objetivo**: Alcanzar 20-25 FPS (mejora del 50-80%)  
**Estado Actual**: 13.6 FPS (baseline con hardware real)  
**Mock Performance**: 66.1 FPS (sin procesamiento real)  
**Fecha Inicio**: 2025-01-14  
**Hardware**: RTX 2060 (6GB VRAM)

---

## 📊 Estado Actual del Sistema

### Baseline Performance
```
FPS: 13.6
YOLO Resolution: 256x256
Depth Resolution: 256x256
Frame Skip: Configuración actual
CUDA Optimizations: No implementadas
GPU Memory: Sin pinned memory
CUDA Streams: No implementados
```

### Cuello de Botella Identificado
- **Transferencias CPU-GPU**: Sin optimizar
- **Resoluciones bajas**: Limitando calidad
- **No hay overlapping**: Operaciones secuenciales
- **Memory copies**: Lentas (no pinned)

---

## 🎯 Cambios a Implementar

### 1️⃣ Modificar `src/utils/config.py`

**Objetivo**: Aumentar resoluciones y habilitar optimizaciones CUDA

#### Código Completo:
```python
# src/utils/config.py

import torch
import logging

log = logging.getLogger(__name__)

class Config:
    def __init__(self):
        # ========== FASE 1: Resoluciones Aumentadas ==========
        self.YOLO_IMAGE_SIZE = 640      # ANTES: 256 → AHORA: 640
        self.DEPTH_INPUT_SIZE = 384     # ANTES: 256 → AHORA: 384
        
        log.info(f"✓ Resoluciones aumentadas: YOLO={self.YOLO_IMAGE_SIZE}, Depth={self.DEPTH_INPUT_SIZE}")
        
        # ========== FASE 1: CUDA Optimizations ==========
        self.CUDA_OPTIMIZATIONS = True
        self.PINNED_MEMORY = True
        self.NON_BLOCKING_TRANSFER = True
        self.CUDA_STREAMS = True  # OBLIGATORIO en FASE 1
        
        # Habilitar optimizaciones CUDA
        if self.CUDA_OPTIMIZATIONS and torch.cuda.is_available():
            self._enable_cuda_optimizations()
        
        # Device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # ========== Frame Skipping ==========
        self.YOLO_SKIP_FRAMES = 0       # Procesar todos (0 = no skip)
        self.DEPTH_SKIP_FRAMES = 6      # Procesar cada 6 frames
        
        log.info("✓ Config FASE 1 cargada")
    
    def _enable_cuda_optimizations(self):
        """FASE 1 Quick Wins - Optimizaciones CUDA"""
        
        # cuDNN benchmark (auto-tune de algoritmos)
        torch.backends.cudnn.benchmark = True
        log.info("  ✓ cuDNN benchmark enabled")
        
        # TensorFloat-32 (RTX 2060 compatible)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        log.info("  ✓ TF32 enabled")
        
        # High precision matmul
        torch.set_float32_matmul_precision('high')
        log.info("  ✓ Float32 precision: high")
        
        # Limpiar cache inicial
        torch.cuda.empty_cache()
        log.info("  ✓ CUDA cache cleared")
```

#### 📝 Notas de Implementación:

```
[Espacio para notas durante implementación]

Cambios realizados:
- 
- 

Problemas encontrados:
- 
- 

Validaciones:
- [ ] Config carga sin errores
- [ ] Logs muestran optimizaciones habilitadas
- [ ] torch.cuda.is_available() = True
```

#### ✅ Checklist:
- [ ] Aumentar YOLO_IMAGE_SIZE a 640
- [ ] Aumentar DEPTH_INPUT_SIZE a 384
- [ ] Agregar flags de CUDA optimizations
- [ ] Habilitar CUDA_STREAMS = True
- [ ] Ajustar frame skipping (DEPTH_SKIP_FRAMES = 6)
- [ ] Validar que no rompe imports
- [ ] Ejecutar y verificar logs

---

### 2️⃣ Modificar `src/core/vision/yolo_processor.py`

**Objetivo**: Implementar GPU pinned memory y non-blocking transfers

#### Localización del Código:
- Método principal: `process_frame()` (~línea 265)
- Clase: `YOLOProcessor`

#### Código de Implementación:

```python
# src/core/vision/yolo_processor.py

# MODIFICAR método process_frame()

def process_frame(self, frame: np.ndarray) -> List[Detection]:
    """
    FASE 1: Optimizado con pinned memory y non-blocking transfer
    """
    
    # Preprocesar
    img = cv2.resize(frame, (self.image_size, self.image_size))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_normalized, (2, 0, 1))  # HWC -> CHW
    
    # ========== FASE 1: Pinned Memory + Async Transfer ==========
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0)
    
    # Pinned memory (permite DMA directo)
    if hasattr(self.config, 'PINNED_MEMORY') and self.config.PINNED_MEMORY:
        img_tensor = img_tensor.pin_memory()
    
    # Async transfer a GPU (non-blocking)
    img_tensor = img_tensor.to(
        self.device,
        non_blocking=getattr(self.config, 'NON_BLOCKING_TRANSFER', False)
    )
    # ==========================================================
    
    # Inferencia (sin cambios)
    with torch.no_grad():
        results = self.model(img_tensor, verbose=False)
    
    # Postprocessing (sin cambios)
    detections = self._parse_yolo_results(results)
    
    return detections
```

#### Instrucciones de Implementación:
1. Abrir `src/core/vision/yolo_processor.py`
2. Localizar método `process_frame()`
3. Buscar la línea donde haces `.to(self.device)`
4. Reemplazar con el código entre las líneas `====`
5. Verificar que `self.image_size` usa `config.YOLO_IMAGE_SIZE`

#### 📝 Notas de Implementación:

```
[Espacio para notas durante implementación]

Problemas encontrados:
- 
- 

Soluciones aplicadas:
- 
- 

Performance observada:
- FPS antes: 
- FPS después: 
- Latencia YOLO antes: ___ms
- Latencia YOLO después: ___ms
```

#### ✅ Checklist:
- [ ] Localizar método `process_frame()`
- [ ] Agregar pinned memory en tensor creation
- [ ] Implementar non_blocking transfer
- [ ] Verificar que image_size = 640
- [ ] Agregar logging de optimizaciones (opcional)
- [ ] Validar que detecciones siguen funcionando
- [ ] Medir FPS antes/después

---

### 3️⃣ Modificar `src/core/vision/depth_estimator.py`

**Objetivo**: Implementar GPU pinned memory y non-blocking transfers

#### Localización del Código:
- Métodos: `_run_midas()` y `_run_depth_anything()` (~líneas 200-250)
- Clase: `DepthEstimator`

#### Código de Implementación:

```python
# src/core/vision/depth_estimator.py

# MODIFICAR _run_depth_anything() (o _run_midas() si lo usas)

def _run_depth_anything(self, frame: np.ndarray) -> np.ndarray:
    """
    FASE 1: Optimizado con pinned memory y async transfer
    """
    
    # Preprocess (sin cambios)
    frame_resized = cv2.resize(frame, (self.depth_size, self.depth_size))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    # Processor de HuggingFace (sin cambios)
    inputs = self.processor(images=frame_rgb, return_tensors="pt")
    
    # ========== FASE 1: Async Transfer ==========
    pixel_values = inputs['pixel_values'].to(
        self.device,
        non_blocking=getattr(self.config, 'NON_BLOCKING_TRANSFER', False)
    )
    # ============================================
    
    # Inferencia (sin cambios)
    with torch.no_grad():
        outputs = self.model(pixel_values)
        depth = outputs.predicted_depth
    
    # Postprocessing (sin cambios)
    depth_map = depth.squeeze().cpu().numpy()
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    
    return depth_normalized.astype(np.uint8)
```

#### Instrucciones de Implementación:
1. Abrir `src/core/vision/depth_estimator.py`
2. Localizar `_run_depth_anything()` (o el método que uses)
3. Buscar donde haces `.to(self.device)`
4. Agregar parámetro `non_blocking=...`
5. Verificar que `self.depth_size` usa `config.DEPTH_INPUT_SIZE` (384)

#### 📝 Notas de Implementación:

```
[Espacio para notas durante implementación]

Particularidades de depth estimation:
- 
- 

Problemas encontrados:
- 
- 

Performance observada:
- Tiempo inferencia antes: ___ms
- Tiempo inferencia después: ___ms
- Depth maps siguen siendo correctos: ✅ / ❌
```

#### ✅ Checklist:
- [ ] Localizar métodos de inferencia
- [ ] Implementar non_blocking transfer
- [ ] Verificar que depth_size = 384
- [ ] Validar que depth maps son correctos
- [ ] Medir latencia antes/después
- [ ] Confirmar que DEPTH_SKIP_FRAMES = 6 funciona

---

### 4️⃣ Modificar `src/core/navigation/navigation_pipeline.py`

**Objetivo**: Implementar CUDA streams para paralelización GPU

⚠️ **CAMBIO IMPORTANTE**: Ya NO es opcional - es obligatorio en FASE 1

**Razón**: Implementación trivial (8 líneas) con ganancia garantizada (+10-15% FPS)

#### Concepto de CUDA Streams:

```
SIN Streams (secuencial):
Frame 1: [YOLO: 10ms] → [Depth: 50ms] = 60ms total

CON Streams (paralelo):
Frame 1: [YOLO: 10ms  ]
         [Depth: 50ms ] ← Al mismo tiempo!
         Total: 50ms (¡15% más rápido!)
```

#### Código de Implementación:

```python
# src/core/navigation/navigation_pipeline.py

class NavigationPipeline:
    def __init__(self, observer, config):
        # ... código existente ...
        
        # ========== FASE 1: CUDA Streams ==========
        if torch.cuda.is_available() and getattr(config, 'CUDA_STREAMS', True):
            self.yolo_stream = torch.cuda.Stream()
            self.depth_stream = torch.cuda.Stream()
            self.use_streams = True
            log.info("✓ CUDA streams creados (YOLO + Depth)")
        else:
            self.use_streams = False
            log.warning("⚠️ CUDA streams deshabilitados")
        # ==========================================
        
        self.frame_count = 0
    
    def run(self):
        """Loop principal con CUDA streams"""
        
        self.observer.start()
        fps_counter = FPSCounter()
        
        try:
            while self.running:
                frame = self.observer.get_latest_frame('rgb')
                if frame is None:
                    time.sleep(0.001)
                    continue
                
                # ========== FASE 1: Parallel Execution ==========
                if self.use_streams:
                    # YOLO en stream 1 (GPU compute units A)
                    with torch.cuda.stream(self.yolo_stream):
                        detections = self.yolo_processor.process_frame(frame)
                    
                    # Depth en stream 2 (GPU compute units B)
                    depth_map = None
                    if self.frame_count % self.config.DEPTH_SKIP_FRAMES == 0:
                        with torch.cuda.stream(self.depth_stream):
                            depth_map = self.depth_estimator.estimate(frame)
                    
                    # Sincronizar ambos streams antes de continuar
                    torch.cuda.synchronize()
                else:
                    # Fallback secuencial (si CUDA no disponible)
                    detections = self.yolo_processor.process_frame(frame)
                    depth_map = None
                    if self.frame_count % self.config.DEPTH_SKIP_FRAMES == 0:
                        depth_map = self.depth_estimator.estimate(frame)
                # ================================================
                
                # Resto del pipeline (rendering, audio, etc)
                # ... tu código existente ...
                
                fps_counter.update()
                self.frame_count += 1
                
                # Stats cada 60 frames
                if self.frame_count % 60 == 0:
                    log.info(f"Frame {self.frame_count} | FPS: {fps_counter.fps:.1f}")
        
        except KeyboardInterrupt:
            log.info("\n=== Stopping Pipeline ===")
        finally:
            self.observer.stop()
```

#### Instrucciones de Implementación:
1. Abrir `src/core/navigation/navigation_pipeline.py`
2. En `__init__()`, agregar creación de streams (después de crear processors)
3. En `run()` o método de procesamiento, envolver YOLO y Depth con streams
4. Agregar `torch.cuda.synchronize()` después de ambos streams
5. Mantener fallback secuencial por si acaso

#### 📝 Notas de Implementación:

```
[Espacio para notas durante implementación]

Complejidad encontrada:
- 
- 

Bugs/problemas:
- 
- 

Mejora de performance:
- FPS sin streams: ___
- FPS con streams: ___
- Ganancia: ___%

Validaciones:
- [ ] Detecciones YOLO correctas
- [ ] Depth maps correctos
- [ ] No hay race conditions
- [ ] torch.cuda.synchronize() funciona
```

#### ✅ Checklist:
- [ ] Crear CUDA streams en __init__
- [ ] Envolver YOLO en yolo_stream
- [ ] Envolver Depth en depth_stream
- [ ] Agregar torch.cuda.synchronize()
- [ ] Validar correctitud de resultados
- [ ] Medir speedup (debe ser +10-15%)
- [ ] Mantener fallback secuencial

---

## 📈 Testing y Validación

### Benchmark con MockObserver

**Propósito**: Medir performance sin depender de gafas físicas

#### Script de Benchmark:

```python
# benchmarks/benchmark_fase1_mock.py

from src.core.mocks.mock_observer import MockObserver
from src.core.navigation.navigation_pipeline import NavigationPipeline
from src.utils.config import Config
import time
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def run_benchmark(duration_seconds=60, label="FASE 1"):
    """Ejecutar benchmark por X segundos con MockObserver"""
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {label}")
    print(f"{'='*60}\n")
    
    # Setup
    config = Config()
    observer = MockObserver(mode='synthetic', fps=30, resolution=(1408, 1408))
    pipeline = NavigationPipeline(observer, config)
    
    # Warmup (10s)
    print("[1/3] Warmup (10s)...")
    observer.start()
    time.sleep(10)
    
    # Benchmark
    print(f"[2/3] Benchmark ({duration_seconds}s)...")
    
    frame_count = 0
    latencies = []
    yolo_latencies = []
    depth_latencies = []
    start_time = time.time()
    
    for _ in range(duration_seconds * 30):  # ~30 FPS target
        frame = observer.get_latest_frame('rgb')
        if frame is None:
            time.sleep(0.001)
            continue
        
        frame_start = time.perf_counter()
        
        # YOLO
        yolo_start = time.perf_counter()
        # detections = pipeline.yolo_processor.process_frame(frame)
        yolo_time = (time.perf_counter() - yolo_start) * 1000
        yolo_latencies.append(yolo_time)
        
        # Depth (cada N frames)
        if frame_count % config.DEPTH_SKIP_FRAMES == 0:
            depth_start = time.perf_counter()
            # depth_map = pipeline.depth_estimator.estimate(frame)
            depth_time = (time.perf_counter() - depth_start) * 1000
            depth_latencies.append(depth_time)
        
        latency = (time.perf_counter() - frame_start) * 1000
        latencies.append(latency)
        frame_count += 1
        
        time.sleep(1/30)  # Throttle
    
    elapsed = time.time() - start_time
    
    # Resultados
    print(f"\n[3/3] Resultados:")
    print(f"  Frames procesados:  {frame_count}")
    print(f"  Tiempo total:       {elapsed:.2f}s")
    print(f"  FPS promedio:       {frame_count / elapsed:.1f}")
    print(f"  Latency media:      {np.mean(latencies):.1f}ms")
    print(f"  Latency p95:        {np.percentile(latencies, 95):.1f}ms")
    print(f"  Latency p99:        {np.percentile(latencies, 99):.1f}ms")
    
    if yolo_latencies:
        print(f"\n  YOLO latency:       {np.mean(yolo_latencies):.1f}ms")
    if depth_latencies:
        print(f"  Depth latency:      {np.mean(depth_latencies):.1f}ms")
    
    observer_stats = observer.get_stats()
    print(f"\nObserver (Mock):")
    print(f"  Generados:          {observer_stats['frames_generated']}")
    print(f"  Observer FPS:       {observer_stats['actual_fps']:.1f}")
    
    observer.stop()
    
    return {
        'fps': frame_count / elapsed,
        'latency_mean': np.mean(latencies),
        'latency_p95': np.percentile(latencies, 95),
        'latency_p99': np.percentile(latencies, 99),
        'yolo_latency': np.mean(yolo_latencies) if yolo_latencies else 0,
        'depth_latency': np.mean(depth_latencies) if depth_latencies else 0,
    }

if __name__ == '__main__':
    # Ejecutar benchmark
    stats = run_benchmark(60, label="FASE 1 - Post Optimizations")
    
    # Guardar resultados (agregar al documento)
    print(f"\n{'='*60}")
    print("📋 RESULTADOS PARA NOTION:")
    print(f"{'='*60}")
    print(f"FPS: {stats['fps']:.1f}")
    print(f"YOLO latency: {stats['yolo_latency']:.1f}ms")
    print(f"Depth latency: {stats['depth_latency']:.1f}ms")
    print(f"Total pipeline (p95): {stats['latency_p95']:.1f}ms")
    print(f"GPU memory usage: [Verificar manualmente con nvidia-smi]")
```

#### Cómo Ejecutar:
```bash
# Desde raíz del proyecto
python benchmarks/benchmark_fase1_mock.py
```

#### 📝 Resultados de Benchmarks:

**Baseline (antes de FASE 1):**
```
FPS: 13.6
YOLO latency: ___ms
Depth latency: ___ms
Total pipeline: ___ms
GPU memory usage: ___MB
```

**Después de config.py (Paso 1):**
```
FPS: ___
YOLO latency: ___ms
Depth latency: ___ms
Total pipeline: ___ms
GPU memory usage: ___MB
Notas: 
- Resoluciones aumentadas
- CUDA opts habilitados
```

**Después de yolo_processor.py (Paso 2):**
```
FPS: ___
YOLO latency: ___ms (esperado: -20% vs baseline)
Depth latency: ___ms
Total pipeline: ___ms
GPU memory usage: ___MB
Notas: 
- Pinned memory activo
- Non-blocking transfers
```

**Después de depth_estimator.py (Paso 3):**
```
FPS: ___
YOLO latency: ___ms
Depth latency: ___ms (esperado: -15% vs baseline)
Total pipeline: ___ms
GPU memory usage: ___MB
Notas: 
- Depth también optimizado
```

**Después de CUDA streams (Paso 4):**
```
FPS: ___ (esperado: 20-25 FPS)
YOLO latency: ___ms
Depth latency: ___ms
Total pipeline: ___ms (esperado: -10-15% vs paso 3)
GPU memory usage: ___MB
Notas: 
- Paralelización funcionando
- Speedup total FASE 1: ___%
```

---

## 🎯 Métricas de Éxito

### Objetivos Cuantitativos:
- [ ] FPS ≥ 20: ❌ / ✅ (actual: ___)
- [ ] FPS ≥ 25: ❌ / ✅ (actual: ___)
- [ ] GPU memory < 5GB: ❌ / ✅ (actual: ___GB)
- [ ] YOLO latency < 15ms: ❌ / ✅ (actual: ___ms)
- [ ] Depth latency < 60ms: ❌ / ✅ (actual: ___ms)
- [ ] Total pipeline < 45ms: ❌ / ✅ (actual: ___ms)

### Objetivos Cualitativos:
- [ ] Detecciones YOLO mantienen calidad con resolución 640
- [ ] Depth maps mantienen calidad con resolución 384
- [ ] Sistema estable sin memory leaks
- [ ] No hay frame drops significativos
- [ ] CUDA streams no causan race conditions

---

## 🐛 Debugging Log

### Problemas Encontrados:

**Problema 1:**
```
Descripción: 
Causa: 
Solución: 
Fecha: 
```

**Problema 2:**
```
Descripción: 
Causa: 
Solución: 
Fecha: 
```

**Problema 3:**
```
Descripción: 
Causa: 
Solución: 
Fecha: 
```

---

## 📝 Notas Generales

### Observaciones Durante Implementación:

```
[Espacio libre para notas generales]

Fecha: 
Observación: 

---

Fecha: 
Observación: 

---

Fecha: 
Observación: 

```

### Decisiones Técnicas:

```
Decisión: Incluir CUDA Streams en FASE 1 (no opcional)
Razón: Implementación trivial (8 líneas) con ganancia +10-15% garantizada
Resultado: [A completar después de implementación]

---

Decisión: DEPTH_SKIP_FRAMES = 6
Razón: Balance entre calidad depth y performance general
Resultado: [A completar]

---

Decisión: 
Razón: 
Resultado: 

```

---

## ✅ Resumen Final FASE 1

**Estado**: 🔄 En Progreso / ✅ Completado / ❌ Bloqueado

**FPS Alcanzado**: ___ FPS

**Mejora sobre Baseline**: ___% (objetivo: 50-80%)

**Cambios Implementados**:
- [ ] config.py modificado (resoluciones + CUDA opts)
- [ ] yolo_processor.py optimizado (pinned memory)
- [ ] depth_estimator.py optimizado (pinned memory)
- [ ] navigation_pipeline.py (CUDA streams)

**¿Pasamos a FASE 2?**: ❌ / ✅

**Razón**:
```
[Explicar por qué sí o por qué no]

Si FPS ≥ 20: ✅ FASE 1 exitosa, podemos continuar a FASE 2
Si FPS < 20: Revisar problemas antes de avanzar

Próximo objetivo FASE 2: 30-35 FPS

```

---

## 🚀 Próximos Pasos

### Si FASE 1 exitosa (FPS ≥ 20):

**FASE 2 - Batch Processing & Model Optimization**
- **Objetivo**: 30-35 FPS
- **Técnicas**: 
  - Batch processing de YOLO (procesar múltiples frames)
  - FP16 precision (half precision)
  - Model pruning
- **Riesgo**: Medio
- **Tiempo estimado**: 2-3 días

### Si FASE 1 no alcanza objetivo (FPS < 20):

**Debugging prioritario**:
1. Verificar que CUDA está disponible: `torch.cuda.is_available()`
2. Confirmar que pinned memory funciona: `tensor.is_pinned()`
3. Validar que streams se crearon: `self.yolo_stream is not None`
4. Revisar logs de CUDA errors
5. Ejecutar `nvidia-smi` para ver uso GPU

---

## 📚 Referencias Útiles

### Documentación CUDA:
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [CUDA Streams Documentation](https://pytorch.org/docs/stable/notes/cuda.html#cuda-streams)

### Debugging:
```bash
# Verificar CUDA
nvidia-smi

# Verificar PyTorch + CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Monitor GPU en tiempo real
watch -n 1 nvidia-smi

# Profiling (si necesario)
python -m torch.utils.bottleneck benchmark_fase1_mock.py
```

---

## 🎯 Checklist General FASE 1

### Pre-implementación:
- [ ] Backup del código actual
- [ ] Git commit: `git commit -m "Baseline antes de FASE 1"`
- [ ] Verificar que MockObserver funciona
- [ ] Tener acceso a nvidia-smi

### Implementación:
- [ ] Paso 1: config.py
- [ ] Paso 2: yolo_processor.py
- [ ] Paso 3: depth_estimator.py
- [ ] Paso 4: navigation_pipeline.py
- [ ] Ejecutar benchmark después de cada paso

### Post-implementación:
- [ ] FPS ≥ 20 alcanzado
- [ ] Documentar resultados en Notion
- [ ] Git commit: `git commit -m "FASE 1 completada - XX FPS alcanzados"`
- [ ] Actualizar plan FASE 2
- [ ] Celebrar 🎉

---

**Última Actualización**: 2025-01-14  
**Responsable**: Roberto  
**Versión**: 2.0 (CUDA Streams obligatorio)  
**Review**: Claude 4 Sonnet