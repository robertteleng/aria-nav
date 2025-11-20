# 🚀 Phase 6: Hybrid Multiprocessing + CUDA Streams

> **Objetivo:** Combinar multiprocessing (workers paralelos) con CUDA Streams (paralelización GPU en main process)  
> **Ganancia esperada:** 48ms → 40ms latency (+20% FPS: 18.4 → 25 FPS)  
> **Riesgo:** BAJO (VRAM: 1.5GB → 2.0GB de 6GB disponibles)  
> **Tiempo estimado:** 4-6 horas

---

## 📋 Paso a Paso (Implementación Completa)

### Paso 1: Modificar `navigation_pipeline.py` (30 min)

**Ubicación:** `src/core/navigation/navigation_pipeline.py`

**Cambio:** Permitir CUDA Streams en main process incluso con multiprocessing activo

```python
# ANTES (líneas 79-92)
if not self.multiproc_enabled:
    self.use_cuda_streams = getattr(Config, 'CUDA_STREAMS', False) and torch.cuda.is_available()
    if self.use_cuda_streams:
        self.yolo_stream = torch.cuda.Stream()
        self.depth_stream = torch.cuda.Stream()
        print("[INFO] ✓ CUDA streams habilitados (YOLO + Depth en paralelo)")
    else:
        self.yolo_stream = None
        self.depth_stream = None
        if torch.cuda.is_available():
            print("[INFO] ⚠️ CUDA streams deshabilitados (ejecución secuencial)")
else:
    self.use_cuda_streams = False
    print("[INFO] 🔄 Multiprocessing mode - GPU work handled by workers")

# DESPUÉS (NUEVO)
# Determinar si usar CUDA Streams
self.camera_id = getattr(Config, 'CAMERA_ID', 'rgb')  # Identificar cámara
enable_streams = getattr(Config, 'CUDA_STREAMS', False) and torch.cuda.is_available()

# PHASE 6: Hybrid mode - Streams en main process, multiproc para workers
if self.multiproc_enabled:
    # Multiprocessing activo
    if self.camera_id == 'rgb':
        # Main process (RGB): SÍ usar streams
        self.use_cuda_streams = enable_streams
        if self.use_cuda_streams:
            self.yolo_stream = torch.cuda.Stream()
            self.depth_stream = torch.cuda.Stream()
            print("[INFO] ✅ PHASE 6: Hybrid mode - CUDA streams in main process")
            print("[INFO]    → Depth + YOLO parallel on RGB camera")
        else:
            self.use_cuda_streams = False
            print("[INFO] 🔄 Main process: Sequential (streams disabled)")
    else:
        # Workers (SLAM1/SLAM2): NO usar streams (solo YOLO, nada que paralelizar)
        self.use_cuda_streams = False
        print(f"[INFO] 🔄 Worker {self.camera_id}: Sequential (YOLO-only, no streams)")
else:
    # Single-process mode (original behavior)
    self.use_cuda_streams = enable_streams
    if self.use_cuda_streams:
        self.yolo_stream = torch.cuda.Stream()
        self.depth_stream = torch.cuda.Stream()
        print("[INFO] ✓ CUDA streams habilitados (YOLO + Depth en paralelo)")
    else:
        self.yolo_stream = None
        self.depth_stream = None
        if torch.cuda.is_available():
            print("[INFO] ⚠️ CUDA streams deshabilitados (ejecución secuencial)")
```

**Líneas a cambiar:** 79-102 (reemplazar bloque completo)

---

### Paso 2: Añadir `CAMERA_ID` al pipeline (15 min)

**Ubicación:** `src/core/navigation/navigation_pipeline.py`

**Cambio:** Añadir parámetro `camera_id` al constructor

```python
# ANTES (línea ~48)
def __init__(
    self,
    yolo_processor,
    depth_estimator=None,
    image_enhancer=None,
):

# DESPUÉS (NUEVO)
def __init__(
    self,
    yolo_processor,
    depth_estimator=None,
    image_enhancer=None,
    camera_id: str = 'rgb',  # ← AÑADIR ESTO
):
    self.camera_id = camera_id  # ← AÑADIR ESTO
```

**Líneas a cambiar:** ~48-53

---

### Paso 3: Pasar `camera_id` desde workers (20 min)

**Ubicación:** `src/core/vision/slam_detection_worker.py`

**Cambio:** Propagar `camera_id` al crear el pipeline en workers

```python
# ANTES (línea ~85, método _worker_loop)
def _worker_loop(self, model_path: str):
    """Worker process loop - loads model and processes frames"""
    # Setup CUDA context for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(0.25, device=0)
    
    # Load YOLO model in worker
    from core.vision.yolo_processor import YoloProcessor
    yolo_proc = YoloProcessor(model_path)
    
    # Load pipeline
    from core.navigation.navigation_pipeline import NavigationPipeline
    pipeline = NavigationPipeline(
        yolo_processor=yolo_proc,
        depth_estimator=None  # SLAM cameras don't use depth
    )

# DESPUÉS (NUEVO)
def _worker_loop(self, model_path: str):
    """Worker process loop - loads model and processes frames"""
    # Setup CUDA context for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(0.25, device=0)
    
    # Load YOLO model in worker
    from core.vision.yolo_processor import YoloProcessor
    yolo_proc = YoloProcessor(model_path)
    
    # Load pipeline
    from core.navigation.navigation_pipeline import NavigationPipeline
    pipeline = NavigationPipeline(
        yolo_processor=yolo_proc,
        depth_estimator=None,  # SLAM cameras don't use depth
        camera_id=self.camera_id  # ← AÑADIR ESTO (propaga el ID)
    )
```

**Líneas a cambiar:** ~85-100

---

### Paso 4: Actualizar `Config` (5 min)

**Ubicación:** `src/utils/config.py`

**Cambio:** Añadir flag para Phase 6

```python
# ANTES (línea ~67)
self.CUDA_STREAMS = True  # OBLIGATORIO en FASE 1

# DESPUÉS (NUEVO - añadir después)
self.CUDA_STREAMS = True  # OBLIGATORIO en FASE 1
self.PHASE6_HYBRID_STREAMS = True  # PHASE 6: Streams en main, multiproc en workers
```

**Líneas a cambiar:** ~67 (añadir línea nueva)

---

### Paso 5: Testing básico (30 min)

**Script de prueba:** `test_phase6_hybrid.py`

```python
#!/usr/bin/env python3
"""
Test Phase 6: Hybrid Multiprocessing + CUDA Streams
Verifica que streams funcionen en main y workers sigan sin ellos
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_hybrid():
    print("=" * 60)
    print("🧪 PHASE 6 HYBRID TEST")
    print("=" * 60)
    
    # Verificar configuración
    from utils.config import Config
    config = Config()
    
    print(f"\n📊 Configuración:")
    print(f"  PHASE2_MULTIPROC_ENABLED: {getattr(Config, 'PHASE2_MULTIPROC_ENABLED', False)}")
    print(f"  CUDA_STREAMS: {config.CUDA_STREAMS}")
    print(f"  PHASE6_HYBRID_STREAMS: {getattr(config, 'PHASE6_HYBRID_STREAMS', False)}")
    
    # Simular main process (RGB)
    print(f"\n🎥 Simulando Main Process (RGB camera):")
    from core.navigation.navigation_pipeline import NavigationPipeline
    from core.vision.yolo_processor import YoloProcessor
    from core.vision.depth_estimator import DepthEstimator
    
    yolo = YoloProcessor("checkpoints/yolo12n.engine")
    depth = DepthEstimator("checkpoints/depth_anything_v2_vits.onnx")
    
    pipeline_main = NavigationPipeline(
        yolo_processor=yolo,
        depth_estimator=depth,
        camera_id='rgb'
    )
    
    print(f"  ✓ Pipeline creado")
    print(f"  → use_cuda_streams: {pipeline_main.use_cuda_streams}")
    print(f"  → Expected: True (streams en main)")
    
    if pipeline_main.use_cuda_streams:
        print(f"  ✅ CUDA Streams ACTIVOS en main process")
    else:
        print(f"  ⚠️ WARNING: Streams NO activos (esperado: True)")
    
    # Simular worker (SLAM)
    print(f"\n🎥 Simulando Worker Process (SLAM camera):")
    yolo_slam = YoloProcessor("checkpoints/yolo12n_slam256.engine")
    
    pipeline_worker = NavigationPipeline(
        yolo_processor=yolo_slam,
        depth_estimator=None,
        camera_id='slam1'
    )
    
    print(f"  ✓ Pipeline worker creado")
    print(f"  → use_cuda_streams: {pipeline_worker.use_cuda_streams}")
    print(f"  → Expected: False (no streams en workers)")
    
    if not pipeline_worker.use_cuda_streams:
        print(f"  ✅ Streams correctamente DESACTIVADOS en worker")
    else:
        print(f"  ⚠️ WARNING: Streams activos (esperado: False)")
    
    print(f"\n" + "=" * 60)
    print(f"✅ Test completado")
    print(f"=" * 60)

if __name__ == "__main__":
    test_hybrid()
```

**Ejecutar:**
```bash
cd /home/roberto/Projects/aria-nav
python test_phase6_hybrid.py
```

**Output esperado:**
```
============================================================
🧪 PHASE 6 HYBRID TEST
============================================================

📊 Configuración:
  PHASE2_MULTIPROC_ENABLED: True
  CUDA_STREAMS: True
  PHASE6_HYBRID_STREAMS: True

🎥 Simulando Main Process (RGB camera):
[INFO] ✅ PHASE 6: Hybrid mode - CUDA streams in main process
[INFO]    → Depth + YOLO parallel on RGB camera
  ✓ Pipeline creado
  → use_cuda_streams: True
  → Expected: True (streams en main)
  ✅ CUDA Streams ACTIVOS en main process

🎥 Simulando Worker Process (SLAM camera):
[INFO] 🔄 Worker slam1: Sequential (YOLO-only, no streams)
  ✓ Pipeline worker creado
  → use_cuda_streams: False
  → Expected: False (no streams en workers)
  ✅ Streams correctamente DESACTIVADOS en worker

============================================================
✅ Test completado
============================================================
```

---

### Paso 6: Benchmark de performance (1 hora)

**Script:** `benchmark_phase6.py`

```python
#!/usr/bin/env python3
"""
Benchmark Phase 6: Medir ganancia de hybrid mode
Compara performance con/sin streams en main process
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import time
import numpy as np
import torch

def benchmark_main_process(use_streams: bool, num_runs: int = 100):
    """Benchmark main process con/sin streams"""
    from core.navigation.navigation_pipeline import NavigationPipeline
    from core.vision.yolo_processor import YoloProcessor
    from core.vision.depth_estimator import DepthEstimator
    from utils.config import Config
    
    # Override config
    Config.CUDA_STREAMS = use_streams
    
    print(f"\n{'='*60}")
    print(f"🎯 Benchmark: {'WITH' if use_streams else 'WITHOUT'} CUDA Streams")
    print(f"{'='*60}")
    
    # Create pipeline
    yolo = YoloProcessor("checkpoints/yolo12n.engine")
    depth = DepthEstimator("checkpoints/depth_anything_v2_vits.onnx")
    
    pipeline = NavigationPipeline(
        yolo_processor=yolo,
        depth_estimator=depth,
        camera_id='rgb'
    )
    
    print(f"Pipeline initialized: use_cuda_streams={pipeline.use_cuda_streams}")
    
    # Dummy frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Warmup
    print(f"Warming up ({10} iterations)...")
    for _ in range(10):
        result = pipeline.process_frame(frame, profile=False)
    
    # Benchmark
    print(f"Benchmarking ({num_runs} iterations)...")
    times = []
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.time()
    for i in range(num_runs):
        iter_start = time.time()
        result = pipeline.process_frame(frame, profile=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        iter_time = (time.time() - iter_start) * 1000  # ms
        times.append(iter_time)
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{num_runs} ({(i+1)/num_runs*100:.0f}%)")
    
    elapsed = time.time() - start
    
    # Stats
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1000 / avg_time
    
    print(f"\n📊 Results:")
    print(f"  Average time: {avg_time:.2f}ms (±{std_time:.2f}ms)")
    print(f"  Min/Max: {min_time:.2f}ms / {max_time:.2f}ms")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total time: {elapsed:.2f}s")
    
    return avg_time, fps

def main():
    print("="*60)
    print("🚀 PHASE 6 PERFORMANCE BENCHMARK")
    print("="*60)
    
    # Test WITHOUT streams (current)
    time_without, fps_without = benchmark_main_process(use_streams=False, num_runs=50)
    
    # Test WITH streams (Phase 6)
    time_with, fps_with = benchmark_main_process(use_streams=True, num_runs=50)
    
    # Comparison
    print(f"\n{'='*60}")
    print(f"📈 COMPARISON")
    print(f"{'='*60}")
    print(f"WITHOUT Streams: {time_without:.2f}ms ({fps_without:.2f} FPS)")
    print(f"WITH Streams:    {time_with:.2f}ms ({fps_with:.2f} FPS)")
    print(f"")
    print(f"Improvement: {time_without - time_with:.2f}ms saved ({(1-time_with/time_without)*100:.1f}% faster)")
    print(f"FPS gain: {fps_with - fps_without:.2f} FPS (+{(fps_with/fps_without-1)*100:.1f}%)")
    print(f"{'='*60}")
    
    # Check VRAM
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n💾 VRAM Usage: {vram_used:.2f}GB / {vram_total:.2f}GB ({vram_used/vram_total*100:.1f}%)")
    
    print(f"\n✅ Benchmark complete!")

if __name__ == "__main__":
    main()
```

**Ejecutar:**
```bash
cd /home/roberto/Projects/aria-nav
python benchmark_phase6.py
```

**Output esperado:**
```
============================================================
🚀 PHASE 6 PERFORMANCE BENCHMARK
============================================================

============================================================
🎯 Benchmark: WITHOUT CUDA Streams
============================================================
Pipeline initialized: use_cuda_streams=False
Warming up (10 iterations)...
Benchmarking (50 iterations)...
  Progress: 20/50 (40%)
  Progress: 40/50 (80%)

📊 Results:
  Average time: 67.23ms (±2.15ms)
  Min/Max: 64.12ms / 72.45ms
  FPS: 14.88
  Total time: 3.42s

============================================================
🎯 Benchmark: WITH CUDA Streams
============================================================
Pipeline initialized: use_cuda_streams=True
Warming up (10 iterations)...
Benchmarking (50 iterations)...
  Progress: 20/50 (40%)
  Progress: 40/50 (80%)

📊 Results:
  Average time: 40.15ms (±1.87ms)
  Min/Max: 38.23ms / 44.67ms
  FPS: 24.91
  Total time: 2.05s

============================================================
📈 COMPARISON
============================================================
WITHOUT Streams: 67.23ms (14.88 FPS)
WITH Streams:    40.15ms (24.91 FPS)

Improvement: 27.08ms saved (40.3% faster)
FPS gain: 10.03 FPS (+67.4%)
============================================================

💾 VRAM Usage: 2.03GB / 6.00GB (33.8%)

✅ Benchmark complete!
```

---

### Paso 7: Validación en sistema completo (1 hora)

**Ejecutar sistema completo:**
```bash
python run.py
```

**Verificar logs:**
```bash
# Buscar mensajes de Phase 6
tail -100 logs/session_*/system.jsonl | grep -i "phase 6\|hybrid\|streams"
```

**Output esperado en logs:**
```
[INFO] ✅ PHASE 6: Hybrid mode - CUDA streams in main process
[INFO]    → Depth + YOLO parallel on RGB camera
[INFO] 🔄 Worker slam-left: Sequential (YOLO-only, no streams)
[INFO] 🔄 Worker slam-right: Sequential (YOLO-only, no streams)
```

**Verificar VRAM:**
```bash
watch -n 1 nvidia-smi
```

Debería mostrar ~2.0GB usado (vs 1.5GB actual).

---

## 📊 Checklist de Validación

### ✅ Antes de commitear:

- [ ] Test unitario pasa (`test_phase6_hybrid.py`)
- [ ] Benchmark muestra mejora 15-20% (`benchmark_phase6.py`)
- [ ] Sistema completo arranca sin errores
- [ ] VRAM usage < 2.5GB (safe margin)
- [ ] Logs muestran "PHASE 6: Hybrid mode"
- [ ] Workers no usan streams (logs confirman)
- [ ] FPS aumenta de 18.4 a ~25 FPS

### ⚠️ Red flags (rollback si ocurre):

- [ ] VRAM usage > 3.0GB (memory leak)
- [ ] Crashes en workers
- [ ] Performance empeora (streams mal implementados)
- [ ] GPU usage baja (streams no funcionando)

---

## 🔧 Troubleshooting

### Problema 1: Streams no se activan en main

**Síntoma:**
```
[INFO] 🔄 Main process: Sequential (streams disabled)
```

**Diagnóstico:**
```python
# En navigation_pipeline.py __init__
print(f"[DEBUG] multiproc_enabled: {self.multiproc_enabled}")
print(f"[DEBUG] camera_id: {self.camera_id}")
print(f"[DEBUG] enable_streams: {enable_streams}")
```

**Solución:** Verificar que `camera_id=='rgb'` y `Config.CUDA_STREAMS==True`

### Problema 2: Workers crashean

**Síntoma:**
```
RuntimeError: CUDA error in worker process
```

**Solución:**
```python
# En slam_detection_worker.py, asegurar:
torch.cuda.set_per_process_memory_fraction(0.2, device=0)  # Reducir a 20%
```

### Problema 3: VRAM overflow

**Síntoma:**
```
CUDA out of memory
```

**Solución temporal:**
```python
# Deshabilitar streams
Config.PHASE6_HYBRID_STREAMS = False
```

---

## 📈 Expected Results

### Before Phase 6 (Current):
```
Main process: 67ms sequential (Depth + YOLO)
FPS: 18.4
VRAM: 1.5GB / 6GB
```

### After Phase 6 (Hybrid):
```
Main process: 40ms parallel (Depth || YOLO)
FPS: ~25
VRAM: 2.0GB / 6GB
Improvement: +36% FPS, +0.5GB VRAM
```

---

## 🎯 Next Steps After Phase 6

Si Phase 6 funciona bien:
1. Monitorear VRAM en sesiones largas (memory leaks?)
2. Considerar añadir depth a SLAM cameras (más beneficio de streams)
3. Experimentar con batch processing
4. Target: 30 FPS con optimizaciones adicionales

---

**¿Preguntas antes de empezar la implementación?**
