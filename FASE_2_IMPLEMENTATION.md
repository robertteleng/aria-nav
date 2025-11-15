---
title: Fase 2 - Multiprocesamiento RTX 2060 (CORREGIDO)
date: 2025-11-15
---

# Fase 2 Implementation Plan

## Executive Summary
- Aprovechar la RTX 2060 para alcanzar **35-40 FPS** rompiendo el procesamiento secuencial.
- Convertir `NavigationPipeline` en un orquestador multiproceso con **2 workers GPU** (central + SLAM compartido).
- Sistema con **3 cámaras**: Central (RGB) → Depth + YOLO | SLAM1/SLAM2 (laterales) → YOLO compartido.
- Mantener compatibilidad con modo legacy, métricas, backpressure, shutdown controlado y monitoreo sistemático.

---

## 1. Estado Actual del Código

### 1.1 Componentes principales

**`NavigationPipeline`** (`navigation_pipeline.py`):
- Ejecuta enhancement (CPU) → depth (GPU) → YOLO (GPU) en el mismo proceso.
- Usa `torch.cuda.Stream` para paralelizar depth/YOLO en cámara central.
- Conserva `latest_depth_map` para frame skipping.
- **Solo procesa cámara CENTRAL**, no laterales.

**`YoloProcessor`** (`yolo_processor.py`):
- Carga `yolo12n.pt` con pinned memory/non-blocking.
- Método: `process_frame(frame, depth_map, depth_raw)` → detecciones.
- No es thread-safe; cada proceso debe instanciar su propio `YoloProcessor`.

**`DepthEstimator`** (`depth_estimator.py`):
- Backend configurable: MiDaS o Depth Anything V2.
- Métodos: `estimate_depth()` y `estimate_depth_with_details()`.
- Cada instancia carga modelo completo en GPU (~2.5-3 GB VRAM).

**`ImageEnhancer`** (`image_enhancer.py`):
- Opcional (configurable).
- Usa OpenCV CLAHE + gamma correction en CPU.
- No accede GPU.

**`SlamDetectionWorker`** (`slam_detection_worker.py`):
- **YA EXISTE** procesamiento asíncrono de cámaras SLAM1/SLAM2.
- Usa `threading.Thread` (bloqueado por GIL) - **PROBLEMA**.
- Target FPS: 8 por cámara.
- Procesa frames en worker thread separado.

**Captura**:
- `main.py` obtiene frames via `observer.get_latest_frame()` y `observer.get_slam_frames()`.
- `MockObserver` genera frames sintéticos/video/estáticos.
- **Actualmente NO hay multiprocesamiento en captura**.

**`Config`**:
- Flags existentes: `CUDA_STREAMS`, `PINNED_MEMORY`, `DEPTH_ENABLED`, `DEPTH_INPUT_SIZE`, `DEPTH_FRAME_SKIP`, `YOLO_MODEL`.
- **Faltan**: `PHASE2_MULTIPROC_ENABLED`, `PHASE2_QUEUE_MAXSIZE`, `PHASE2_STATS_INTERVAL`, `PHASE2_BACKPRESSURE_TIMEOUT`.

### 1.2 Arquitectura de cámaras REAL

```
Sistema tiene 3 cámaras:
├── Central (RGB 1408×1408)
│   ├── Depth estimation (DepthEstimator)
│   └── YOLO detection (YoloProcessor)
├── SLAM1 (lateral izquierda, menor resolución)
│   └── YOLO detection (SlamDetectionWorker con threading)
└── SLAM2 (lateral derecha, menor resolución)
    └── YOLO detection (SlamDetectionWorker con threading)
```

---

## 2. Bottlenecks y Motivación Técnica

### 2.1 Problemas identificados

**Latencia por frame**:
- Central: **90-100 ms** cuando depth activo (depth 50-60ms + YOLO 40-45ms).
- Objetivo: **25 ms** (40 FPS).
- GPU subutilizada: procesamiento secuencial en un solo thread.

**GIL de Python**:
- `SlamDetectionWorker` usa `threading` → bloqueado por GIL.
- No puede procesar múltiples cámaras verdaderamente en paralelo.
- CUDA streams NO rompen el GIL.

**Tamaño de frames**:
- Central: 1408×1408 × 3 bytes = **~6 MB** por frame.
- Serialización (pickle) en `multiprocessing.Queue` es costosa.
- Solución: colas pequeñas (maxsize=2) + considerar shared memory si bottleneck.

### 2.2 Por qué CUDA streams no son suficientes

CUDA streams permiten overlap depth/YOLO **dentro de un proceso**, pero:
- No escalan para múltiples cámaras (SLAM1, SLAM2).
- No rompen el GIL de Python.
- Pipeline sigue atado a un solo thread principal.
- No permiten procesar frames N, N+1, N+2 simultáneamente en etapas diferentes.

**Conclusión**: Multiprocesamiento es necesario.

---

## 3. Arquitectura Multiproceso Propuesta

### 3.1 Estructura de procesos

| Proceso | Responsabilidad | Input | Output | Modelos GPU | VRAM | CUDA sync |
|---------|----------------|-------|--------|-------------|------|-----------|
| **Main Process** (Coordinator) | Captura 3 frames, enhancement central, distribución, fusión de resultados, visualización | Observer | Queues | Ninguno | 0 GB | N/A |
| **Central Worker** | Depth + YOLO para cámara central | `central_queue` | `result_queue` | DepthEstimator + YoloProcessor | ~5 GB | `torch.cuda.synchronize()` |
| **SLAM Worker** | YOLO para SLAM1 y SLAM2 (secuencial, mismo modelo) | `slam_queue` | `result_queue` | YoloProcessor (compartido) | ~2.5 GB | `torch.cuda.synchronize()` |

**TOTAL VRAM: 5-6 GB** ✅ (dentro del límite RTX 2060)

### 3.2 Flujo de datos

```
Main Process:
1. Captura frame_central, frame_slam1, frame_slam2
2. Enhancement en frame_central (CPU)
3. Encola:
   - central_queue.put(FrameMessage(frame_id, 'central', frame_central))
   - slam_queue.put(FrameMessage(frame_id, 'slam1', frame_slam1))
   - slam_queue.put(FrameMessage(frame_id, 'slam2', frame_slam2))
4. Espera 3 resultados en result_queue
5. Fusiona detecciones de 3 cámaras
6. Coordinator.process() + visualización

Central Worker (GPU Process):
1. msg = central_queue.get(timeout=1.0)
2. Con CUDA streams:
   - Stream 1: depth = depth_estimator.estimate_depth(msg.frame)
   - Stream 2: detections = yolo_processor.process_frame(msg.frame, depth)
3. torch.cuda.synchronize()
4. result_queue.put(ResultMessage('central', detections, depth_map, latency))

SLAM Worker (GPU Process):
1. msg = slam_queue.get(timeout=1.0)
2. detections = yolo_processor.process_frame(msg.frame)
3. torch.cuda.synchronize()
4. result_queue.put(ResultMessage(msg.camera, detections, None, latency))
5. Loop para procesar SLAM1 y SLAM2 secuencialmente
```

### 3.3 Comunicación entre procesos

**Colas (`torch.multiprocessing.Queue`)**:

```python
central_queue = mp.Queue(maxsize=2)   # Solo cámara central
slam_queue = mp.Queue(maxsize=4)      # 2 frames × 2 cámaras
result_queue = mp.Queue(maxsize=6)    # 2 frames × 3 cámaras
stop_event = mp.Event()               # Shutdown signal
```

**Estructura de mensajes**:

```python
# FrameMessage (input a workers)
{
    'frame_id': int,          # Monotonic counter para sincronización
    'camera': str,            # 'central', 'slam1', 'slam2'
    'frame': np.ndarray,      # Frame RGB/grayscale
    'timestamp': float        # time.time()
}

# ResultMessage (output de workers)
{
    'frame_id': int,
    'camera': str,
    'detections': List[dict],
    'depth_map': Optional[np.ndarray],  # Solo central
    'depth_raw': Optional[np.ndarray],  # Solo central
    'latency_ms': float,
    'profiling': {
        'depth_ms': float,      # Solo central
        'yolo_ms': float,
        'gpu_mem_mb': float
    }
}
```

**Políticas de timeout**:
- `Queue.put()`: timeout=0.1s → si llena, log warning + drop frame.
- `Queue.get()`: timeout=1.0s → si vacía, fallback a modo secuencial.
- Después de 3 timeouts consecutivos: desactiva `PHASE2_MULTIPROC_ENABLED`.

### 3.4 Sincronización y memoria

**Multiprocessing setup**:
```python
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)  # Contextos CUDA independientes
```

**GPU sync en cada worker**:
```python
torch.cuda.set_device(Config.DEVICE)  # Al inicio del worker
torch.cuda.synchronize()               # Después de cada inferencia
```

**Shared memory**:
- Inicialmente usar `mp.Queue` con pickle (simple).
- Si profiling muestra bottleneck en IPC: migrar a `multiprocessing.shared_memory.SharedMemory`.
- Benchmark antes de optimizar prematuramente.

### 3.5 Modelos GPU y gestión de VRAM

**Central Worker**:
```python
depth_estimator = DepthEstimator()      # ~2.5 GB
yolo_processor = YoloProcessor()        # ~2.5 GB
depth_stream = torch.cuda.Stream()
yolo_stream = torch.cuda.Stream()
# TOTAL: ~5 GB
```

**SLAM Worker**:
```python
yolo_processor = YoloProcessor.from_profile("slam")  # ~2.5 GB
# Procesa SLAM1 y SLAM2 secuencialmente con el MISMO modelo
# TOTAL: ~2.5 GB
```

**Total VRAM: 5-6 GB** ✅

**Estrategias si OOM**:
1. Reducir `DEPTH_INPUT_SIZE` (1408 → 1024).
2. Usar modelo YOLO más pequeño (yolo12n → yolo11n).
3. Frame skip más agresivo en SLAM (cada 3 frames).
4. `torch.cuda.empty_cache()` periódico en workers.

### 3.6 Manejo de errores y shutdown

**Error handling en workers**:
```python
def central_worker(central_queue, result_queue, stop_event):
    try:
        depth_estimator = DepthEstimator()
        yolo_processor = YoloProcessor()
        
        while not stop_event.is_set():
            try:
                msg = central_queue.get(timeout=1.0)
                # Procesamiento...
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Central worker error: {e}")
                result_queue.put({'error': True, 'camera': 'central'})
                
    except Exception as e:
        logging.critical(f"Central worker fatal: {e}")
        stop_event.set()  # Signal shutdown
    finally:
        torch.cuda.empty_cache()
```

**Shutdown sequence**:
```python
def shutdown(self):
    if hasattr(self, 'stop_event'):
        # 1. Signal all workers
        self.stop_event.set()
        
        # 2. Join with timeout
        for proc in self.processes:
            proc.join(timeout=2)
            if proc.is_alive():
                logging.warning(f"Force terminating {proc.name}")
                proc.terminate()
                proc.join(timeout=1)
        
        # 3. Drain queues
        for q in [self.central_queue, self.slam_queue, self.result_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except:
                    break
        
        logging.info("All workers shutdown complete")
```

**Fallback a modo secuencial**:
- Contador de timeouts consecutivos.
- Si >= 3: `Config.PHASE2_MULTIPROC_ENABLED = False` (runtime).
- Log warning y continúa con `_process_sequential()`.

### 3.7 Backpressure y drop policy

**Prioridades**:
1. **Central** (más importante): depth + YOLO.
2. **SLAM1/SLAM2**: contexto periférico, menos crítico.

**Políticas**:

```python
# En Main Process al encolar
try:
    central_queue.put(msg, timeout=0.1)
except queue.Full:
    logging.warning("Central queue full, dropping frame")
    metrics['dropped_central'] += 1

try:
    slam_queue.put(msg, timeout=0.05)  # Timeout más corto
except queue.Full:
    logging.debug("SLAM queue full, dropping frame")
    metrics['dropped_slam'] += 1
```

**En SLAM Worker**:
- Si `result_queue` llena por >0.5s → drop frame más antiguo de `slam_queue`.
- Prioriza frames recientes sobre antiguos.

---

## 4. Plan de Modificación Paso a Paso

### Fase 4.1: Config y tipos (Día 1)

**Archivo**: `utils/config.py`
```python
# Añadir flags
PHASE2_MULTIPROC_ENABLED = False
PHASE2_QUEUE_MAXSIZE = 2
PHASE2_SLAM_QUEUE_MAXSIZE = 4
PHASE2_RESULT_QUEUE_MAXSIZE = 6
PHASE2_STATS_INTERVAL = 5.0
PHASE2_BACKPRESSURE_TIMEOUT = 0.5
```

**Nuevo archivo**: `core/processing/multiproc_types.py`
```python
from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np

@dataclass
class FrameMessage:
    frame_id: int
    camera: str  # 'central', 'slam1', 'slam2'
    frame: np.ndarray
    timestamp: float

@dataclass
class ResultMessage:
    frame_id: int
    camera: str
    detections: List[dict]
    depth_map: Optional[np.ndarray]
    depth_raw: Optional[np.ndarray]
    latency_ms: float
    profiling: Dict[str, float]
```

### Fase 4.2: Workers GPU (Día 2-3)

**Nuevo archivo**: `core/processing/central_worker.py`
```python
import torch
import time
import logging
from core.vision.depth_estimator import DepthEstimator
from core.vision.yolo_processor import YoloProcessor
from utils.config import Config

def central_gpu_worker(central_queue, result_queue, stop_event):
    """
    Worker para cámara central: Depth + YOLO con CUDA streams.
    """
    logging.info("[Central Worker] Starting...")
    
    try:
        # Setup GPU
        torch.cuda.set_device(Config.DEVICE)
        depth_estimator = DepthEstimator()
        yolo_processor = YoloProcessor()
        
        depth_stream = torch.cuda.Stream()
        yolo_stream = torch.cuda.Stream()
        
        logging.info(f"[Central Worker] Models loaded, VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        while not stop_event.is_set():
            try:
                msg = central_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            frame_id = msg['frame_id']
            frame = msg['frame']
            
            start = time.perf_counter()
            
            # Parallel execution
            depth_map, depth_raw = None, None
            with torch.cuda.stream(depth_stream):
                depth_start = time.perf_counter()
                result = depth_estimator.estimate_depth_with_details(frame)
                if result:
                    depth_map = result.map_8bit
                    depth_raw = result.raw
                depth_ms = (time.perf_counter() - depth_start) * 1000
            
            with torch.cuda.stream(yolo_stream):
                yolo_start = time.perf_counter()
                detections = yolo_processor.process_frame(frame, depth_map, depth_raw)
                yolo_ms = (time.perf_counter() - yolo_start) * 1000
            
            torch.cuda.synchronize()
            
            total_ms = (time.perf_counter() - start) * 1000
            
            result_queue.put({
                'frame_id': frame_id,
                'camera': 'central',
                'detections': detections,
                'depth_map': depth_map,
                'depth_raw': depth_raw,
                'latency_ms': total_ms,
                'profiling': {
                    'depth_ms': depth_ms,
                    'yolo_ms': yolo_ms,
                    'gpu_mem_mb': torch.cuda.memory_allocated() / 1e6
                }
            })
            
    except Exception as e:
        logging.critical(f"[Central Worker] Fatal error: {e}")
        stop_event.set()
    finally:
        torch.cuda.empty_cache()
        logging.info("[Central Worker] Shutdown complete")
```

**Nuevo archivo**: `core/processing/slam_worker.py`
```python
import torch
import time
import logging
import queue
from core.vision.yolo_processor import YoloProcessor
from utils.config import Config

def slam_gpu_worker(slam_queue, result_queue, stop_event):
    """
    Worker para SLAM1/SLAM2: YOLO compartido (procesa secuencialmente).
    """
    logging.info("[SLAM Worker] Starting...")
    
    try:
        torch.cuda.set_device(Config.DEVICE)
        yolo_processor = YoloProcessor.from_profile("slam")
        
        logging.info(f"[SLAM Worker] Model loaded, VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        while not stop_event.is_set():
            try:
                msg = slam_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            start = time.perf_counter()
            
            detections = yolo_processor.process_frame(msg['frame'])
            torch.cuda.synchronize()
            
            latency_ms = (time.perf_counter() - start) * 1000
            
            result_queue.put({
                'frame_id': msg['frame_id'],
                'camera': msg['camera'],
                'detections': detections,
                'depth_map': None,
                'depth_raw': None,
                'latency_ms': latency_ms,
                'profiling': {
                    'yolo_ms': latency_ms,
                    'gpu_mem_mb': torch.cuda.memory_allocated() / 1e6
                }
            })
            
    except Exception as e:
        logging.critical(f"[SLAM Worker] Fatal error: {e}")
        stop_event.set()
    finally:
        torch.cuda.empty_cache()
        logging.info("[SLAM Worker] Shutdown complete")
```

### Fase 4.3: Modificar NavigationPipeline (Día 4-5)

**Modificaciones en `__init__`**:
```python
def __init__(self, yolo_processor, *, image_enhancer=None, depth_estimator=None):
    # Legacy components (para fallback)
    self.yolo_processor = yolo_processor
    self.image_enhancer = image_enhancer
    self.depth_estimator = depth_estimator or self._build_depth_estimator()
    
    # Frame tracking
    self.frame_counter = 0
    self.latest_depth_map = None
    self.latest_depth_raw = None
    
    # Phase 2: Multiprocessing setup
    if getattr(Config, 'PHASE2_MULTIPROC_ENABLED', False):
        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)
        
        from core.processing.central_worker import central_gpu_worker
        from core.processing.slam_worker import slam_gpu_worker
        
        self.central_queue = mp.Queue(maxsize=Config.PHASE2_QUEUE_MAXSIZE)
        self.slam_queue = mp.Queue(maxsize=Config.PHASE2_SLAM_QUEUE_MAXSIZE)
        self.result_queue = mp.Queue(maxsize=Config.PHASE2_RESULT_QUEUE_MAXSIZE)
        self.stop_event = mp.Event()
        
        self.workers = [
            mp.Process(
                target=central_gpu_worker,
                args=(self.central_queue, self.result_queue, self.stop_event),
                name="CentralWorker"
            ),
            mp.Process(
                target=slam_gpu_worker,
                args=(self.slam_queue, self.result_queue, self.stop_event),
                name="SLAMWorker"
            )
        ]
        
        for worker in self.workers:
            worker.start()
        
        logging.info("[Pipeline] Multiprocessing workers started")
        
        # Stats
        self.stats = {
            'frames_processed': 0,
            'dropped_central': 0,
            'dropped_slam': 0,
            'timeout_errors': 0,
            'last_stats_time': time.time()
        }
    else:
        logging.info("[Pipeline] Running in sequential mode")
```

**Nuevo método `process()`**:
```python
def process(self, frames_dict, *, profile=False):
    """
    frames_dict = {
        'central': np.ndarray,
        'slam1': np.ndarray,
        'slam2': np.ndarray
    }
    """
    if not getattr(Config, 'PHASE2_MULTIPROC_ENABLED', False):
        # Legacy: solo procesa central
        return self._process_sequential(frames_dict['central'], profile)
    
    return self._process_multiproc(frames_dict, profile)

def _process_multiproc(self, frames_dict, profile):
    frame_id = self.frame_counter
    self.frame_counter += 1
    timestamp = time.time()
    
    # Enhancement solo en central
    central_frame = frames_dict['central']
    if self.image_enhancer:
        central_frame = self.image_enhancer.enhance_frame(central_frame)
    
    # Distribute to workers
    try:
        self.central_queue.put({
            'frame_id': frame_id,
            'camera': 'central',
            'frame': central_frame,
            'timestamp': timestamp
        }, timeout=0.1)
    except queue.Full:
        self.stats['dropped_central'] += 1
        logging.warning(f"Dropped central frame {frame_id}")
    
    for camera in ['slam1', 'slam2']:
        try:
            self.slam_queue.put({
                'frame_id': frame_id,
                'camera': camera,
                'frame': frames_dict[camera],
                'timestamp': timestamp
            }, timeout=0.05)
        except queue.Full:
            self.stats['dropped_slam'] += 1
    
    # Collect results (espera 3 cámaras)
    results = {}
    timeout_count = 0
    
    for _ in range(3):
        try:
            result = self.result_queue.get(timeout=1.0)
            results[result['camera']] = result
            timeout_count = 0
        except queue.Empty:
            timeout_count += 1
            self.stats['timeout_errors'] += 1
            
            if timeout_count >= 3:
                logging.error("3 consecutive timeouts, falling back to sequential")
                Config.PHASE2_MULTIPROC_ENABLED = False
                return self._process_sequential(frames_dict['central'], profile)
    
    # Update stats
    self.stats['frames_processed'] += 1
    if time.time() - self.stats['last_stats_time'] > Config.PHASE2_STATS_INTERVAL:
        self._print_stats()
    
    # Merge results
    return self._merge_results(results, frames_dict)

def _merge_results(self, results, frames_dict):
    """Fusiona detecciones de 3 cámaras."""
    central_result = results.get('central', {})
    
    all_detections = []
    for camera, result in results.items():
        for det in result.get('detections', []):
            det['camera'] = camera
            all_detections.append(det)
    
    # Update latest depth
    if 'depth_map' in central_result:
        self.latest_depth_map = central_result['depth_map']
        self.latest_depth_raw = central_result.get('depth_raw')
    
    return PipelineResult(
        frame=frames_dict['central'],
        detections=all_detections,
        depth_map=self.latest_depth_map,
        depth_raw=self.latest_depth_raw,
        timings={
            'multiproc': True,
            'central_ms': central_result.get('latency_ms', 0),
            'slam1_ms': results.get('slam1', {}).get('latency_ms', 0),
            'slam2_ms': results.get('slam2', {}).get('latency_ms', 0)
        }
    )
```

**Shutdown**:
```python
def shutdown(self):
    if not hasattr(self, 'stop_event'):
        return
    
    logging.info("[Pipeline] Initiating shutdown...")
    self.stop_event.set()
    
    for worker in self.workers:
        worker.join(timeout=2)
        if worker.is_alive():
            logging.warning(f"Terminating {worker.name}")
            worker.terminate()
            worker.join(timeout=1)
    
    # Drain queues
    for q in [self.central_queue, self.slam_queue, self.result_queue]:
        while not q.empty():
            try:
                q.get_nowait()
            except:
                break
    
    logging.info("[Pipeline] Shutdown complete")
```

### Fase 4.4: Integración en main.py (Día 5)

```python
# En main.py
try:
    while True:
        # Obtener frames de 3 cámaras
        frame_central = observer.get_latest_frame()
        slam_frames = observer.get_slam_frames()
        
        frames_dict = {
            'central': frame_central,
            'slam1': slam_frames.get('slam1'),
            'slam2': slam_frames.get('slam2')
        }
        
        # Procesar (modo multi o secuencial según config)
        result = pipeline.process(frames_dict, profile=True)
        
        # Coordinator y visualización
        coordinator.process_preprocessed_frame(result)
        presentation_manager.render(result)
        
except KeyboardInterrupt:
    logging.info("Shutting down...")
finally:
    pipeline.shutdown()
    observer.close()
```

---

## 5. Validación y Testing

### 5.1 Test básico (Día 6)
```bash
# Activar multiproceso
export PHASE2_MULTIPROC_ENABLED=True

# Run con mock
python src/main.py --mock

# Verificar logs:
# - "Multiprocessing workers started"
# - "[Central Worker] Models loaded"
# - "[SLAM Worker] Model loaded"
# - FPS stats cada 5s
```

### 5.2 Benchmark FPS (Día 7)
```python
# Comparación:
# Secuencial: 13.6 FPS
# Multiproceso: ¿35-40 FPS?

# Medir con:
nvidia-smi dmon -s u -d 1  # GPU utilization
```

### 5.3 Stress tests (Día 8-9)

**Backpressure test**:
```python
# En depth_worker, añadir:
time.sleep(0.5)  # Simular lentitud

# Verificar:
# - Logs "SLAM queue full"
# - Frames dropped en SLAM
# - Central continúa a buen FPS
```

**Crash recovery**:
```python
# En central_worker, forzar:
if frame_id == 100:
    raise RuntimeError("Simulated crash")

# Verificar:
# - stop_event.set() llamado
# - Fallback a modo secuencial
# - No deadlock
```

**VRAM monitoring**:
```bash
watch -n 1 nvidia-smi

# Verificar:
# - Total < 6 GB
# - Central worker: ~5 GB
# - SLAM worker: ~2.5 GB
```

### 5.4 Profiling (Día 10)

```python
# Añadir en NavigationPipeline:
def _print_stats(self):
    elapsed = time.time() - self.stats['last_stats_time']
    fps = self.stats['frames_processed'] / elapsed
    
    logging.info(f"""
    [STATS] {elapsed:.1f}s window:
      FPS: {fps:.1f}
      Dropped central: {self.stats['dropped_central']}
      Dropped SLAM: {self.stats['dropped_slam']}
      Timeout errors: {self.stats['timeout_errors']}
    """)
    
    # Reset
    self.stats['frames_processed'] = 0
    self.stats['dropped_central'] = 0
    self.stats['dropped_slam'] = 0
    self.stats['last_stats_time'] = time.time()
```

---

## 6. Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| **OOM (VRAM overflow)** | Media | Alto | - Reducir input_size depth (1408→1024)<br>- Modelo YOLO más pequeño<br>- Frame skip SLAM agresivo<br>- `torch.cuda.empty_cache()` |
| **Deadlock en colas** | Baja | Alto | - Timeouts estrictos (0.1-1s)<br>- Fallback automático<br>- Logs de debugging |
| **Worker crash sin notificación** | Media | Medio | - `stop_event.set()` en exception<br>- `process.is_alive()` monitoring<br>- Restart automático (fase 3) |
| **Latencia IPC alta** | Media | Medio | - Benchmark pickle vs shared_memory<br>- Colas pequeñas (maxsize=2)<br>- Profiling de queue.put/get |
| **Desincronización frame_id** | Baja | Alto | - Monotonic counter<br>- Validación en merge<br>- Logs si mismatch |

---

## 7. Estimación de Ganancia

### 7.1 Performance actual
- **FPS**: 13.6 (solo cámara central)
- **Latencia**: 90-100 ms/frame
- **GPU utilization**: 40-50% (subutilizada)
- **Bottleneck**: GIL + procesamiento secuencial

### 7.2 Performance esperada Fase 2

**Caso optimista (40 FPS)**:
- Central: depth + YOLO en paralelo (CUDA streams) → 60 ms
- SLAM: procesamiento concurrente → no bloquea central
- Overlap: mientras depth(N) ejecuta, YOLO(N-1) termina
- GPU utilization: 80-90%

**Caso pesimista (28-30 FPS)**:
- IPC overhead: +10-15 ms por serialización
- Queue contention: ocasional blocking
- GPU utilization: 60-70%

**Caso realista (35 FPS)**:
- **Ganancia: +157%** sobre baseline
- Suficiente para navegación en tiempo real
- Permite Fase 3 (GStreamer) y Fase 4 (TensorRT) posteriores

### 7.3 Análisis de overlap

```
Secuencial (actual):
Frame N: [Capture] → [Enhance] → [Depth 60ms] → [YOLO 45ms] = 105ms total
FPS = 9.5

Multiproceso (objetivo):
Frame N:   [Capture] → [Enhance] → Queue
Frame N:                          [Depth 60ms] (worker 1)
Frame N:                          [YOLO 45ms] (worker 1, parallel)
Frame N-1:                                    [SLAM1 YOLO 30ms] (worker 2)
Frame N-2:                                    [SLAM2 YOLO 30ms] (worker 2)

Throughput limitado por etapa más lenta = Depth 60ms
Pero con overlap → FPS teórico = 1000ms / 60ms = 16.6 (central)
Con 3 cámaras procesando: FPS efectivo ~35-40
```

---

## 8. Respuesta Final

### ¿Vale la pena multiprocesamiento o CUDA streams ya solucionan el problema?

**Respuesta: SÍ, multiprocesamiento es esencial.**

**Razones**:

1. **CUDA streams solo paraleliza dentro de un proceso**:
   - Permite depth + YOLO concurrente en GPU.
   - NO rompe el GIL de Python.
   - NO permite procesar múltiples cámaras (SLAM1/SLAM2).

2. **Sistema tiene 3 cámaras que deben procesarse simultáneamente**:
   - Central: depth + YOLO (más complejo).
   - SLAM1/SLAM2: solo YOLO (más simple).
   - Threading actual bloqueado por GIL → desperdicia CPU/GPU.

3. **Objetivo 35-40 FPS requiere overlap real**:
   - Frame N en depth, frame N+1 en YOLO, frame N+2 en captura.
   - Solo posible con procesos independientes.

4. **GPU RTX 2060 está subutilizada**:
   - Uso actual: 40-50%.
   - Objetivo: 80-90% con workers paralelos.

**Fase 2 multiprocessing es prerequisito para Fase 3 (GStreamer) y Fase 4 (TensorRT).**

---

## 9. Próximos Pasos

### Después de Fase 2 (si éxito)

**Fase 3: GStreamer Zero-Copy** (40→50 FPS):
- Reemplazar OpenCV por GStreamer pipeline.
- appsink con GPU memory mapping.
- Evitar CPU↔GPU transfers.

**Fase 4: TensorRT Optimization** (50→60+ FPS):
- Convertir YOLO + Depth a TensorRT engines.
- FP16 precision.
- Batch inference para SLAM.

**Fase 5: Advanced Features**:
- Motion detection (solo procesar si cambio).
- Predictive tracking (Kalman filters).
- Temporal fusion depth maps.

---

## Apéndice A: Checklist de Implementación

### Setup
- [ ] Añadir flags en `Config`
- [ ] Crear `multiproc_types.py`
- [ ] Instalar dependencies si faltan

### Workers
- [ ] Implementar `central_worker.py`
- [ ] Implementar `slam_worker.py`
- [ ] Tests unitarios de workers (mock queues)

### Pipeline
- [ ] Modificar `NavigationPipeline.__init__`
- [ ] Implementar `_process_multiproc()`
- [ ] Implementar `shutdown()`
- [ ] Implementar `_merge_results()`
- [ ] Mantener `_process_sequential()` intacto

### Integration
- [ ] Modificar `main.py` para pasar `frames_dict`
- [ ] Añadir shutdown en finally block
- [ ] Mock de 3 cámaras en `MockObserver`

### Testing
- [ ] Test básico con mock
- [ ] Benchmark FPS (sequential vs multiproc)
- [ ] Stress test backpressure
- [ ] Crash recovery test
- [ ] VRAM monitoring (nvidia-smi)
- [ ] Profiling latencias por etapa

### Documentación
- [ ] Comentarios en código
- [ ] README Fase 2
- [ ] Troubleshooting guide
- [ ] Performance benchmarks

---

**Estimación total: 10 días desarrollo + testing**

**Go/No-Go decision point**: Día 7 (después de benchmark FPS)
- Si FPS < 25: revisar arquitectura
- Si 25-35 FPS: continuar con optimizaciones menores
- Si > 35 FPS: ✅ Fase 2 completa, iniciar Fase 3