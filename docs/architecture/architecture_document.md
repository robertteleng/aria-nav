# 🧭 Aria Navigation System — Arquitectura y Flujo de Datos (Innovación)

> Referencia técnica consolidada para el build de innovación (no producción comercial).
> Target actual: 18-22 FPS en RTX 2060 con Meta Aria Glasses usando YOLO TensorRT + Depth ONNX.

## 🏃‍♂️ Lectura Rápida (30s)
- Propósito: navegación asistida en tiempo real con audio espacial para usuarios con discapacidad visual.
- Arquitectura: Observer → Coordinator → Pipelines (RGB/SLAM) → Decision Engine → Audio Router.
- Ritmo: YOLO cada 3er frame, Depth cada 12º frame; latencia típica captura→decisión ~107ms, E2E con audio ~150ms.
- Hardware objetivo: Meta Aria Glasses + compute unit (Intel NUC + RTX 2060, 6GB VRAM).
- Estado: build de investigación, optimizado con TensorRT FP16 + ONNX CUDA; telemetry y MLflow locales.

---

## 📚 Índice
1. [Visión General](#visión-general)
2. [Stack de Hardware y Runtime](#stack-de-hardware-y-runtime)
3. [Patrón Arquitectónico y Directorios](#patrón-arquitectónico-y-directorios)
4. [Pipelines y Flujos de Datos](#pipelines-y-flujos-de-datos)
5. [Component Deep Dive](#component-deep-dive)
6. [Modelo de Objetos y Prioridades](#modelo-de-objetos-y-prioridades)
7. [Configuración Esencial](#configuración-esencial)
8. [Performance y Recursos](#performance-y-recursos)
9. [Decisiones de Diseño](#decisiones-de-diseño)
10. [Testing, Telemetría y Observabilidad](#testing-telemetría-y-observabilidad)
11. [Roadmap de Innovación](#roadmap-de-innovación)
12. [Referencias](#referencias)

---

## Visión General
- Solución de navegación asistida en tiempo real con **CV + audio espacial**.
- Iteración actual orientada a **aprendizaje/experimentación** (no SLA productivo).
- Flujo núcleo: captura → realce → detección → profundidad → fusión → tracking → priorización → audio TTS direccional.

## Stack de Hardware y Runtime
- **Dispositivo**: Meta Aria (RGB 640x480@60fps, SLAM 640x480@30fps, IMU 1000Hz).
- **Compute unit**: Intel i7 + RTX 2060 (6GB VRAM), 32GB RAM, Ubuntu 22.04.
- **Targets**: 18-22 FPS sostenidos; YOLO ~40ms, Depth ~27ms; uso VRAM ~1.5GB (25% de 6GB).

## Patrón Arquitectónico y Directorios
- **Separación de responsabilidades**: Observer (I/O), Coordinator (orquestación), Pipelines (RGB/SLAM), Decision Engine, Audio Router, Presentation.
- **Mapa de código (extracto)**:
```
src/
├── main.py                       # Entrada y selección de modo
├── core/
│   ├── observer.py / mock_observer.py
│   ├── navigation/
│   │   ├── coordinator.py / navigation_pipeline.py
│   │   ├── navigation_decision_engine.py / builder.py
│   │   ├── rgb_audio_router.py / slam_audio_router.py
│   ├── vision/                   # YOLO, depth, enhancers, tracking
│   ├── audio/                    # audio_system.py, navigation_audio_router.py
│   ├── telemetry/                # loggers async
│   └── processing/               # workers (experimental)
├── presentation/                 # dashboards y renderers
└── utils/                        # config, profiler, monitors
```
- **Razonamiento**: testabilidad (mocks), intercambiabilidad de hardware, y habilitar pipelines paralelos (RGB + SLAM opcional).

## Pipelines y Flujos de Datos

### Main Loop (alto nivel)
```
Aria SDK → Observer → Coordinator
  ├─ RGB pipeline (60fps in, frame skip activo)
  ├─ SLAM pipeline (periférico, opcional)
  └─ Decision Engine → Audio Router → Audio System (TTS + espacial)
```

### Línea de Tiempo (Frame N = 180, 60 FPS)
```
Time(ms) | Componente             | Operación                      | Data
0        | Aria SDK               | Captura RGB                    | (480,640,3) u8
2        | Observer               | Undistort + rotación           | (480,640,3) u8
6        | ImageEnhancer          | Brillo/contraste               | (480,640,3) u8
8        | NavPipeline            | Skip check (180%3==0 → YOLO)   | -
10       | YOLO TensorRT FP16     | Resize + BCHW + inferencia     | 40ms
54       | YOLOProcessor          | NMS                            | detections
56       | NavPipeline            | Depth? (180%12==0 → sí)        | -
58       | DepthEstimator ONNX    | Resize + normaliza             | (518,518,3)
64       | ONNX Runtime (CUDA)    | Inferencia                     | 27ms
91       | DepthEstimator         | Reescalar depth map            | (480,640)
93       | NavPipeline            | Fusión depth+detecciones       | detections+depth
95       | ObjectTracker          | Tracking                       | tracked objs
97       | NavPipeline            | Clasificar zonas L/C/R         | tracked objs
99       | DecisionEngine         | Calcular prioridad             | list prioridades
101      | DecisionEngine         | Comando top                    | "Close person center"
103      | AudioRouter            | Cooldown check (2s)            | ok
105      | AudioRouter            | Enqueue TTS                    | queue
107      | AudioSystem            | Síntesis                       | wav buffer
150      | AudioSystem            | Play                           | salida
```
- **Latencia captura→decisión**: ~107ms. **E2E con audio**: ~150ms.
- **Frame skip**: YOLO cada 3er frame; Depth cada 12º; efectivo 18-22 FPS.

### RGB Pipeline (detallado)
1) Realce de imagen (~2ms)
2) YOLO TensorRT FP16 (40ms) @640x640
3) Depth-Anything v2 ONNX CUDA (27ms, 1/12 frames) @518x518
4) Fusión depth+detecciones (~1ms)
5) Tracking temporal (~2ms)
6) Clasificación espacial zonas (~1ms)
7) Decision Engine (~3ms)
8) Audio routing (~2ms)

### SLAM Pipeline (periférico, opcional)
- SLAM L/R @30fps → rectificación (~5ms) → YOLO paralelo (~40ms c/u) → detección lateral → audio "Caution left/right". Sin depth (usa tamaño bbox).

### Memoria (flujo)
- YOLO TensorRT ~800MB, Depth ONNX ~500MB, contexto CUDA ~200MB → ~1.5GB VRAM en uso; headroom ~4.5GB.

## Component Deep Dive

### Observer (`src/core/observer.py`)
- Abstracción de hardware (RGB, SLAM, IMU). Implementaciones: `AriaObserver` (real) y `MockObserver` (desarrollo sin hardware).
- Garantiza frames contiguos y rotación correcta para YOLO.

### Coordinator (`src/core/navigation/coordinator.py`)
- Inicializa modelos (YOLO, Depth), pipelines RGB/SLAM y Decision Engine.
- Loop principal: obtiene frames, aplica frame-skip, encola detecciones, actualiza dashboards, maneja shutdown limpio.
- Multiprocessing experimental (Phase 6) deshabilitado por defecto.

### Navigation Pipeline (`src/core/navigation/navigation_pipeline.py`)
- Etapas modulables (enhance → detect → depth → track → zone → priority).
- Estrategia de skip configurable: `YOLO_FRAME_SKIP=3`, `DEPTH_FRAME_SKIP=12`.
- Fusión depth/bboxes y clasificación espacial antes de priorizar.

### YOLO Processor (`src/core/vision/yolo_processor.py`)
- Exportado a TensorRT FP16 (`workspace=4GB`, batch=1, input 640x640). Latencia ~40ms vs ~120ms PyTorch.
- Optimiza: layer fusion, FP16, input fijo (sin dynamic shapes).

### Depth Estimator (`src/core/vision/depth_estimator.py`)
- Modelo Depth-Anything v2 Small en ONNX Runtime CUDA (~27ms). Input 518x518.
- Fusión a nivel de bbox:
```python
def fuse_depth(detections, depth_map):
    for det in detections:
        x1,y1,x2,y2 = det.bbox
        mean = depth_map[y1:y2, x1:x2].mean()
        det.distance = 1.0 / (mean + 1e-6)
        det.distance_bucket = bucket(det.distance)  # close/medium/far
```

### Decision Engine (`src/core/navigation/navigation_decision_engine.py`)
- Prioridad = f(distancia, zona, clase, motion_state).
```python
def calculate_priority(det, motion_state):
    p = 100 if det.distance_bucket=="close" else 50 if det.distance_bucket=="medium" else 10
    p += 30 if det.zone=="center" else 10
    if det.class_name in ["car","truck","bus"]: p += 40
    elif det.class_name in ["person","bicycle","motorcycle"]: p += 20
    if motion_state=="stationary": p *= 0.5
    return p
```
- Genera comando natural: `"Close {class} {zone}"` → TTS.

### Audio Router (`src/core/audio/navigation_audio_router.py`)
- Cola prioritaria + cooldown (2s por comando) para evitar spam.
- Puede interrumpir comandos de menor prioridad; soporta beeps proporcionales a distancia.

### Telemetry (`src/core/telemetry/loggers/telemetry_logger.py`)
- Logger async con cola `maxsize=2000` y flush en background. No bloquea loop principal, bufferiza y escribe en `logs/session_*/telemetry/*.jsonl`.
- Señales clave: FPS, latencia por etapa, eventos de audio y detección.

### Presentation Layer
- Dashboards OpenCV / Rerun / Web para debugging visual; no requerido para uso ciego pero útil para iteración.

## Modelo de Objetos y Prioridades
- **Críticos (P8-10):** person, stop sign, car, truck, bus.
- **Importantes (P5-7):** bicycle, motorcycle, traffic light, stairs.
- **Contextuales (P1-4):** door, chair, bench.
- **Modificadores:** distancia (close x2, medium x1.5), zona (center +30%).
- Ajustables en `Config` según entorno (indoor/outdoor).

## Configuración Esencial
```yaml
vision:
  model: yolov11/12n TensorRT
  input_resolution: 640x640
  confidence_threshold: 0.5
  yolo_frame_skip: 3
  depth_frame_skip: 12
  depth_model: depth-anything-v2-small.onnx

audio:
  tts_engine: pyttsx3
  speech_rate: 150
  volume: 0.9
  cooldown_seconds: 2.0
  queue_max_size: 3

spatial:
  zones: left[0,213], center[213,426], right[426,640]
  distance_pixels:
    person_close: 200
    car_close: 150
```
- **Parámetros críticos para perf**: skips, input size, precision (FP16), cola de audio (tamaño/cooldown).

## Performance y Recursos
- **Latencia (frame con depth):** ~78ms pipeline + ~43ms TTS → ~150ms E2E.
- **Latencia sin depth:** ~51ms (se usa en frames intermedios).
- **Evolución por fases:**

| Fase | Enfoque | FPS | Cambio clave |
|------|---------|-----|--------------|
| Baseline | PyTorch puro | 3.5 | N/A |
| 1-2 | TensorRT + ONNX | 10.2 | Optimización de modelos |
| 3 | Shared Memory | 6.6 | Race conditions (revertido) |
| 4 | Frame skip + tuning | 18.4 | Skip inteligente |
| 5 | Queues no bloqueantes | 18.8 | Timeouts en colas |
| 6 | CUDA Streams híbrido | 19.0 | Depth || YOLO (+0.6 FPS) |

- **CUDA Streams (híbrido):** YOLO TensorRT + Depth ONNX en streams separados; sincronización limita la ganancia a ~3%.
- **Uso VRAM:** ~1.5GB de 6GB (25%); suficiente margen para tracking avanzado o VLM experimental.
- **Bottlenecks actuales:** YOLO 40ms, Depth 27ms, preprocessing 5ms, resto 6ms.

## Decisiones de Diseño
- **Arquitectura separada (Observer/Coordinator/Pipelines)**: testabilidad y reemplazo de hardware sin tocar lógica.
- **Frame skip (YOLO/Depth)**: trade-off rendimiento/respuesta; sin skip → ~8 FPS (inviable).
- **TensorRT + ONNX**: cada modelo en su framework óptimo; PyTorch-only → 3.5 FPS.
- **Telemetry async**: evita bloqueos I/O y picos de 250ms; batch writes.
- **MLflow local con SQLite**: portabilidad y consultas rápidas; sin server remoto.

## Testing, Telemetría y Observabilidad
- **Unit/integration**: `tests/` cubre visión, audio, spatial, priority.
- **Profiling**: `utils/profiler.py`, `memory_profiler.py`, `resource_monitor.py`.
- **Debugging rápido**:
  - FPS bajo → `nvidia-smi`, habilitar profiler, revisar skips.
  - Audio lag → cooldown/queue, TTS backend.
  - CUDA OOM → bajar resolución depth, desactivar SLAM, reducir buffers.
- **Logs**: `logs/session_*/telemetry/*.jsonl` + `decision_engine.log`, `audio_system.log`.

## Roadmap de Innovación (2025)
- Q1: multi-idioma (ES/EN), optimizar undistortion, tracking (ByteTrack).
- Q2: VLM ligero (descripciones de escena), app móvil compañera, dashboard de telemetría cloud-lite.
- Q3-Q4: piloto real con usuarios, refinar audio espacial 3D, publicación y apertura parcial.

## Referencias
- YOLO (Ultralytics), TensorRT docs, ONNX Runtime CUDA.
- Project Aria SDK.
- Depth Anything v2.
- MLflow Tracking (SQLite backend).

