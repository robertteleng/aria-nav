# Estrategia de Testing: Navigation & Vision

Este documento resume la cobertura actual de pruebas unitarias/integración ligera para los módulos clave del sistema (audio, navegación, visión, hardware) y las áreas pendientes.

## Estructura de carpetas

```
tests/
├─ core/
│  ├─ audio/
│  │  ├─ test_audio_system.py
│  │  └─ test_navigation_audio_router.py
│  ├─ hardware/
│  │  └─ test_device_manager.py
│  ├─ imu/
│  │  └─ test_motion_detector.py
│  ├─ navigation/
│  │  ├─ test_builder.py
│  │  ├─ test_coordinator_integration.py
│  │  ├─ test_navigation_decision_engine.py
│  │  ├─ test_navigation_pipeline.py
│  │  ├─ test_rgb_audio_router.py
│  │  └─ test_slam_audio_router.py
│  ├─ test_observer.py
│  └─ vision/
│     ├─ test_image_enhancer.py
│     ├─ test_mps_utils.py
│     ├─ test_object_tracker.py
│     ├─ test_slam_detection_worker.py
│     ├─ test_yolo_processor.py
│     └─ test_yolo_runtime_config.py
```

## Cobertura actual

### Audio & Routing
- `test_audio_system.py`: cooldowns, cola TTS, `speak_async` (forzado y normal), actualización de dimensiones.
- `test_navigation_audio_router.py`: cola prioritaria (cooldowns global/source, drops, skip de TTS), ciclo `start/stop`, fallback automático.
- `test_rgb_audio_router.py`: formateo de mensajes, fallback al `AudioSystem`, metadatos ausentes y router inexistente.
- `test_slam_audio_router.py`: deduplicación por `frame_index`, limpieza de `latest_events`, prioridades por objeto, funcionamiento sin router común.

### Motor de navegación
- `test_navigation_decision_engine.py`: filtrado de objetos, ranking por prioridad, cooldowns dependientes de movimiento, actualización de timestamp.
- `test_navigation_pipeline.py`: ejecución de todas las etapas (enhancer, depth estimator, YOLO) con mocks, respeto de `DEPTH_FRAME_SKIP`, manejo de depth deshabilitado.
- `test_builder.py`: verificación de wiring (instancias stub), configuración de SLAM según `Config`.
- `test_coordinator_integration.py`: smoke test end-to-end (pipeline → decision engine → audio router) incluyendo SLAM.

### Visión
- `test_image_enhancer.py`: `LOW_LIGHT_ENHANCEMENT`, auto detección y aplicación de CLAHE/gamma.
- `test_object_tracker.py`: requiere historial suficiente, devuelve sólo detecciones consistentes.
- `test_mps_utils.py`: selección de dispositivo, configuración de entorno y caché MPS.
- `test_yolo_runtime_config.py`: defaults, perfiles (`slam`), overrides y errores.
- `test_yolo_processor.py`: constructor con overrides, `runtime_config`, conflicto de argumentos (sin cargar modelos reales).
- `test_slam_detection_worker.py`: flujo `submit/start/stop`, frame skipping, manejo de cola y eventos generados.

### Hardware / IMU / Observer
- `test_device_manager.py`: conexión wifi/usb, streaming profile, cleanup con mocks de Aria SDK.
- `test_motion_detector.py`: umbrales, transición stationary/walking, zona de histéresis.
- `test_observer.py`: callbacks de imágenes (rotación, conversión), almacenamiento de frames y datos IMU.

## Lista de comandos útiles

```bash
# Ejecutar todas las pruebas (una vez resuelto NumPy)
conda run -n aria_x86 python -m pytest tests

# Ejecutar archivo específico
a. python3 -m pytest tests/core/navigation/test_navigation_pipeline.py
b. python3 -m pytest tests/core/vision/test_slam_detection_worker.py

# Validar sintaxis sin pytest
PYTHONPYCACHEPREFIX=./.pyc_cache python3 -m compileall tests
```

> Nota: la ejecución actualmente está bloqueada por un bug de NumPy/Accelerate en macOS. Solución: instalar NumPy desde conda-forge (`numpy>=2.1.2`) o desactivar el chequeo `_mac_os_check`. Mientras tanto usamos `compileall` para sanity check.

## Cobertura pendiente / siguientes pasos

- Integración profunda con modelos reales (YOLO, MiDaS): pruebas end-to-end con inferencia controlada o snapshots.
- Tests de rendimiento/estrés para colas (`NavigationAudioRouter`, `SlamDetectionWorker`).
- Validaciones adicionales en `depth_estimator` (simular outputs planos, caché de mapas).
- Pruebas en `NavigationPipeline` que verifiquen propagación de excepciones y contadores internos.
- Escenarios multi-cámara (SLAM1 + SLAM2 simultáneos) para el router y el worker.

Con estos elementos cubiertos, el sistema cuenta con una capa robusta de tests unitarios que protege la lógica principal y los caminos de integración más críticos. El backlog anterior se centra ahora en pruebas con dependencias de terceros y métricas de rendimiento.
