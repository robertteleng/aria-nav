# Aria Navigation System

Sistema de navegación asistida para personas con discapacidad visual usando gafas Meta Aria. El proyecto implementa un pipeline modular que combina visión por computador, análisis espacial y comandos de audio con prioridades y cooldown por fuente.

## 🧭 Resumen rápido
- ✅ Pipeline RGB modular: `ImageEnhancer` → `DepthEstimator` (MiDaS/Depth-Anything) → `YoloProcessor` (perfiles RGB/SLAM) → `NavigationDecisionEngine`.
- ✅ Audio centralizado: `NavigationAudioRouter` coordina `RgbAudioRouter`, `SlamAudioRouter` y `AudioSystem`, aplica cooldowns dinámicos y registra métricas.
- ✅ Visión periférica asíncrona: `SlamDetectionWorker` procesa SLAM1/SLAM2 en paralelo y genera eventos priorizados.
- ✅ Presentación desacoplada: `PresentationManager` + `FrameRenderer` ofrecen dashboard OpenCV/Rerun/Web, overlays de navegación y mini-mapa de profundidad.
- ✅ Suite de pruebas `pytest` cubriendo pipeline, audio router, SLAM, MPS utilities y configuraciones clave.

## 📚 Índice
1. [Visión general](#visión-general)
2. [Arquitectura en breve](#arquitectura-en-breve)
3. [Requisitos](#requisitos)
4. [Instalación](#instalación)
5. [Ejecución](#ejecución)
6. [Telemetría y observabilidad](#telemetría-y-observabilidad)
7. [Estructura del repositorio](#estructura-del-repositorio)
8. [Configuración](#configuración)
9. [Flujo de trabajo y pruebas](#flujo-de-trabajo-y-pruebas)
10. [Roadmap](#roadmap)
11. [Créditos](#créditos)

## Visión general
- **Objetivo**: ofrecer navegación asistida en tiempo real aprovechando cámaras RGB/SLAM e IMU de las Meta Aria.
- **Arquitectura**: `DeviceManager` y `Observer` gestionan el SDK; `Coordinator` orquesta pipeline, audio y SLAM; `PresentationManager` maneja UI; `NavigationAudioRouter` unifica prioridades por fuente.
- **Modularidad**: cada capa está desacoplada para permitir mejoras independientes (hardware ↔ visión ↔ audio ↔ presentación ↔ telemetría).
- Documentación adicional en `docs/architecture/pipeline_overview.md` y `docs/architecture_document.md`.

## Arquitectura en breve
1. `DeviceManager` configura streaming (USB/Wi-Fi) y obtiene calibración RGB.
2. `Observer` recibe frames RGB/SLAM e IMU, normaliza orientación y estima `motion_state`.
3. `NavigationPipeline` (enhancer + depth + YOLO) produce un `PipelineResult` con timings opcionales.
4. `NavigationDecisionEngine` calcula prioridades; `RgbAudioRouter` formatea mensajes y los envía al `NavigationAudioRouter`, que decide si hablar vía `AudioSystem`.
5. Si `PERIPHERAL_VISION_ENABLED` está activo, `SlamDetectionWorker` procesa SLAM1/SLAM2 en background y `SlamAudioRouter` integra sus eventos en el audio centralizado.
6. `PresentationManager` usa `FrameRenderer` y dashboards (OpenCV/Rerun/Web) para overlays RGB, mini-mapa de profundidad, estado de audio y eventos SLAM.

## Requisitos
- **Hardware**
  - Gafas Meta Aria con perfil `profile28` o equivalente habilitado.
  - Mac con macOS 13+ (Apple Silicon recomendado) para modo local.
  - (Opcional) Host remoto (Jetson/Linux) para modo híbrido vía ImageZMQ.
- **Software**
  - Python 3.10+ con `pip` o Conda/Mamba.
  - Paquetes principales: `torch`, `torchvision`, `ultralytics`, `opencv-python`, `numpy`, `projectaria-tools`, `aria-sdk` (suministrado por Meta), `transformers` (opcional, Depth Anything v2), `pytest`.
  - `say` disponible en macOS para TTS (`which say`).

## Instalación
```bash
# 1. Clonar el repositorio
git clone https://github.com/<tu-usuario>/aria-navigation.git
cd aria-navigation

# 2. Crear entorno (ejemplo con venv; usa Conda si lo prefieres)
python3 -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install --upgrade pip wheel

# 3. Instalar dependencias principales
pip install torch torchvision torchaudio  # Metal MPS soportado por defecto en macOS
pip install ultralytics opencv-python numpy projectaria-tools transformers pytest
# Instala aria.sdk siguiendo la guía oficial de Meta Aria (distribución privada).

# 4. (Opcional) Verificar TTS y cámara
python -c "import torch; print(torch.__version__)"
which say
```

## Ejecución
```bash
# Modo principal con hardware real
python src/main.py

# Modo debug sin hardware (frames sintéticos + TTS)
python src/main.py debug

# Modo híbrido Mac → Jetson (ImageZMQ sender, procesamiento remoto en desarrollo)
python src/main.py hybrid
```

Controles principales:
- `q`: salir del sistema.
- `t`: prueba del sistema de audio (cola RGB).
- `Ctrl+C`: parada segura gestionada por `CtrlCHandler`.

El arranque pregunta por dashboard (`opencv`, `rerun`, `web`) y habilita el flujo correspondiente. En debug se limitan a OpenCV simplificado.

## Telemetría y observabilidad
- `logs/audio_telemetry.jsonl`: `NavigationAudioRouter` registra eventos (`enqueued`, `spoken`, `skipped`, `dropped`) con metadata y resúmen de sesión.
- `NavigationAudioRouter.get_metrics()`: métricas por fuente (RGB, SLAM1, SLAM2), tamaños de cola y cooldown efectivo.
- `Coordinator`: emite métricas `PROFILE` del pipeline (`enhance`, `depth`, `yolo`, `nav_audio`, `render`, `total`) cada `PROFILE_WINDOW_FRAMES`.
- `PresentationManager.log_audio_command()`: historial de comandos reproducidos en la UI.

## Estructura del repositorio
```
aria-navigation/
├── README.md
├── docs/
│   ├── architecture/
│   └── ...
├── experiments/
│   └── meta_stream_all.py
├── logs/
│   └── audio_telemetry.jsonl
├── src/
│   ├── main.py
│   ├── core/
│   │   ├── audio/
│   │   │   ├── audio_system.py
│   │   │   └── navigation_audio_router.py
│   │   ├── hardware/
│   │   │   └── device_manager.py
│   │   ├── imu/
│   │   │   └── motion_detector.py
│   │   ├── navigation/
│   │   │   ├── builder.py
│   │   │   ├── coordinator.py
│   │   │   ├── navigation_decision_engine.py
│   │   │   ├── navigation_pipeline.py
│   │   │   ├── rgb_audio_router.py
│   │   │   └── slam_audio_router.py
│   │   └── vision/
│   │       ├── depth_estimator.py
│   │       ├── image_enhancer.py
│   │       ├── slam_detection_worker.py
│   │       └── yolo_processor.py
│   ├── presentation/
│   │   ├── presentation_manager.py
│   │   └── renderers/frame_renderer.py
│   └── utils/config.py
└── tests/
    └── core/...
```

## Configuración
`src/utils/config.py` centraliza los toggles:
- `PERIPHERAL_VISION_ENABLED`, `SLAM_TARGET_FPS`: control de visión periférica y workers SLAM.
- `YOLO_*`: modelo, dispositivo (MPS), thresholds y frame skipping para perfiles RGB/SLAM.
- `DEPTH_*`, `MIDAS_*`, `DEPTH_ANYTHING_VARIANT`: selección de backend y parámetros de profundidad.
- `LOW_LIGHT_ENHANCEMENT`, `AUTO_ENHANCEMENT`, `GAMMA_CORRECTION`: estrategia de realce en baja iluminación.
- `ZONE_SYSTEM`, `CENTER_ZONE_*`: definición de zonas y prioridades espaciales.
- `PROFILE_PIPELINE`, `PROFILE_WINDOW_FRAMES`: métricas de rendimiento.
- `STREAMING_INTERFACE`, `STREAMING_PROFILE_*`: configuración de streaming Aria (USB/Wi-Fi).

## Flujo de trabajo y pruebas
- `Builder.build_full_system()` crea todas las dependencias (pipeline, audio router, frame renderer, SLAM workers).
- `main_debug()` permite validar integración sin hardware real (frames mock, SLAM sintetizado, TTS).
- Tests unitarios/integración en `tests/` (usar `pytest`). Incluyen pruebas para pipeline, routers RGB/SLAM, audio queue, MPS utils y motion detection.
- Recomendación: tras cambios en cooldowns o thresholds, ejecutar una sesión corta y revisar `logs/audio_telemetry.jsonl`.

## Roadmap
- [ ] Empaquetar dependencias (requirements/environment) para instalación reproducible.
- [ ] Completar modo híbrido end-to-end (Mac sender ↔ Jetson processor) y compartir telemetría.
- [ ] Integrar métricas de `NavigationAudioRouter` y profundidad en dashboards interactivos.
- [ ] Documentar troubleshooting de Aria SDK, calibraciones SLAM y requisitos de red.

## Créditos
- **Autor**: Roberto Rojas Sahuquillo (TFM 2025).
- **Agradecimientos**: Comunidad Project Aria y colaboradores del laboratorio de accesibilidad.
