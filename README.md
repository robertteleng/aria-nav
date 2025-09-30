git clone [tu-repo-url]
git checkout dev
git checkout -b feature-name
git add .
git commit -m "feature-name: description"
git checkout dev
git merge feature-name
# Aria Navigation System

Sistema de navegación asistida para personas con discapacidad visual usando gafas Meta Aria. El proyecto implementa un pipeline modular que combina visión por computador, análisis espacial y comandos de audio priorizados.

## 🧭 Resumen rápido
- ✅ Pipeline RGB completo: `ImageEnhancer` → `DepthEstimator` → `YoloProcessor` → `NavigationDecisionEngine`.
- ✅ Audio unificado: `NavigationAudioRouter` gestiona eventos RGB/SLAM y aplica cooldowns; `AudioSystem` reproduce TTS en macOS.
- ✅ Coordinador refactorizado: `Coordinator` orquesta pipeline, SLAM, routing y métricas de profiling.
- 🔄 Visión periférica (SLAM) activa y en evolución: eventos dedicados con prioridades y logs.
- 🔄 Próximo paso: aislar el helper de routing SLAM y ejecutar sesiones end-to-end para afinar cooldowns.

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
- **Objetivo**: ofrecer navegación asistida en tiempo real aprovechando las cámaras RGB/SLAM y sensores de las Meta Aria.
- **Core loop**: captura → mejora → detección → decisión → audio → dashboards.
- **Modularidad**: cada capa (hardware, pipeline, audio, presentación) está desacoplada para facilitar iteraciones y despliegues híbridos Mac/Jetson.

Para más contexto arquitectónico consulta `docs/architecture/pipeline_overview.md` y `docs/architecture_document.md`.

## Arquitectura en breve
1. `DeviceManager` conecta con las gafas y alimenta al `Observer` (frames RGB, SLAM y estado de movimiento).
2. `Coordinator.process_frame` ejecuta `NavigationPipeline` (enhancer + depth + YOLO) y genera detecciones con métricas de profiling.
3. `NavigationDecisionEngine` analiza las detecciones, calcula zonas/distancias/prioridades y decide si emitir un evento de audio (con metadata y `EventPriority`).
4. `NavigationAudioRouter` (si está disponible) recibe eventos RGB/SLAM, aplica cooldowns por fuente y registra telemetría; en fallback, `AudioSystem` gestiona el TTS directamente.
5. `PresentationManager` muestra overlays (OpenCV por defecto, opción `rerun` o `web`) y sáturas de estado.

## Requisitos
- **Hardware**
	- Gafas Meta Aria con perfil `profile28` habilitado.
	- Mac con macOS 13+ (Apple Silicon recomendado) para el modo local.
	- (Opcional) Jetson/host Linux para procesado remoto vía ImageZMQ (modo híbrido en desarrollo).
- **Software**
	- Conda o Mamba.
	- Python 3.10 (provisionado por `environment.yml`).
	- Meta Aria SDK instalado y funcionando (ver documentación oficial de Meta).

## Instalación
```bash
# 1. Clonar el repositorio
git clone https://github.com/<tu-usuario>/aria-navigation.git
cd aria-navigation

# 2. Crear y activar el entorno Conda
conda env create -f environment.yml
conda activate aria-navigation

# 3. (Opcional) Verificar versión de Python y disponibilidad de 'say'
python --version
which say  # Debe existir en macOS para TTS
```

## Ejecución
```bash
# Modo principal (hardware real)
python src/main.py

# Modo debug sin hardware (frames mock + TTS)
python src/main.py debug

# Placeholder modo híbrido Mac → Jetson (en construcción)
python src/main.py hybrid
```

Controles en el modo principal:
- `q`: salir del sistema.
- `t`: disparar prueba del sistema de audio.
- `Ctrl+C`: parada segura gestionada por `CtrlCHandler`.

El script preguntará si deseas habilitar dashboard y el tipo (`opencv`, `rerun`, `web`). La ruta por defecto usa OpenCV.

## Telemetría y observabilidad
- `logs/audio_telemetry.jsonl`: respaldo del `NavigationAudioRouter` con cada evento (enqueued, spoken, skipped, dropped) y resumen final de sesión.
- `Coordinator.print_stats()`: métricas agregadas de pipeline y perfilado (`enhance`, `depth`, `yolo`, `nav_audio`, etc.).
- `PresentationManager.log_audio_command()`: histórico de comandos reproducidos en la UI.
- Ajusta la ventana de profiling con `Config.PROFILE_WINDOW_FRAMES`.

## Estructura del repositorio
```
aria-navigation/
├── README.md
├── environment.yml
├── docs/
│   ├── architecture/
│   │   └── pipeline_overview.md
│   └── development_diary.md
├── experiments/
│   └── meta_stream_all.py
├── src/
│   ├── main.py
│   ├── core/
│   │   ├── navigation/
│   │   │   ├── builder.py
│   │   │   ├── coordinator.py
│   │   │   ├── navigation_decision_engine.py
│   │   │   └── navigation_pipeline.py
│   │   ├── audio/
│   │   │   ├── audio_system.py
│   │   │   └── navigation_audio_router.py
│   │   ├── vision/
│   │   │   ├── yolo_processor.py
│   │   │   ├── depth_estimator.py
│   │   │   └── image_enhancer.py
│   │   ├── hardware/device_manager.py
│   │   └── observer.py
│   ├── communication/
│   │   └── mac_client.py
│   ├── presentation/
│   │   ├── presentation_manager.py
│   │   └── dashboards/
│   └── utils/config.py
├── logs/
│   └── audio_telemetry.jsonl
└── quick_deploy.sh
```

## Configuración
La configuración central está en `src/utils/config.py` (`Config`):
- `YOLO_*`: parámetros del detector (modelo, dispositivo MPS, thresholds).
- `PERIPHERAL_VISION_ENABLED`: activa/desactiva el pipeline SLAM y los `SlamDetectionWorker`.
- `DEPTH_*`: control del estimador de profundidad (`midas` o `depth_anything_v2`).
- `AUDIO_*`: cooldown base, tamaño de cola y velocidad de TTS.
- `PROFILE_*`: ventanas de profiling y métricas para el coordinador.

Actualiza estos valores antes de ejecutar para adaptar el sistema a tu hardware o a campañas de pruebas específicas.

## Flujo de trabajo y pruebas
- `Builder.build_full_system()` fabrica todas las dependencias con wiring actualizado (pipeline + decision engine + audio router + SLAM).
- `main_debug()` permite validar la integración sin hardware real (frames sintéticos, toggles de audio).
- Se recomienda ejecutar sesiones cortas tras cualquier cambio en cooldowns o prioridades para revisar `logs/audio_telemetry.jsonl`.
- Pipeline de tests automatizados aún no disponible; las validaciones son manuales/experimentales.

## Roadmap
- [ ] Extraer el helper de routing SLAM a un módulo independiente con métricas dedicadas.
- [ ] Ejecutar sesiones end-to-end con usuarios internos para ajustar cooldowns y prioridades.
- [ ] Completar modo híbrido Mac ↔ Jetson usando ImageZMQ.
- [ ] Documentar guías de troubleshooting para Aria SDK y sincronización SLAM.

## Créditos
- **Autor**: Roberto Rojas Sahuquillo (TFM 2025).
- **Agradecimientos**: Comunidad Project Aria y colaboradores del laboratorio de accesibilidad.

---

> Última actualización: septiembre 2025.