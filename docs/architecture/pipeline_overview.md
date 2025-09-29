# Pipeline Overview

## Capas y Componentes Clave

- **Hardware / Captura**
  - `DeviceManager`: gestiona conexión con Meta Aria, inicializa streams.
  - `Observer`: bufferiza frames RGB y SLAM, expone getters y estado de movimiento.

- **Procesamiento RGB**
  - `ImageEnhancer`: preprocesa frames (baja iluminación, contraste).
  - `YoloProcessor` (perfil RGB): detección de objetos.
  - `DepthEstimator`: genera depth map normalizado.

- **Visión Periférica (SLAM)**
  - `SlamDetectionWorker`: threads por cámara SLAM, YOLO perfil slam.

- **Pipeline / Navegación**
  - `Coordinator`: orquesta enhancer, detección, depth, fusión, audio y stats.
  - `Builder`: fabrica dependencias y adjunta SLAM workers + audio router.

- **Audio**
  - `AudioSystem`: TTS (`say`), cooldown por frase.
  - `NavigationAudioRouter`: cola prioritaria de eventos (actualmente SLAM).

- **Presentación**
  - `PresentationManager`: actualiza dashboards (OpenCV/Web/Rerun).
  - `WebDashboard`: servidor Flask con streams y métricas.

## Recorrido Actual

1. **Captura**
   - `DeviceManager` se conecta a Aria y alimenta al `Observer` con frames RGB/SLAM y movimiento.

2. **Loop Principal (`main.py`)**
   - En cada iteración se consultan frames RGB/SLAM y el estado de movimiento.

3. **Procesamiento RGB**
   - `Coordinator.process_frame` invoca `ImageEnhancer`.
   - Ejecuta `YoloProcessor.process_frame` y obtiene detecciones.
   - Solicita depth map a `DepthEstimator` (según configuración `Config`).
   - Fusiona detecciones con depth para estimar distancia normalizada.

4. **Decisión RGB + Audio**
   - Ordena detecciones por prioridad (clase, zona, distancias).
   - Si el objeto top supera el umbral y respeta el cooldown, construye un mensaje y lo encola en `NavigationAudioRouter` como evento `rgb`.
   - `AudioSystem` queda limitado a reproducción TTS (`say`), aplicando sólo cooldowns de frase/repetición.

5. **Visión Periférica (SLAM)**
   - `Coordinator` envía frames SLAM a cada `SlamDetectionWorker`.
   - Cada worker aplica YOLO (perfil slam) y emite eventos con `bbox`, zona periférica y distancia estimada.
   - `_submit_and_route` actualiza `latest_slam_events` y, si hay novedades, encola mensajes en `NavigationAudioRouter` con prioridad calculada.

6. **Audio (SLAM + RGB)**
   - `NavigationAudioRouter` corre en un hilo separado; aplica cooldown global y por fuente, tanto para eventos SLAM como para RGB.
   - Cuando procede, usa `AudioSystem.speak_async` para reproducir el mensaje.

7. **Presentación**
   - `PresentationManager.update_display` recibe frame procesado, detecciones, depth y eventos SLAM.
   - Según dashboard, actualiza HUDs (OpenCV) o endpoints Flask en `WebDashboard`.

## Cambios Recentes Relevantes

- Limpieza de overlays SLAM cuando no hay eventos (evita recuadros ghost).
- Ajustes de cooldown dinámico en `AudioSystem` según `motion_state`.
- Preparación para compartir métricas con dashboards.

## Checklist de Métricas / Logs

Para instrumentar antes de refactorizar audio:

- **RGB Pipeline**
  - `frames_processed`, número de detecciones por frame.
  - Objeto top: clase, zona, prioridad, distancia (depth).
  - Razón de skip (cooldown insuficiente, prioridad baja).

- **AudioSystem**
  - Frase generada, `repeat_cooldown`, `_should_announce` (true/false).
  - Estado `tts_speaking`, duración estimada, cola actual.

- **NavigationAudioRouter**
  - Eventos encolados, hablados, descartados (`events_processed`, `events_spoken`, `events_dropped`).
  - Tiempos desde último anuncio global y por fuente (SLAM1/SLAM2/RGB cuando se integre).
  - Prioridad del evento final seleccionado.

- **SLAM Workers**
  - Frecuencia de eventos nuevos, frame_index, latencia procesamiento.
  - Clasificación de zona (far_left/right) y distancia.

- **Contexto General**
  - `motion_state`, FPS actual, thresholds de depth aplicados.
  - Estado de toggles (`PERIPHERAL_VISION_ENABLED`, `DEPTH_ENABLED`).

- **Formato de Logging**
  - Archivo dedicado `logs/audio_telemetry.jsonl` (JSON por línea).
  - Campos base: `timestamp`, `source`, `event_type`, `message`, `priority`, `cooldown_ms`, `decision`.
  - Resumen de sesión al final (`session_summary` con totales y métricas promedio).

Con esta instrumentación podremos validar más tarde una versión donde todo el audio (RGB + SLAM) pase exclusivamente por `NavigationAudioRouter`.
