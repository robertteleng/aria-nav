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
  - `NavigationAudioRouter`: cola prioritaria centralizada (RGB + SLAM) con cooldown global/por fuente, expone métricas y emite `logs/audio_telemetry.jsonl` con eventos.
  - `AudioSystem`: motor TTS (`say`) en inglés con cooldown interno configurable desde el router.

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
  - Si el objeto top supera el umbral configurado, construye un mensaje en inglés con metadatos (prioridad, zona, cooldown sugerido) y lo encola en `NavigationAudioRouter` como evento `rgb`.
  - `Coordinator` ya no gestiona cooldowns directamente: delega en el router para decidir si se anuncia o se descarta.

5. **Visión Periférica (SLAM)**
   - `Coordinator` envía frames SLAM a cada `SlamDetectionWorker`.
   - Cada worker aplica YOLO (perfil slam) y emite eventos con `bbox`, zona periférica y distancia estimada.
   - `_submit_and_route` actualiza `latest_slam_events` y, si hay novedades, encola mensajes en `NavigationAudioRouter` con prioridad calculada.

6. **Audio (SLAM + RGB)**
  - `NavigationAudioRouter` corre en un hilo separado; aplica cooldown global y por fuente con base en los metadatos recibidos.
  - Cuando procede, ajusta el cooldown del `AudioSystem` y usa `speak_async` para reproducir el mensaje en inglés.

7. **Presentación**
   - `PresentationManager.update_display` recibe frame procesado, detecciones, depth y eventos SLAM.
   - Según dashboard, actualiza HUDs (OpenCV) o endpoints Flask en `WebDashboard`.

## Cambios Recentes Relevantes

- Limpieza de overlays SLAM cuando no hay eventos (evita recuadros ghost).
- Centralización del audio: `NavigationAudioRouter` recibe tanto RGB como SLAM y controla cooldowns; `AudioSystem` queda como motor TTS en inglés.
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
  - Eventos encolados, procesados, hablados, omitidos o descartados (`events_enqueued`, `events_processed`, `events_spoken`, `events_skipped`, `events_dropped`).
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
  - Campos base: `timestamp`, `action`, `source`, `priority`, `priority_value`, `message`, `metadata`, `reason` (si aplica) y `raw_event` simplificado.
  - Resumen de sesión al final (`session_summary`) con métrica agregada y duración.

Con esta instrumentación podremos validar más tarde una versión donde todo el audio (RGB + SLAM) pase exclusivamente por `NavigationAudioRouter`.
