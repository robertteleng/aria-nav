# Estrategia de Testing: Navigation Audio

Este documento describe la cobertura actual de pruebas para los módulos de audio de navegación (RGB + SLAM), el objetivo de cada suite y qué escenarios permanecen pendientes.

## Estructura de carpetas

```
tests/
├─ core/
│  ├─ audio/
│  │  └─ test_navigation_audio_router.py
│  └─ navigation/
│     ├─ test_navigation_decision_engine.py
│     ├─ test_rgb_audio_router.py
│     ├─ test_slam_audio_router.py
│     └─ test_coordinator_integration.py
```

Cada archivo cubre una capa concreta:

- `test_navigation_decision_engine.py`: lógica de priorización y cooldown por detección.
- `test_rgb_audio_router.py`: formateo de mensajes y fallback del canal RGB.
- `test_slam_audio_router.py`: orquestación de eventos SLAM, deduplicación y prioridades.
- `test_navigation_audio_router.py`: cola priorizada compartida (cooldowns, drops, rechazo de TTS).
- `test_coordinator_integration.py`: smoke test end-to-end con dependencias stub para asegurar el wiring.

## Cobertura actual

### NavigationDecisionEngine
- Filtrado de objetos no soportados (p. ej. `dog`).
- Cálculo de zonas y distancia en función del `bbox`.
- Ordenamiento descendente por prioridad.
- Aplicación de cooldown diferenciada por estado (`stationary`, `walking`).
- Rechazo de candidatos con prioridad insuficiente y actualización del timestamp interno.

### RgbAudioRouter
- Uso esperado del router compartido (`set_source_cooldown`, `enqueue_from_rgb`).
- Fallback al `AudioSystem` cuando el router falla o no existe.
- Mensajes para diferentes distancias (`cerca`, `medio`) y ausencia de metadata de cooldown.

### SlamAudioRouter
- Submissions al `SlamDetectionWorker` y almacenamiento de `frame_index`.
- Limpieza del estado `latest_events` cuando el worker no detecta objetos.
- Priorización (CRITICAL/HIGH) según tipo y distancia.
- Comportamiento cuando no hay `NavigationAudioRouter` disponible (solo actualiza `last_indices`).

### NavigationAudioRouter
- Reglas de `_should_announce`: cooldown global, cooldown por fuente y espaciamiento SLAM.
- Empaquetado de eventos SLAM (`enqueue_from_slam`).
- Procesamiento de la cola con `start/stop` y `speak_async` exitoso.
- Manejo de cola llena (`events_dropped`) y rechazo de TTS (`events_skipped`).

### Coordinator (integración)
- Flujo RGB completo: pipeline stub → decision engine → `RgbAudioRouter` → `NavigationAudioRouter`.
- Flujo SLAM: adjuntar workers, enviar frames y verificar mensajes generados.

## Lista de comandos útiles

```bash
# Ejecutar todas las pruebas (cuando pytest esté instalado)
python3 -m pytest tests

# Ejecutar un archivo concreto
python3 -m pytest tests/core/navigation/test_rgb_audio_router.py

# Validar sintaxis sin pytest
PYTHONPYCACHEPREFIX=./.pyc_cache python3 -m compileall tests
```

> Nota: el entorno actual no incluye `pytest` por defecto; instálalo con `pip install pytest` para ejecutar la suite completa.

## Cobertura pendiente

- `NavigationPipeline`: tests unitarios o de integración que verifiquen la salida de `process` con distintas configuraciones (depth map, timings, errores).
- `Builder`: smoke test que construya el sistema completo y confirme que las dependencias esenciales se crean sin lanzar excepciones.
- Casos de políticas de cooldown extremos en `NavigationAudioRouter` (configuración dinámica, recuperación tras `stop()` sin `start()` previo).
- Pruebas de regresión para `SlamAudioRouter` con múltiples cámaras (`SLAM1`, `SLAM2`) simultáneas.
- Tests de rendimiento/estrés (por ejemplo, encolando más de 16 eventos en la cola prioritaria).

Mantener esta lista viva ayudará a priorizar nuevas pruebas conforme se amplíe la funcionalidad del sistema de audio.
