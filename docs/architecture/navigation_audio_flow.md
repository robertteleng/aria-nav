# Flujo de Audio de Navegación

Este documento resume cómo los eventos RGB y SLAM convergen en el `NavigationAudioRouter` tras la refactorización que separa decisión y formateo.

## Visión general

1. `NavigationPipeline` procesa cada frame RGB y entrega detecciones al `NavigationDecisionEngine`.
2. El engine genera un `DecisionCandidate`: incluye el objeto priorizado, metadatos (zona, distancia, cooldown sugerido) y la prioridad, pero **no** crea el mensaje hablado.
3. `RgbAudioRouter` recibe el candidato, arma la frase en inglés, ajusta el cooldown de la fuente y encola el resultado en el `NavigationAudioRouter`. Si la cola no está disponible, recurre directamente al `AudioSystem`.
4. De forma paralela, cada `SlamDetectionWorker` produce `SlamDetectionEvent` que `SlamAudioRouter` filtra, prioriza y transforma en mensajes antes de encolarlos en el router común.
5. `NavigationAudioRouter` es la cola priorizada compartida. Aplica cooldown global, mide métricas, escribe telemetría y envía los mensajes al `AudioSystem`.

## Diagrama de flujo

```plantuml
@startuml NavigationAudioFlow
scale 0.9
skinparam monochrome true
skinparam shadowing false
skinparam defaultFontName Courier
skinparam arrowThickness 1.2
skinparam rectangle {
  BackgroundColor White
  BorderColor Black
}

rectangle "NavigationPipeline" as pipeline
rectangle "NavigationDecisionEngine" as decision
rectangle "DecisionCandidate\n(data + cooldown)" as candidate
rectangle "RgbAudioRouter" as rgb
rectangle "SlamDetectionWorker" as slam_worker
rectangle "SlamAudioRouter" as slam_router
rectangle "NavigationAudioRouter" as shared_router
rectangle "AudioSystem" as audio
rectangle "Fallback Queue\n(AudioSystem only)" as fallback

pipeline --> decision : detections
decision --> candidate : analyze + evaluate
candidate --> rgb : DecisionCandidate
rgb --> shared_router : formatted message

slam_worker --> slam_router : SlamDetectionEvent
slam_router --> shared_router : message + priority

shared_router --> audio : speak_async
rgb -[#grey,dashed]-> fallback : router unavailable
fallback -[#grey,dashed]-> audio : queue_message

note right of rgb
  - Construye mensaje en inglés
  - Ajusta cooldown por source
  - Reusa fallback legado
end note

note left of decision
  - Ranking por prioridad
  - Aplica cooldown por movimiento
  - No construye mensaje
end note

note bottom of shared_router
  Cola priorizada común (RGB + SLAM).
  Gestiona cooldown global y métricas.
end note

@enduml
```

> Exporta el `.puml` desde `docs/architecture/navigation_audio_flow.puml` si necesitas la imagen en otros formatos.

## Impacto del cambio

- Las decisiones RGB ahora se representan como datos puros (`DecisionCandidate`), alineando la arquitectura con SLAM y facilitando nuevas salidas (p. ej. HUD o telemetría).
- `RgbAudioRouter` encapsula la lógica de formateo y mantiene el fallback legado, lo que simplifica al coordinator y aísla la presentación de la lógica de negocio.
- El `NavigationAudioRouter` sigue siendo el punto de unión; no se modifica su contrato, sólo recibe mensajes más consistentes desde ambas fuentes.
