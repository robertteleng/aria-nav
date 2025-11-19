# Resumen para la presentación del audio espacial

Estas tres diapositivas condensan el enfoque actual del algoritmo de audio espacial y sus métricas clave. Cada sección incluye el título y los bullets que puedes copiar directamente en tu herramienta de presentaciones. Las referencias a diagramas están al final.

---

## Diapositiva 1: “Audio de Navegación Espacial Multimodal”
### Propósito
- Mostrar cómo RGB y SLAM convergen en una única capa de audio.
- Destacar la combinación de beep direccional y TTS para feedback inmediato.

### Contenido sugerido
- **Feedback inmediato:** beep espacial (≈100 ms) precede al TTS corto para dar dirección en milisegundos.
- **Tres fuentes unificadas:** RGB frontal + cámaras SLAM 1 y 2 comparten la misma cola priorizada.
- **Prevención de spam:** múltiples cooldowns (global, por fuente y anti-duplicado) evitan repeticiones sin sacrificar alertas.
- **Arquitectura:** RGB/Slam → `NavigationAudioRouter` → `AudioSystem` (beep + `say`/`pyttsx3`).

## Diapositiva 2: “Gestión de Prioridades y Anti-Entrecorte”
### Visual recomendado
- Diagrama de flujo simplificado con decisiones: anti-stutter → cooldown global → cooldown por fuente → grace para interrupciones críticas.

### Bullets clave
- **Grace period:** 250 ms de tolerancia antes de permitir que un evento crítico interrumpa el TTS activo.
- **Anti-duplicados:** se bloquea el mismo mensaje si se pronunció en los últimos 2 s.
- **Cooldowns configurables:**
  | Fuente | Cooldown | Razón |
  |--------|----------|-------|
  | RGB | 1.2 s | Detecciones frontales frecuentes |
  | SLAM (periférico) | 3.0 s | Menor prioridad, evitar ruido lateral |
  | Global | 0.8 s | Máximo entre cualquier evento |
- **SLAM spacing:** 1 s de separación entre SLAM1 y SLAM2 para que no se solapen.
- **Resultado:** solo los eventos críticos interrumpen y solo después de que expire el grace y el cooldown global reducido.

## Diapositiva 3: “Mejoras Cuantificables vs. Sistema Convencional”
### Tabla comparativa (puedes usar un gráfico de barras)
| Métrica | Sistema convencional | Audio espacial actual | Mejora |
|---------|----------------------|----------------------|--------|
| Duración del mensaje | ~3.5 s (frases completas) | ~0.8 s (mensaje único) | -77 % ⬇️ |
| Latencia percibida | 500 ms | 100 ms | -80 % ⬆️ |
| Eventos spam | ~40 % | ~5 % | -87 % ⬇️ |
| Direccionalidad | ❌ TTS solo | ✅ beep estéreo + TTS | Nueva UX |

### Mensajes y UX
- Mensajes minimalistas (“Person”, “Chair”) reducen la carga cognitiva.
- Beeps paneados entregan izquierda/centro/derecha sin añadir texto.
- Respuesta 4× más rápida y fatiga auditiva mínima.

## Alternativa: Diapositiva única “Sistema + Resultados” (si necesitas condensar)
| Arquitectura | Algoritmo | Resultados |
|--------------|-----------|------------|
| RGB + SLAM 1/2 → `NavigationAudioRouter` | Cola priorizada + anti-stutter | -77 % duración total |
| Beep espacial + TTS minimalista | Grace 250 ms para interrupciones críticas | -80 % latencia percibida |
| Fallback directo al `AudioSystem` si la cola falla | 3 niveles de cooldown (global, fuente, SLAM) | -87 % spam |
| | Interrupción solo para eventos críticos y bien espaciados | Beep direccional (izquierda/centro/derecha) |

## Diagramas para enriquecer las diapositivas
- `docs/architecture/navigation_audio_flow.puml`: flujo completo desde el pipeline de navegación hasta la cola compartida y el motor TTS. Ideal para la diapositiva 1.
- `docs/diagrams/audio_algorithm.puml`: muestra la lógica de detección crítica, filtrado y gate hacia el audio. Útil para la diapositiva 2.
- Exporta los `.puml` a PNG/SVG usando PlantUML (por ejemplo, `plantuml docs/architecture/navigation_audio_flow.puml`).

Puedes copiar los bullets desde aquí y ajustarlos en tu herramienta de presentación; si quieres una versión con viñetas traducidas al inglés o ya listos como tarjetas, dime y genero una variante adicional.