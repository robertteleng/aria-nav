# Handoff Document: Aria Navigation Project

**Fecha:** 2025-11-30
**Branch actual:** `feature/audio-tracking-improvements`
**Último commit:** `117739a` - "fix: beeps no longer block TTS announcements"

---

## 🎯 Estado Actual del Proyecto

### Hardware
- **Meta Aria glasses** con 3 cámaras calibradas:
  - RGB (frontal): 1408x1408, FOV central, 19 FPS
  - SLAM1 (left fisheye): Periférica izquierda, 8 FPS
  - SLAM2 (right fisheye): Periférica derecha, 8 FPS
- Calibraciones disponibles desde Aria SDK (extrinsics + intrinsics)

### Funcionalidades Implementadas

#### ✅ Sistema de Audio (LIMPIO Y FUNCIONAL)
- **TTS (Text-to-Speech)**: macOS (`say`) + Linux (`pyttsx3`)
- **Beeps espaciales**: Stereo panning + volumen dinámico por distancia
- **Threading independiente**: TTS y beeps NO se bloquean entre sí (bug arreglado)
- **Cooldowns**: Por clase y por instancia
- **Scan mode**: Resume audible de top 5 objetos en escena (NOA-inspired)

Archivos clave:
- `src/core/audio/audio_system.py` - Sistema principal (REFACTORIZADO)
- `docs/AUDIO_FLOW.md` - Documentación completa del flujo
- `examples/test_beep_tts_fix.py` - Test de verificación

#### ✅ Object Tracking (RGB ONLY)
- **ObjectTracker**: IoU-based tracking para RGB
- **Per-instance cooldowns**: Cada objeto (person_0, person_1) tiene su propio cooldown
- **Track IDs**: Únicos dentro de RGB camera

Archivos clave:
- `src/core/navigation/object_tracker.py` - Tracker simple (funcional)
- `src/core/navigation/navigation_decision_engine.py` - Integración del tracker

#### ✅ Multiprocessing Pipeline
- Workers dedicados para RGB + SLAM1 + SLAM2
- GPU processing con YOLO + Depth Anything V2
- Event-driven architecture con queues

Archivos clave:
- `src/core/navigation/coordinator.py` - Orquestador principal
- `src/core/vision/slam_detection_worker.py` - SLAM workers
- `src/core/vision/rgb_detection_worker.py` - RGB worker

#### ✅ Audio Routing
- **RGB Audio Router**: Maneja anuncios de RGB camera
- **SLAM Audio Router**: Maneja anuncios de SLAM cameras
- **Deduplicación básica**: Por clase y timestamp (muy simple)

Archivos clave:
- `src/core/navigation/rgb_audio_router.py`
- `src/core/navigation/slam_audio_router.py`

---

## ❌ Problemas Actuales

### 1. **Código Muerto y Duplicación**
El proyecto ha crecido orgánicamente y tiene:
- Variables sin usar
- Lógica duplicada entre RGB y SLAM routers
- Checks redundantes en `navigation_decision_engine.py`
- Posible duplicación de eventos de audio

**Impacto:** Difícil de debuggear, riesgo de bugs ocultos

### 2. **Cross-Camera Tracking NO Existe**
**Problema crítico:**
```
t=0s: SLAM1 detecta "person" → anuncia "Person approaching from left"
t=2s: Person entra en RGB → anuncia OTRA VEZ "Person ahead" ❌
```

Actualmente:
- SLAM y RGB tienen trackers **completamente independientes**
- Mismo objeto = 2 track_ids diferentes = 2 anuncios duplicados
- Deduplicación actual es solo temporal por clase (muy básica)

**Solución diseñada pero NO implementada:**
- `GlobalObjectTracker` con handoff temporal (ver `docs/wild-beaming-fog.md`)
- Comparte track_ids entre las 3 cámaras
- Matching cross-camera: clase + zona + tiempo
- Geometría 3D opcional (solo si temporal falla)

---

## 📁 Archivos Clave para Leer

### Documentación
```
docs/AUDIO_FLOW.md                      # Flujo de audio (beeps + TTS)
docs/AUDIO_TRACKING_IMPROVEMENTS.md     # Mejoras implementadas
.claude/plans/wild-beaming-fog.md       # Plan cross-camera tracking (NO implementado)
```

### Core System
```
src/core/navigation/coordinator.py                 # Orquestador principal
src/core/navigation/navigation_decision_engine.py  # Lógica de decisión
src/core/navigation/object_tracker.py              # Tracker RGB (simple IoU)
```

### Audio
```
src/core/audio/audio_system.py         # Sistema de audio (LIMPIO)
src/core/navigation/rgb_audio_router.py
src/core/navigation/slam_audio_router.py
```

### Vision
```
src/core/vision/slam_detection_worker.py
src/core/vision/rgb_detection_worker.py
```

---

## 🎯 Próximo Objetivo: Limpieza + Cross-Camera Tracking

### Fase 1: Limpieza (PRIORIDAD)
**Objetivo:** Hacer el código mantenible antes de añadir complejidad

**Tareas:**
1. Identificar código muerto (variables, funciones no usadas)
2. Consolidar lógica duplicada entre routers
3. Simplificar checks redundantes en decision engine
4. Eliminar duplicación de eventos
5. Documentar flujo completo en PlantUML

**Output esperado:**
- `docs/REFACTORING_REPORT.md` - Qué se eliminó/consolidó
- Código limpio y comprensible
- Tests de regresión pasando

### Fase 2: Cross-Camera Tracking
**Objetivo:** Implementar GlobalObjectTracker según plan

**Tareas:**
1. Crear `src/core/vision/global_object_tracker.py`
2. Implementar matching intra-camera (IoU, igual que ahora)
3. Implementar matching cross-camera (temporal + zona)
4. Integrar en `navigation_decision_engine.py`
5. Añadir track_id a eventos SLAM
6. Testing con escenarios reales

**Output esperado:**
- GlobalObjectTracker funcional
- Deduplicación cross-camera operativa
- Sin anuncios duplicados SLAM → RGB

---

## 🚨 Cosas a NO Romper

1. **Audio system** - Está limpio y funcional, no tocar
2. **Per-instance tracking RGB** - Funciona bien
3. **Performance** - Mantener 19 FPS (RGB) y 8 FPS (SLAM)
4. **Multiprocessing architecture** - No refactorizar workers

---

## 📊 Git Status

```
Branch: feature/audio-tracking-improvements
Commits desde main:
- bef6618 Refactor telemetry logging system
- d8154b3 feat: integrate MLflow experiment tracking
- 117739a fix: beeps no longer block TTS announcements (ÚLTIMO)

Archivos sin commitear:
- Ninguno (todo está limpio)
```

---

## 💡 Recomendaciones para Nuevo Chat

### Empezar con Fase 1 (Limpieza)

**Por qué:**
- El código actual es "ingobernable" (palabras del desarrollador)
- Difícil encontrar bugs sin entender el flujo
- Añadir cross-camera tracking a código desordenado = desastre

**Estrategia sugerida:**
1. Leer archivos core (coordinator, decision_engine, routers)
2. Crear diagrama de flujo PlantUML del estado actual
3. Identificar dead code con búsquedas
4. Consolidar duplicación
5. Documentar en REFACTORING_REPORT.md
6. Commit limpieza
7. ENTONCES implementar GlobalObjectTracker

### Evitar ByteTrack

**Razón:** Excesivo para este caso
- Diseñado para single-camera + multitudes + 30 FPS
- Tu caso: 3 cameras + escenas simples + cooldowns
- Añadiría 5-10ms latency → bajaría FPS
- No soporta multi-cámara nativamente

**Usar:** GlobalObjectTracker custom (plan en `wild-beaming-fog.md`)

---

## 🎬 Prompt Sugerido para Nuevo Chat

```markdown
Estoy trabajando en un sistema de navegación asistida con Meta Aria glasses (3 cámaras: RGB + SLAM1 + SLAM2). El proyecto funciona pero se ha vuelto "monstruoso e ingobernable" y necesito limpieza antes de implementar cross-camera tracking.

**Estado actual:**
- Branch: feature/audio-tracking-improvements
- Audio system: LIMPIO y funcional (no tocar)
- Object tracking: Solo RGB (falta cross-camera)
- Problema: Código muerto, duplicación, difícil de debuggear

**Objetivo inmediato (Fase 1):**
Limpiar y refactorizar el código para hacerlo mantenible:
1. Identificar código muerto (variables/funciones no usadas)
2. Consolidar lógica duplicada entre routers
3. Simplificar checks redundantes en decision_engine
4. Documentar flujo completo

**Objetivo siguiente (Fase 2):**
Implementar GlobalObjectTracker para evitar anuncios duplicados SLAM→RGB.

**Lee primero:**
- docs/HANDOFF_TO_NEW_CHAT.md (este archivo)
- docs/AUDIO_FLOW.md
- src/core/navigation/coordinator.py
- src/core/navigation/navigation_decision_engine.py

**Empecemos con Fase 1: Analiza el código y crea un plan de limpieza detallado.**
```

---

## 📚 Glosario Técnico

- **Track ID:** Identificador único de un objeto detectado
- **IoU (Intersection over Union):** Métrica de overlap entre bboxes (0-1)
- **Handoff:** Transición de objeto entre cámaras (SLAM→RGB)
- **Temporal matching:** Reconocer objeto por tiempo + clase + zona
- **Cross-camera tracking:** Mantener IDs entre cámaras diferentes
- **Intra-camera tracking:** Tracking dentro de misma cámara
- **Beep spatial:** Tono con panning estéreo para indicar dirección
- **Cooldown:** Tiempo mínimo entre anuncios del mismo objeto
- **Persistence:** Frames consecutivos que objeto debe estar presente

---

**FIN DEL HANDOFF DOCUMENT**

Este archivo contiene toda la información necesaria para continuar el proyecto en un nuevo chat con contexto fresco.
