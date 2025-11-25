=== 2025-11-25 - TELEMETRY CONSOLIDATION + MLFLOW ===

CAMBIOS PRINCIPALES:

1. REORGANIZACIÓN TELEMETRY
   - Movido loggers a src/core/telemetry/loggers/
   - Estructura unificada de logs:
     logs/session_YYYY-MM-DD_HH-MM-SS/
     ├── telemetry/
     │   ├── performance.jsonl
     │   ├── detections.jsonl
     │   ├── audio_events.jsonl
     │   └── summary.json
     ├── decision_engine.log
     └── audio_system.log

2. MLFLOW INTEGRADO
   - Export automático al finalizar sesión
   - Backend SQLite local (~/mlruns/mlflow.db)
   - Métricas: FPS, latencia, detecciones por clase
   - Ver resultados:
     mlflow ui --backend-store-uri "sqlite:////home/$USER/mlruns/mlflow.db"

3. LIMPIEZA CÓDIGO
   - Eliminados 19 archivos test redundantes
   - Movidos benchmarks a carpeta dedicada
   - Renombrado central→rgb para consistencia

DECISIONES:
✓ MLflow con SQLite (sin servidor)
✓ Export async (no bloquea shutdown)
✓ Integrado en AsyncTelemetryLogger existente
❌ MLflow FileStore deprecated → usar SQLite

USO:
```python
telemetry = AsyncTelemetryLogger(mlflow_experiment="aria-nav")
# ... inferencia ...
telemetry.finalize_session(model_name="yolo12n", resolution=640)
```

---

=== DÍA 1 - STREAMING + YOLO ===

PIPELINE CONSEGUIDO:
┌──────┐    ┌────────┐    ┌──────┐    ┌────────┐
│ Aria │───▶│ Observer│───▶│ YOLO │───▶│ Display│
│ USB  │    │ RGB     │    │ v11n │    │ OpenCV │
└──────┘    └────────┘    └──────┘    └────────┘
  60fps      rot90(-1)    CPU+nano     realtime

✓ RGB 60fps funcionando
✓ YOLOv11n tiempo real
✓ Performance fluida

FIXES CRÍTICOS:
- np.rot90(image, -1) → orientación OK
- np.ascontiguousarray() → YOLO happy
- CPU not MPS → evita bug NMS
- YOLOn not YOLOs → mejor speed

DECISIONES:
❌ Undistortion → empeora detección
✓ Profile28 → 60fps
✓ USB → más estable

BREAKTHROUGH:
Sistema base funcional ✓
No overthink arquitectura

PRÓXIMO: Audio commands
┌──────┐    ┌────────┐    ┌──────┐    ┌────────┐    ┌──────┐
│ Aria │───▶│Observer│───▶│ YOLO │───▶│ Audio  │───▶│User  │
│      │    │        │    │      │    │Commands│    │Blind │
└──────┘    └────────┘    └──────┘    └────────┘    └──────┘

GIT: streaming → dev (merged)