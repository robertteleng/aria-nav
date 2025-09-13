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