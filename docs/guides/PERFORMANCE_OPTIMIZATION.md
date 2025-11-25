# ⚡ Performance Optimization Guide (RTX 2060, Aria Innovation Build)

> Resumen práctico para sostener 18–22 FPS con Meta Aria Glasses usando YOLO TensorRT + Depth ONNX. Enfoque de innovación (no producto), prioriza reproducibilidad y rapidez de iteración.

## 🏃‍♂️ TL;DR
- Objetivo: ~107ms captura→decisión, ~150ms con audio; 18–22 FPS efectivos (YOLO cada 3er frame, Depth cada 12º).
- Hardware: Intel i7 + RTX 2060 (6GB), Ubuntu 22.04, drivers CUDA recientes.
- Modelos: YOLOv11/12n → TensorRT FP16 (640x640); Depth-Anything v2 Small → ONNX Runtime CUDA (518x518).
- Config crítica: `YOLO_FRAME_SKIP=3`, `DEPTH_FRAME_SKIP=12`, input 640x640, FP16, cooldown de audio 2s.
- Telemetría: habilita profiler y logs para verificar latencias por etapa.

## 📊 Métricas actuales
| Métrica | Valor | Notas |
|---------|-------|-------|
| FPS efectivo | 18–22 | YOLO skip 3, Depth skip 12 |
| YOLO latency | ~40ms | TensorRT FP16, batch=1 |
| Depth latency | ~27ms | ONNX CUDA, 518x518 |
| Latencia pipeline (con depth) | ~78ms | Sin TTS |
| Latencia audio (TTS) | ~43ms | pyttsx3 |
| E2E (captura→audio) | ~150ms | Con depth |
| VRAM | ~1.5GB / 6GB | ~25% uso |

## ✅ Checklist rápido (para reproducir)
1) Exportar YOLO a TensorRT FP16 (input fijo 640x640, workspace 4GB).
2) Exportar Depth-Anything v2 Small a ONNX y ejecutar con CUDA EP.
3) Configurar frame skip: YOLO=3, Depth=12; resolución entrada 640x480.
4) Habilitar image enhancement ligero (brillo/contraste) para confianza.
5) Verificar cola de audio: `queue_max_size=3`, `cooldown=2s`.
6) Correr con profiler/telemetría activada y revisar `performance.jsonl`.

## 🔧 Knobs de rendimiento (impacto alto)
| Parámetro | Default | Impacto | Trade-off |
|-----------|---------|---------|-----------|
| `YOLO_FRAME_SKIP` | 3 | FPS↑, latencia↓ | Menos frescura visual |
| `DEPTH_FRAME_SKIP` | 12 | FPS↑ | Depth menos frecuente |
| Input size YOLO | 640x640 | Latencia↓ | Menor precisión |
| Precision | FP16 | Latencia↓ | Ligera pérdida de precisión |
| CUDA Streams | Híbrido | +0.6 FPS | Ganancia limitada (TRT+ONNX) |
| Audio cooldown | 2s | Evita spam | Menos avisos repetidos |
| Depth resolution | 518x518 | Latencia≃ | Si bajas, depth menos precisa |

## 🛠️ Recetas clave
- **YOLO → TensorRT FP16 (batch=1):** exporta desde PyTorch, fija input 640x640, desactiva dynamic shapes para máxima velocidad.
- **Depth → ONNX CUDA:** exporta el modelo small; ejecuta con CUDA EP. Reescalar a 518x518 y de vuelta a 480x640.
- **Frame Skip Inteligente:** YOLO cada 3er frame, Depth cada 12º; tracking y fusión mantienen consistencia entre frames.
- **Fusión Depth+BBoxes:** calcula media de depth en el bbox, clasifica en close/medium/far y alimenta el Decision Engine.
- **Audio no bloqueante:** cola prioritaria + cooldown 2s; TTS en hilo separado.
- **Telemetry async:** logger con cola `maxsize=2000`, flush en background; evita picos de 250ms por I/O.

## 🧪 Validación rápida
1) Corre el pipeline con telemetría: revisa `logs/session_*/telemetry/performance.jsonl`.
2) Mide latencia por etapa: YOLO ~40ms, Depth ~27ms, resto ~11ms.
3) FPS: usa `nvidia-smi dmon` y el profiler interno; espera 18–22 FPS.
4) Audio: confirma que la cola no se desborda (`queue_max_size=3`).

## 🐛 Performance troubleshooting
- **FPS <15:** verifica skips (3/12), input size, que se esté usando TensorRT/ONNX (no PyTorch).
- **CUDA OOM:** reduce depth res (518→384), desactiva SLAM, limita buffers de detección.
- **Audio lag:** revisa cooldown y TTS backend; cola no bloqueante habilitada.
- **Latencia inestable:** profiler on; revisa GC y operaciones I/O en el loop.

## 🧭 Referencias
- Arquitectura consolidada: [`docs/architecture/architecture_document.md`](../architecture/architecture_document.md)
- Historial completo de optimizaciones (archivo): [`docs/archive/cuda/`](../archive/cuda/)

*Última actualización: noviembre 2025. Enfoque de innovación; ajustar parámetros según experimentación.*
