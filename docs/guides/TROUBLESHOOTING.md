# 🆘 Troubleshooting Catalog (Aria Navigation)

> Índice de síntomas → acciones rápidas para el build de innovación (18–22 FPS, YOLO TRT + Depth ONNX). Usa el índice para saltar a tu problema.

## 📋 Quick Navigation
- [Instalación/Setup](#instalacionsetup)
- [TensorRT/ONNX](#tensorrtonnx)
- [Depth](#depth)
- [YOLO](#yolo)
- [Audio](#audio)
- [Multiprocessing](#multiprocessing)
- [Performance](#performance)
- [Aria Hardware](#aria-hardware)
- [GPU/CUDA](#gpucuda)
- [Logging/Telemetry](#loggingtelemetry)
- [Estrategias Generales](#estrategias-generales)

## Instalación/Setup
### Issue: `projectaria-tools` Not Found
- Instala con `pip install projectaria-tools`; verifica virtualenv activo.
- Revisa `PYTHONPATH` si usas editable install.

### Issue: CUDA Not Available
- Confirma `nvidia-smi`; versión de driver ≥ toolkit.
- Reinstala/activa entorno con PyTorch CUDA.

### Issue: Permission Denied on USB Device
- Usa puertos USB 3.0, evita hubs; agrega udev rules si aplica.

## TensorRT/ONNX
### Issue: TensorRT Export Fails
- Simplifica modelo o usa ONNX Runtime como fallback; fija input size 640x640.

### Issue: ONNX Model Wrong Input Size
- Re-exporta con tamaño correcto (YOLO 640x640, Depth 518x518) y valida shapes.

### Issue: TensorRT Engine Crashes
- Rebuild con workspace 4GB y FP16; valida versión de TensorRT/driver.

## Depth
### Issue: Depth Returns None
- Verifica carga de modelo ONNX y provider CUDA; revisa rutas de modelo.

### Issue: Depth Very Slow (>200ms)
- Asegura ONNX CUDA activo; input 518x518; sincróniza streams si paralelo.

### Issue: Depth Map Quality Poor
- Ajusta realce de imagen; prueba resolución 384–518; revisa calibración.

## YOLO
### Issue: No Detections
- Confirma modelo TRT cargado; threshold 0.5; input 640x640; revisa luz/posición.

### Issue: Low Detection Accuracy
- Ajusta threshold/NMS; verifica enhancement y calibración; modelo correcto (v11/12n).

## Audio
### Issue: TTS Not Working (macOS/Linux)
- Comprueba backend (pyttsx3 dependencias); revisa salida de audio por OS; cola activa.

### Issue: Audio Telemetry Spikes
- Reduce verbosidad de logs; asegura flush async activo.

## Multiprocessing
### Issue: Worker Process Crashes
- Revisa excepciones en logs de worker; valida versión de libs en ambos lados.

### Issue: Queue Overflow/Deadlock
- Limita tamaños de cola; usa timeouts; evita blocking puts.

### Issue: Serialization Slow (pickle)
- Minimiza payloads; usa shared memory cuando sea seguro.

## Performance
### Issue: Low FPS (<10 FPS)
- Verifica skips (YOLO=3, Depth=12), input 640x640, TensorRT/ONNX en uso.
- Consulta `guides/PERFORMANCE_OPTIMIZATION.md` para checklist completo.

### Issue: High Latency (>100ms)
- Perfil por etapa (telemetría); revisa I/O y audio cooldown; confirma FP16.

## Aria Hardware
### Issue: Cannot Connect to Aria
- Usa USB 3.0, cable directo; reinicia dispositivo; espera LED azul.

### Issue: Streaming Starts Then Stops
- Revisa alimentación/consumo; evita hubs; comprueba estabilidad del puerto.

## GPU/CUDA
### Issue: CUDA Out of Memory
- Baja resolución depth (518→384), desactiva SLAM, reduce buffers.

### Issue: CUDA Driver Version Mismatch
- Alinea versiones driver/toolkit; reinstala PyTorch CUDA si rompe.

## Logging/Telemetry
### Issue: Log Files Not Created
- Revisa permisos y rutas de `logs/`; verifica que logger async arranque.

### Issue: JSONL Files Corrupted
- Evita kills abruptos; permite flush del hilo; reduce verbosidad.

## Estrategias Generales
- Habilita logging verbose en el módulo afectado.
- Usa Mock Observer para aislar hardware.
- Perf: habilita profiler y revisa telemetría (`performance.jsonl`).
- Sistema: monitorea CPU/GPU (`nvidia-smi`, `htop`).

## Recursos
- Setup: `docs/setup/SETUP.md`
- Arquitectura: `docs/architecture/architecture_document.md`
- Performance: `docs/guides/PERFORMANCE_OPTIMIZATION.md`
- Contributing/debug rápido: `docs/development/CONTRIBUTING.md`

## Archivado
Los fixes históricos y listas largas están en `docs/archive/` (ver `archive/cuda/` y `archive/development/`).
