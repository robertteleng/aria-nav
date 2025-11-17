# Recomendaciones de Optimización - FASE 4+

## 🔴 Problema Crítico Identificado: I/O Bloqueante

### Síntomas
- Spikes periódicos de **350-400ms** cada ~50 frames (~2.5 segundos)
- FPS estable a 19.2 pero con drops puntuales a 14-15 FPS
- Patrón consistente en toda la sesión

### Causa Raíz
`TelemetryLogger` escribe **síncronamente a disco** en cada frame:
```python
def _write_jsonl(self, path: Path, data: Dict[str, Any]) -> None:
    with self._write_lock:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')  # ← Bloqueante cuando OS hace flush
```

**Impacto:**
- 3+ escrituras por frame (performance.jsonl, detections.jsonl, audio_events.jsonl)
- Sistema operativo sincroniza buffers periódicamente → 300-400ms stall
- Main thread bloqueado esperando I/O

---

## ✅ Soluciones Recomendadas

### 1. **Telemetry Asíncrona con Background Thread** ⭐ PRIORITARIO

**Implementación:**
```python
class AsyncTelemetryLogger(TelemetryLogger):
    def __init__(self, output_dir=None, flush_interval=1.0, buffer_size=100):
        super().__init__(output_dir)
        self._write_queue = queue.Queue(maxsize=1000)
        self._flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self._flush_interval = flush_interval
        self._buffer_size = buffer_size
        self._flush_thread.start()
    
    def _write_jsonl(self, path: Path, data: Dict[str, Any]) -> None:
        """Queue write instead of blocking."""
        try:
            self._write_queue.put_nowait((path, data))
        except queue.Full:
            # Log error but don't block main thread
            pass
    
    def _flush_worker(self) -> None:
        """Background thread for batched disk writes."""
        buffers = {}  # path -> list of lines
        
        while True:
            try:
                # Collect writes with timeout
                path, data = self._write_queue.get(timeout=self._flush_interval)
                
                if path not in buffers:
                    buffers[path] = []
                
                line = json.dumps(data, ensure_ascii=True)
                buffers[path].append(line)
                
                # Flush if buffer full or timeout
                for file_path, lines in list(buffers.items()):
                    if len(lines) >= self._buffer_size:
                        self._flush_buffer(file_path, lines)
                        buffers[file_path] = []
                        
            except queue.Empty:
                # Timeout: flush all buffers
                for file_path, lines in list(buffers.items()):
                    if lines:
                        self._flush_buffer(file_path, lines)
                        buffers[file_path] = []
    
    def _flush_buffer(self, path: Path, lines: List[str]) -> None:
        """Batch write to disk."""
        try:
            with open(path, 'a', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
        except Exception as e:
            print(f"[TELEMETRY ERROR] {e}")
```

**Beneficios:**
- Elimina bloqueos de I/O del thread principal
- Batch writes (100 líneas de una vez) → menos syscalls
- **Gana: ~300-400ms cada 2-3 segundos** = ~10-15% FPS improvement

**Riesgo:**
- Pérdida de últimos frames si el programa crashea (mitigable con `atexit` flush)

---

### 2. **CUDA Streams para Overlapping** ✅ YA IMPLEMENTADO

**Estado: COMPLETADO en central_worker.py**

```python
# Lines 83-85: Ya implementado
self.depth_stream = torch.cuda.Stream()
self.yolo_stream = torch.cuda.Stream()

# Lines 107-145: Ya en uso
with torch.cuda.stream(self.depth_stream):
    depth_tensor = self.depth_model.infer_image_gpu(frame_tensor, 384)

with torch.cuda.stream(self.yolo_stream):
    detections = self.yolo_processor.process_frame(frame, depth_map, depth_raw)

torch.cuda.synchronize()  # Wait for both
```

**Beneficios realizados:**
- ✅ Depth (27ms) y YOLO (7ms) se ejecutan en paralelo
- ✅ Overlap aprovecha idle time de GPU
- ✅ Latencia total ~32ms en vez de ~34ms secuencial

**Nota:** Ya está funcionando en producción desde commit 8e4e69a (multiprocessing).

---

### 3. **Optimizar TTS** 🔊 BAJO

Alternativas a `pyttsx3`:

**Opción A: Generación offline + playback**
```python
# Pre-generar WAV files al inicio
self.tts_cache = {
    "laptop": "audio/laptop.wav",
    "person": "audio/person.wav",
    ...
}

def speak_async(self, message):
    wav_file = self.tts_cache.get(message.lower())
    if wav_file:
        # sounddevice.play() es no-bloqueante
        sd.play(wav_data, samplerate=22050, blocking=False)
```

**Opción B: Festival/espeak directo (más rápido)**
```bash
# Festival es ~50ms más rápido que pyttsx3
festival --tts <<< "Laptop"
```

**Beneficios:**
- Elimina drops de 150ms durante TTS
- **Gana: ~100-150ms** cada vez que habla

---

### 4. **Frame Skipping Adaptativo** 📊 BAJO

Saltar frames automáticamente si latencia > threshold:

```python
if latency_ms > 60:  # Target: 50ms @ 20 FPS
    self.adaptive_skip = min(self.adaptive_skip + 1, 3)
else:
    self.adaptive_skip = max(self.adaptive_skip - 1, 0)
```

**Beneficios:**
- Mantiene FPS consistente bajo carga
- Evita acumulación de latencia

---

## 📊 Impacto Estimado

| Optimización | Ganancia FPS | Reducción Latencia | Prioridad | Estado |
|--------------|--------------|-------------------|-----------|--------|
| **Async Telemetry** | +2-3 FPS | -300ms spikes | ⭐⭐⭐ | ❌ Pendiente |
| **CUDA Streams** | ~2ms gained | Overlap depth+yolo | ✅ | ✅ HECHO |
| **TTS Optimizado** | +0.2 FPS | -100ms spikes | ⭐ | ❌ Pendiente |
| **Adaptive Skip** | Estabiliza | Previene acumulación | ⭐ | ❌ Pendiente |

**Target realista: 19.2 → 21-22 FPS** con Async Telemetry (CUDA streams ya aplicado)

---

## 🚀 Plan de Implementación

### Fase 1: Async Telemetry (1-2 horas) ⭐ ÚNICO PENDIENTE CRÍTICO
1. Crear `AsyncTelemetryLogger` class
2. Reemplazar en `main.py`
3. Testing: 5 minutos de ejecución continua
4. Validar: No más spikes >100ms después de warmup

### ~~Fase 2: CUDA Streams~~ ✅ YA IMPLEMENTADO
- Commit 8e4e69a: Multiprocessing con CUDA streams
- CentralWorker usa depth_stream y yolo_stream
- Funcionando en producción

### Fase 3: TTS Optimización (1 hora) - OPCIONAL
1. Pre-generar WAVs para palabras comunes
2. Fallback a pyttsx3 para frases dinámicas
3. Testing: Validar calidad de audio

---

## 📝 Notas

- **Actual:** 19.2 FPS promedio, spikes cada 2-3 segundos de 350-400ms
- **Target:** 21-22 FPS sostenido, sin spikes >100ms
- **Bottleneck principal:** I/O síncrono (300-400ms cada flush) ← ÚNICO PROBLEMA REAL
- **Quick win:** Async Telemetry elimina el 90% de los spikes
- **Ya optimizado:** ✅ Multiprocessing, ✅ CUDA streams, ✅ TensorRT RGB, ✅ ONNX CUDA depth

---

**Fecha:** 17 Nov 2025  
**Estado:** Análisis completado, pendiente implementación
