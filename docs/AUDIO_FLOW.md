# Audio System Architecture - Beeps + TTS

## Resumen

El sistema de audio de Aria Navigation combina **dos canales independientes**:

1. **TTS (Text-to-Speech)**: Anuncios verbales de objetos y eventos
2. **Beeps espaciales**: Tonos direccionales que indican posición y urgencia

**Característica clave**: Ambos sistemas funcionan **completamente en paralelo** mediante threading, sin bloquearse entre sí.

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────┐
│                   AudioSystem                           │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────────┐   ┌──────────────────────┐   │
│  │   TTS Channel        │   │   Beep Channel       │   │
│  │  (speak_async)       │   │ (play_spatial_beep)  │   │
│  └──────────────────────┘   └──────────────────────┘   │
│           │                           │                 │
│           │                           │                 │
│           ▼                           ▼                 │
│  ┌──────────────────────┐   ┌──────────────────────┐   │
│  │ Thread(daemon=True)  │   │ Thread(daemon=True)  │   │
│  │   _speak()           │   │   _play_beeps()      │   │
│  └──────────────────────┘   └──────────────────────┘   │
│           │                           │                 │
│           │                           │                 │
│           ▼                           ▼                 │
│  ┌──────────────────────┐   ┌──────────────────────┐   │
│  │  macOS: `say`        │   │ sounddevice.play()   │   │
│  │  Linux: pyttsx3      │   │ (stereo panning)     │   │
│  └──────────────────────┘   └──────────────────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Flujo de Ejecución

### 1. TTS Announcements (speak_async)

**Ejemplo**: "Person ahead"

```python
# Llamada desde NavigationDecisionEngine
audio_system.speak_async("Person ahead", force=False)

# Flujo interno:
1. _should_announce(message) → check cooldowns (SOLO TTS, NO beeps)
   - Repeat cooldown: Mismo mensaje < 2s → skip
   - Announcement cooldown: Cualquier mensaje < 0s → skip

2. threading.Thread(target=_speak, daemon=True).start()
   ↓
3. _speak() ejecuta en thread separado:
   - Set tts_speaking = True
   - Ejecuta TTS backend (say o pyttsx3)
   - Set tts_speaking = False
```

**Puntos clave:**
- ✅ **NO** bloquea el main thread
- ✅ **NO** se ve afectado por beeps
- ✅ Cooldowns solo afectan TTS, no beeps

### 2. Spatial Beeps (play_spatial_beep)

**Ejemplo**: Beep en zona "left" con distancia "close"

```python
# Llamada desde RgbAudioRouter
audio_system.play_spatial_beep(zone="left", is_critical=False, distance="close")

# Flujo interno:
1. threading.Thread(target=_play_beeps, daemon=True).start()
   ↓
2. _play_beeps() ejecuta en thread separado:
   - Si critical: 1 beep largo (1000Hz, 0.3s)
   - Si normal: 2 beeps cortos (500Hz, 0.1s, gap 0.05s)
   ↓
3. Para cada beep → _play_tone(freq, duration, zone, distance)
   - Genera tono con numpy
   - Aplica volumen dinámico por distancia:
     * very_close: 100% volumen
     * close: 70% volumen
     * medium: 45% volumen
     * far: 25% volumen
   - Aplica panning espacial:
     * left: L=100%, R=20%
     * right: L=20%, R=100%
     * center: L=100%, R=100%
   - sounddevice.play(audio_data, blocking=False)
```

**Puntos clave:**
- ✅ **NO** bloquea el main thread
- ✅ **NO** afecta a TTS
- ✅ `time.sleep(gap)` entre beeps es OK (estamos en thread separado)

## Independencia de Canales

### ❌ Problema Anterior (ANTES DEL FIX)

```python
# _should_announce (líneas 127-128 - VERSIÓN ANTIGUA)
if self.tts_speaking and self.announcement_cooldown > 0.1:
    return False  # ❌ TTS bloqueado si ya estaba hablando!

# play_spatial_beep (líneas 194-197 - VERSIÓN ANTIGUA)
for i in range(count):
    self._play_tone(...)
    if i < count - 1:
        time.sleep(gap)  # ❌ Bloqueaba main thread!
```

**Resultado**: Beeps bloqueaban TTS porque:
1. `play_spatial_beep()` ejecutaba en main thread
2. `time.sleep(0.05)` pausaba todo el proceso
3. `speak_async()` llamado durante el sleep → `tts_speaking` check → rechazado

### ✅ Solución Implementada (DESPUÉS DEL FIX)

```python
# _should_announce (líneas 123-146 - VERSIÓN NUEVA)
def _should_announce(self, phrase: str) -> bool:
    """Check if a TTS announcement should be made.

    Beeps and TTS are completely independent - beeps never block TTS.
    Only TTS cooldowns affect TTS announcements.
    """
    if not self.tts_backend:
        return False

    now = time.time()

    # Check if it's a repeated phrase
    if phrase == self.last_phrase:
        return (now - self.last_phrase_time) >= self.repeat_cooldown

    # Different phrase - check announcement cooldown
    return (now - self.last_announcement_time) >= self.announcement_cooldown
    # ✅ NO check de tts_speaking!

# play_spatial_beep (líneas 192-225 - VERSIÓN NUEVA)
def play_spatial_beep(self, zone: str, is_critical: bool = False, distance: Optional[str] = None) -> None:
    """Play spatial audio beep in a separate thread to avoid blocking TTS."""

    def _play_beeps():
        """Play beeps in background thread."""
        try:
            # ... beep logic ...
            for i in range(count):
                self._play_tone(freq, duration, zone, distance)
                if i < count - 1:
                    time.sleep(gap)  # ✅ OK en thread separado!
        except Exception as e:
            print(f"[WARN] Beep error: {e}")

    # ✅ Thread daemon - NO bloquea nada
    threading.Thread(target=_play_beeps, daemon=True).start()
```

## Casos de Uso

### Caso 1: TTS + Beep simultáneos

```python
# Usuario acercándose a obstáculo
audio_system.speak_async("Obstacle ahead", force=True)  # Thread 1
audio_system.play_spatial_beep("center", is_critical=True, distance="very_close")  # Thread 2

# ✅ Resultado: TTS habla MIENTRAS beeps suenan (paralelo)
```

### Caso 2: Múltiples beeps durante TTS

```python
audio_system.speak_async("Multiple objects around you", force=True)
time.sleep(0.1)
audio_system.play_spatial_beep("left", distance="close")    # No bloquea
time.sleep(0.2)
audio_system.play_spatial_beep("right", distance="medium")  # No bloquea
time.sleep(0.2)
audio_system.play_spatial_beep("center", distance="far")    # No bloquea

# ✅ Resultado: TTS continúa mientras beeps suenan en diferentes momentos
```

### Caso 3: Beeps rápidos + TTS

```python
# Detección rápida de múltiples objetos
for obj in objects:
    audio_system.play_spatial_beep(obj.zone, distance=obj.distance)  # Threads en paralelo

audio_system.speak_async(f"{len(objects)} objects detected", force=True)

# ✅ Resultado: Todos los beeps + TTS ejecutan sin bloquearse
```

## Configuración

### TTS Settings

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `repeat_cooldown` | Tiempo mínimo entre repeticiones del mismo mensaje | 2.0s |
| `announcement_cooldown` | Tiempo mínimo entre mensajes diferentes | 0.0s |
| `tts_rate` | Velocidad de habla (macOS: 190, Linux: 130) | Platform-dependent |

### Beep Settings

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `AUDIO_SPATIAL_BEEPS_ENABLED` | Habilitar/deshabilitar beeps | True |
| `BEEP_CRITICAL_FREQUENCY` | Frecuencia para objetos críticos | 1000 Hz |
| `BEEP_CRITICAL_DURATION` | Duración de beep crítico | 0.3s |
| `BEEP_NORMAL_FREQUENCY` | Frecuencia para objetos normales | 500 Hz |
| `BEEP_NORMAL_DURATION` | Duración de beep normal | 0.1s |
| `BEEP_NORMAL_GAP` | Pausa entre beeps normales | 0.05s |
| `BEEP_NORMAL_COUNT` | Número de beeps normales | 2 |
| `BEEP_VOLUME` | Volumen base (antes de distance multiplier) | 0.7 |

### Distance Multipliers (Volumen dinámico)

```python
distance_multipliers = {
    "very_close": 1.0,   # 100% - máximo volumen
    "close": 0.7,        # 70% - volumen medio-alto
    "medium": 0.45,      # 45% - volumen medio-bajo
    "far": 0.25          # 25% - volumen suave
}
```

**Fórmula final**:
```
volume_final = BEEP_VOLUME * distance_multiplier
```

**Ejemplo**:
- `BEEP_VOLUME = 0.7`
- `distance = "close"` → multiplier = 0.7
- `volume_final = 0.7 * 0.7 = 0.49` (49% del máximo)

## Testing

### Test Automático

```bash
python3 examples/test_beep_tts_fix.py
```

**Verifica**:
- ✅ TTS + beeps simultáneos
- ✅ Beeps durante TTS
- ✅ Múltiples beeps sin bloqueo
- ✅ Rapid fire (stress test)

### Test Manual

1. **TTS solo**:
   ```python
   audio.speak_async("Test message", force=True)
   ```
   Deberías escuchar el mensaje hablado.

2. **Beep solo**:
   ```python
   audio.play_spatial_beep("left", is_critical=False, distance="close")
   ```
   Deberías escuchar 2 beeps en el canal izquierdo.

3. **Ambos simultáneos**:
   ```python
   audio.speak_async("Simultaneous test", force=True)
   audio.play_spatial_beep("center", is_critical=True, distance="very_close")
   ```
   Deberías escuchar TTS + beep al mismo tiempo.

## Troubleshooting

### "TTS no se escucha"

- **macOS**: Verifica que `say` esté instalado: `which say`
- **Linux**: Verifica que `pyttsx3` esté instalado: `pip install pyttsx3`
- **Check logs**: Busca `[AUDIO] 🔊` en la salida

### "Beeps no se escuchan"

- Verifica que `sounddevice` esté instalado: `pip install sounddevice`
- Verifica que `numpy` esté instalado: `pip install numpy`
- Check config: `Config.AUDIO_SPATIAL_BEEPS_ENABLED = True`

### "Beeps todavía bloquean TTS"

- Verifica que tengas la versión corregida de `audio_system.py`
- Check línea 130: `_should_announce` NO debe tener check de `tts_speaking`
- Check línea 225: `play_spatial_beep` debe usar `threading.Thread`

### "Volumen de beeps no cambia con distancia"

- Verifica que el objeto tenga campo `distance`:
  ```python
  obj = {"class": "person", "zone": "left", "distance": "close"}
  ```
- Check que `rgb_audio_router.py` pase `distance` a `play_spatial_beep()`

## Métricas

### Beep Statistics

```python
stats = audio_system.get_beep_stats()
print(stats)
# {
#   'critical_beeps': 5,
#   'normal_beeps': 12,
#   'critical_frequency': 1000,
#   'normal_frequency': 500
# }
```

### Audio Queue Size

```python
queue_size = audio_system.get_queue_size()
print(f"TTS queue: {queue_size} messages")
```

## Referencias

- **Archivo principal**: [src/core/audio/audio_system.py](../src/core/audio/audio_system.py)
- **Test del fix**: [examples/test_beep_tts_fix.py](../examples/test_beep_tts_fix.py)
- **Documentación de mejoras**: [AUDIO_TRACKING_IMPROVEMENTS.md](./AUDIO_TRACKING_IMPROVEMENTS.md)
