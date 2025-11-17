# Solución de Audio para Linux

## Resumen

Se ha implementado exitosamente el soporte de audio multiplataforma siguiendo la guía `AUDIO_MIGRATION_GUIDE.md`. El sistema ahora funciona tanto en macOS como en Linux usando librerías Python portables.

## Problema Original

El sistema de audio usaba comandos específicos de macOS:
- `say` para text-to-speech (TTS)
- `afplay` para reproducción de audio

Estos comandos **no existen en Linux**, causando que el audio no funcionara.

## Solución Implementada

### 1. Text-to-Speech (TTS)

#### Dependencias Instaladas
```bash
# Sistema
sudo apt-get install espeak-ng

# Python
pip install pyttsx3==2.98
```

#### Implementación
- **macOS**: Continúa usando el comando nativo `say`
- **Linux**: Usa `pyttsx3` con backend `espeak-ng`
- Detección automática del sistema operativo en `_setup_tts()`
- Ejecución asíncrona en ambas plataformas (threading)

### 2. Beeps Espaciales

#### Tecnología
- Usa `sounddevice` (ya estaba instalado)
- Reproduce arrays numpy directamente al hardware de audio
- Elimina necesidad de archivos temporales y reproductores externos

#### Ventajas
- Más eficiente (sin I/O de archivos)
- Multiplataforma por diseño
- Control preciso del audio estéreo para direccionalidad

### 3. Cambios en el Código

#### `src/core/audio/audio_system.py`

**Importaciones**:
```python
# TTS multiplataforma
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

# Config al nivel del módulo (evita import errors)
try:
    from utils.config import Config
except ImportError:
    Config = None
```

**Configuración de TTS**:
```python
def _setup_tts(self):
    system = platform.system()
    
    if system == "Darwin" and shutil.which('say'):
        self.tts_backend = "say"  # macOS
    elif system == "Linux" and pyttsx3:
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', self.tts_rate)
        self.tts_backend = "pyttsx3"  # Linux
```

**Ejecución de TTS**:
```python
if self.tts_backend == "say":
    # macOS: subprocess sin wait (async)
    subprocess.Popen(run_cmd)
elif self.tts_backend == "pyttsx3":
    # Linux: pyttsx3 (bloqueante, por eso en thread)
    self.tts_engine.say(message)
    self.tts_engine.runAndWait()
```

**Beeps Espaciales** (sin cambios, ya era multiplataforma):
```python
def _play_tone(self, frequency, duration, zone):
    # Genera numpy array con tono
    audio_data = np.column_stack((left, right))
    # Reproduce directamente con sounddevice
    sd.play(audio_data, samplerate=sample_rate, blocking=False)
```

#### `requirements.txt`
```
pyttsx3==2.98
```

### 4. Testing

#### Script de Test
`test_audio_linux.py` - Prueba completa del sistema:
1. TTS en español
2. Beeps espaciales (left, right, center)
3. Beeps críticos y normales
4. Estadísticas

#### Resultados
✅ TTS: Funciona correctamente con espeak-ng
✅ Beeps espaciales: Direccionalidad correcta (stereo)
✅ Beeps críticos: Frecuencia y duración correctas
✅ Sistema asíncrono: Sin bloqueos

```bash
[INFO] ✓ AudioSystem: Using pyttsx3 for TTS on Linux.
[AUDIO] 🔊 Sistema de audio en Linux funcionando correctamente
✓ All audio tests passed successfully!
```

## Arquitectura

```
AudioSystem (audio_system.py)
├── TTS Backend Detection
│   ├── macOS → 'say' command
│   └── Linux → pyttsx3 + espeak-ng
├── Spatial Beeps
│   └── sounddevice + numpy (multiplataforma)
└── Thread Management
    └── Daemon threads for async execution
```

## Ventajas de la Solución

1. **Multiplataforma**: Funciona en macOS y Linux sin cambios
2. **Portable**: Solo dependencias Python + espeak-ng
3. **Eficiente**: Sin archivos temporales ni procesos externos pesados
4. **Mantenible**: Código Python puro, más fácil de debuggear
5. **Robusto**: Fallbacks y manejo de errores apropiados

## Configuración en Nuevos Sistemas

### Ubuntu/Debian
```bash
sudo apt-get install espeak-ng
pip install pyttsx3==2.98
```

### Arch Linux
```bash
sudo pacman -S espeak-ng
pip install pyttsx3==2.98
```

### Fedora/RHEL
```bash
sudo dnf install espeak-ng
pip install pyttsx3==2.98
```

## Pruebas Realizadas

- ✅ Instalación en Ubuntu 24.04 con RTX 2060
- ✅ TTS en español con espeak-ng
- ✅ Beeps espaciales estéreo
- ✅ Integración con pipeline completo
- ✅ Sin bloqueos ni latencia adicional

## Próximos Pasos

1. **Opcional**: Ajustar voz de espeak-ng (velocidad, tono, idioma)
2. **Opcional**: Explorar otros backends TTS para Linux (festival, flite)
3. **Testing**: Validar en sesiones largas del sistema completo

## Referencias

- `AUDIO_MIGRATION_GUIDE.md` - Guía original de migración
- `test_audio_linux.py` - Script de validación
- Commit: `2a14673` - "feat: Add Linux audio support with pyttsx3 and espeak-ng"

## Notas Técnicas

### pyttsx3 vs subprocess
- `pyttsx3.runAndWait()` es **bloqueante**
- Se ejecuta en daemon thread para no bloquear el pipeline
- En macOS, `subprocess.Popen()` sin `wait()` es no-bloqueante (ya arreglado en FASE 4)

### espeak-ng
- Motor TTS ligero y rápido
- Soporte para múltiples idiomas
- Voz robótica pero inteligible
- Latencia mínima (~100-200ms)

### Fallbacks
- Si Config no se importa: valores por defecto (volumen 0.7, etc.)
- Si pyttsx3 no disponible: TTS deshabilitado con warning
- Si sounddevice no disponible: Beeps deshabilitados con warning

---

**Autor**: Sistema de IA con documentación de migración  
**Fecha**: 17 de Noviembre, 2025  
**Estado**: ✅ Implementado y validado
