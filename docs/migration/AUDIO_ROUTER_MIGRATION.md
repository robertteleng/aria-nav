# Guía de Migración del Sistema de Audio (`AudioSystem`)

Este documento detalla el proceso técnico para refactorizar el `AudioSystem` del proyecto y hacerlo compatible con múltiples plataformas (especialmente Linux), eliminando las dependencias exclusivas de macOS.

## 1. Objetivo Principal

El `AudioSystem` actual utiliza las herramientas de línea de comandos `say` (para Texto-a-Voz) y `afplay` (para reproducir sonidos), las cuales solo existen en macOS. El objetivo es reemplazar estas herramientas por librerías de Python que ofrezcan la misma funcionalidad en diferentes sistemas operativos.

## 2. Herramientas Recomendadas

Para lograr la compatibilidad multiplataforma, se recomienda usar las siguientes librerías:

1.  **Para la Voz (TTS): `pyttsx3`**
    *   **¿Por qué?** Es un wrapper que interactúa con los motores de TTS nativos de cada sistema operativo.
    *   **En Linux:** Utiliza `espeak-ng`. Requiere instalación previa (`sudo apt-get install espeak-ng`).
    *   **En macOS:** Puede usar el motor `NSSpeechSynthesizer` nativo.
    *   **En Windows:** Utiliza la API de voz de Windows (SAPI5).

2.  **Para los Sonidos (Beeps): `sounddevice`**
    *   **¿Por qué?** Permite reproducir arrays de `numpy` directamente en el hardware de audio, eliminando la necesidad de crear archivos `.wav` temporales y de depender de un programa reproductor externo. Es más eficiente y limpio.

## 3. Pasos para la Migración

A continuación se describen los cambios necesarios en el archivo `src/core/audio/audio_system.py`.

### Paso 1: Actualizar Dependencias

Asegúrate de que las siguientes librerías estén en tu archivo `requirements.txt` y se instalen en el entorno virtual:

```
pyttsx3
sounddevice
numpy
```

### Paso 2: Refactorizar la Inicialización (`__init__` y `_setup_tts`)

El objetivo es detectar el sistema operativo al inicio y configurar el motor de audio correspondiente.

**Lógica Propuesta:**

Se introduce una variable de instancia, `self.tts_backend`, para saber qué motor usar en las demás funciones.

```python
# Concepto para el método _setup_tts
import platform
import shutil

def _setup_tts(self):
    """Configura el motor de TTS según el sistema operativo."""
    self.tts_backend = None
    system = platform.system()

    if system == "Linux":
        try:
            import pyttsx3
            # Inicializa el motor de pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.tts_rate)
            self.tts_backend = "pyttsx3"
            print("[INFO] ✓ AudioSystem: Usando pyttsx3 para TTS en Linux.")
        except Exception as e:
            print(f"[ERROR] No se pudo inicializar pyttsx3 en Linux: {e}")

    elif system == "Darwin":
        if shutil.which('say'):
            # Mantiene la lógica actual para macOS
            self.tts_backend = "say"
            print("[INFO] ✓ AudioSystem: Usando 'say' para TTS en macOS.")
        else:
            print("[ERROR] Comando 'say' no encontrado en macOS.")
            
    else:
        print(f"[WARN] Sistema operativo {system} no soportado para TTS.")

```

### Paso 3: Refactorizar la Reproducción de Voz (`speak_async`)

Esta función debe modificarse para usar el motor de TTS que se inicializó en el paso anterior. La lógica se ejecuta dentro del hilo `_speak`.

**Lógica Propuesta:**

```python
# Concepto para la función interna _speak()
def _speak():
    try:
        if not self.tts_speaking:
            self.tts_speaking = True
            print(f"[AUDIO] 🔊 {message}")
            
            # --- INICIO DEL CAMBIO ---
            if self.tts_backend == "say":
                # Lógica actual para macOS
                run_cmd = ["say", "-r", str(self.tts_rate)]
                # ... (resto del comando)
                subprocess.Popen(run_cmd)
                # ... (estimación de duración)

            elif self.tts_backend == "pyttsx3":
                # Nueva lógica para Linux (y otros)
                self.engine.say(message)
                self.engine.runAndWait() # Esta función es bloqueante
            # --- FIN DEL CAMBIO ---

    except Exception as e:
        print(f"[WARN] TTS error: {e}")
    finally:
        self.tts_speaking = False
```
**Nota:** Dado que `runAndWait()` es bloqueante, es fundamental que siga ejecutándose dentro de un hilo (`threading.Thread`), tal como está diseñado actualmente, para no congelar la aplicación principal.

### Paso 4: Refactorizar la Reproducción de Sonidos (`_play_tone`)

Aquí se reemplaza la creación de archivos temporales y la llamada a `afplay` por una llamada directa a `sounddevice`.

**Lógica Propuesta:**

Se mantiene toda la generación del array de `numpy`, pero se simplifica drásticamente la parte de la reproducción.

```python
# Concepto para el método _play_tone
def _play_tone(self, frequency: float, duration: float, zone: str) -> None:
    import numpy as np
    # Se asume que 'sounddevice' se importa al inicio del archivo o aquí
    import sounddevice as sd

    # 1. Se mantiene toda la lógica que genera el array estéreo 'audio_data'
    # ... (generación de onda, volumen, fades, canales estéreo)
    sample_rate = 44100 # Esta variable debe estar disponible

    # 2. Se reemplaza el bloque try/except que usa tempfile y afplay
    try:
        # La nueva forma de reproducir, simple y directa:
        sd.play(audio_data, samplerate=sample_rate, blocking=False)
    except Exception as e:
        print(f"[WARN] Failed to play spatial beep with sounddevice: {e}")

```
`blocking=False` asegura que la reproducción se inicie y el programa continúe, replicando el comportamiento asíncrono de `subprocess.Popen`.

## 4. Conclusión

Al implementar estos cambios, el `AudioSystem` se volverá agnóstico al sistema operativo, cumpliendo un requisito fundamental del plan de migración. Esto no solo permitirá que el proyecto funcione en el NUC con Linux, sino que también lo hará más robusto y fácil de mantener o portar a otras plataformas en el futuro.
