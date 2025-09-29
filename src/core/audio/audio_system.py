import platform
import shutil
import subprocess
import threading
import time
from collections import deque
from typing import Optional

class AudioSystem:
    """Directional audio command system optimized to avoid spam"""
    
    def __init__(self):
        self._setup_tts()
        
        # Audio control (solo TTS, sin lógica de selección de objetos)
        self.audio_queue = deque(maxlen=3)
        self.last_announcement_time = time.time()
        # Cooldown base entre frases (ajustable desde el coordinador/router)
        self.announcement_cooldown = 0.0
        # Cooldown adicional para repetir la misma frase
        self.repeat_cooldown = 2.0
        # Seguimiento de la última frase emitida
        self.last_phrase: Optional[str] = None
        self.last_phrase_time: float = 0.0
        
        # TTS state
        self.tts_speaking = False
        self._say_warned = False
        self._last_debug_ts = 0.0
        self._debug_interval = 2.0
        
        # Frame dimensions (updated by observer)
        self.frame_width = None
        self.frame_height = None
        
        print("[INFO] ✓ Audio navigation system initialized")
    
    @property
    def is_speaking(self) -> bool:
        """Current speaking state for observer compatibility"""
        return self.tts_speaking
    
    def _setup_tts(self):
        """Configure TTS using macOS `say` command"""
        self.tts_rate = 190
        self.say_available = platform.system() == 'Darwin' and shutil.which('say') is not None
        
        self.voice_preferences = ['Samantha', 'Alex', 'Victoria', 'Daniel']
        self.selected_voice = None
        
    def update_frame_dimensions(self, width: int, height: int) -> None:
        """Store frame dimensions for spatial processing (compatibilidad futura)"""
        self.frame_width = width
        self.frame_height = height

    # ------------------------------------------------------------------
    # Configuración expuesta al coordinador/router
    # ------------------------------------------------------------------

    def set_repeat_cooldown(self, seconds: float) -> None:
        """Actualizar cooldown para repetir la misma frase"""
        try:
            self.repeat_cooldown = max(0.1, float(seconds))
        except (TypeError, ValueError):
            pass

    def set_announcement_cooldown(self, seconds: float) -> None:
        """Actualizar cooldown base entre frases distintas"""
        try:
            value = float(seconds)
            if value < 0.0:
                value = 0.0
            self.announcement_cooldown = value
        except (TypeError, ValueError):
            pass

    def queue_message(self, message: str, *, force: bool = False) -> bool:
        """Encolar un mensaje para reproducción TTS."""
        if not message:
            return False
        return self.speak_async(message, force=force)
    
    def _should_announce(self, phrase: str) -> bool:
        """Decidir si se anuncia ahora en función de cooldowns"""
        if not self.say_available or self.tts_speaking:
            return False
        now = time.time()
        if phrase != self.last_phrase:
            return (now - self.last_announcement_time) >= self.announcement_cooldown
        return (now - self.last_announcement_time) >= self.repeat_cooldown
    
    def speak_async(self, message: str, *, force: bool = False) -> bool:
        """Speak message asynchronously respecting cooldowns.

        Args:
            message: Texto a reproducir.
            force: Si es True, ignora _should_announce y lanza el TTS igualmente.

        Returns:
            bool: True si se encola reproducción, False si se omite.
        """
        def _speak():
            try:
                if not self.tts_speaking and self.say_available:
                    self.tts_speaking = True
                    print(f"[AUDIO] 🔊 {message}")
                    
                    self.audio_queue.append(message)
                    
                    # Execute TTS
                    run_cmd = ["say", "-r", str(self.tts_rate)]
                    if self.selected_voice:
                        run_cmd.extend(["-v", self.selected_voice])
                    run_cmd.append(message)
                    subprocess.Popen(run_cmd)
                    
                    # Estimate speaking duration
                    words = max(1, len(message.split()))
                    duration = (words / max(100, self.tts_rate)) * 60.0 + 0.15
                    time.sleep(duration)
                    
            except Exception as e:
                print(f"[WARN] TTS error: {e}")
            finally:
                if self.audio_queue and self.audio_queue[-1] == message:
                    try:
                        self.audio_queue.pop()
                    except Exception:
                        pass
                self.tts_speaking = False

        if force or self._should_announce(message):
            self.last_phrase = message
            self.last_announcement_time = time.time()
            self.last_phrase_time = self.last_announcement_time
            threading.Thread(target=_speak, daemon=True).start()
            return True
        return False
    
    def speak_force(self, message: str):
        """Force speak bypassing cooldown (for testing)"""
        if not self.say_available:
            print("[WARN] TTS not available for force speak")
            return
            
        print(f"[AUDIO][FORCE] 🔊 {message}")
        try:
            run_cmd = ["say", "-r", str(self.tts_rate)]
            if self.selected_voice:
                run_cmd.extend(["-v", self.selected_voice])
            run_cmd.append(message)
            subprocess.Popen(run_cmd)
        except Exception as e:
            print(f"[WARN] Force speak failed: {e}")
    
    def close(self):
        """Cleanup TTS resources"""
        # No persistent processes to clean up in this implementation
        pass
