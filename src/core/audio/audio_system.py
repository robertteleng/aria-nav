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
    
    def play_spatial_beep(self, zone: str, is_critical: bool = False) -> None:
        """Play spatialized beep based on zone and priority.
        
        Args:
            zone: "left", "center", or "right"
            is_critical: True for CRITICAL (high pitch, long), False for NORMAL (low pitch, short)
        """
        from utils.config import Config
        
        if not getattr(Config, "AUDIO_SPATIAL_BEEPS_ENABLED", True):
            return
        
        if is_critical:
            # CRITICAL: Single long high-pitched beep
            freq = getattr(Config, "BEEP_CRITICAL_FREQUENCY", 1000)
            duration = getattr(Config, "BEEP_CRITICAL_DURATION", 0.3)
            self._play_tone(freq, duration, zone)
        else:
            # NORMAL: Two short low-pitched beeps
            freq = getattr(Config, "BEEP_NORMAL_FREQUENCY", 500)
            duration = getattr(Config, "BEEP_NORMAL_DURATION", 0.1)
            gap = getattr(Config, "BEEP_NORMAL_GAP", 0.05)
            count = getattr(Config, "BEEP_NORMAL_COUNT", 2)
            
            for i in range(count):
                self._play_tone(freq, duration, zone)
                if i < count - 1:
                    time.sleep(gap)
    
    def _play_tone(self, frequency: float, duration: float, zone: str) -> None:
        """Generate and play a tone with spatial positioning using macOS afplay.
        
        Args:
            frequency: Tone frequency in Hz
            duration: Duration in seconds
            zone: "left", "center", or "right" for spatial positioning
        """
        import numpy as np
        import tempfile
        import os
        from utils.config import Config
        
        sample_rate = 44100
        volume = getattr(Config, "BEEP_VOLUME", 0.7)
        
        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2 * np.pi * frequency * t)
        
        # Apply volume and fade in/out to avoid clicks
        fade_samples = int(sample_rate * 0.01)  # 10ms fade
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        tone[:fade_samples] *= fade_in
        tone[-fade_samples:] *= fade_out
        tone *= volume
        
        # Create stereo with spatial positioning
        if zone == "left":
            left = tone
            right = tone * 0.2  # Reduce right channel
        elif zone == "right":
            left = tone * 0.2  # Reduce left channel
            right = tone
        else:  # center
            left = tone
            right = tone
        
        # Interleave stereo channels
        stereo = np.empty((len(tone) * 2,), dtype=tone.dtype)
        stereo[0::2] = left
        stereo[1::2] = right
        
        # Convert to 16-bit PCM
        audio_data = (stereo * 32767).astype(np.int16)
        
        # Write to temporary WAV file
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_path = f.name
                
                # Write WAV header manually
                import struct
                import wave
                
                with wave.open(temp_path, 'w') as wav_file:
                    wav_file.setnchannels(2)  # Stereo
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data.tobytes())
            
            # Play using afplay (macOS)
            subprocess.Popen(['afplay', temp_path], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            
            # Schedule deletion after playback (async)
            def cleanup():
                time.sleep(duration + 0.1)
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
            threading.Thread(target=cleanup, daemon=True).start()
            
        except Exception as e:
            print(f"[WARN] Failed to play spatial beep: {e}")
    
    def close(self):
        """Cleanup TTS resources"""
        # No persistent processes to clean up in this implementation
        pass
