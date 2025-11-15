import platform
import shutil
import subprocess
import threading
import time
from collections import deque
from typing import Optional

# Try to import dependencies and handle missing ones
try:
    import numpy as np
except ImportError:
    np = None
    print("[WARN] Numpy not found. Beep functionality will be disabled.")

try:
    import sounddevice as sd
except ImportError:
    sd = None
    print("[WARN] Sounddevice not found. Beep functionality will be disabled.")

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None
    print("[WARN] pyttsx3 not found. TTS will be disabled on non-macOS systems.")


class AudioSystem:
    """Directional audio command system optimized to avoid spam, now multi-platform."""
    
    def __init__(self):
        self.tts_rate = 190
        self.tts_engine = None
        self.tts_backend: Optional[str] = None
        self._setup_tts()
        
        # Audio control
        self.audio_queue = deque(maxlen=3)
        self.last_announcement_time = time.time()
        self.announcement_cooldown = 0.0
        self.repeat_cooldown = 2.0
        self.last_phrase: Optional[str] = None
        self.last_phrase_time: float = 0.0
        
        # TTS state
        self.tts_speaking = False
        
        # Beep statistics
        self.beep_stats = {
            'critical_beeps': 0,
            'normal_beeps': 0,
            'critical_frequency': 0,
            'normal_frequency': 0,
        }
        
        # Frame dimensions
        self.frame_width = None
        self.frame_height = None
        
        print("[INFO] ✓ Audio navigation system initialized")
    
    @property
    def is_speaking(self) -> bool:
        return self.tts_speaking
    
    def _setup_tts(self):
        """Configure TTS based on the operating system."""
        system = platform.system()
        
        if system == "Darwin" and shutil.which('say'):
            self.tts_backend = "say"
            self.voice_preferences = ['Samantha', 'Alex', 'Victoria', 'Daniel']
            self.selected_voice = None
            print("[INFO] ✓ AudioSystem: Using 'say' for TTS on macOS.")
        
        elif system == "Linux" and pyttsx3:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', self.tts_rate)
                self.tts_backend = "pyttsx3"
                print("[INFO] ✓ AudioSystem: Using pyttsx3 for TTS on Linux.")
            except Exception as e:
                print(f"[ERROR] Failed to initialize pyttsx3 on Linux: {e}")
                self.tts_backend = None
        else:
            print(f"[WARN] No supported TTS backend found for {system}.")
            self.tts_backend = None

    def update_frame_dimensions(self, width: int, height: int) -> None:
        self.frame_width = width
        self.frame_height = height

    def set_repeat_cooldown(self, seconds: float) -> None:
        try:
            self.repeat_cooldown = max(0.1, float(seconds))
        except (TypeError, ValueError):
            pass

    def set_announcement_cooldown(self, seconds: float) -> None:
        try:
            self.announcement_cooldown = max(0.0, float(seconds))
        except (TypeError, ValueError):
            pass

    def queue_message(self, message: str, *, force: bool = False) -> bool:
        if not message:
            return False
        return self.speak_async(message, force=force)
    
    def _should_announce(self, phrase: str) -> bool:
        if not self.tts_backend or self.tts_speaking:
            return False
        now = time.time()
        if phrase != self.last_phrase:
            return (now - self.last_announcement_time) >= self.announcement_cooldown
        return (now - self.last_announcement_time) >= self.repeat_cooldown
    
    def speak_async(self, message: str, *, force: bool = False) -> bool:
        def _speak():
            try:
                self.tts_speaking = True
                print(f"[AUDIO] 🔊 {message}")
                self.audio_queue.append(message)
                
                if self.tts_backend == "say":
                    run_cmd = ["say", "-r", str(self.tts_rate)]
                    if hasattr(self, 'selected_voice') and self.selected_voice:
                        run_cmd.extend(["-v", self.selected_voice])
                    run_cmd.append(message)
                    # Use Popen for async execution
                    proc = subprocess.Popen(run_cmd)
                    proc.wait() # Wait for it to finish to manage state correctly
                
                elif self.tts_backend == "pyttsx3" and self.tts_engine:
                    self.tts_engine.say(message)
                    self.tts_engine.runAndWait() # This is blocking, perfect for a thread
                    
            except Exception as e:
                print(f"[WARN] TTS error: {e}")
            finally:
                if self.audio_queue and self.audio_queue[-1] == message:
                    try:
                        self.audio_queue.pop()
                    except IndexError:
                        pass
                self.tts_speaking = False

        if self.tts_backend and (force or self._should_announce(message)):
            self.last_phrase = message
            self.last_announcement_time = time.time()
            self.last_phrase_time = self.last_announcement_time
            threading.Thread(target=_speak, daemon=True).start()
            return True
        return False
    
    def play_spatial_beep(self, zone: str, is_critical: bool = False) -> None:
        from utils.config import Config
        
        if not getattr(Config, "AUDIO_SPATIAL_BEEPS_ENABLED", True):
            return
        
        if is_critical:
            freq = getattr(Config, "BEEP_CRITICAL_FREQUENCY", 1000)
            duration = getattr(Config, "BEEP_CRITICAL_DURATION", 0.3)
            self._play_tone(freq, duration, zone)
            self.beep_stats['critical_beeps'] += 1
            self.beep_stats['critical_frequency'] = freq
        else:
            freq = getattr(Config, "BEEP_NORMAL_FREQUENCY", 500)
            duration = getattr(Config, "BEEP_NORMAL_DURATION", 0.1)
            gap = getattr(Config, "BEEP_NORMAL_GAP", 0.05)
            count = getattr(Config, "BEEP_NORMAL_COUNT", 2)
            
            for i in range(count):
                self._play_tone(freq, duration, zone)
                if i < count - 1:
                    time.sleep(gap)
            self.beep_stats['normal_beeps'] += count
            self.beep_stats['normal_frequency'] = freq
    
    def _play_tone(self, frequency: float, duration: float, zone: str) -> None:
        if not np or not sd:
            if time.time() - getattr(self, '_last_beep_warn_ts', 0) > 5.0:
                print("[WARN] Cannot play beep. Numpy or Sounddevice not installed.")
                self._last_beep_warn_ts = time.time()
            return

        from utils.config import Config
        
        sample_rate = 44100
        volume = getattr(Config, "BEEP_VOLUME", 0.7)
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2 * np.pi * frequency * t)
        
        fade_samples = int(sample_rate * 0.01)
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        if len(tone) > fade_samples * 2:
            tone[:fade_samples] *= fade_in
            tone[-fade_samples:] *= fade_out
        tone *= volume
        
        if zone == "left":
            left = tone
            right = tone * 0.2
        elif zone == "right":
            left = tone * 0.2
            right = tone
        else:
            left = tone
            right = tone
        
        # Combine channels for stereo
        audio_data = np.column_stack((left, right))
        
        try:
            sd.play(audio_data, samplerate=sample_rate, blocking=False)
        except Exception as e:
            print(f"[WARN] Failed to play spatial beep with sounddevice: {e}")
    
    def get_beep_stats(self) -> dict:
        return dict(self.beep_stats)
    
    def get_queue_size(self) -> int:
        return len(self.audio_queue)
    
    def close(self):
        if self.tts_backend == "pyttsx3" and self.tts_engine:
            try:
                self.tts_engine.stop()
            except Exception:
                pass
        print("[INFO] AudioSystem closed.")