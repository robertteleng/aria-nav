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

# Import config at module level to avoid import errors
try:
    from utils.config import Config
except ImportError:
    Config = None
    print("[WARN] Config not found. Using default audio settings.")

# Import logger at module level
try:
    from core.telemetry.loggers.navigation_logger import get_navigation_logger
except ImportError:
    get_navigation_logger = None
    print("[WARN] Navigation logger not found. Debug logging will be disabled.")


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
        self.tts_rate = 130  # Default rate, will be adjusted per platform
        
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
            self.tts_rate = 190
            self.voice_preferences = ['Samantha', 'Alex', 'Victoria', 'Daniel']
            self.selected_voice = None
            print("[INFO] ✓ AudioSystem: Using 'say' for TTS on macOS.")
        
        elif system == "Linux" and pyttsx3:
            try:
                self.tts_engine = pyttsx3.init()
                # Slower rate for Linux (espeak-ng is fast by default)
                self.tts_rate = 130  # Much slower for clarity
                self.tts_engine.setProperty('rate', self.tts_rate)
                self.tts_engine.setProperty('volume', 1.0)
                self.tts_backend = "pyttsx3"
                print(f"[INFO] ✓ AudioSystem: Using pyttsx3 for TTS on Linux (rate={self.tts_rate}).")
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
                    # FASE 4 FIX: Use Popen WITHOUT wait() for true async
                    # The daemon thread will exit naturally, no need to block
                    subprocess.Popen(run_cmd)
                    # Give TTS process time to start
                    time.sleep(0.1)
                
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

        should_speak = force or self._should_announce(message)
        if get_navigation_logger:
            logger = get_navigation_logger().audio
            logger.debug(f"speak_async('{message}', force={force}) -> should={should_speak}, backend={self.tts_backend}")
        
        if self.tts_backend and should_speak:
            self.last_phrase = message
            self.last_announcement_time = time.time()
            self.last_phrase_time = self.last_announcement_time
            threading.Thread(target=_speak, daemon=True).start()
            return True
        return False
    
    def play_spatial_beep(self, zone: str, is_critical: bool = False, distance: Optional[str] = None) -> None:
        """Play spatial audio beep in a separate thread to avoid blocking TTS.

        Beeps run asynchronously and never interfere with TTS announcements.
        """
        if not getattr(Config, "AUDIO_SPATIAL_BEEPS_ENABLED", True) if Config else True:
            return

        def _play_beeps():
            """Play beeps in background thread."""
            try:
                if is_critical:
                    freq = getattr(Config, "BEEP_CRITICAL_FREQUENCY", 1000)
                    duration = getattr(Config, "BEEP_CRITICAL_DURATION", 0.3)
                    self._play_tone(freq, duration, zone, distance)
                    self.beep_stats['critical_beeps'] += 1
                    self.beep_stats['critical_frequency'] = freq
                else:
                    freq = getattr(Config, "BEEP_NORMAL_FREQUENCY", 500)
                    duration = getattr(Config, "BEEP_NORMAL_DURATION", 0.1)
                    gap = getattr(Config, "BEEP_NORMAL_GAP", 0.05)
                    count = getattr(Config, "BEEP_NORMAL_COUNT", 2)

                    for i in range(count):
                        self._play_tone(freq, duration, zone, distance)
                        if i < count - 1:
                            time.sleep(gap)  # Sleep is OK here - we're in a separate thread
                    self.beep_stats['normal_beeps'] += count
                    self.beep_stats['normal_frequency'] = freq
            except Exception as e:
                print(f"[WARN] Beep error: {e}")

        # Run beeps in daemon thread - completely independent from TTS
        threading.Thread(target=_play_beeps, daemon=True).start()
    
    def _play_tone(self, frequency: float, duration: float, zone: str, distance: Optional[str] = None) -> None:
        if not np or not sd:
            if time.time() - getattr(self, '_last_beep_warn_ts', 0) > 5.0:
                print("[WARN] Cannot play beep. Numpy or Sounddevice not installed.")
                self._last_beep_warn_ts = time.time()
            return

        sample_rate = 44100
        base_volume = getattr(Config, "BEEP_VOLUME", 0.7) if Config else 0.7

        # 🆕 VOLUMEN DINÁMICO por distancia (mantiene panning intacto)
        distance_multipliers = {
            "very_close": 1.0,   # 100% - máximo volumen
            "close": 0.7,        # 70% - volumen medio-alto
            "medium": 0.45,      # 45% - volumen medio-bajo
            "far": 0.25          # 25% - volumen suave
        }
        distance_multiplier = distance_multipliers.get(distance, 0.7) if distance else 0.7
        volume = base_volume * distance_multiplier

        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(2 * np.pi * frequency * t)

        fade_samples = int(sample_rate * 0.01)
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        if len(tone) > fade_samples * 2:
            tone[:fade_samples] *= fade_in
            tone[-fade_samples:] *= fade_out
        tone *= volume

        # Panning espacial (ratio entre canales, NO volumen absoluto)
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
    
    def scan_scene(self, navigation_objects: list) -> None:
        """🆕 MODO SCAN: Resume audible de 3-5 objetos principales en la escena.

        Inspirado en NOA - genera frase breve con objetos agrupados por zona.
        Ejemplo: "Scanning. Ahead: person, chair. Left: table."
        """
        if not navigation_objects:
            self.speak_async("Scanning. No objects detected.", force=True)
            return

        # Agrupar top 5 objetos por zona
        zone_objects = {"center": [], "left": [], "right": []}
        for obj in navigation_objects[:5]:
            zone = obj.get("zone", "center")
            class_name = obj.get("class", "object")

            # Traducir nombres a speech-friendly
            speech_labels = {
                "person": "person",
                "car": "car",
                "truck": "truck",
                "bus": "bus",
                "bicycle": "bike",
                "motorcycle": "motorcycle",
                "chair": "chair",
                "table": "table",
                "bottle": "bottle",
                "door": "door",
                "laptop": "laptop",
                "couch": "couch",
                "bed": "bed",
            }
            label = speech_labels.get(class_name, class_name)
            zone_objects[zone].append(label)

        # Construir frase
        parts = ["Scanning."]
        zone_names = {"center": "Ahead", "left": "Left", "right": "Right"}

        for zone in ["center", "left", "right"]:
            if zone_objects[zone]:
                objects_str = ", ".join(zone_objects[zone][:3])  # Max 3 por zona
                parts.append(f"{zone_names[zone]}: {objects_str}.")

        message = " ".join(parts)
        self.speak_async(message, force=True)  # Force=True para override cooldowns

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