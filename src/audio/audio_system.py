import platform
import shutil
import subprocess
import threading
import time
from collections import deque
from typing import List, Optional

class AudioSystem:
    """Directional audio command system optimized to avoid spam"""
    
    def __init__(self):
        self._setup_tts()
        
        # Audio control
        self.audio_queue = deque(maxlen=3)
        self.last_announcement_time = time.time()
        # Lower base cooldown to reduce perceived delay
        self.announcement_cooldown = 0.8
        # Additional repeat cooldown for the same phrase
        self.repeat_cooldown = 2.0
        # Track last spoken phrase to allow immediate speech on changes
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
        """Store frame dimensions for spatial processing"""
        self.frame_width = width
        self.frame_height = height
    
    def process_detections(self, detections: List[dict], motion_state: str = "stationary") -> None:
        """Process detections and generate adaptive audio commands based on motion"""
        if not detections:
            return
        
        # NUEVO: Configuración adaptativa según movimiento
        if motion_state == "walking":
            # Más frecuente cuando caminas (navegación activa)
            self.repeat_cooldown = 1.5  # segundos
            max_objects = 1  # Solo objeto más importante
            prefix = "[WALKING]"
        else:
            # Menos frecuente cuando parado (exploración)
            self.repeat_cooldown = 3.0  # segundos  
            max_objects = 2  # Hasta 2 objetos
            prefix = "[STATIONARY]"
        
        # Filtrar objetos según estado de movimiento
        relevant_detections = detections[:max_objects]
        
        # Generar comando (usar primer objeto por ahora)
        detection = relevant_detections[0]
        print(f"[AUDIO DEBUG] {prefix} Detection: {detection}")
    
        command = self._generate_command(detection)
        print(f"[AUDIO DEBUG] {prefix} Generated command: '{command}'")
        
        if command and self._should_announce(command):
            print(f"[AUDIO DEBUG] {prefix} Sending to TTS: '{command}'")
            self.speak_async(command)
    
    def _generate_command(self, detection: dict) -> Optional[str]:
        """Generate simple directional command"""
        zone_mapping = {
            'center': 'center',
            'top_left': 'upper left',
            'top_right': 'upper right', 
            'bottom_left': 'lower left',
            'bottom_right': 'lower right'
        }
        
        zone_text = zone_mapping.get(detection['zone'], 'center')
        distance = detection.get('distance', '')
        
        # AÑADIR LA DISTANCIA AL MENSAJE:
        if distance:
            return f"{detection['name']} {distance} {zone_text}"
        else:
            return f"{detection['name']} {zone_text}"
    
    def _should_announce(self, phrase: str) -> bool:
        """Decide if we should speak now based on cooldown and phrase changes"""
        if not self.say_available or self.tts_speaking:
            return False
        now = time.time()
        # If phrase changed, allow immediate (no base cooldown)
        if phrase != self.last_phrase:
            return True
        # Same phrase: enforce repeat cooldown
        return (now - self.last_announcement_time) >= self.repeat_cooldown
    
    def speak_async(self, message: str):
        """Speak message asynchronously with cooldown control"""
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
        
        if self._should_announce(message):
            self.last_phrase = message
            self.last_announcement_time = time.time()
            self.last_phrase_time = self.last_announcement_time
            threading.Thread(target=_speak, daemon=True).start()
    
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
