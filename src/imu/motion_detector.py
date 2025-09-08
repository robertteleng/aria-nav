from collections import deque
from typing import Literal

MotionState = Literal["stationary", "walking"]

class SimpleMotionDetector:
    def __init__(self):
        self.magnitude_history = deque(maxlen=50)  # ~0.5 segundos de historia
        self.last_motion_state: MotionState = "stationary"
        
        # Thresholds para motion detection
        self.stationary_threshold = 0.5  # m/s² variación para "parado"
        self.walking_threshold = 1.0     # m/s² variación para "caminando"
        
    def update(self, magnitude: float, timestamp_ns: int) -> MotionState:
        """Actualiza estado de movimiento basado en variación de aceleración"""

        self.last_magnitude = magnitude
        self.magnitude_history.append(magnitude)
        
        if len(self.magnitude_history) < 10:  # Necesitamos historia mínima
            return self.last_motion_state
            
        # Calcular variación en la ventana
        magnitudes = list(self.magnitude_history)
        variance = self._calculate_variance(magnitudes)
        
        # Determinar estado basado en variación
        if variance < self.stationary_threshold:
            new_state: MotionState = "stationary"
        elif variance > self.walking_threshold:
            new_state: MotionState = "walking"
        else:
            new_state = self.last_motion_state  # Mantener estado anterior (hysteresis)
            
        self.last_motion_state = new_state
        return new_state
    
    def _calculate_variance(self, values):
        """Calcular standard deviation de la lista de valores"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5  # Standard deviation