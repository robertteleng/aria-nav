import numpy as np
import time
import math
from collections import deque
from typing import Optional
from projectaria_tools.core.sensor_data import MotionData

from motion.orientation_state import OrientationState

class MotionProcessor:
    """Simple yaw tracking with automatic magnetometer correction"""
    
    def __init__(self, history_size=20):
        # Current orientation state
        self.orientation = OrientationState(
            heading=0.0, pitch=0.0, yaw=0.0, roll=0.0,
            is_moving=False, movement_speed=0.0,
            last_updated=time.time()
        )
        
        # Data buffers
        self.accel_history = deque(maxlen=history_size)
        self.gyro_history = deque(maxlen=history_size)
        self.magneto_history = deque(maxlen=history_size)
        
        # ===== YAW TRACKING SIMPLE =====
        self.yaw_angle = 0.0  # Radianes, integrado del giroscopio
        self.yaw_zero = 0.0   # Baseline de calibración
        self._prev_timestamp = None
        
        # ===== CORRECCIÓN AUTOMÁTICA CON MAGNETÓMETRO =====
        self._correction_counter = 0
        self._auto_calibrated = False
        self.correction_interval = 200  # Cada N muestras (~5 segundos a 100Hz)
        
        # ===== DETECCIÓN DE MOVIMIENTO SIMPLE =====
        self.accel_magnitudes = deque(maxlen=10)
        self.movement_variance = 0.0
        self.movement_threshold = 0.5  # Varianza threshold
        
        # Movement detection thresholds
        self.stationary_time = 0.0
        
        print("[INFO] MotionProcessor - Yaw simple con corrección automática magnetómetro")
        print(f"[INFO] Corrección cada {self.correction_interval} muestras")
    
    def update_motion(self, samples, imu_idx: int) -> None:
        """Yaw del giroscopio con corrección automática por magnetómetro"""
        for sample in samples:
            timestamp_ns = sample.capture_timestamp_ns
            timestamp_s = timestamp_ns * 1e-9
            accel = sample.accel_msec2
            gyro = sample.gyro_radsec
            
            # Store raw data
            self.accel_history.append((timestamp_s, accel))
            self.gyro_history.append((timestamp_s, gyro))
            
            # ===== 1. INTEGRAR YAW DEL GIROSCOPIO =====
            if self._prev_timestamp is not None:
                dt = timestamp_s - self._prev_timestamp
                if 0 < dt < 0.1:  # Filtrar timestamps razonables
                    yaw_rate = gyro[0]  # Eje X confirmado para yaw
                    self.yaw_angle += yaw_rate * dt
                    # Mantener en rango [-π, π]
                    self.yaw_angle = ((self.yaw_angle + math.pi) % (2 * math.pi)) - math.pi
            
            self._prev_timestamp = timestamp_s
            
            # ===== 2. CORRECCIÓN AUTOMÁTICA CON MAGNETÓMETRO =====
            self._correction_counter += 1
            
            if (self._correction_counter % self.correction_interval == 0 and 
                hasattr(self.orientation, 'heading') and self.orientation.heading != 0):
                
                self._auto_correct_with_magnetometer()
            
            # ===== 3. DETECCIÓN DE MOVIMIENTO CON VARIANZA =====
            self._detect_movement_variance(accel)
            
            # ===== 4. ACTUALIZAR ESTADO =====
            # Convertir yaw calibrado a grados para el estado
            yaw_calibrated_rad = self._wrap_pi(self.yaw_angle - self.yaw_zero)
            self.orientation.yaw = math.degrees(yaw_calibrated_rad)
            self.orientation.last_updated = timestamp_s
            
            # Debug cada 200 muestras
            if self._correction_counter % 200 == 0:
                print(f"[YAW] Gyro: {math.degrees(self.yaw_angle):.1f}° "
                      f"Calibrado: {self.orientation.yaw:.1f}° "
                      f"Mag: {getattr(self.orientation, 'heading', 0):.1f}° "
                      f"Moving: {self.orientation.is_moving}")
    
    def update_heading(self, sample: MotionData) -> None:
        """Procesar magnetómetro para heading de referencia"""
        timestamp = sample.capture_timestamp_ns * 1e-9
        
        # Store magnetometer data
        self.magneto_history.append((timestamp, sample.mag_tesla))
        
        # Calculate compass heading
        heading = self._calculate_compass_heading(sample.mag_tesla)
        if heading is not None:
            # Aplicar suavizado al heading del magnetómetro
            if hasattr(self.orientation, 'heading') and self.orientation.heading != 0.0:
                heading_diff = heading - self.orientation.heading
                
                # Handle 360° -> 0° wrap
                if heading_diff > 180:
                    heading_diff -= 360
                elif heading_diff < -180:
                    heading_diff += 360
                
                # Smooth transition
                self.orientation.heading += 0.1 * heading_diff
                self.orientation.heading = (self.orientation.heading + 360) % 360
            else:
                self.orientation.heading = heading
                # Inicialización automática del yaw con magnetómetro
                if not self._auto_calibrated:
                    self._initialize_yaw_with_magnetometer(heading)
            
            self.orientation.last_updated = timestamp
    
    def _auto_correct_with_magnetometer(self):
        """Corrección automática del drift del giroscopio usando magnetómetro"""
        try:
            mag_heading_deg = self.orientation.heading
            mag_yaw_rad = math.radians(mag_heading_deg)
            gyro_yaw_rad = self.yaw_angle
            
            # Calcular diferencia entre magnetómetro y giroscopio
            diff = mag_yaw_rad - gyro_yaw_rad
            
            # Manejar wrap-around (-π to π)
            if diff > math.pi:
                diff -= 2 * math.pi
            elif diff < -math.pi:
                diff += 2 * math.pi
            
            # Aplicar corrección gradual (20% de la diferencia)
            correction_factor = 0.2
            correction = diff * correction_factor
            self.yaw_angle += correction
            
            # Logging solo para correcciones significativas
            if abs(correction) > math.radians(2):  # > 2 grados
                print(f"[AUTO-CORRECT] Gyro drift corregido: {math.degrees(correction):.1f}° "
                      f"(Mag: {mag_heading_deg:.1f}° vs Gyro: {math.degrees(gyro_yaw_rad):.1f}°)")
            
        except Exception as e:
            print(f"[AUTO-CORRECT] Error en corrección: {e}")
    
    def _initialize_yaw_with_magnetometer(self, mag_heading_deg):
        """Inicializar yaw del giroscopio con magnetómetro en el arranque"""
        self.yaw_angle = math.radians(mag_heading_deg)
        self.yaw_zero = self.yaw_angle  # Establecer como referencia inicial
        self._auto_calibrated = True
        print(f"[INIT] Yaw inicializado con magnetómetro: {mag_heading_deg:.1f}°")
    
    def _detect_movement_variance(self, accel):
        """Detección de movimiento usando varianza de magnitudes de aceleración"""
        # Calcular magnitud de aceleración
        accel_magnitude = math.sqrt(accel[0]**2 + accel[1]**2 + accel[2]**2)
        self.accel_magnitudes.append(accel_magnitude)
        
        # Calcular varianza cuando tenemos suficientes muestras
        if len(self.accel_magnitudes) >= 5:
            recent_magnitudes = list(self.accel_magnitudes)[-5:]
            mean_magnitude = sum(recent_magnitudes) / len(recent_magnitudes)
            variance = sum((m - mean_magnitude)**2 for m in recent_magnitudes) / len(recent_magnitudes)
            
            self.movement_variance = variance
            
            # Determinar si está en movimiento
            was_moving = self.orientation.is_moving
            self.orientation.is_moving = variance > self.movement_threshold
            
            # Log cambios de estado
            if was_moving != self.orientation.is_moving:
                state = "CAMINANDO" if self.orientation.is_moving else "QUIETO"
                print(f"[MOVEMENT] Usuario {state} (varianza: {variance:.3f})")
            
            self.orientation.movement_speed = variance
    
    def _calculate_compass_heading(self, mag_data):
        """Cálculo básico de heading compass"""
        try:
            mag_x, mag_y, mag_z = mag_data
            
            # Heading usando X,Y (plano horizontal)
            heading_rad = math.atan2(mag_z, -mag_x) # atan2(Este, Norte)
            heading_deg = math.degrees(heading_rad)
            
            # Normalizar a 0-360°
            heading_deg = (heading_deg + 360) % 360
            
            return heading_deg
        except Exception as e:
            print(f"[COMPASS] Error calculando heading: {e}")
            return None
    
    @staticmethod
    def _wrap_pi(angle):
        """Normalizar ángulo a rango [-π, π]"""
        return (angle + math.pi) % (2 * math.pi) - math.pi
    
    def get_contextual_direction(self, object_center_x, frame_width):
        """Generar dirección contextual usando yaw corregido"""
        # Determinar zona básica en frame
        if object_center_x < frame_width * 0.33:
            zone_offset = -45  # Izquierda
        elif object_center_x > frame_width * 0.67:
            zone_offset = 45   # Derecha
        else:
            zone_offset = 0    # Centro
        
        # Calcular heading absoluto del objeto
        user_heading = getattr(self.orientation, 'heading', 0.0)
        object_heading = (user_heading + zone_offset) % 360
        
        # Convertir a dirección natural en español
        directions = [
            "frente a ti",           # 0°
            "adelante a la derecha", # 45°
            "a tu derecha",          # 90°
            "atrás a la derecha",    # 135°
            "detrás de ti",          # 180°
            "atrás a la izquierda",  # 225°
            "a tu izquierda",        # 270°
            "adelante a la izquierda" # 315°
        ]
        
        idx = int((object_heading + 22.5) / 45) % 8
        return directions[idx]
    
    def get_yaw_degrees(self):
        """Obtener yaw actual en grados (calibrado)"""
        yaw_cal = self._wrap_pi(self.yaw_angle - self.yaw_zero)
        return math.degrees(yaw_cal)
    
    def get_absolute_heading(self):
        """Obtener heading absoluto del magnetómetro"""
        return getattr(self.orientation, 'heading', 0.0)
    
    def force_recalibration(self):
        """Forzar recalibración con magnetómetro actual"""
        if hasattr(self.orientation, 'heading') and self.orientation.heading != 0:
            self._initialize_yaw_with_magnetometer(self.orientation.heading)
        else:
            print("[RECAL] No hay datos de magnetómetro para recalibrar")
    
    # ===== MÉTODOS DE COMPATIBILIDAD =====
    
    def get_relative_direction(self, object_center_x: float, frame_width: int) -> str:
        """Compatibilidad con sistema existente"""
        return self.get_contextual_direction(object_center_x, frame_width)
    
    def get_movement_context(self) -> str:
        """Contexto de movimiento para comandos de audio"""
        if self.orientation.is_moving:
            return "mientras caminas"
        else:
            return ""
    
    def reset_orientation(self):
        """Reset completo del sistema de orientación"""
        print("[RESET] Reiniciando sistema de orientación...")
        
        self.yaw_angle = 0.0
        self.yaw_zero = 0.0
        self._prev_timestamp = None
        self._correction_counter = 0
        self._auto_calibrated = False
        
        self.orientation.heading = 0.0
        self.orientation.yaw = 0.0
        self.orientation.is_moving = False
        self.movement_variance = 0.0
        
        # Clear buffers
        self.accel_history.clear()
        self.gyro_history.clear()
        self.magneto_history.clear()
        self.accel_magnitudes.clear()
        
        print("[RESET] Sistema listo para nueva inicialización con magnetómetro")

    def test_all_magnetometer_combinations(self, mag_data):
        """Test todas las combinaciones de ejes para encontrar la correcta"""
        mag_x, mag_y, mag_z = mag_data
        
        combinations = [
            ("X,Y", math.atan2(mag_y, mag_x)),
            ("Y,X", math.atan2(mag_x, mag_y)),
            ("X,Z", math.atan2(mag_z, mag_x)),
            ("Z,X", math.atan2(mag_x, mag_z)),
            ("Y,Z", math.atan2(mag_z, mag_y)),
            ("Z,Y", math.atan2(mag_y, mag_z)),
            ("-X,Y", math.atan2(mag_y, -mag_x)),
            ("X,-Y", math.atan2(-mag_y, mag_x)),
            ("-X,Z", math.atan2(mag_z, -mag_x)),
            ("Z,-X", math.atan2(-mag_x, mag_z)),
        ]
        
        print(f"\n[MAG-TEST] Valores raw: X={mag_x:.1f}, Y={mag_y:.1f}, Z={mag_z:.1f}")
        for name, heading_rad in combinations:
            heading_deg = (math.degrees(heading_rad) + 360) % 360
            print(f"[MAG-TEST] {name:>4}: {heading_deg:6.1f}°")
        print("[MAG-TEST] Tu brújula marca: ¿cuántos grados?")
        print("")