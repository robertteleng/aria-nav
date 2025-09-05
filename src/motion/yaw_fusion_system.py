#!/usr/bin/env python3
"""
Sistema de Fusión IMU + Magnetómetro para Gafas Aria
Basado en análisis exhaustivo: usa atan2(mz, -mx) + integración de giroscopio
"""


import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

# Declinación magnética local (Alicante aprox.)
DECLINATION_DEG = -0.4

@dataclass
class SensorReading:
    """Lectura de sensores con timestamp"""
    timestamp: float
    # Magnetómetro (μT)
    mx: float
    my: float 
    mz: float
    # Giroscopio (rad/s)
    gx: float
    gy: float
    gz: float
    # Acelerómetro (m/s²)
    ax: float
    ay: float
    az: float

class YawFusionSystem:
    """
    Sistema de fusión de sensores para cálculo de yaw (heading)
    Combina giroscopio (alta frecuencia) + magnetómetro (referencia absoluta)
    """
    
    def __init__(self):
        # Estado del sistema
        self.yaw = 0.0  # Heading actual en grados
        self.gyro_bias = 0.0  # Bias estimado del giroscopio
        self.last_timestamp = None
        
        # Parámetros de fusión
        self.mag_weight = 0.02  # Peso del magnetómetro (corrección suave)
        self.bias_alpha = 0.001  # Factor de aprendizaje del bias
        self.gyro_threshold = 0.1  # Umbral para detección de movimiento (rad/s)
        
        # Validación de magnetómetro
        self.mag_min_strength = 20.0  # μT mínimo
        self.mag_max_strength = 80.0  # μT máximo
        self.max_tilt_degrees = 30.0  # Inclinación máxima para confiar en magnetómetro
        
        # Buffer para suavizado
        self.yaw_buffer = []
        self.buffer_size = 5
        
        # Estadísticas
        self.updates_count = 0
        self.mag_corrections = 0
        
        print("🧭 YawFusionSystem inicializado")
    
    def calculate_tilt_angles(self, ax: float, ay: float, az: float) -> Tuple[float, float]:
        """
        Calcular ángulos de inclinación (pitch, roll) desde acelerómetro
        """
        # Normalizar vector de aceleración
        norm = math.sqrt(ax*ax + ay*ay + az*az)
        if norm < 0.1:  # Evitar división por cero
            return 0.0, 0.0
        
        ax_n, ay_n, az_n = ax/norm, ay/norm, az/norm
        
        # Calcular pitch y roll
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, ax_n))))
        roll = math.degrees(math.atan2(ay_n, az_n))
        
        return pitch, roll
    
    def is_magnetometer_reliable(self, mx: float, my: float, mz: float, 
                                ax: float, ay: float, az: float) -> bool:
        """
        Determinar si la lectura del magnetómetro es confiable
        """
        # Verificar magnitud del campo magnético
        mag_strength = math.sqrt(mx*mx + my*my + mz*mz)
        if not (self.mag_min_strength <= mag_strength <= self.mag_max_strength):
            return False
        
        # Verificar inclinación
        pitch, roll = self.calculate_tilt_angles(ax, ay, az)
        tilt_magnitude = math.sqrt(pitch*pitch + roll*roll)
        if tilt_magnitude > self.max_tilt_degrees:
            return False
        
        return True
    
    def calculate_magnetic_heading(self, mx: float, my: float, mz: float) -> float:
        """
        Calcular heading desde magnetómetro usando la fórmula encontrada: atan2(mz, -mx)
        """
        heading_rad = math.atan2(mz, -mx)
        heading_deg = (math.degrees(heading_rad) + DECLINATION_DEG + 360) % 360
        return heading_deg
    
    def estimate_gyro_bias(self, gx: float, gy: float, gz: float):
        """
        Estimar bias del giroscopio when el usuario está quieto
        """
        # Detectar si está quieto (magnitud baja del gyro)
        gyro_magnitude = math.sqrt(gx*gx + gy*gy + gz*gz)
        
        if gyro_magnitude < self.gyro_threshold:
            # Actualizar estimación de bias usando filtro de primer orden
            # Asumiendo que gx es el yaw axis para las gafas Aria
            self.gyro_bias = (1 - self.bias_alpha) * self.gyro_bias + self.bias_alpha * gx
    
    def update(self, reading: SensorReading) -> dict:
        """
        Actualizar fusión con nueva lectura de sensores
        """
        current_time = reading.timestamp
        
        # Inicialización en primera lectura
        if self.last_timestamp is None:
            # Usar magnetómetro para inicializar yaw si es confiable
            if self.is_magnetometer_reliable(reading.mx, reading.my, reading.mz,
                                           reading.ax, reading.ay, reading.az):
                self.yaw = self.calculate_magnetic_heading(reading.mx, reading.my, reading.mz)
            else:
                self.yaw = 0.0  # Valor por defecto
            
            self.last_timestamp = current_time
            return self._create_result(reading, initialized=True)
        
        # Calcular delta de tiempo
        dt = current_time - self.last_timestamp
        if dt <= 0 or dt > 1.0:  # Saltar lecturas anómalas
            self.last_timestamp = current_time
            return self._create_result(reading, skipped=True)
        
        # Estimar bias del giroscopio
        self.estimate_gyro_bias(reading.gx, reading.gy, reading.gz)
        
        # Integrar giroscopio (componente principal)
        # Nota: gx podría ser el eje yaw para las Aria según orientación
        gyro_corrected = reading.gx - self.gyro_bias
        yaw_delta = math.degrees(gyro_corrected * dt)
        self.yaw += yaw_delta
        
        # Normalizar yaw
        self.yaw = (self.yaw + 360) % 360
        
        # Corrección con magnetómetro (si es confiable)
        mag_correction_applied = False
        if self.is_magnetometer_reliable(reading.mx, reading.my, reading.mz,
                                       reading.ax, reading.ay, reading.az):
            mag_heading = self.calculate_magnetic_heading(reading.mx, reading.my, reading.mz)
            
            # Calcular diferencia angular (considerando wraparound)
            diff = mag_heading - self.yaw
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            
            # Aplicar corrección suave
            self.yaw += self.mag_weight * diff
            self.yaw = (self.yaw + 360) % 360
            
            mag_correction_applied = True
            self.mag_corrections += 1
        
        # Suavizado temporal
        self.yaw_buffer.append(self.yaw)
        if len(self.yaw_buffer) > self.buffer_size:
            self.yaw_buffer.pop(0)
        
        # Calcular yaw suavizado
        if len(self.yaw_buffer) >= 3:
            # Suavizado circular
            x_sum = sum(math.cos(math.radians(y)) for y in self.yaw_buffer)
            y_sum = sum(math.sin(math.radians(y)) for y in self.yaw_buffer)
            smoothed_yaw = math.degrees(math.atan2(y_sum, x_sum))
            smoothed_yaw = (smoothed_yaw + 360) % 360
        else:
            smoothed_yaw = self.yaw
        
        self.updates_count += 1
        self.last_timestamp = current_time
        
        return self._create_result(reading, smoothed_yaw=smoothed_yaw, 
                                 mag_correction=mag_correction_applied)
    
    def _create_result(self, reading: SensorReading, **kwargs) -> dict:
        """Crear resultado estructurado"""
        pitch, roll = self.calculate_tilt_angles(reading.ax, reading.ay, reading.az)
        mag_strength = math.sqrt(reading.mx**2 + reading.my**2 + reading.mz**2)
        mag_reliable = self.is_magnetometer_reliable(reading.mx, reading.my, reading.mz,
                                                   reading.ax, reading.ay, reading.az)
        
        result = {
            'success': True,
            'yaw': kwargs.get('smoothed_yaw', self.yaw),
            'raw_yaw': self.yaw,
            'pitch': pitch,
            'roll': roll,
            'gyro_bias': self.gyro_bias,
            'mag_strength': mag_strength,
            'mag_reliable': mag_reliable,
            'mag_corrections_total': self.mag_corrections,
            'updates_count': self.updates_count,
            'timestamp': reading.timestamp
        }
        
        # Estados especiales
        if kwargs.get('initialized'):
            result['status'] = 'initialized'
        elif kwargs.get('skipped'):
            result['status'] = 'skipped'
        elif kwargs.get('mag_correction'):
            result['status'] = 'corrected'
        else:
            result['status'] = 'integrated'
        
        return result
    
    def get_cardinal_direction(self, yaw: float) -> str:
        """Convertir yaw a dirección cardinal"""
        if yaw < 22.5 or yaw >= 337.5:
            return "Norte"
        elif yaw < 67.5:
            return "Noreste"
        elif yaw < 112.5:
            return "Este"
        elif yaw < 157.5:
            return "Sureste"
        elif yaw < 202.5:
            return "Sur"
        elif yaw < 247.5:
            return "Suroeste"
        elif yaw < 292.5:
            return "Oeste"
        else:
            return "Noroeste"
    
    def get_status_summary(self) -> dict:
        """Obtener resumen del estado del sistema"""
        correction_rate = self.mag_corrections / max(1, self.updates_count)
        
        return {
            'total_updates': self.updates_count,
            'mag_corrections': self.mag_corrections,
            'correction_rate': correction_rate,
            'current_yaw': self.yaw,
            'current_direction': self.get_cardinal_direction(self.yaw),
            'gyro_bias': self.gyro_bias,
            'buffer_size': len(self.yaw_buffer)
        }

# Clase de integración para el Observer principal
class AriaYawIntegrator:
    """
    Integrador para usar YawFusionSystem en el observer de Aria
    """
    
    def __init__(self):
        self.fusion = YawFusionSystem()
        self.last_mag_data = None
        self.last_imu_data = None
        self.last_result = None
        
    def on_magneto_received(self, *args):
        """Callback del magnetómetro (firma flexible). Acepta (sample) o (label, sample) o (label, ts, sample)."""
        if not args:
            return
        sample = args[-1]
        try:
            # sample.mag_tesla viene en Tesla (SI). Convertimos a μT para las validaciones de rango.
            mx_t, my_t, mz_t = sample.mag_tesla
            mx_uT, my_uT, mz_uT = mx_t * 1e6, my_t * 1e6, mz_t * 1e6
            self.last_mag_data = {
                'mx': mx_uT,
                'my': my_uT,
                'mz': mz_uT,
                'timestamp': time.time()
            }
            self._try_fusion_update()
        except Exception as e:
            print(f"[MAG] error: {e}")
    
    def on_imu_received(self, *args):
        """Callback del IMU (firma flexible). Acepta (sample) o (label, sample) o (label, ts, sample) o listas/tuplas."""
        if not args:
            return
        sample = args[-1]
        try:
            if hasattr(sample, 'accel_mps2') and hasattr(sample, 'gyro_rps'):
                self.last_imu_data = {
                    'gx': sample.gyro_rps[0],
                    'gy': sample.gyro_rps[1],
                    'gz': sample.gyro_rps[2],
                    'ax': sample.accel_mps2[0],
                    'ay': sample.accel_mps2[1],
                    'az': sample.accel_mps2[2],
                    'timestamp': time.time()
                }
            elif isinstance(sample, (list, tuple)) and len(sample) >= 6:
                self.last_imu_data = {
                    'gx': sample[0], 'gy': sample[1], 'gz': sample[2],
                    'ax': sample[3], 'ay': sample[4], 'az': sample[5],
                    'timestamp': time.time()
                }
            elif isinstance(sample, dict):
                self.last_imu_data = {
                    'gx': sample.get('gx', 0), 'gy': sample.get('gy', 0), 'gz': sample.get('gz', 0),
                    'ax': sample.get('ax', 0), 'ay': sample.get('ay', 0), 'az': sample.get('az', 0),
                    'timestamp': time.time()
                }
            else:
                # No interpretable → ignorar silenciosamente
                return
            self._try_fusion_update()
        except Exception as e:
            print(f"[IMU] error: {e}")
    
    def _try_fusion_update(self):
        """Intentar actualizar fusión si tenemos datos de ambos sensores"""
        if self.last_mag_data is None or self.last_imu_data is None:
            return
        
        # Crear SensorReading combinando datos más recientes
        reading = SensorReading(
            timestamp=max(self.last_mag_data['timestamp'], self.last_imu_data['timestamp']),
            mx=self.last_mag_data['mx'],
            my=self.last_mag_data['my'],
            mz=self.last_mag_data['mz'],
            gx=self.last_imu_data['gx'],
            gy=self.last_imu_data['gy'],
            gz=self.last_imu_data['gz'],
            ax=self.last_imu_data['ax'],
            ay=self.last_imu_data['ay'],
            az=self.last_imu_data['az']
        )
        
        # Actualizar fusión
        self.last_result = self.fusion.update(reading)
    
    def get_current_heading(self) -> Optional[dict]:
        """Obtener heading actual"""
        if self.last_result is None:
            return None
        
        return {
            'heading': self.last_result['yaw'],
            'direction': self.fusion.get_cardinal_direction(self.last_result['yaw']),
            'quality': 'good' if self.last_result['mag_reliable'] else 'fair',
            'status': self.last_result['status'],
            'pitch': self.last_result['pitch'],
            'roll': self.last_result['roll']
        }

def main():
    """Test básico del sistema"""
    print("🧭 YAW FUSION SYSTEM TEST")
    print("Este es un test básico - integra con tu observer principal")
    
    # Crear sistema
    fusion = YawFusionSystem()
    
    # Datos de ejemplo (reemplazar con datos reales)
    test_reading = SensorReading(
        timestamp=time.time(),
        mx=-20.0, my=15.0, mz=30.0,  # Magnetómetro
        gx=0.1, gy=0.0, gz=0.05,    # Giroscopio
        ax=0.0, ay=0.0, az=9.81     # Acelerómetro
    )
    
    result = fusion.update(test_reading)
    print(f"Resultado de prueba: {result}")
    
    print("\n💻 Para integrar en tu sistema:")
    print("```python")
    print("# En tu observer principal:")
    print("from yaw_fusion_system import AriaYawIntegrator")
    print("")
    print("class EnhancedAriaObserver:")
    print("    def __init__(self):")
    print("        self.yaw_integrator = AriaYawIntegrator()")
    print("        # ... resto de tu código")
    print("")
    print("    def on_magneto_received(self, sample):")
    print("        self.yaw_integrator.on_magneto_received(sample)")
    print("        # ... tu código existente")
    print("")
    print("    def on_imu_received(self, sample):")
    print("        self.yaw_integrator.on_imu_received(sample)")
    print("")
    print("    def get_heading_for_audio(self):")
    print("        heading_info = self.yaw_integrator.get_current_heading()")
    print("        if heading_info:")
    print("            return f'Orientado hacia el {heading_info[\"direction\"]}'")
    print("        return None")
    print("```")

if __name__ == "__main__":
    main()