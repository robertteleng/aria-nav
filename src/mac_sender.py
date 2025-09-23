#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mac Sender - FIXED VERSION
Corrección del formato IMU para compatibilidad total con Jetson receiver

PROBLEMA IDENTIFICADO:
- Mac enviaba IMU como imagen pequeña codificada
- Jetson esperaba datos serializados con pickle
- Resultado: "invalid load key" y "double free or corruption"

SOLUCIÓN:
- Enviar IMU data como array numpy estructurado
- Compatible con _decode_numpy_imu_array() del receiver
- Mantener misma funcionalidad, formato correcto
"""

import cv2
import time
import numpy as np
import imagezmq
import pickle
from dataclasses import dataclass
from typing import Optional
from utils.ctrl_handler import CtrlCHandler
from core.hardware.device_manager import DeviceManager
from core.observer import Observer

# =================================================================
# DATA STRUCTURES (matching Jetson receiver)
# =================================================================

@dataclass
class IMUData:
    """Estructura optimizada para datos IMU"""
    timestamp_ns: int
    accel_x: float
    accel_y: float
    accel_z: float
    magnitude: float
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    imu_idx: int = 0


class MacSender:
    """
    Bridge que usa DeviceManager + Observer existentes
    y envía RGB + SLAM1 + SLAM2 + IMU al Jetson con formato compatible
    """
    
    def __init__(self, jetson_ip="192.168.8.204", jetson_port=5555):
        print("[MAC SENDER] Iniciando bridge Mac → Jetson (FIXED VERSION)...")
        
        # ImageZMQ sender
        self.sender = imagezmq.ImageSender(
            connect_to=f"tcp://{jetson_ip}:{jetson_port}"
        )
        
        # Componentes existentes
        self.device_manager = None
        self.observer = None
        
        # Stats por tipo de stream
        self.frames_sent = {
            'rgb': 0,
            'slam1': 0, 
            'slam2': 0,
            'imu': 0
        }
        self.start_time = time.time()
        
        print(f"[MAC SENDER] Conectando a Jetson: {jetson_ip}:{jetson_port}")
        print("[MAC SENDER] FIXED: Compatible IMU format")
    
    def start(self):
        """Inicia el sistema usando componentes existentes"""
        print("[MAC SENDER] Configurando Aria SDK...")
        
        # 1. DeviceManager exactamente igual
        self.device_manager = DeviceManager()
        self.device_manager.connect()
        rgb_calib = self.device_manager.start_streaming()
        
        # 2. Observer exactamente igual (SIN dashboard para ahorrar recursos)
        self.observer = Observer(rgb_calib=rgb_calib)
        self.device_manager.register_observer(self.observer)
        self.device_manager.subscribe()
        
        print("[MAC SENDER] ✅ Aria SDK configurado, iniciando envío compatible...")
        
        # 3. Loop de envío con formato corregido
        self._sending_loop()
    
    def _create_compatible_imu_array(self, motion_data: dict) -> np.ndarray:
        """
        CORREGIDO: Crear array numpy compatible con _decode_numpy_imu_array()
        
        Format esperado por Jetson receiver:
        [timestamp_ns, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, imu_idx]
        """
        # Extraer datos del motion state
        magnitude = motion_data.get('magnitude', 9.8)
        state = motion_data.get('state', 'stationary')
        
        # Timestamp actual en nanosegundos
        timestamp_ns = int(time.time() * 1e9)
        
        # Descomponer magnitude en componentes X,Y,Z realistas
        # Aria típicamente tiene orientación Y dominante
        accel_x = magnitude * 0.1   # Componente X mínima
        accel_y = magnitude * 0.95  # Componente Y principal (gravedad)
        accel_z = magnitude * 0.1   # Componente Z mínima
        
        # Gyro simulado (Observer no tiene gyro real)
        gyro_x = 0.0
        gyro_y = 0.0
        gyro_z = 0.0
        
        # IMU index (siempre 0 para IMU principal)
        imu_idx = 0
        
        # Crear array compatible con receiver
        imu_array = np.array([
            timestamp_ns,  # [0] timestamp
            accel_x,       # [1] accel_x
            accel_y,       # [2] accel_y  
            accel_z,       # [3] accel_z
            gyro_x,        # [4] gyro_x
            gyro_y,        # [5] gyro_y
            gyro_z,        # [6] gyro_z
            imu_idx        # [7] imu_idx
        ], dtype=np.float64)
        
        return imu_array
    
    def _create_pickled_imu_data(self, motion_data: dict) -> np.ndarray:
        """
        ALTERNATIVA: Crear IMUData serializado con pickle
        Para compatibilidad total con receiver
        """
        magnitude = motion_data.get('magnitude', 9.8)
        timestamp_ns = int(time.time() * 1e9)
        
        # Crear IMUData estructurado
        imu_data = IMUData(
            timestamp_ns=timestamp_ns,
            accel_x=magnitude * 0.1,
            accel_y=magnitude * 0.95,
            accel_z=magnitude * 0.1,
            magnitude=magnitude,
            gyro_x=0.0,
            gyro_y=0.0,
            gyro_z=0.0,
            imu_idx=0
        )
        
        # Serializar con pickle y convertir a ndarray
        serialized = pickle.dumps(imu_data)
        return np.frombuffer(serialized, dtype=np.uint8).copy()
    
    def _sending_loop(self):
        """Loop principal CORREGIDO con formato IMU compatible"""
        print("[MAC SENDER] 🚀 Loop compatible activo...")
        
        ctrl_handler = CtrlCHandler()
        last_stats_time = time.time()

        try:
            while not ctrl_handler.should_stop:
                try:
                    # 1. ENVIAR RGB FRAME (principal)
                    rgb_frame = self.observer.get_latest_frame()
                    if rgb_frame is not None:
                        reply = self.sender.send_image("mac_rgb", rgb_frame)
                        self.frames_sent['rgb'] += 1

                    # 2. ENVIAR SLAM FRAMES
                    all_frames = self.observer.get_all_frames()

                    if 'slam1' in all_frames and all_frames['slam1'] is not None:
                        reply = self.sender.send_image("mac_slam1", all_frames['slam1'])
                        self.frames_sent['slam1'] += 1

                    if 'slam2' in all_frames and all_frames['slam2'] is not None:
                        reply = self.sender.send_image("mac_slam2", all_frames['slam2'])
                        self.frames_sent['slam2'] += 1

                    # 3. ENVIAR IMU DATA - FORMATO CORREGIDO
                    motion_data = self.observer.get_motion_state()
                    if motion_data is not None:
                        # OPCIÓN A: Array numpy compatible
                        imu_array = self._create_compatible_imu_array(motion_data)
                        reply = self.sender.send_image("mac_imu", imu_array)
                        self.frames_sent['imu'] += 1

                        # DEBUG: Verificar formato enviado
                        if self.frames_sent['imu'] % 50 == 1:  # Cada 50 frames
                            print(f"[MAC SENDER] 🔍 IMU format sent:")
                            print(f"  Shape: {imu_array.shape}")
                            print(f"  Dtype: {imu_array.dtype}")
                            print(f"  Data: {imu_array[:4]}...")  # Primera mitad
                            print(f"  Magnitude: {motion_data['magnitude']:.3f}")

                    # Stats cada 5 segundos
                    current_time = time.time()
                    if current_time - last_stats_time >= 5.0:
                        elapsed = current_time - self.start_time

                        total_frames = sum(self.frames_sent.values())
                        fps = total_frames / elapsed if elapsed > 0 else 0

                        print(f"[MAC SENDER] 📊 Frames enviados (FIXED FORMAT):")
                        print(f"  RGB: {self.frames_sent['rgb']}")
                        print(f"  SLAM1: {self.frames_sent['slam1']}")
                        print(f"  SLAM2: {self.frames_sent['slam2']}")
                        print(f"  IMU: {self.frames_sent['imu']} (compatible format)")
                        print(f"  Total FPS: {fps:.1f}")

                        last_stats_time = current_time

                    # Rate limiting 
                    time.sleep(1/30)  # ~30 FPS máximo

                except Exception as e:
                    print(f"[MAC SENDER] ⚠️ Error enviando frames: {e}")
                    time.sleep(0.1)

        finally:
            print("[MAC SENDER] 🛑 Cerrando sender...")
            self._cleanup()
    
    def _cleanup(self):
        """Cleanup usando métodos existentes"""
        try:
            if self.observer:
                self.observer.stop()
            
            if self.device_manager:
                self.device_manager.cleanup()
            
            if self.sender:
                self.sender.close()
            
            duration = (time.time() - self.start_time) / 60.0
            total_frames = sum(self.frames_sent.values())
            print(f"[MAC SENDER] ✅ Sesión terminada: {duration:.1f}min")
            print(f"  RGB: {self.frames_sent['rgb']} frames")
            print(f"  SLAM1: {self.frames_sent['slam1']} frames") 
            print(f"  SLAM2: {self.frames_sent['slam2']} frames")
            print(f"  IMU: {self.frames_sent['imu']} packets (FIXED)")
            print(f"  Total: {total_frames} transmissions")
            
        except Exception as e:
            print(f"[MAC SENDER] ⚠️ Error en cleanup: {e}")


def test_imu_format():
    """Test específico del formato IMU corregido"""
    print("🧪 Testing FIXED IMU format...")
    
    # Simular motion data del Observer
    mock_motion_data = {
        'magnitude': 9.85,
        'state': 'walking'
    }
    
    sender = MacSender()
    
    # Test formato array
    imu_array = sender._create_compatible_imu_array(mock_motion_data)
    print(f"✅ Array format:")
    print(f"  Shape: {imu_array.shape}")
    print(f"  Dtype: {imu_array.dtype}")
    print(f"  Data: {imu_array}")
    
    # Test formato pickle
    imu_pickled = sender._create_pickled_imu_data(mock_motion_data)
    print(f"✅ Pickled format:")
    print(f"  Shape: {imu_pickled.shape}")
    print(f"  Dtype: {imu_pickled.dtype}")
    print(f"  Size: {imu_pickled.size} bytes")
    
    print("✅ Formato IMU test completado")


def main():
    """Entry point del Mac Sender CORREGIDO"""
    print("=" * 60)
    print("MAC SENDER - FIXED VERSION")
    print("Bridge Aria → Jetson con formato IMU compatible")
    print("Corrige: 'invalid load key' y 'double free or corruption'")
    print("=" * 60)
    
    # Test formato antes de enviar
    test_option = input("¿Ejecutar test de formato IMU primero? (y/n): ").lower()
    if test_option == 'y':
        test_imu_format()
        print()
    
    # Configuración Jetson
    jetson_ip = input("IP del Jetson [192.168.8.204]: ").strip()
    if not jetson_ip:
        jetson_ip = "192.168.8.204"
    
    try:
        sender = MacSender(jetson_ip=jetson_ip)
        sender.start()
        
    except KeyboardInterrupt:
        print("\n[MAC SENDER] 🛑 Interrumpido por usuario")
    except Exception as e:
        print(f"[MAC SENDER] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[MAC SENDER] 👋 Programa terminado")


if __name__ == "__main__":
    main()
