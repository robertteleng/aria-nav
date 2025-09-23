#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JETSON RECEIVER - ImageZMQ con IMU y SLAM
TFM: Sistema Navegación para Ciegos - Migración Híbrida

Recibe datos del Mac Aria Sender y los adapta para el Observer existente.
Usa toda la arquitectura modular ya establecida.
"""

import cv2
import numpy as np
import imagezmq
import time
import pickle
import threading
from dataclasses import dataclass
from typing import Dict, Optional, List
from collections import deque

# =================================================================
# DATA STRUCTURES (matching Mac sender)
# =================================================================

@dataclass
class IMUData:
    """Estructura IMU data"""
    timestamp_ns: int
    accel_x: float
    accel_y: float
    accel_z: float
    magnitude: float
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    imu_idx: int = 0

@dataclass
class FramePacket:
    """Paquete de datos completo"""
    timestamp_ns: int
    frame_type: str  # 'rgb', 'slam1', 'slam2'
    frame_data: bytes
    shape: tuple
    camera_id: str
    imu_data: Optional[IMUData] = None
    frame_index: int = 0

# =================================================================
# JETSON IMAGEZMQ RECEIVER
# =================================================================

class JetsonReceiver:
    """
    Receptor ImageZMQ que recibe datos del Mac y los convierte
    para ser procesados por el Observer existente del Jetson.
    """
    
    def __init__(self, port: int = 5555, bind_address: str = "*"):
        print("[RECEIVER] Inicializando Jetson ImageZMQ Receiver...")
        
        self.port = port
        self.bind_address = bind_address
        
        # ImageZMQ setup
        self.image_hub = None
        self.is_connected = False
        self.is_running = False
        
        # Data storage for Observer
        self.current_frames = {
            'rgb': None,
            'slam1': None,
            'slam2': None
        }
        self.latest_imu_data = None
        self.frame_counts = {'rgb': 0, 'slam1': 0, 'slam2': 0}
        
        # Threading
        self._lock = threading.Lock()
        self._stop = False
        
        # Stats
        self.frames_received = 0
        self.last_frame_time = 0
        self.start_time = time.time()
        
        print("[RECEIVER] ✅ Jetson ImageZMQ Receiver configurado")
    
    def start(self):
        """Iniciar el receptor ImageZMQ"""
        try:
            bind_address = f"tcp://{self.bind_address}:{self.port}"
            self.image_hub = imagezmq.ImageHub(
                open_port=bind_address,
                REQ_REP=True  # Request-reply pattern
            )
            self.is_connected = True
            self.is_running = True
            
            print(f"[RECEIVER] ✅ ImageZMQ receptor iniciado en {bind_address}")
            
            # Start receiving thread
            self.receiving_thread = threading.Thread(target=self._receiving_worker, daemon=True)
            self.receiving_thread.start()
            
            return True
            
        except Exception as e:
            print(f"[RECEIVER] ❌ Error iniciando ImageZMQ: {e}")
            self.is_connected = False
            self.is_running = False
            return False
    
    def _receiving_worker(self):
        """Worker thread que recibe datos del Mac"""
        print("[RECEIVER] 📡 Receiving worker activo")
        
        while self.is_running and not self._stop:
            try:
                # Receive data from Mac with timeout
                sender_name, payload = self.image_hub.recv_image()

                # Ensure payload is bytes before deserialization
                if isinstance(payload, np.ndarray):
                    serialized_data = payload.tobytes()
                else:
                    serialized_data = payload

                # Deserialize packet
                try:
                    packet = pickle.loads(serialized_data)
                except Exception as e:
                    print(f"[RECEIVER] ⚠️ Deserialization error: {e}")
                    # Send error reply
                    self.image_hub.send_reply(b'ERROR_DESERIALIZE')
                    continue

                # Route packet por tipo
                if isinstance(packet, IMUData) or sender_name == "mac_imu_data":
                    self._process_imu_packet(packet)
                    self.image_hub.send_reply(b'OK')
                    continue

                # Process frame packet
                success = self._process_packet(packet)
                
                # Send reply
                if success:
                    self.image_hub.send_reply(b'OK')
                    self.frames_received += 1
                    self.last_frame_time = time.time()
                else:
                    self.image_hub.send_reply(b'ERROR_PROCESS')
                
                # Debug every 30 frames
                if self.frames_received % 30 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frames_received / elapsed if elapsed > 0 else 0
                    frame_type = getattr(packet, 'frame_type', 'unknown')
                    print(f"[RECEIVER] 📊 Frames: {self.frames_received}, "
                          f"FPS: {fps:.1f}, Type: {frame_type}")
                
            except Exception as e:
                if "Resource temporarily unavailable" not in str(e):
                    print(f"[RECEIVER] ⚠️ Error receiving: {e}")
                time.sleep(0.01)
    
    def _process_packet(self, packet: FramePacket) -> bool:
        """Procesar packet recibido y almacenar datos"""
        try:
            # Decompress frame
            frame = self._decompress_frame(packet.frame_data, packet.shape)
            if frame is None:
                return False
            
            # Store frame by type
            frame_type = packet.frame_type
            
            with self._lock:
                if frame_type in self.current_frames:
                    self.current_frames[frame_type] = frame
                    self.frame_counts[frame_type] += 1
                
                # Store IMU data if available
                if packet.imu_data:
                    self.latest_imu_data = packet.imu_data
            
            return True
            
        except Exception as e:
            print(f"[RECEIVER] ⚠️ Error processing packet: {e}")
            return False

    def _process_imu_packet(self, imu_packet: IMUData) -> None:
        """Actualizar IMU data recibida"""
        try:
            with self._lock:
                self.latest_imu_data = imu_packet
        except Exception as e:
            print(f"[RECEIVER] ⚠️ Error processing IMU packet: {e}")
    
    def _decompress_frame(self, frame_data: bytes, shape: tuple) -> Optional[np.ndarray]:
        """Descomprimir frame data"""
        try:
            # Try JPEG decompression first
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                return frame
            
            # Fallback: raw bytes
            if len(shape) == 3:
                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(shape)
                return frame
            
            return None
            
        except Exception as e:
            print(f"[RECEIVER] ⚠️ Decompression error: {e}")
            return None
    
    # =================================================================
    # INTERFACE FOR OBSERVER
    # =================================================================
    
    def get_latest_frame(self, camera_type: str = 'rgb') -> Optional[np.ndarray]:
        """Obtener último frame para el Observer"""
        with self._lock:
            return self.current_frames.get(camera_type, None)
    
    def get_all_frames(self) -> Dict[str, Optional[np.ndarray]]:
        """Obtener todos los frames disponibles"""
        with self._lock:
            return self.current_frames.copy()
    
    def get_latest_imu(self) -> Optional[IMUData]:
        """Obtener últimos datos IMU"""
        with self._lock:
            return self.latest_imu_data
    
    def get_frame_counts(self) -> Dict[str, int]:
        """Obtener contadores de frames"""
        with self._lock:
            return self.frame_counts.copy()
    
    def is_receiving_data(self) -> bool:
        """Verificar si está recibiendo datos"""
        current_time = time.time()
        return (current_time - self.last_frame_time) < 5.0  # 5 second timeout
    
    def get_connection_status(self) -> str:
        """Obtener estado de conexión"""
        if not self.is_connected:
            return "disconnected"
        elif not self.is_receiving_data():
            return "connected_no_data"
        else:
            return "active"
    
    # =================================================================
    # LIFECYCLE MANAGEMENT
    # =================================================================
    
    def stop(self):
        """Detener receptor"""
        print("[RECEIVER] 🛑 Deteniendo receptor...")
        
        self._stop = True
        self.is_running = False
        
        # Wait for receiving thread
        if hasattr(self, 'receiving_thread') and self.receiving_thread.is_alive():
            self.receiving_thread.join(timeout=1.0)
        
        # Close ImageZMQ
        if self.image_hub:
            try:
                self.image_hub.close()
            except:
                pass
        
        self.is_connected = False
        print("[RECEIVER] ✅ Receptor detenido")
    
    def get_stats(self) -> Dict:
        """Obtener estadísticas del receiver"""
        elapsed = time.time() - self.start_time
        fps = self.frames_received / elapsed if elapsed > 0 else 0
        
        return {
            'frames_received': self.frames_received,
            'fps': fps,
            'uptime_seconds': elapsed,
            'connection_status': self.get_connection_status(),
            'frame_counts': self.get_frame_counts(),
            'last_frame_time': self.last_frame_time
        }
    
    def print_stats(self):
        """Imprimir estadísticas"""
        stats = self.get_stats()
        print(f"[RECEIVER] 📊 STATS:")
        print(f"   📡 Frames received: {stats['frames_received']}")
        print(f"   📊 FPS: {stats['fps']:.1f}")
        print(f"   🔗 Status: {stats['connection_status']}")
        print(f"   ⏱️ Uptime: {stats['uptime_seconds']/60:.1f} min")
        print(f"   📦 Frame counts: {stats['frame_counts']}")

# =================================================================
# CONVENIENCE FUNCTIONS
# =================================================================

def create_jetson_receiver(port: int = 5555) -> JetsonReceiver:
    """Factory function para crear receiver"""
    return JetsonReceiver(port=port)

def test_receiver():
    """Test básico del receiver"""
    print("🧪 Testing Jetson Receiver...")
    
    receiver = create_jetson_receiver()
    
    if not receiver.start():
        print("❌ Failed to start receiver")
        return False
    
    print("✅ Receiver started, waiting for data...")
    print("💡 Start Mac sender to test data flow")
    
    try:
        for i in range(10):
            time.sleep(1)
            receiver.print_stats()
            
            # Check if receiving data
            if receiver.is_receiving_data():
                print("✅ Receiving data successfully!")
                
                # Test frame access
                rgb_frame = receiver.get_latest_frame('rgb')
                if rgb_frame is not None:
                    print(f"✅ RGB frame: {rgb_frame.shape}")
                
                imu_data = receiver.get_latest_imu()
                if imu_data:
                    print(f"✅ IMU data: magnitude={imu_data.magnitude:.3f}")
                
                break
        else:
            print("⚠️ No data received in 10 seconds")
    
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted")
    
    finally:
        receiver.stop()
    
    print("✅ Test completed")
    return True

# =================================================================
# MAIN EXECUTION
# =================================================================

if __name__ == "__main__":
    test_receiver()
