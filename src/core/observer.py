#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Observer Desacoplado - Solo Aria SDK Interface
Maneja únicamente la conexión con Aria SDK y streaming de datos

Responsabilidades ÚNICAMENTE:
- Recibir callbacks del Aria SDK
- Procesar frames RGB, SLAM1, SLAM2  
- Procesar datos IMU
- Almacenar frames más recientes
- NO procesamiento, NO audio, NO dashboard

Fecha: Septiembre 2025
Versión: 2.0 - Decoupled Architecture
"""

import numpy as np
import cv2
import threading
import time
import aria.sdk as aria
from projectaria_tools.core.sensor_data import ImageDataRecord, MotionData
from typing import Sequence, Optional, Dict, Any
from collections import deque

from utils.config import Config


class Observer:
    """
    Observer puramente dedicado al Aria SDK
    
    Este Observer SOLO maneja:
    - Callbacks del SDK (on_image_received, on_imu_received)
    - Rotación y almacenamiento de frames
    - Threading thread-safe para acceso a frames
    - Estadísticas básicas de captura
    
    NO maneja:
    - Procesamiento YOLO
    - Audio commands
    - Dashboards o UI
    - Navigation logic
    """
    
    def __init__(self, rgb_calib=None):
        """
        Inicializar Observer puro para Aria SDK
        
        Args:
            rgb_calib: Calibración RGB opcional del dispositivo
        """
        self.rgb_calib = rgb_calib
        
        # Frame storage thread-safe
        self._lock = threading.Lock()
        self.current_frames = {
            'rgb': None,      # Center camera (main)
            'slam1': None,    # Left camera 
            'slam2': None     # Right camera
        }
        
        # Frame statistics
        self.frame_counts = {
            'rgb': 0, 
            'slam1': 0, 
            'slam2': 0
        }
        self.start_time = time.time()
        
        # IMU data storage
        self.imu_data = {
            'imu0': deque(maxlen=100),
            'imu1': deque(maxlen=100)
        }
        self.motion_magnitude_history = deque(maxlen=50)
        
        # Threading control
        self._stop = False
        
        print("[OBSERVER] AriaObserver inicializado (SDK-only)")
        print("[OBSERVER] Monitoring: RGB + SLAM1 + SLAM2 + IMU0/1")
    
    def on_image_received(self, image: np.array, record: ImageDataRecord) -> None:
        """
        Callback del SDK para nuevas imágenes - TODAS las cámaras
        
        Args:
            image: Imagen raw del SDK
            record: Metadatos de la imagen
        """
        camera_id = record.camera_id
        
        # Procesar según el tipo de cámara
        if camera_id == aria.CameraId.Rgb:
            processed_image = self._process_rgb_image(image)
            camera_key = 'rgb'
            
        elif camera_id == aria.CameraId.Slam1:
            processed_image = self._process_slam_image(image)
            camera_key = 'slam1'
            
        elif camera_id == aria.CameraId.Slam2:
            processed_image = self._process_slam_image(image)
            camera_key = 'slam2'
            
        else:
            # Ignorar EyeTrack y otras cámaras
            return
        
        # Almacenamiento thread-safe
        with self._lock:
            self.current_frames[camera_key] = processed_image
            self.frame_counts[camera_key] += 1
        
        # Debug periódico
        if self.frame_counts[camera_key] % 200 == 0:
            fps = self._calculate_fps(camera_key)
            print(f"[OBSERVER] {camera_key.upper()}: {self.frame_counts[camera_key]} frames ({fps:.1f} FPS)")
    
    def on_imu_received(self, samples: Sequence[MotionData], imu_idx: int) -> None:
        """
        Callback del SDK para datos IMU
        
        Args:
            samples: Secuencia de datos de movimiento
            imu_idx: Índice del IMU (0 o 1)
        """
        if not samples:
            return
        
        sample = samples[0]
        accelerometer = sample.accel_msec2
        timestamp = sample.capture_timestamp_ns
        
        # Calcular magnitud de aceleración
        magnitude = (accelerometer[0]**2 + accelerometer[1]**2 + accelerometer[2]**2)**0.5
        
        # Almacenar datos thread-safe
        imu_key = f'imu{imu_idx}'
        with self._lock:
            self.imu_data[imu_key].append({
                'timestamp': timestamp,
                'acceleration': accelerometer,
                'magnitude': magnitude
            })
            
            # Solo para IMU0, mantener historial de magnitud
            if imu_idx == 0:
                self.motion_magnitude_history.append(magnitude)
        
        # Debug periódico (cada ~3 segundos para IMU0)
        if imu_idx == 0 and len(self.imu_data[imu_key]) % 300 == 0:
            motion_state = self._estimate_motion_state()
            print(f"[OBSERVER] IMU0: magnitude={magnitude:.2f} m/s², state={motion_state}")
    
    def on_streaming_client_failure(self, reason, message: str) -> None:
        """
        Callback del SDK para errores de streaming
        
        Args:
            reason: Razón del error
            message: Mensaje descriptivo del error
        """
        print(f"[OBSERVER ERROR] Streaming failure: {reason} - {message}")
    
    def _process_rgb_image(self, image: np.array) -> np.array:
        """
        Procesar imagen RGB (cámara central)
        
        Args:
            image: Imagen raw BGR del SDK
            
        Returns:
            np.array: Imagen procesada y rotada
        """
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        # Convertir a espacio de color esperado por el resto del pipeline
        if getattr(Config, 'RGB_CAMERA_COLOR_SPACE', 'BGR').upper() == 'RGB':
            return cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR)

        return rotated
    
    def _process_slam_image(self, image: np.array) -> np.array:
        """
        Procesar imágenes SLAM (cámaras periféricas izquierda/derecha)
        
        Args:
            image: Imagen raw (puede ser grayscale)
            
        Returns:
            np.array: Imagen procesada y rotada
        """
        # Convertir grayscale a RGB si es necesario
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif getattr(Config, 'RGB_CAMERA_COLOR_SPACE', 'BGR').upper() == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Rotación consistente con RGB
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return rotated
    
    def _calculate_fps(self, camera_key: str) -> float:
        """
        Calcular FPS aproximado para una cámara
        
        Args:
            camera_key: Clave de la cámara ('rgb', 'slam1', 'slam2')
            
        Returns:
            float: FPS estimado
        """
        uptime = time.time() - self.start_time
        if uptime > 0:
            return self.frame_counts[camera_key] / uptime
        return 0.0
    
    def _estimate_motion_state(self) -> str:
        """
        Estimar estado de movimiento basado en historial de magnitud IMU
        
        Returns:
            str: 'stationary', 'walking', o 'unknown'
        """
        if len(self.motion_magnitude_history) < 10:
            return 'unknown'
        
        # Calcular variación en ventana reciente
        recent_magnitudes = list(self.motion_magnitude_history)[-20:]
        if len(recent_magnitudes) < 5:
            return 'unknown'
        
        mean_mag = np.mean(recent_magnitudes)
        std_mag = np.std(recent_magnitudes)
        
        # Thresholds simples para clasificación
        if std_mag < 0.5:
            return 'stationary'
        elif std_mag > 1.0:
            return 'walking'
        else:
            return 'stationary'  # Default conservador
    
    # ============================================================================
    # PUBLIC API - Thread-safe access methods
    # ============================================================================
    
    def get_latest_frame(self, camera: str = 'rgb') -> Optional[np.array]:
        """
        Obtener el frame más reciente de una cámara específica
        
        Args:
            camera: 'rgb', 'slam1', o 'slam2'
            
        Returns:
            np.array o None: Frame más reciente o None si no disponible
        """
        with self._lock:
            return self.current_frames.get(camera, None)
    
    def get_all_frames(self) -> Dict[str, Optional[np.array]]:
        """
        Obtener todos los frames actuales de todas las cámaras
        
        Returns:
            dict: Diccionario con frames de todas las cámaras
        """
        with self._lock:
            return self.current_frames.copy()
    
    def get_frame_counts(self) -> Dict[str, int]:
        """
        Obtener contadores de frames de todas las cámaras
        
        Returns:
            dict: Contadores de frames por cámara
        """
        with self._lock:
            return self.frame_counts.copy()
    
    def get_latest_imu_data(self, imu_idx: int = 0) -> Optional[Dict[str, Any]]:
        """
        Obtener datos IMU más recientes
        
        Args:
            imu_idx: Índice del IMU (0 o 1)
            
        Returns:
            dict o None: Datos IMU más recientes o None si no disponible
        """
        imu_key = f'imu{imu_idx}'
        with self._lock:
            if self.imu_data[imu_key]:
                return self.imu_data[imu_key][-1].copy()
            return None
    
    def get_motion_state(self) -> Dict[str, Any]:
        """
        Obtener estado de movimiento estimado
        
        Returns:
            dict: Estado de movimiento con metadatos
        """
        with self._lock:
            latest_imu = None
            if self.imu_data['imu0']:
                latest_imu = self.imu_data['imu0'][-1]
            
            return {
                'state': self._estimate_motion_state(),
                'magnitude': latest_imu['magnitude'] if latest_imu else 9.8,
                'timestamp': latest_imu['timestamp'] if latest_imu else None,
                'history_length': len(self.motion_magnitude_history)
            }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas generales del sistema
        
        Returns:
            dict: Estadísticas completas del Observer
        """
        uptime = time.time() - self.start_time
        stats = {
            'uptime_seconds': uptime,
            'uptime_minutes': uptime / 60.0,
            'frame_counts': self.get_frame_counts(),
            'fps_estimates': {},
            'motion_state': self.get_motion_state(),
            'total_imu_samples': {
                'imu0': len(self.imu_data['imu0']),
                'imu1': len(self.imu_data['imu1'])
            }
        }
        
        # Calcular FPS para cada cámara
        for camera in ['rgb', 'slam1', 'slam2']:
            stats['fps_estimates'][camera] = self._calculate_fps(camera)
        
        return stats
    
    def print_stats(self) -> None:
        """Imprimir estadísticas de captura"""
        stats = self.get_system_stats()
        
        print(f"\n[OBSERVER STATS] Uptime: {stats['uptime_minutes']:.1f} min")
        print(f"  RGB frames: {stats['frame_counts']['rgb']} ({stats['fps_estimates']['rgb']:.1f} FPS)")
        print(f"  SLAM1 frames: {stats['frame_counts']['slam1']} ({stats['fps_estimates']['slam1']:.1f} FPS)")
        print(f"  SLAM2 frames: {stats['frame_counts']['slam2']} ({stats['fps_estimates']['slam2']:.1f} FPS)")
        print(f"  Motion state: {stats['motion_state']['state']}")
        print(f"  IMU samples: IMU0={stats['total_imu_samples']['imu0']}, IMU1={stats['total_imu_samples']['imu1']}")
    
    def stop(self) -> None:
        """
        Señalar parada del Observer
        
        Nota: El Observer no tiene hilos propios que detener,
        pero este método mantiene consistencia de API
        """
        self._stop = True
        print("[OBSERVER] Stop signal received")
