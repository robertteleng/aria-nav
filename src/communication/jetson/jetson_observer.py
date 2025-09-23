#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JETSON OBSERVER - ADAPTADO PARA IMAGEZMQ
TFM: Sistema Navegación para Ciegos - Migración Híbrida

Observer que usa JetsonImageZMQReceiver en lugar del Aria SDK directo.
Reutiliza toda la arquitectura modular existente exactamente igual que el Observer del Mac.
"""

import numpy as np
import cv2
import threading
import time
from typing import Optional, Dict

# Import existing modular architecture (TUS componentes)
from core.navigation.builder import build_navigation_system
from core.vision.depth_estimator import DepthEstimator
from core.imu.motion_detector import SimpleMotionDetector
from presentation.dashboards.opencv_dashboard import OpenCVDashboard

# Import the receiver
from communication.jetson_receiver import JetsonReceiver, IMUData

class JetsonObserver:
    """
    Observer adaptado para datos de ImageZMQ manteniendo
    toda la funcionalidad del Observer original.
    """
    
    def __init__(self, receiver_port: int = 5555, enable_dashboard: bool = False):
        print("[JETSON OBSERVER] Inicializando Observer adaptado...")
        
        # Core processing components (using existing architecture)
        self.coordinator = build_navigation_system(enable_dashboard=False)
        self.depth_estimator = DepthEstimator()
        self.motion_detector = SimpleMotionDetector()
        
        # ImageZMQ receiver instead of Aria SDK
        self.receiver = JetsonImageZMQReceiver(port=receiver_port)
        
        # Dashboard integration (optional)
        self.dashboard = None
        if enable_dashboard:
            try:
                self.dashboard = OpenCVDashboard()
                print("[JETSON OBSERVER] ✅ Dashboard activado")
            except ImportError:
                print("[JETSON OBSERVER] ⚠️ Dashboard no disponible")
        
        # Multi-camera frame storage
        self.current_frames = {
            'rgb': None,
            'slam1': None,
            'slam2': None
        }
        
        # Multi-camera detections
        self.camera_detections = {
            'rgb': [],
            'slam1': [],
            'slam2': []
        }
        
        # Processing state
        self.frame_counts = {'rgb': 0, 'slam1': 0, 'slam2': 0}
        self.current_depth_map = None
        self.current_motion_state = "stationary"
        
        # Async processing
        self._lock = threading.Lock()
        self._stop = False
        self._processing_thread = None
        
        print("[JETSON OBSERVER] ✅ Observer inicializado")
    
    def start(self):
        """Iniciar el observer y todos sus componentes"""
        print("[JETSON OBSERVER] 🚀 Iniciando sistema...")
        
        # Start receiver
        if not self.receiver.start():
            print("[JETSON OBSERVER] ❌ Error iniciando receiver")
            return False
        
        # Start processing thread
        self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._processing_thread.start()
        
        print("[JETSON OBSERVER] ✅ Sistema iniciado")
        print("[JETSON OBSERVER] Esperando datos del Mac...")
        
        return True
    
    def _processing_loop(self):
        """
        Loop principal de procesamiento adaptado para ImageZMQ.
        Reemplaza on_image_received y on_imu_received callbacks.
        """
        frame_counter = 0
        frames_without_data = 0
        max_wait_frames = 150  # ~5 seconds at 30fps
        
        while not self._stop:
            try:
                loop_start = time.time()
                
                # Get frames from receiver instead of Aria callbacks
                frames = self.receiver.get_all_frames()
                imu_data = self.receiver.get_latest_imu()
                
                # Check if we're receiving data
                if not any(frame is not None for frame in frames.values()):
                    frames_without_data += 1
                    if frames_without_data >= max_wait_frames:
                        print("[JETSON OBSERVER] ⚠️ No frames received for 5 seconds")
                        frames_without_data = 0
                    time.sleep(0.01)
                    continue
                
                frames_without_data = 0
                
                # Process IMU data for motion detection
                if imu_data:
                    self.current_motion_state = self.motion_detector.update(
                        imu_data.magnitude, imu_data.timestamp_ns
                    )
                    
                    # Debug IMU every 300 samples
                    if not hasattr(self, 'imu_count'):
                        self.imu_count = 0
                    self.imu_count += 1
                    
                    if self.imu_count % 300 == 0:
                        print(f"[JETSON OBSERVER] 📊 IMU: magnitude={imu_data.magnitude:.3f}, "
                              f"state={self.current_motion_state}")
                        if self.dashboard:
                            self.dashboard.log_motion_state(self.current_motion_state, imu_data.magnitude)
                
                # Process frames (similar to original processing logic)
                processed_frames = {}
                
                for camera, frame in frames.items():
                    if frame is None:
                        continue
                    
                    if camera == 'rgb':
                        # Enhanced processing on RGB (main camera)
                        annotated_frame = self.coordinator.process_frame(frame)
                        processed_frames[camera] = annotated_frame
                        self.camera_detections[camera] = self.coordinator.current_detections
                        
                        # Motion-aware audio processing
                        if hasattr(self.coordinator, 'audio_system'):
                            self.coordinator.audio_system.process_detections(
                                self.coordinator.current_detections, 
                                self.current_motion_state
                            )
                    else:
                        # SLAM cameras - store as-is for now
                        processed_frames[camera] = frame
                        self.camera_detections[camera] = []
                
                # Update current frames
                with self._lock:
                    self.current_frames.update(processed_frames)
                    for camera in frames:
                        if frames[camera] is not None:
                            self.frame_counts[camera] += 1
                
                # Depth estimation (only on RGB when enabled)
                if 'rgb' in processed_frames and self.depth_estimator.model is not None:
                    if not hasattr(self, 'depth_frame_skip'):
                        self.depth_frame_skip = 10
                        self.cached_depth_map = None
                    
                    if self.frame_counts['rgb'] % self.depth_frame_skip == 0:
                        depth_map = self.depth_estimator.estimate_depth(processed_frames['rgb'])
                        self.cached_depth_map = depth_map
                        self.current_depth_map = depth_map
                    else:
                        depth_map = self.cached_depth_map
                
                # Dashboard updates (optimized)
                frame_counter += 1
                if self.dashboard and frame_counter % 3 == 0:
                    # RGB + detections every 3 frames
                    if 'rgb' in processed_frames:
                        self.dashboard.log_rgb_frame(processed_frames['rgb'])
                        rgb_detections = self.camera_detections.get('rgb', [])
                        self.dashboard.log_detections(rgb_detections, processed_frames['rgb'].shape)
                    
                    # SLAM frames every 5 frames
                    if frame_counter % 5 == 0:
                        if 'slam1' in processed_frames:
                            self.dashboard.log_slam1_frame(processed_frames['slam1'])
                        if 'slam2' in processed_frames:
                            self.dashboard.log_slam2_frame(processed_frames['slam2'])
                    
                    # Depth every 10 frames
                    if frame_counter % 10 == 0 and self.current_depth_map is not None:
                        self.dashboard.log_depth_map(self.current_depth_map)
                    
                    # Update dashboard
                    key = self.dashboard.update_all()
                    if key == ord('q'):
                        print("[JETSON OBSERVER] 'q' pressed - stopping system")
                        self._stop = True
                        break
                
                # Performance monitoring
                loop_time = time.time() - loop_start
                
                # Frame rate limiting
                target_frame_time = 1.0 / 30  # 30 FPS
                sleep_time = target_frame_time - loop_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                print(f"[JETSON OBSERVER] ⚠️ Error in processing loop: {e}")
                time.sleep(0.1)
    
    # =================================================================
    # API COMPATIBILITY METHODS (para mantener compatibilidad)
    # =================================================================
    
    def get_latest_frame(self, camera_type: str = 'rgb') -> Optional[np.ndarray]:
        """Get most recent processed frame"""
        with self._lock:
            return self.current_frames.get(camera_type, None)
    
    def get_latest_detections(self) -> list:
        """Get most recent RGB detections"""
        return self.camera_detections.get('rgb', [])
    
    def get_latest_depth_map(self) -> Optional[np.ndarray]:
        """Get most recent depth map"""
        return self.current_depth_map
    
    def get_motion_state(self) -> Dict:
        """Get current motion state"""
        latest_imu = self.receiver.get_latest_imu()
        return {
            'state': self.current_motion_state,
            'magnitude': latest_imu.magnitude if latest_imu else 9.8
        }
    
    def get_camera_detections(self) -> Dict:
        """Get detections from all cameras"""
        return self.camera_detections.copy()
    
    def get_all_frames(self) -> Dict:
        """Get frames from all cameras"""
        with self._lock:
            return self.current_frames.copy()
    
    def test_audio(self):
        """Test audio system"""
        try:
            if hasattr(self.coordinator, 'audio_system'):
                self.coordinator.audio_system.speak_force("Test audio desde Jetson")
                if self.dashboard:
                    self.dashboard.log_audio_command("Sistema de audio test", 1)
        except Exception as e:
            print(f"[JETSON OBSERVER] ⚠️ Audio test failed: {e}")
    
    def print_stats(self):
        """Print comprehensive statistics"""
        print(f"[JETSON OBSERVER] 📊 COMPREHENSIVE STATS:")
        
        # Receiver stats
        receiver_stats = self.receiver.get_stats()
        print(f"  📡 Receiver: {receiver_stats['frames_received']} frames, "
              f"{receiver_stats['fps']:.1f} FPS")
        
        # Frame counts
        print(f"  📦 Processing: RGB:{self.frame_counts['rgb']} "
              f"SLAM1:{self.frame_counts['slam1']} SLAM2:{self.frame_counts['slam2']}")
        
        # Motion state
        print(f"  🚶 Motion: {self.current_motion_state}")
        
        # Component stats
        if hasattr(self.coordinator, 'yolo_processor'):
            yolo_count = getattr(self.coordinator.yolo_processor, 'detection_count', 0)
            print(f"  🎯 YOLO: {yolo_count} detections")
        
        if hasattr(self.coordinator, 'audio_system'):
            audio_queue = getattr(self.coordinator.audio_system, 'audio_queue', [])
            print(f"  🔊 Audio: {len(audio_queue)} queued commands")
    
    def stop(self):
        """Stop processing and cleanup"""
        print("[JETSON OBSERVER] 🛑 Stopping system...")
        
        if self.dashboard:
            self.dashboard.log_system_message("Sistema Jetson cerrándose - Cleanup iniciado", "SYSTEM")
        
        self._stop = True
        
        # Wait for processing thread
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=1.0)
        
        # Stop receiver
        self.receiver.stop()
        
        # Cleanup coordinator
        try:
            self.coordinator.cleanup()
        except Exception as e:
            print(f"[JETSON OBSERVER] ⚠️ Coordinator cleanup error: {e}")
        
        # Shutdown dashboard
        if self.dashboard:
            self.dashboard.shutdown()
        
        print("[JETSON OBSERVER] ✅ Sistema detenido")


if __name__ == "__main__":
    # Test básico
    observer = JetsonObserver(enable_dashboard=True)
    
    if observer.start():
        print("Observer started, press Ctrl+C to stop")
        try:
            while True:
                time.sleep(5)
                observer.print_stats()
        except KeyboardInterrupt:
            pass
        finally:
            observer.stop()