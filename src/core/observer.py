import numpy as np
import cv2
import threading
import time
import aria.sdk as aria
from projectaria_tools.core.sensor_data import ImageDataRecord, MotionData
from typing import Sequence, Dict

from vision.yolo_proccesor import YoloProcessor
from audio.audio_system import AudioSystem
from utils.visualization import FrameRenderer
from vision.depth_estimator import DepthEstimator
from vision.image_enhancer import ImageEnhancer
from utils.opencv_dashboard import OpenCVDashboard


class Observer:
    """
    Enhanced observer with multi-camera capture: RGB + SLAM1 + SLAM2
    + Rerun Dashboard integration
    """
    
    def __init__(self, rgb_calib=None, enable_dashboard=True):
        # Core processing components
        self.yolo_processor = YoloProcessor()
        self.audio_system = AudioSystem()
        self.frame_renderer = FrameRenderer()
        self.depth_estimator = DepthEstimator()
        self.image_enhancer = ImageEnhancer()
        
        # Dashboard integration
        # self.dashboard = RerunDashboard() if enable_dashboard else None
        self.dashboard = OpenCVDashboard() if enable_dashboard else None

        if self.dashboard:
            self.dashboard.log_system_message("Sistema iniciado - Observer multicámara activo", "SYSTEM")
        
        # Multi-camera frame storage
        self.current_frames = {
            'rgb': None,      # Center camera (main)
            'slam1': None,    # Left camera 
            'slam2': None     # Right camera
        }
        
        # Multi-camera detections (separate for now)
        self.camera_detections = {
            'rgb': [],
            'slam1': [],
            'slam2': []
        }
        
        # Processing state
        self.frame_counts = {'rgb': 0, 'slam1': 0, 'slam2': 0}
        self.current_depth_map = None
        
        # Async processing
        self._lock = threading.Lock()
        self._latest_raw_frames = {'rgb': None, 'slam1': None, 'slam2': None}
        self._stop = False
        self._processing_thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )
        self._processing_thread.start()
        
        print("[INFO] Multi-camera Observer initialized")
        print("[INFO] Monitoring: RGB (center) + SLAM1 (left) + SLAM2 (right)")
        if self.dashboard:
            self.dashboard.log_system_message("Cámaras configuradas: RGB + SLAM1 + SLAM2", "SYSTEM")
    
    def on_image_received(self, image: np.array, record: ImageDataRecord) -> None:
        """SDK callback for new images - ALL three cameras"""
        camera_id = record.camera_id
        
        # Process RGB camera (center view)
        if camera_id == aria.CameraId.Rgb:
            processed_image = self._process_rgb_image(image)
            camera_key = 'rgb'
            
        # Process SLAM1 camera (left peripheral)
        elif camera_id == aria.CameraId.Slam1:
            processed_image = self._process_slam_image(image)
            camera_key = 'slam1'
            
        # Process SLAM2 camera (right peripheral) 
        elif camera_id == aria.CameraId.Slam2:
            processed_image = self._process_slam_image(image)
            camera_key = 'slam2'
            
        else:
            # Ignore EyeTrack and other cameras
            return
            
        # Update frame dimensions for spatial processing (using RGB as reference)
        if camera_key == 'rgb':
            h, w = processed_image.shape[:2]
            self.audio_system.update_frame_dimensions(w, h)
        
        # Store processed frame for async processing
        with self._lock:
            self._latest_raw_frames[camera_key] = processed_image
            
        self.frame_counts[camera_key] += 1
        
        # Debug every 100 frames per camera
        if self.frame_counts[camera_key] % 100 == 0:
            print(f"[DEBUG] {camera_key.upper()} frames processed: {self.frame_counts[camera_key]}")
            if self.dashboard:
                self.dashboard.log_system_message(f"{camera_key.upper()} frames: {self.frame_counts[camera_key]}", "DEBUG")

    def _process_rgb_image(self, image: np.array) -> np.array:
        """Process RGB camera image (center view)"""
        # Fix color channels
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Rotate for correct orientation
        rotated_image = np.rot90(image, -1)
        return np.ascontiguousarray(rotated_image)
    
    def _process_slam_image(self, image: np.array) -> np.array:
        """Process SLAM camera images (left/right peripheral)"""
        # SLAM cameras are grayscale, convert to RGB for YOLO consistency
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        # Rotate SLAM images (same as RGB for consistency)
        rotated_image = np.rot90(image, -1)
        return np.ascontiguousarray(rotated_image)

    def on_imu_received(self, samples: Sequence[MotionData], imu_idx: int) -> None:
        """Motion Detection - unchanged pero con log al dashboard"""
        sample = samples[0]
        accelerometer = sample.accel_msec2
        timestamp = sample.capture_timestamp_ns
        
        magnitude = (accelerometer[0]**2 + accelerometer[1]**2 + accelerometer[2]**2)**0.5
        
        if not hasattr(self, 'motion_detector'):
            from imu.motion_detector import SimpleMotionDetector
            self.motion_detector = SimpleMotionDetector()
        
        if imu_idx == 0:
            motion_state = self.motion_detector.update(magnitude, timestamp)
            
            if not hasattr(self, 'imu_count'):
                self.imu_count = 0
            self.imu_count += 1
                
            if self.imu_count % 300 == 0:
                print(f"[MOTION DEBUG] Magnitude: {magnitude:.3f} m/s² - State: {motion_state}")
                if self.dashboard:
                    self.dashboard.log_motion_state(motion_state, magnitude)

    def on_streaming_client_failure(self, reason, message: str) -> None:
        """SDK callback for streaming errors"""
        print(f"[ERROR] Streaming failure: {reason}: {message}")
        if self.dashboard:
            self.dashboard.log_system_message(f"Streaming error: {message}", "ERROR")
    
    def _processing_loop(self):
        """Multi-camera processing - YOLO on each camera separately + Dashboard logging"""
        frame_counter = 0
        
        while not self._stop:
            # Get latest frames from all cameras
            frames = {}
            with self._lock:
                for camera in ['rgb', 'slam1', 'slam2']:
                    if self._latest_raw_frames[camera] is not None:
                        frames[camera] = self._latest_raw_frames[camera]
                        self._latest_raw_frames[camera] = None
            
            # Skip if no new frames
            if not frames:
                time.sleep(0.01)
                continue
                
            try:
                # Process each camera independently
                processed_frames = {}
                
                for camera, frame in frames.items():
                    # Apply image enhancement
                    enhanced_frame = self.image_enhancer.enhance_frame(frame)
                    processed_frames[camera] = enhanced_frame
                    
                    # Run YOLO detection on each camera separately
                    detections = self.yolo_processor.process_frame(enhanced_frame, None)
                    
                    # Store detections per camera (for debugging/analysis)
                    self.camera_detections[camera] = detections
                    
                    # Debug: print detections per camera
                    if detections:
                        print(f"[{camera.upper()}] Detected {len(detections)} objects: {[d['name'] for d in detections]}")
                        if self.dashboard:
                            for det in detections[:2]:  # Top 2 por cámara
                                name = det.get('name', 'unknown')
                                zone = det.get('zone', 'unknown')
                                priority = det.get('priority', 0)
                                self.dashboard.log_system_message(
                                    f"{camera.upper()}: {name} en {zone} (P{priority})", "DETECT"
                                )
                
                # Store current frames
                self.current_frames.update(processed_frames)

                # Enviar frames al dashboard
                if self.dashboard:
                    # Panel 1: RGB + YOLO (arriba-izquierda)
                    if 'rgb' in processed_frames:
                        rgb_detections = self.camera_detections.get('rgb', [])
                        # Aplicar overlay primero
                        rgb_with_overlay = self.frame_renderer.draw_navigation_overlay(
                            processed_frames['rgb'], rgb_detections, self.audio_system, None
                        )
                        self.dashboard.log_rgb_frame(rgb_with_overlay)
                        self.dashboard.log_detections(rgb_detections, processed_frames['rgb'].shape)
                    
                    # Panel 3: SLAM1 (arriba-derecha)
                    if 'slam1' in processed_frames:
                        self.dashboard.log_slam1_frame(processed_frames['slam1'])
                    
                    # Panel 6: SLAM2 (abajo-derecha)
                    if 'slam2' in processed_frames:
                        self.dashboard.log_slam2_frame(processed_frames['slam2'])

                # Depth estimation (only on RGB)
                depth_map = None
                if 'rgb' in frames and self.depth_estimator.model is not None:
                    if not hasattr(self, 'depth_frame_skip'):
                        self.depth_frame_skip = 5
                        self.cached_depth_map = None
                    
                    if self.frame_counts['rgb'] % self.depth_frame_skip == 0:
                        depth_map = self.depth_estimator.estimate_depth(processed_frames['rgb'])
                        self.cached_depth_map = depth_map
                        self.current_depth_map = depth_map
                        if self.dashboard:
                            self.dashboard.log_depth_map(depth_map)
                    else:
                        depth_map = self.cached_depth_map

                # Audio processing: Use ONLY RGB detections for now (unchanged behavior)
                rgb_detections = self.camera_detections.get('rgb', [])
                motion_state = getattr(self.motion_detector, 'last_motion_state', 'stationary')
                
                # Capturar comandos de audio antes de procesarlos
                audio_commands_before = len(self.audio_system.audio_queue)
                self.audio_system.process_detections(rgb_detections, motion_state)
                audio_commands_after = len(self.audio_system.audio_queue)
                
                # Si se añadieron comandos, logear al dashboard
                if self.dashboard and audio_commands_after > audio_commands_before:
                    # Obtener último comando añadido
                    if hasattr(self.audio_system, 'last_command_sent'):
                        self.dashboard.log_audio_command(self.audio_system.last_command_sent, 5)

                # Visual overlay: Use RGB as main display + add camera info
                if 'rgb' in processed_frames:
                    annotated_frame = self.frame_renderer.draw_navigation_overlay(
                        processed_frames['rgb'], rgb_detections, self.audio_system, depth_map
                    )
                    # Add multi-camera indicators
                    annotated_frame = self._add_camera_info_overlay(annotated_frame)
                else:
                    # Fallback to any available frame
                    annotated_frame = list(processed_frames.values())[0]
                
                # Store final result
                self.current_frame = annotated_frame
                
                # Actualizar métricas del dashboard cada 30 frames
                frame_counter += 1
                if self.dashboard and frame_counter % 30 == 0:
                    self.dashboard.log_performance_metrics()
                
            except Exception as e:
                print(f"[WARN] Multi-camera processing failed: {e}")
                if self.dashboard:
                    self.dashboard.log_system_message(f"Processing error: {str(e)}", "ERROR")
                
                # Fallback to single camera
                if 'rgb' in frames:
                    self.current_frame = frames['rgb']
                elif frames:
                    self.current_frame = list(frames.values())[0]

    def _add_camera_info_overlay(self, frame: np.array) -> np.array:
        """Add simple indicators showing multi-camera status"""
        h, w = frame.shape[:2]
        
        # Count detections by camera
        rgb_count = len(self.camera_detections.get('rgb', []))
        slam1_count = len(self.camera_detections.get('slam1', []))
        slam2_count = len(self.camera_detections.get('slam2', []))
        
        # Add indicators (top of screen)
        y_pos = 15
        
        # Center (RGB) - Always show
        cv2.putText(frame, f"CENTER: {rgb_count}", (w//2 - 50, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Left (SLAM1) - Yellow if active
        color = (255, 255, 0) if slam1_count > 0 else (100, 100, 100)
        cv2.putText(frame, f"LEFT: {slam1_count}", (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Right (SLAM2) - Orange if active  
        color = (255, 165, 0) if slam2_count > 0 else (100, 100, 100)
        cv2.putText(frame, f"RIGHT: {slam2_count}", (w - 100, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame

    # API compatibility methods
    def get_latest_frame(self):
        """Get most recent processed frame (RGB main view)"""
        return getattr(self, 'current_frame', None)

    def get_latest_detections(self):
        """Get most recent RGB detections for dashboard (unchanged behavior)"""
        return self.camera_detections.get('rgb', [])

    def get_latest_depth_map(self):
        """Get most recent depth map for dashboard"""
        return self.current_depth_map

    def get_motion_state(self):
        """Get current motion state and magnitude for dashboard"""
        if hasattr(self, 'motion_detector'):
            return {
                'state': self.motion_detector.last_motion_state,
                'magnitude': getattr(self.motion_detector, 'last_magnitude', 9.8)
            }
        return None

    def get_camera_detections(self):
        """Get detections from all cameras for analysis"""
        return self.camera_detections.copy()

    def get_all_frames(self):
        """Get frames from all cameras"""
        return self.current_frames.copy()

    def test_audio(self):
        """Test audio system"""
        try:
            self.audio_system.speak_force("audio test")
            if self.dashboard:
                self.dashboard.log_audio_command("Sistema de audio test", 1)
        except Exception as e:
            print(f"[WARN] Audio test failed: {e}")
            if self.dashboard:
                self.dashboard.log_system_message(f"Audio test error: {str(e)}", "ERROR")

    def print_stats(self):
        """Print multi-camera statistics"""
        print(f"[INFO] Multi-camera Statistics:")
        print(f"  - RGB frames: {self.frame_counts['rgb']}")
        print(f"  - SLAM1 frames: {self.frame_counts['slam1']}")
        print(f"  - SLAM2 frames: {self.frame_counts['slam2']}")
        print(f"  - YOLO detections: {self.yolo_processor.detection_count}")
        print(f"  - Audio commands: {len(self.audio_system.audio_queue)}")
        
        if self.dashboard:
            stats_msg = f"RGB:{self.frame_counts['rgb']} SLAM1:{self.frame_counts['slam1']} SLAM2:{self.frame_counts['slam2']} YOLO:{self.yolo_processor.detection_count}"
            self.dashboard.log_system_message(f"Stats: {stats_msg}", "STATS")
        
        # Show recent detections per camera
        for camera, detections in self.camera_detections.items():
            if detections:
                objects = [d['name'] for d in detections]
                print(f"  - {camera.upper()} last detections: {objects}")

    def stop(self):
        """Stop processing and cleanup"""
        if self.dashboard:
            self.dashboard.log_system_message("Sistema cerrándose - Cleanup iniciado", "SYSTEM")
        
        self._stop = True
        try:
            if hasattr(self, '_processing_thread') and self._processing_thread.is_alive():
                self._processing_thread.join(timeout=0.5)
        except Exception:
            pass
        
        try:
            self.audio_system.close()
        except Exception:
            pass
        
        if self.dashboard:
            self.dashboard.shutdown()
