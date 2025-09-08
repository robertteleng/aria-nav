import numpy as np
import cv2
import threading
import time
import aria.sdk as aria
from projectaria_tools.core.sensor_data import ImageDataRecord, MotionData
from typing import Sequence, Dict, Optional

from vision.yolo_proccesor import YoloProcessor
from audio.audio_system import AudioSystem
from utils.visualization import FrameRenderer
from vision.depth_estimator import DepthEstimator
from vision.image_enhancer import ImageEnhancer

class Observer:
    """
    Enhanced observer with peripheral vision using RGB + SLAM1 + SLAM2 cameras.
    Implements 180° field of view for complete spatial awareness.
    """
    
    def __init__(self, rgb_calib=None):
        # Core processing components
        self.yolo_processor = YoloProcessor()
        self.audio_system = AudioSystem()
        self.frame_renderer = FrameRenderer()
        self.depth_estimator = DepthEstimator()
        self.image_enhancer = ImageEnhancer()
        
        # Multi-camera frame storage
        self.current_frames = {
            'rgb': None,      # Center camera (main)
            'slam1': None,    # Left camera 
            'slam2': None     # Right camera
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
        
        print("[INFO] ✓ Peripheral Vision Observer initialized")
        print("[INFO] ✓ Monitoring: RGB (center) + SLAM1 (left) + SLAM2 (right)")
    
    def on_image_received(self, image: np.array, record: ImageDataRecord) -> None:
        """SDK callback for new images - ALL cameras"""
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
        
        # Debug every 100 frames
        if self.frame_counts[camera_key] % 100 == 0:
            print(f"[DEBUG] {camera_key.upper()} frames processed: {self.frame_counts[camera_key]}")

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
        # SLAM cameras are grayscale, convert to RGB for consistency
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        # Rotate SLAM images (same as RGB for consistency)
        rotated_image = np.rot90(image, -1)
        return np.ascontiguousarray(rotated_image)

    def on_imu_received(self, samples: Sequence[MotionData], imu_idx: int) -> None:
        """Motion Detection - same as before"""
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

    def on_streaming_client_failure(self, reason, message: str) -> None:
        """SDK callback for streaming errors"""
        print(f"[ERROR] Streaming failure: {reason}: {message}")
    
    def _processing_loop(self):
        """Enhanced async processing with multi-camera fusion"""
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
                time.sleep(0.003)
                continue
                
            try:
                # Process each camera stream
                processed_frames = {}
                all_detections = []
                
                for camera, frame in frames.items():
                    # Apply image enhancement
                    enhanced_frame = self.image_enhancer.enhance_frame(frame)
                    processed_frames[camera] = enhanced_frame
                    
                    # Run YOLO detection on each camera
                    detections = self.yolo_processor.process_frame(enhanced_frame, None)
                    
                    # Add camera source info to detections
                    for detection in detections:
                        detection['camera'] = camera
                        detection['camera_zone'] = self._get_camera_zone(camera)
                    
                    all_detections.extend(detections)
                
                # Store current frames
                self.current_frames.update(processed_frames)
                
                # Depth estimation (only on RGB for now)
                depth_map = None
                if 'rgb' in frames and self.depth_estimator.model is not None:
                    if not hasattr(self, 'depth_frame_skip'):
                        self.depth_frame_skip = 5
                        self.cached_depth_map = None
                    
                    if self.frame_counts['rgb'] % self.depth_frame_skip == 0:
                        depth_map = self.depth_estimator.estimate_depth(processed_frames['rgb'])
                        self.cached_depth_map = depth_map
                        self.current_depth_map = depth_map
                    else:
                        depth_map = self.cached_depth_map

                # Fusion: Combine detections from all cameras
                fused_detections = self._fuse_multi_camera_detections(all_detections)
                
                # Audio processing with enhanced spatial awareness
                motion_state = getattr(self.motion_detector, 'last_motion_state', 'stationary')
                self.audio_system.process_detections(fused_detections, motion_state)

                # Visual overlay (use RGB as main display)
                if 'rgb' in processed_frames:
                    annotated_frame = self.frame_renderer.draw_navigation_overlay(
                        processed_frames['rgb'], fused_detections, self.audio_system, depth_map
                    )
                    # Add peripheral vision indicators
                    annotated_frame = self._add_peripheral_indicators(annotated_frame, all_detections)
                else:
                    # Fallback to any available frame
                    annotated_frame = list(processed_frames.values())[0]
                
                # Store final result
                self.current_frame = annotated_frame
                
            except Exception as e:
                print(f"[WARN] Multi-camera processing failed: {e}")
                # Fallback to single camera
                if 'rgb' in frames:
                    self.current_frame = frames['rgb']
                elif frames:
                    self.current_frame = list(frames.values())[0]

    def _get_camera_zone(self, camera: str) -> str:
        """Map camera to spatial zone"""
        zone_mapping = {
            'rgb': 'center',
            'slam1': 'left_peripheral', 
            'slam2': 'right_peripheral'
        }
        return zone_mapping.get(camera, 'unknown')

    def _fuse_multi_camera_detections(self, all_detections: list) -> list:
        """
        Intelligent fusion of detections from multiple cameras.
        Removes duplicates and prioritizes by importance and camera.
        """
        if not all_detections:
            return []
        
        # Priority by camera (center > peripherals)
        camera_priority = {'rgb': 3, 'slam1': 2, 'slam2': 2}
        
        # Priority by object type
        object_priority = {
            'person': 10, 'car': 9, 'truck': 9, 'bus': 9,
            'bicycle': 8, 'motorcycle': 8,
            'stop sign': 7, 'traffic light': 6,
            'chair': 4, 'door': 3, 'bench': 2
        }
        
        # Add combined priority score
        for detection in all_detections:
            camera_score = camera_priority.get(detection.get('camera', 'rgb'), 1)
            object_score = object_priority.get(detection.get('name', ''), 1)
            confidence_score = detection.get('confidence', 0.5)
            
            # Combined score: camera importance + object importance + confidence
            detection['fusion_score'] = (camera_score * 2) + object_score + (confidence_score * 5)
        
        # Sort by fusion score (highest first)
        sorted_detections = sorted(all_detections, key=lambda x: x['fusion_score'], reverse=True)
        
        # Remove duplicates and limit to top detections
        unique_detections = []
        max_detections = 3  # Prevent audio overload
        
        for detection in sorted_detections:
            if len(unique_detections) >= max_detections:
                break
                
            # Simple duplicate check (same object type in similar area)
            is_duplicate = False
            for existing in unique_detections:
                if (detection['name'] == existing['name'] and 
                    detection.get('camera_zone') == existing.get('camera_zone')):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_detections.append(detection)
        
        return unique_detections

    def _add_peripheral_indicators(self, frame: np.array, all_detections: list) -> np.array:
        """Add visual indicators for peripheral camera detections"""
        h, w = frame.shape[:2]
        
        # Count detections by camera
        camera_counts = {'slam1': 0, 'slam2': 0}
        for detection in all_detections:
            camera = detection.get('camera', 'rgb')
            if camera in camera_counts:
                camera_counts[camera] += 1
        
        # Add left indicator
        if camera_counts['slam1'] > 0:
            cv2.rectangle(frame, (10, 10), (60, 40), (255, 255, 0), 2)
            cv2.putText(frame, f"L:{camera_counts['slam1']}", (15, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Add right indicator  
        if camera_counts['slam2'] > 0:
            cv2.rectangle(frame, (w-70, 10), (w-10, 40), (255, 165, 0), 2)
            cv2.putText(frame, f"R:{camera_counts['slam2']}", (w-65, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        
        return frame

    # API compatibility methods (same as before)
    def get_latest_frame(self):
        """Get most recent processed frame (RGB main view)"""
        return getattr(self, 'current_frame', None)

    def get_latest_detections(self):
        """Get most recent fused detections for dashboard"""
        return getattr(self.yolo_processor, 'latest_detections', [])

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

    def get_peripheral_frames(self):
        """Get all camera frames for advanced dashboard"""
        return self.current_frames.copy()

    def test_audio(self):
        """Test audio system"""
        try:
            self.audio_system.speak_force("peripheral vision test - center, left, right")
        except Exception as e:
            print(f"[WARN] Audio test failed: {e}")

    def print_stats(self):
        """Print comprehensive statistics"""
        print(f"[INFO] Peripheral Vision Statistics:")
        print(f"  - RGB frames: {self.frame_counts['rgb']}")
        print(f"  - SLAM1 frames: {self.frame_counts['slam1']}")
        print(f"  - SLAM2 frames: {self.frame_counts['slam2']}")
        print(f"  - YOLO detections: {self.yolo_processor.detection_count}")
        print(f"  - Audio commands: {len(self.audio_system.audio_queue)}")

    def stop(self):
        """Stop processing and cleanup"""
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