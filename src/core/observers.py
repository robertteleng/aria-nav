import numpy as np
import cv2
import threading
import time
import aria.sdk as aria
from projectaria_tools.core.sensor_data import ImageDataRecord

from vision.yolo_proccesor import YoloProcessor
from audio.audio_system import AudioSystem
from utils.visualization import FrameRenderer
from vision.depth_estimator import DepthEstimator

class Observer:
    """
    Main observer integrating vision, audio, and IMU processing.
    Implements the observer pattern for Aria SDK callbacks.
    """
    
    def __init__(self, rgb_calib=None):
        # Core processing components
        self.yolo_processor = YoloProcessor()
        self.audio_system = AudioSystem()
        self.frame_renderer = FrameRenderer()

        self.depth_estimator = DepthEstimator()
        self.current_depth_map = None

        
        # Frame processing
        self.current_frame = None
        self.frame_count = 0
        
        # Async processing to avoid blocking callbacks
        self._lock = threading.Lock()
        self._latest_raw = None
        self._stop = False
        self._processing_thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )
        self._processing_thread.start()
        
        print("[INFO] ✓ Navigation observer initialized")
    
    def on_image_received(self, image: np.array, record: ImageDataRecord) -> None:
        """SDK callback for new images - RGB only"""
        if record.camera_id == aria.CameraId.Rgb:
            # FIX COLOR: Convertir BGR a RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Rotate for correct orientation
            rotated_image = np.rot90(image, -1)
            contiguous_image = np.ascontiguousarray(rotated_image)
            
            # Update frame dimensions for spatial processing
            h, w = contiguous_image.shape[:2]
            self.audio_system.update_frame_dimensions(w, h)
            
            # Pass to async processing
            with self._lock:
                self._latest_raw = contiguous_image
            self.frame_count += 1
            
            if self.frame_count % 100 == 0:
                print(f"[DEBUG] RGB frames processed: {self.frame_count}")
    
    
    def on_streaming_client_failure(self, reason, message: str) -> None:
        """SDK callback for streaming errors"""
        print(f"[ERROR] Streaming failure: {reason}: {message}")
    
    def _processing_loop(self):
        """Async thread: consume latest frame and run YOLO + overlay"""
        while not self._stop:
            frame = None
            with self._lock:
                if self._latest_raw is not None:
                    frame = self._latest_raw
                    self._latest_raw = None
            
            if frame is None:
                time.sleep(0.003)
                continue
            
            try:
                # # Depth estimation (si está habilitado)
                # depth_map = None
                # if self.depth_estimator.model is not None:
                #     depth_map = self.depth_estimator.estimate_depth(frame)
                #     self.current_depth_map = depth_map
            
                # THROTTLING: Procesar depth cada 5 frames para mejor performance
                depth_map = None
                if self.depth_estimator.model is not None:
                    if not hasattr(self, 'depth_frame_skip'):
                        self.depth_frame_skip = 5
                        self.cached_depth_map = None
                    
                    if self.frame_count % self.depth_frame_skip == 0:
                        # Procesar depth cada 5 frames
                        depth_map = self.depth_estimator.estimate_depth(frame)
                        self.cached_depth_map = depth_map
                        self.current_depth_map = depth_map
                    else:
                        # Usar depth map cacheado
                        depth_map = self.cached_depth_map

                # Vision processing (pasando depth_map)
                detections = self.yolo_processor.process_frame(frame, depth_map)
                
                # Audio processing  
                self.audio_system.process_detections(detections)
                
                # Visual overlay
                annotated_frame = self.frame_renderer.draw_navigation_overlay(
                frame, detections, self.audio_system, depth_map
                )
            
                self.current_frame = annotated_frame
                
            except Exception as e:
                print(f"[WARN] Processing pipeline failed: {e}")
                self.current_frame = frame
    
    def get_latest_frame(self):
        """Get most recent processed frame"""
        return self.current_frame
    
    # Añadir método para obtener depth map:
    def get_latest_depth(self):
        """Get most recent depth map"""
        return self.current_depth_map
    
    def test_audio(self):
        """Test audio system - triggered by 't' key"""
        try:
            self.audio_system.speak_force("audio test")
        except Exception as e:
            print(f"[WARN] Audio test failed: {e}")
    
    def print_stats(self):
        """Print final statistics"""
        print(f"[INFO] Final stats:")
        print(f"  - RGB frames received: {self.frame_count}")
        print(f"  - YOLO detections: {self.yolo_processor.detection_count}")
        print(f"  - Audio commands sent: {len(self.audio_system.audio_queue)}")

    def stop(self):
        """Stop async processing and cleanup"""
        self._stop = True
        try:
            if hasattr(self, '_processing_thread') and self._processing_thread.is_alive():
                self._processing_thread.join(timeout=0.5)
        except Exception:
            pass
        
        # Cleanup subsystems
        try:
            self.audio_system.close()
        except Exception:
            pass
