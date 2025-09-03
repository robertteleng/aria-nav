import numpy as np
import cv2
import threading
import time
import aria.sdk as aria
from projectaria_tools.core.sensor_data import MotionData, BarometerData, ImageDataRecord

from vision.yolo_proccesor import YoloProcessor
from audio.audio_system import AudioSystem
from utils.visualization import FrameRenderer
from motion.motion_processor import MotionProcessor


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
        self.motion_processor = MotionProcessor()

        # Contadores para debug
        self.imu_sample_count = 0
        self.magneto_sample_count = 0
        self._last_imu_debug = 0
        
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
    
    def on_imu_received(self, samples, imu_idx) -> None:
        """SDK callback for IMU"""
        self.imu_sample_count += len(samples)
        
        # Debug cada 100 muestras para no saturar
        if self.imu_sample_count - self._last_imu_debug >= 100:
            print(f"[IMU] Received {len(samples)} samples from IMU {imu_idx}")
            if samples:
                sample = samples[0]
                accel = sample.accel_msec2
                gyro = sample.gyro_radsec
                # print(f"[IMU] Accel: [{accel[0]:.2f}, {accel[1]:.2f}, {accel[2]:.2f}] m/s²")
                # print(f"[IMU] Gyro: [{gyro[0]:.3f}, {gyro[1]:.3f}, {gyro[2]:.3f}] rad/s")
            self._last_imu_debug = self.imu_sample_count
        
        # Pasar al procesador
        self.motion_processor.update_motion(samples, imu_idx)
    
    def on_magneto_received(self, sample: MotionData) -> None:
        """SDK callback for magnetometer"""
        self.magneto_sample_count += 1
    
        # Debug cada 20 muestras
        if self.magneto_sample_count % 20 == 0:
            mag = sample.mag_tesla
            print(f"[MAGNETO] Sample #{self.magneto_sample_count}")
            print(f"[MAGNETO] Mag: [{mag[0]:.6f}, {mag[1]:.6f}, {mag[2]:.6f}] Tesla")
            heading = self.motion_processor.orientation.heading
            print(f"[MAGNETO] Current heading: {heading:.1f}°")

            # ===== AÑADE ESTO - TEST DE EJES =====
            mag_x, mag_y, mag_z = mag
            combo1 = (np.degrees(np.arctan2(mag_y, mag_x)) + 360) % 360  # Original
            combo2 = (np.degrees(np.arctan2(mag_x, mag_z)) + 360) % 360  # X,Z  
            combo3 = (np.degrees(np.arctan2(mag_y, mag_z)) + 360) % 360  # Y,Z
            
            print(f"[TEST] Heading X,Y: {combo1:.1f}°")
            print(f"[TEST] Heading X,Z: {combo2:.1f}°")
            print(f"[TEST] Heading Y,Z: {combo3:.1f}°")
            print(f"[TEST] Tu brújula: 206° (SW)")
            print("")
        
        # Pasar al procesador
        self.motion_processor.update_heading(sample)
        
    def on_baro_received(self, sample) -> None:
        """SDK callback for barometer - optional for altitude"""
        pass
    
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
                # Vision processing
                detections = self.yolo_processor.process_frame(frame)
                
                # Audio processing  
                self.audio_system.process_detections(detections)
                
                # Visual overlay
                annotated_frame = self.frame_renderer.draw_navigation_overlay(
                    frame, detections, self.audio_system
                )
                self.current_frame = annotated_frame
                
            except Exception as e:
                print(f"[WARN] Processing pipeline failed: {e}")
                self.current_frame = frame
    
    def get_latest_frame(self):
        """Get most recent processed frame"""
        return self.current_frame
    
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
        print(f"  - IMU samples processed: {self.imu_sample_count}")
        print(f"  - Magnetometer samples: {self.magneto_sample_count}")
        if hasattr(self.motion_processor, 'orientation'):
            orientation = self.motion_processor.orientation
            print(f"  - Final heading: {orientation.heading:.1f}°")
            print(f"  - Movement detected: {orientation.is_moving}")

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
