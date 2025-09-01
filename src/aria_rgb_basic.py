"""
Navigation system for blind users using Meta Aria glasses
TFM - Day 1: Basic RGB stream with custom observer

Date: 2025-08-30
Version: 1.0 - Basic RGB streaming
"""

import signal
import cv2
import numpy as np
import torch
import threading
import time
import aria.sdk as aria
from ultralytics import YOLO


from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)

from audio_navigation_system import AudioNavigationSystem



class CtrlCHandler:
    """
    Handle Ctrl+C for clean shutdown to avoid data corruption
    and abrupt device disconnect.
    """
    def __init__(self):
        self.should_stop = False
        # Registrar handler para SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Callback executed when Ctrl+C is detected"""
        print("\n[INFO] Interrupt signal detected, closing cleanly...")
        self.should_stop = True


class AriaRgbObserver:
    """
    Custom observer to receive only the RGB stream from Aria glasses.
    Implements observer callbacks from the Aria SDK and filters RGB images.
    """
    
    def __init__(self, rgb_calib=None):
        # Most recent frame produced
        self.current_frame = None
        # Counter for stats
        self.frame_count = 0

        # Official calibration if available
        self.rgb_calib = rgb_calib
        self.dst_calib = None
        # Counter for stats
        self.frame_count = 0

        print("[INFO] Loading YOLO model...")
        self.yolo_model = YOLO('yolo11n.pt')  
        
        self.device = 'cpu'
    
        # Move model to selected device
        self.yolo_model.to(self.device)
        print("[INFO] ✓ YOLOv11 loaded and configured")

        # Initialize audio navigation system
        self.audio_system = AudioNavigationSystem()

        # Procesamiento asíncrono para evitar bloqueo del observer
        self._lock = threading.Lock()
        self._latest_raw = None
        self._stop = False
        self._proc_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._proc_thread.start()

    
    def on_image_received(self, image: np.array, record) -> None:
        """
        SDK callback when a new image arrives
        Args:
            image: NumPy array with image data (BGR)
            record: Image metadata (timestamp, camera_id, ...)
        """
        # Filter only RGB camera
        if record.camera_id == aria.CameraId.Rgb:
            # --- Undistort with official calibration (optional) ---
            img_bgr = image  # as provided by the SDK

            # if self.rgb_calib is not None and self.dst_calib is None:
            #     h, w = img_bgr.shape[:2]
            #     self.dst_calib = get_linear_camera_calibration(w, h, 120, "camera-rgb")
            #     print(f"[INFO] Undistort activo → destino {w}x{h}")

            # if self.rgb_calib is not None and self.dst_calib is not None:
            #     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            #     undist_rgb = distort_by_calibration(img_rgb, self.dst_calib, self.rgb_calib)
            #     img_bgr = cv2.cvtColor(undist_rgb, cv2.COLOR_RGB2BGR)

            # Rotate 90° for correct orientation (physical mount)
            rotated_image = np.rot90(img_bgr, -1)
            contiguous_image = np.ascontiguousarray(rotated_image)

            h, w = contiguous_image.shape[:2]
            self.audio_system.update_frame_dimensions(w, h)

            # Hand off the frame to the async pipeline (keep only the latest)
            with self._lock:
                self._latest_raw = contiguous_image
            self.frame_count += 1
            
            # Debug: print stats every 100 frames
            if self.frame_count % 100 == 0:
                print(f"[DEBUG] RGB frames processed: {self.frame_count}")

    def _draw_enhanced_frame(self, frame, detections):
        """Draw overlays with zones and navigation detections"""
        height, width = frame.shape[:2]

        # Copy frame to annotate
        annotated_frame = frame.copy()

        # === 4-quadrant grid ===
        midx, midy = width // 2, height // 2
        cv2.line(annotated_frame, (midx, 0), (midx, height - 1), (255, 255, 255), 1)
        cv2.line(annotated_frame, (0, midy), (width - 1, midy), (255, 255, 255), 1)

        # Quadrant labels (Title Case)
        cv2.putText(annotated_frame, "Top Left", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, "Top Right", (midx + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, "Bottom Left", (10, midy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, "Bottom Right", (midx + 10, midy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Colors per quadrant (visible over the image)
        quadrant_colors = {
            'top_left': (0, 255, 255),     # yellow
            'top_right': (0, 165, 255),    # orange
            'bottom_left': (0, 0, 255),    # red
            'bottom_right': (255, 0, 255)  # magenta
        }

        # Draw detections
        zone_title = {
            'top_left': 'Top Left',
            'top_right': 'Top Right',
            'bottom_left': 'Bottom Left',
            'bottom_right': 'Bottom Right',
            'center': 'Center',
        }
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            zone = det.get('zone', det.get('quadrant'))
            color = quadrant_colors.get(zone, (255, 255, 255))

            # Colorized bbox by quadrant
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Label with class, quadrant and distance
            zone_text = zone_title.get(zone, zone.replace('_', ' ').title())
            label = f"{det['name']} ({zone_text}) - {det['distance']}"
            cv2.putText(annotated_frame, label, (int(x1), max(20, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # Audio system status
        queue_len = len(self.audio_system.audio_queue) if hasattr(self.audio_system, 'audio_queue') else 0
        audio_status = "🔊 SPEAKING" if self.audio_system.is_speaking else f"Queue: {queue_len}"
        cv2.putText(annotated_frame, audio_status, (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated_frame
    
    def get_latest_frame(self):
        """
        Get the most recent available frame
        Returns:
            np.array or None: Latest rotated RGB frame, or None if no data
        """
        return self.current_frame

    def _processing_loop(self):
        """Hilo: consume último frame y corre YOLO + overlay"""
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
                results = self.yolo_model(frame, device=self.device, verbose=False)
                detections = self.audio_system.process_detections(results)
                annotated = self._draw_enhanced_frame(frame, detections)
                self.current_frame = annotated
            except Exception as e:
                print(f"[WARN] Pipeline de procesamiento falló: {e}")
                self.current_frame = frame

    def stop(self):
        self._stop = True
        try:
            if hasattr(self, '_proc_thread') and self._proc_thread.is_alive():
                self._proc_thread.join(timeout=0.5)
        except Exception:
            pass


def connect_aria_device():
    """
    Establish connection with the Aria device
    Returns:
        tuple: (device_client, device)
    Raises:
        Exception: if connection fails
    """
    print("[INFO] Starting connection with Aria glasses...")
    
    # Create device client
    device_client = aria.DeviceClient()
    
    # TODO: add IP config if using WiFi
    # client_config = aria.DeviceClientConfig()
    # client_config.ip_v4_address = "192.168.1.100"
    # device_client.set_client_config(client_config)
    
    # Connect to device (USB by default)
    device = device_client.connect()
    
    print("[INFO] ✓ Connection established successfully")
    return device_client, device


def setup_rgb_streaming(device):
    """
    Configure streaming to capture only image data
    Args:
        device: Connected Aria device
    Returns:
        StreamingManager configured and started
    """
    print("[INFO] Configuring RGB streaming...")
    
    # Get streaming manager
    streaming_manager = device.streaming_manager
    
    # Create streaming config
    streaming_config = aria.StreamingConfig()
    
    # Use predefined profile (example: profile28 includes RGB)
    streaming_config.profile_name = "profile28"
    
    # Use USB interface (more stable than WiFi for dev)
    streaming_config.streaming_interface = aria.StreamingInterface.Usb
    
    # Use ephemeral certs (no manual cert setup)
    streaming_config.security_options.use_ephemeral_certs = True
    
    # Apply config to manager
    streaming_manager.streaming_config = streaming_config
    
        # Start streaming on device
    streaming_manager.start_streaming()

    # Get official calibration JSON
    sensors_calib_json = streaming_manager.sensors_calibration()
    sensors_calib = device_calibration_from_json_string(sensors_calib_json)

    rgb_calib = None
    try:
        rgb_calib = sensors_calib.get_camera_calib("camera-rgb")
        print("[INFO] ✓ RGB calibration obtained from SDK")
    except Exception as e:
        print(f"[WARN] Could not fetch RGB calibration: {e}")

    print("[INFO] ✓ RGB streaming started")

    return streaming_manager, rgb_calib

def main():
    """
    Main entrypoint: handle connection, streaming and visualization
    """
    print("=" * 60)
    print("TFM - Navigation system for blind users")
    print("Day 1: Basic RGB stream from Aria glasses")
    print("=" * 60)
    
    # Setup handler for clean exit
    ctrl_handler = CtrlCHandler()
    
    # Variables for cleanup in case of error
    device_client = None
    streaming_manager = None
    streaming_client = None
    
    try:
        # 1. Device connection
        device_client, device = connect_aria_device()
        
        # 2. Streaming configuration
        streaming_manager, rgb_calib = setup_rgb_streaming(device)
        
        # 3. Observer setup
        print("[INFO] Setting up RGB observer...")
        
        # Create custom observer with calibration
        rgb_observer = AriaRgbObserver(rgb_calib=rgb_calib)
        
        # Get streaming client
        streaming_client = streaming_manager.streaming_client
        
        # Register observer
        streaming_client.set_streaming_client_observer(rgb_observer)
        
        # TODO: Configure filter for images only (optional)
        # subscription_config = streaming_client.subscription_config
        # subscription_config.subscriber_data_type = aria.StreamingDataType.Image
        
        # Start subscription
        streaming_client.subscribe()
        print("[INFO] ✓ RGB stream subscription active")
        
        # 4. Visualization setup
        window_name = "Aria RGB Stream - TFM Navigation"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        print("[INFO] Stream active - Press 'q' to quit or Ctrl+C")
        print("[INFO] Waiting for RGB frames...")
        
        # 5. Main visualization loop
        frames_displayed = 0
        
        while not ctrl_handler.should_stop:
            # Get latest frame
            current_frame = rgb_observer.get_latest_frame()
            
            # If a frame is available, show it
            if current_frame is not None:
                cv2.imshow(window_name, current_frame)
                frames_displayed += 1
                
                # Stats every 200 displayed frames
                if frames_displayed % 200 == 0:
                    print(f"[INFO] Frames displayed: {frames_displayed}")
            
            # Check if 'q' was pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] 'q' detected, closing application...")
                break
            if key == ord('t'):
                print("[INFO] Test key 't' pressed → speaking 'audio test' (force)")
                try:
                    rgb_observer.audio_system.speak_force("audio test")
                except Exception as e:
                    print(f"[WARN] Test speak failed: {e}")
        
        # Final stats
        print(f"[INFO] Final stats:")
        print(f"  - RGB frames received: {rgb_observer.frame_count}")
        print(f"  - Frames displayed: {frames_displayed}")
        
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt detected")
        
    except Exception as e:
        print(f"[ERROR] Error during execution: {e}")
        print("[ERROR] Check device connection and dependencies")
        
    finally:
        # 6. Ordered cleanup of resources
        print("[INFO] Starting resource cleanup...")

        try:
            # Stop processing thread from observer
            if 'rgb_observer' in locals() and rgb_observer:
                rgb_observer.stop()
        except Exception:
            pass

        try:
            # Unsubscribe from stream
            if streaming_client:
                streaming_client.unsubscribe()
                print("[INFO] ✓ Unsubscribed successfully")
        except Exception as e:
            print(f"[WARN] Error during unsubscribe: {e}")
        
        try:
            # Stop streaming
            if streaming_manager:
                streaming_manager.stop_streaming()
                print("[INFO] ✓ Streaming stopped")
        except Exception as e:
            print(f"[WARN] Error stopping streaming: {e}")
        
        try:
            # Disconnect device
            if device_client and 'device' in locals():
                device_client.disconnect(device)
                print("[INFO] ✓ Device disconnected")
        except Exception as e:
            print(f"[WARN] Error on disconnect: {e}")

        try:
            # Close audio system resources
            if 'rgb_observer' in locals() and rgb_observer and hasattr(rgb_observer, 'audio_system'):
                rgb_observer.audio_system.close()
        except Exception:
            pass
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        print("[INFO] ✓ Windows closed")
        
        print("[INFO] Program finished successfully")


if __name__ == "__main__":
    # Ejecutar programa principal
    main()
