"""
Navigation system for blind users using Meta Aria glasses - MAC SENDER VERSION
Author: Roberto Rojas Sahuquillo
Date: 2025-09  
Version: 0.60 - Mac Sender to Jetson Processor

ARCHITECTURE:
Mac (Aria SDK) → ImageZMQ → Jetson (Processing) → ImageZMQ → Mac (Display)
"""

import cv2
import time
import numpy as np
import imagezmq
from utils.ctrl_handler import CtrlCHandler
from core.hardware.device_manager import DeviceManager
from core.observer import Observer


class MacSender:
    """
    Mac sender que captura desde Aria SDK y envía al Jetson para procesamiento
    """
    
    def __init__(self, jetson_ip="192.168.0.25", jetson_port=5555):
        self.jetson_ip = jetson_ip
        self.jetson_port = jetson_port
        
        # ImageZMQ sender setup
        self.sender = imagezmq.ImageSender(
            connect_to=f"tcp://{jetson_ip}:{jetson_port}",
        )
        
        # Stats tracking
        self.frames_sent = 0
        self.start_time = time.time()
        self.last_fps_time = time.time()
        self.connection_errors = 0
        
        print(f"[SENDER] Configurado para enviar a Jetson: {jetson_ip}:{jetson_port}")

    def send_frame_to_jetson(self, frame: np.ndarray, camera_id: str = "rgb") -> bool:
        try:
            # RESIZE TODOS los frames a tamaño estándar
            frame = cv2.resize(frame, (640, 480))
            
            # Verificar que ahora está correcto
            if frame.shape != (480, 640, 3):
                print(f"[SENDER ERROR] Resize falló: {frame.shape}")
                return False
                
            # Send frame with camera identifier
            frame_id = f"aria_{camera_id}_{self.frames_sent}"
            reply = self.sender.send_image(frame_id, frame)
            
            if reply == b'OK':
                self.frames_sent += 1
                return True
            else:
                print(f"[SENDER WARN] Respuesta inesperada del Jetson: {reply}")
                return False
                
        except Exception as e:
            self.connection_errors += 1
            if self.connection_errors % 10 == 1:
                print(f"[SENDER ERROR] Error enviando frame: {e}")
            return False
    
    def print_sender_stats(self):
        """Print estadísticas del sender"""
        elapsed = time.time() - self.start_time
        fps = self.frames_sent / elapsed if elapsed > 0 else 0
        
        print(f"[SENDER STATS] Frames enviados: {self.frames_sent}")
        print(f"[SENDER STATS] FPS promedio: {fps:.1f}")
        print(f"[SENDER STATS] Errores conexión: {self.connection_errors}")
        print(f"[SENDER STATS] Tiempo activo: {elapsed:.1f}s")
    
    def close(self):
        """Cerrar sender"""
        try:
            self.sender.close()
        except:
            pass


class AriaObserverSender(Observer):
    """
    Observer modificado que envía frames al Jetson en lugar de procesarlos localmente
    """
    
    def __init__(self, rgb_calib=None, jetson_ip="192.168.8.204"):
        # Initialize parent Observer but WITHOUT coordinator (no local processing)
        self.depth_estimator = None  # Disabled for sender mode
        self.dashboard = None        # Disabled for sender mode (Jetson will process)
        
        # Mac sender specific setup
        self.sender = MacSender(jetson_ip=jetson_ip)
        
        # Multi-camera frame storage (same as parent)
        self.current_frames = {
            'rgb': None,
            'slam1': None,
            'slam2': None
        }
        
        # Processing state
        self.frame_counts = {'rgb': 0, 'slam1': 0, 'slam2': 0}
        self.send_counts = {'rgb': 0, 'slam1': 0, 'slam2': 0}
        
        # Latest frame for display (will come from Jetson eventually)
        self.current_frame = None
        
        print("[SENDER OBSERVER] Initialized - Mode: Send to Jetson")
        print("[SENDER OBSERVER] Local processing: DISABLED")
        print("[SENDER OBSERVER] Target: All frames → Jetson")
    
    def on_image_received(self, image: np.array, record) -> None:
        """
        SDK callback modificado para enviar frames al Jetson
        """
        import aria.sdk as aria
        
        camera_id = record.camera_id
        
        # # Process and identify camera
        # if camera_id == aria.CameraId.Rgb:
        #     processed_image = self._process_rgb_image(image)
        #     camera_key = 'rgb'
        # elif camera_id == aria.CameraId.Slam1:
        #     processed_image = self._process_slam_image(image)
        #     camera_key = 'slam1'
        # elif camera_id == aria.CameraId.Slam2:
        #     processed_image = self._process_slam_image(image)
        #     camera_key = 'slam2'
        # else:
        #     return  # Ignore other cameras
        
        # # Update counters
        # self.frame_counts[camera_key] += 1
        
        # # Send to Jetson (priority: RGB > SLAM)
        # if camera_key == 'rgb' or self.frame_counts[camera_key] % 3 == 0:  # Send all RGB, every 3rd SLAM
        #     if self.sender.send_frame_to_jetson(processed_image, camera_key):
        #         self.send_counts[camera_key] += 1
        
        # # Store for local display (until we get processed frames back)
        # self.current_frames[camera_key] = processed_image
        
        # # Use RGB as main display
        # if camera_key == 'rgb':
        #     self.current_frame = processed_image
        
        # # Debug every 100 frames
        # if self.frame_counts[camera_key] % 100 == 0:
        #     print(f"[SENDER] {camera_key.upper()}: received={self.frame_counts[camera_key]}, sent={self.send_counts[camera_key]}")
    

        if camera_id == aria.CameraId.Rgb:
            processed_image = self._process_rgb_image(image)
            camera_key = 'rgb'
            
            # Update counters
            self.frame_counts[camera_key] += 1
            
            # Enviar TODOS los frames RGB
            if self.sender.send_frame_to_jetson(processed_image, camera_key):
                self.send_counts[camera_key] += 1
            
            # Store for local display
            self.current_frames[camera_key] = processed_image
            self.current_frame = processed_image
            
            # Debug every 100 frames
            if self.frame_counts[camera_key] % 100 == 0:
                print(f"[SENDER] {camera_key.upper()}: received={self.frame_counts[camera_key]}, sent={self.send_counts[camera_key]}")
    def _process_rgb_image(self, image: np.array) -> np.array:
        """Procesamiento RGB consistente con Observer base"""
        return super()._process_rgb_image(image)
    
    def _process_slam_image(self, image: np.array) -> np.array:
        """Procesamiento SLAM consistente con Observer base"""
        return super()._process_slam_image(image)
    
    def get_latest_frame(self):
        """Get most recent frame for display"""
        return self.current_frame
    
    def print_stats(self):
        """Print sender statistics"""
        print(f"[SENDER STATS] Frame counts: {self.frame_counts}")
        print(f"[SENDER STATS] Send counts: {self.send_counts}")
        self.sender.print_sender_stats()
    
    def stop(self):
        """Stop sender"""
        print("[SENDER] Cerrando sender...")
        self.sender.close()
    
    def test_audio(self):
        """Audio testing disabled in sender mode"""
        print("[SENDER] Audio testing deshabilitado - se maneja en Jetson")


def main():
    """
    Main modificado: Mac Sender que envía todo al Jetson
    """
    print("=" * 60)
    print("ARIA MAC SENDER: Frames to Jetson for processing")
    print("=" * 60)
    
    # Setup clean exit handler
    ctrl_handler = CtrlCHandler()

    # Jetson configuration
    jetson_ip = input("IP del Jetson [192.168.8.204]: ").strip() or "192.168.8.204"
    
    print(f"[MAIN] Configurando sender para Jetson: {jetson_ip}")
    
    # Core components
    device_manager = None
    observer = None
    
    try:
        # 1. Device connection and streaming setup (same as before)
        device_manager = DeviceManager()
        device_manager.connect()
        rgb_calib = device_manager.start_streaming()
        
        # 2. Observer setup - SENDER VERSION (no local processing)
        observer = AriaObserverSender(rgb_calib=rgb_calib, jetson_ip=jetson_ip)
        device_manager.register_observer(observer)
        device_manager.subscribe()
        
        # 3. Simple display loop (raw frames until Jetson integration complete)
        window_name = "Aria Mac Sender - Raw Frames"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        print("[INFO] Mac Sender active - sending to Jetson")
        print("[INFO] Display shows RAW frames (processed frames from Jetson coming soon)")
        print("[INFO] Press 'q' to quit or Ctrl+C")
        
        frames_displayed = 0
        last_stats_time = time.time()
        
        while not ctrl_handler.should_stop:
            current_frame = observer.get_latest_frame()
            
            if current_frame is not None:
                # Simple display (will be replaced by processed frames from Jetson)
                cv2.imshow(window_name, current_frame)
                frames_displayed += 1
                
                # Add sender info overlay
                if frames_displayed % 30 == 0:  # Every 30 frames
                    info_text = f"Frames sent: RGB={observer.send_counts['rgb']} SLAM1={observer.send_counts['slam1']} SLAM2={observer.send_counts['slam2']}"
                    print(f"[INFO] {info_text}")
            
            # Stats every 10 seconds
            if time.time() - last_stats_time > 10.0:
                observer.print_stats()
                last_stats_time = time.time()
            
            # UI handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] 'q' detected, closing sender...")
                break
            elif key == ord('s'):
                observer.print_stats()
        
        # Final statistics
        print("\n" + "=" * 40)
        print("SENDER SESSION COMPLETE")
        observer.print_stats()
        
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt detected")
        
    except Exception as e:
        print(f"[ERROR] Error during sender execution: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print("[INFO] Starting sender cleanup...")
        
        if observer:
            observer.stop()
        
        if device_manager:
            device_manager.cleanup()
        
        cv2.destroyAllWindows()
        print("[INFO] Mac Sender finished successfully")


if __name__ == "__main__":
    main()
