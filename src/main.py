"""
Navigation system for blind users using Meta Aria glasses
Author: Roberto Rojas Sahuquillo
Date: 2025-09  
Version: 0.50 - Clean main with Rerun dashboard
"""

import cv2
from utils.ctrl_handler import CtrlCHandler
from core.hardware.device_manager import DeviceManager
from core.observer import Observer


def main():
    """Clean entry point orchestrating all components"""
    print("=" * 60)
    print("ARIA project: Navigation system for blind users")
    print("=" * 60)
    
    # Setup clean exit handler
    ctrl_handler = CtrlCHandler()

    # Initialize dashboard flag (Observer will own the dashboard instance)
    enable_dashboard = input("Habilitar dashboard Rerun? (y/n): ").lower() == 'y'
    if enable_dashboard:
        print("[MAIN] Dashboard OpenCV activado (gestionado por Observer)")
    
    # Core components
    device_manager = None
    observer = None
    
    try:
        # 1. Device connection and streaming setup
        device_manager = DeviceManager()
        device_manager.connect()
        rgb_calib = device_manager.start_streaming()
        
        # 2. Observer setup with all modules integrated
        # Pass user's choice so only one dashboard gets created (inside Observer)
        observer = Observer(rgb_calib=rgb_calib, enable_dashboard=enable_dashboard)
        device_manager.register_observer(observer)
        device_manager.subscribe()
        
        # 3. Main visualization loop
        if not enable_dashboard:
            window_name = "Aria Navigation - TFM"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
        
        print("[INFO] Stream active - Press 'q' to quit or Ctrl+C")
        print("[INFO] Press 't' to test audio system")
        
        frames_displayed = 0
        
        while not ctrl_handler.should_stop:
            current_frame = observer.get_latest_frame()
            
            if current_frame is not None:
                # OpenCV display (only if no dashboard)
                if not enable_dashboard:
                    cv2.imshow(window_name, current_frame)
                
                frames_displayed += 1
                
                # if frames_displayed % 200 == 0:
                #     print(f"[INFO] Frames displayed: {frames_displayed}")
            
            # Refrescar dashboard en hilo principal (macOS requiere main-thread para GUI)
            if enable_dashboard and getattr(observer, 'dashboard', None):
                key = observer.dashboard.update_all()
                if key == ord('q'):
                    print("[INFO] 'q' detected in dashboard, closing application...")
                    break
                # No llamar a waitKey dos veces cuando dashboard activo
                key = 255
            else:
                key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] 'q' detected, closing application...")
                break
            elif key == ord('t'):
                print("[INFO] Testing audio system...")
                observer.test_audio()
        
        # Final statistics
        observer.print_stats()
        
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt detected")
        
    except Exception as e:
        print(f"[ERROR] Error during execution: {e}")
        
    finally:
        # Cleanup
        print("[INFO] Starting resource cleanup...")
        
        if observer:
            observer.stop()
        
        if device_manager:
            device_manager.cleanup()
        
        # Dashboard is owned and closed by Observer
        
        cv2.destroyAllWindows()
        print("[INFO] Program finished successfully")


if __name__ == "__main__":
    main()
