"""
Navigation system for blind users using Meta Aria glasses
Author: Roberto Rojas Sahuquillo
Date: 2025-09  
Version: 0.50 - Clean main with Rerun dashboard
"""

import cv2
from utils.ctrl_handler import CtrlCHandler
from core.device_manager import DeviceManager
from core.observer import Observer
from utils.rerun_dashboard import RerunDashboard


def main():
    """Clean entry point orchestrating all components"""
    print("=" * 60)
    print("ARIA project: Navigation system for blind users")
    print("=" * 60)
    
    # Setup clean exit handler
    ctrl_handler = CtrlCHandler()

    # Initialize dashboard
    dashboard = None
    enable_dashboard = input("Habilitar dashboard Rerun? (y/n): ").lower() == 'y'
    
    if enable_dashboard:
        dashboard = RerunDashboard()
        print("[MAIN] Dashboard Rerun activado")
    
    # Core components
    device_manager = None
    observer = None
    
    try:
        # 1. Device connection and streaming setup
        device_manager = DeviceManager()
        device_manager.connect()
        rgb_calib = device_manager.start_streaming()
        
        # 2. Observer setup with all modules integrated
        observer = Observer(rgb_calib=rgb_calib)
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
                # Dashboard logging
                if dashboard:
                    dashboard.log_rgb_frame(current_frame)

                    # Log SLAM frames
                    slam_frames = observer.get_all_frames()
                    if 'slam1' in slam_frames or 'slam2' in slam_frames:
                        dashboard.log_slam_frames(
                            slam1_frame=slam_frames.get('slam1'),
                            slam2_frame=slam_frames.get('slam2')
    )
                    dashboard.log_performance_metrics()
                    
                    # Log detections
                    detections = observer.get_latest_detections()
                    if detections:
                        print(f"[DEBUG] {len(detections)} detections found")
                        dashboard.log_detections(detections, current_frame.shape)
                    else:
                        print("[DEBUG] No detections found")
                    
                    # Log depth map
                    depth_map = observer.get_latest_depth_map()
                    if depth_map is not None:
                        dashboard.log_depth_map(depth_map)
                    
                    # Log motion state
                    motion_state = observer.get_motion_state()
                    if motion_state:
                        dashboard.log_motion_state(motion_state['state'], motion_state['magnitude'])
                
                # OpenCV display (only if no dashboard)
                else:
                    cv2.imshow(window_name, current_frame)
                
                frames_displayed += 1
                
                if frames_displayed % 200 == 0:
                    print(f"[INFO] Frames displayed: {frames_displayed}")
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] 'q' detected, closing application...")
                break
            elif key == ord('t'):
                print("[INFO] Testing audio system...")
                observer.test_audio()
                if dashboard:
                    dashboard.log_audio_command("Test audio command", priority=5)
        
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
        
        if dashboard:
            dashboard.shutdown()
        
        cv2.destroyAllWindows()
        print("[INFO] Program finished successfully")


if __name__ == "__main__":
    main()