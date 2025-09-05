#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Navigation system for blind users using Meta Aria glasses
TFM - Day 3: Refactored modular architecture

Date: 2025-09-01  
Version: 3.0 - Clean modular architecture
"""

import cv2
from utils.ctrl_handler import CtrlCHandler
from core.device_manager import DeviceManager
from core.observers import Observer


def main():
    """Clean entry point orchestrating all components"""
    print("=" * 60)
    print("TFM - Navigation system for blind users")
    print("Day 3: Modular architecture with IMU ready")
    print("=" * 60)
    
    # Setup clean exit handler
    ctrl_handler = CtrlCHandler()
    
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
        window_name = "Aria Navigation - TFM (Modular)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        print("[INFO] Stream active - Press 'q' to quit or Ctrl+C")
        print("[INFO] Press 't' to test audio system")
        
        frames_displayed = 0
        
        while not ctrl_handler.should_stop:
            current_frame = observer.get_latest_frame()
            
            if current_frame is not None:
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
            elif key == ord('r'):  # Force recalibration
                observer.motion_processor.force_recalibration()
            elif key == ord('y'):  # Show yaw info
                yaw = observer.motion_processor.get_yaw_degrees()
                heading = observer.motion_processor.get_absolute_heading()
                print(f"[YAW-INFO] Gyro yaw: {yaw:.1f}° | Mag heading: {heading:.1f}°")
            elif key == ord('m'):  # Test magnetometer combinations
                if observer.motion_processor.magneto_history:
                    last_mag_data = observer.motion_processor.magneto_history[-1][1]
                    observer.motion_processor.test_all_magnetometer_combinations(last_mag_data)
                    print("INSTRUCCIONES:")
                    print("1. Apunta las gafas al NORTE según tu brújula física")
                    print("2. Presiona 'm' para ver todas las combinaciones")
                    print("3. Busca la combinación que dé ~0° (o ~360°)")
                    print("4. Esa será la combinación correcta para tu hardware")

            elif key == ord('h'):  # Show current heading
                if observer.motion_processor.magneto_history:
                    last_mag_data = observer.motion_processor.magneto_history[-1][1]
                    heading = observer.motion_processor._calculate_compass_heading(last_mag_data)
                    print(f"[HEADING] Magnetómetro da: {heading:.1f}°")
                    print(f"[HEADING] Con tu brújula debería ser: ¿cuántos grados?")
        # Final statistics
        observer.print_stats()
        
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt detected")
        
    except Exception as e:
        print(f"[ERROR] Error during execution: {e}")
        
    finally:
        # Ordered cleanup
        print("[INFO] Starting resource cleanup...")
        
        if observer:
            observer.stop()
        
        if device_manager:
            device_manager.cleanup()
        
        cv2.destroyAllWindows()
        print("[INFO] Program finished successfully")


if __name__ == "__main__":
    main()
