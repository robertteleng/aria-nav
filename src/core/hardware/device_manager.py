from typing import Optional

import aria.sdk as aria
from projectaria_tools.core.calibration import device_calibration_from_json_string

from utils.config import Config

class DeviceManager:
    """Manages connection and streaming configuration for Aria device"""
    
    def __init__(self):
        self.device_client = None
        self.device = None
        self.streaming_manager = None
        self.streaming_client = None

    def connect(self, ip_address: Optional[str] = None):
        """Establish connection with Aria device"""
        print("[INFO] Starting connection with Aria glasses...")

        self.device_client = aria.DeviceClient()

        desired_interface = getattr(Config, "STREAMING_INTERFACE", "usb").lower()
        use_wifi = desired_interface == "wifi"

        target_ip = ip_address
        if use_wifi and not target_ip:
            target_ip = getattr(Config, "STREAMING_WIFI_DEVICE_IP", None)

        if target_ip:
            client_config = aria.DeviceClientConfig()
            client_config.ip_v4_address = target_ip
            self.device_client.set_client_config(client_config)
            print(f"[INFO] Connecting over Wi-Fi to {target_ip}...")
        else:
            print("[INFO] Connecting over USB (no IP required)...")

        self.device = self.device_client.connect()
        print("[INFO] ✓ Connection established successfully")

    def start_streaming(self, use_wifi: Optional[bool] = None):
        """Configure and start RGB streaming, return calibration"""
        print("[INFO] Configuring RGB streaming...")
        
        self.streaming_manager = self.device.streaming_manager
        
        # Streaming configuration
        streaming_config = aria.StreamingConfig()
        desired_interface = getattr(Config, "STREAMING_INTERFACE", "usb").lower()
        if use_wifi is None:
            use_wifi = desired_interface == "wifi"

        if use_wifi:
            profile = getattr(
                Config,
                "STREAMING_PROFILE_WIFI",
                getattr(Config, "STREAMING_PROFILE", "profile18"),
            )
        else:
            profile = getattr(
                Config,
                "STREAMING_PROFILE_USB",
                getattr(Config, "STREAMING_PROFILE", "profile28"),
            )

        streaming_config.profile_name = profile
        streaming_config.security_options.use_ephemeral_certs = True

        if use_wifi:
            streaming_config.streaming_interface = aria.StreamingInterface.WifiStation
            print(f"[INFO] Using Wi-Fi streaming profile '{profile}'")
        else:
            streaming_config.streaming_interface = aria.StreamingInterface.Usb
            print(f"[INFO] Using USB streaming profile '{profile}'")
        
        self.streaming_manager.streaming_config = streaming_config
        self.streaming_manager.start_streaming()
        
        # Get calibrations from SDK (RGB + SLAM1 + SLAM2)
        rgb_calib = None
        slam1_calib = None
        slam2_calib = None
        
        try:
            sensors_calib_json = self.streaming_manager.sensors_calibration()
            sensors_calib = device_calibration_from_json_string(sensors_calib_json)
            
            rgb_calib = sensors_calib.get_camera_calib("camera-rgb")
            slam1_calib = sensors_calib.get_camera_calib("camera-slam-left")
            slam2_calib = sensors_calib.get_camera_calib("camera-slam-right")
            
            print("[INFO] ✓ Camera calibrations obtained (RGB + SLAM1 + SLAM2)")
        except Exception as e:
            print(f"[WARN] Could not fetch calibrations: {e}")
        
        self.streaming_client = self.streaming_manager.streaming_client
        print("[INFO] ✓ RGB streaming configured")
        
        return rgb_calib, slam1_calib, slam2_calib
    
    def register_observer(self, observer):
        """Register observer with streaming client"""
        self.streaming_client.set_streaming_client_observer(observer)
        
    def subscribe(self):
        """Start subscription to stream"""
        self.streaming_client.subscribe()
        print("[INFO] ✓ Stream subscription active")
    
    def cleanup(self):
        """Clean shutdown of all connections"""
        try:
            if self.streaming_client:
                self.streaming_client.unsubscribe()
                print("[INFO] ✓ Unsubscribed")
        except Exception as e:
            print(f"[WARN] Error during unsubscribe: {e}")
        
        try:
            if self.streaming_manager:
                self.streaming_manager.stop_streaming()
                print("[INFO] ✓ Streaming stopped")
        except Exception as e:
            print(f"[WARN] Error stopping streaming: {e}")
        
        try:
            if self.device_client and self.device:
                self.device_client.disconnect(self.device)
                print("[INFO] ✓ Device disconnected")
        except Exception as e:
            print(f"[WARN] Error on disconnect: {e}")
