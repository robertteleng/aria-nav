import aria.sdk as aria
from projectaria_tools.core.calibration import device_calibration_from_json_string

class DeviceManager:
    """Manages connection and streaming configuration for Aria device"""
    
    def __init__(self):
        self.device_client = None
        self.device = None
        self.streaming_manager = None
        self.streaming_client = None

    def connect(self, ip_address: str = "192.168.0.204"):
        """Establish connection with Aria device"""
        print("[INFO] Starting connection with Aria glasses...")
        
        self.device_client = aria.DeviceClient()

        if ip_address:
            client_config = aria.DeviceClientConfig()
            client_config.ip_v4_address = ip_address
            self.device_client.set_client_config(client_config)
            print(f"[INFO] Connecting over Wi-Fi to {ip_address}...")
        else:
            print("[INFO] Connecting over USB...")

        self.device = self.device_client.connect()
        print("[INFO] ✓ Connection established successfully")
    
    def start_streaming(self, use_wifi: bool = True):
        """Configure and start RGB streaming, return calibration"""
        print("[INFO] Configuring RGB streaming...")
        
        self.streaming_manager = self.device.streaming_manager
        
        # Streaming configuration
        streaming_config = aria.StreamingConfig()
        streaming_config.profile_name = "profile28"
        streaming_config.security_options.use_ephemeral_certs = True

        if use_wifi:
            streaming_config.streaming_interface = aria.StreamingInterface.WifiStation
        else:
            streaming_config.streaming_interface = aria.StreamingInterface.Usb
        
        self.streaming_manager.streaming_config = streaming_config
        self.streaming_manager.start_streaming()
        
        # Get calibration
        rgb_calib = None
        try:
            sensors_calib_json = self.streaming_manager.sensors_calibration()
            sensors_calib = device_calibration_from_json_string(sensors_calib_json)
            rgb_calib = sensors_calib.get_camera_calib("camera-rgb")
            print("[INFO] ✓ RGB calibration obtained")
        except Exception as e:
            print(f"[WARN] Could not fetch RGB calibration: {e}")
        
        self.streaming_client = self.streaming_manager.streaming_client
        print("[INFO] ✓ RGB streaming configured")
        
        return rgb_calib
    
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

