#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import cv2
import imagezmq
from core.hardware.device_manager import DeviceManager
from utils.ctrl_handler import CtrlCHandler

class MacSender:
    def __init__(self):
        self.sender = imagezmq.ImageSender(connect_to="tcp://192.168.8.204:5555")
        self.device_manager = DeviceManager()
    
    def on_image_received(self, image, record):
        if record.camera_id.name == "RGB":
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            try:
                reply = self.sender.send_image("mac", rotated)
            except Exception as e:
                print(f"Send error: {e}")
    
    def start(self):
        self.device_manager.connect()
        self.device_manager.start_streaming()
        self.device_manager.register_observer(self)
        self.device_manager.subscribe()
        
        ctrl_handler = CtrlCHandler()
        while not ctrl_handler.should_stop:
            pass
        
        self.device_manager.cleanup()

if __name__ == "__main__":
    sender = MacSender()
    sender.start()
