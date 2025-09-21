#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📡 Mac Aria ImageZMQ Sender
Captura frames del Aria SDK y los envía al Jetson via ImageZMQ

Autor: Roberto Rojas Sahuquillo
Fecha: TFM - Bloque 13 Mac-Jetson Bridge
Basado en: Observer existente + ImageZMQ bridge
"""

import cv2
import numpy as np
import time
import threading
import imagezmq
import aria.sdk as aria
from projectaria_tools.core.sensor_data import ImageDataRecord
from projectaria_tools.core.calibration import device_calibration_from_json_string
from typing import Optional
import socket
import sys


class MacSender:
    """
    Sender que captura frames del Aria SDK y los envía al Jetson
    """
    
    def __init__(self, jetson_ip: str = "192.168.8.204", jetson_port: int = 5555):
        self.jetson_ip = jetson_ip
        self.jetson_port = jetson_port
        
        # ImageZMQ sender
        self.sender = None
        self.is_connected = False
        
        # Aria SDK components
        self.device_client = None
        self.device = None
        self.streaming_manager = None
        self.streaming_client = None
        
        # Frame processing
        self.current_frame = None
        self.frames_sent = 0
        self.last_frame_time = 0
        
        # Threading
        self._lock = threading.Lock()
        self._stop = False
        
        print(f"🚀 Mac Aria Sender initialized")
        print(f"🎯 Target Jetson: {jetson_ip}:{jetson_port}")
    
    def connect_imagezmq(self):
        """Conectar al Jetson via ImageZMQ"""
        try:
            # Test connection first
            self._test_jetson_connection()
            
            # Create sender
            send_address = f"tcp://{self.jetson_ip}:{self.jetson_port}"
            self.sender = imagezmq.ImageSender(connect_to=send_address)
            self.is_connected = True
            
            print(f"✅ ImageZMQ connected to {send_address}")
            return True
            
        except Exception as e:
            print(f"❌ ImageZMQ connection failed: {e}")
            print(f"💡 Make sure Jetson is running on {self.jetson_ip}:{self.jetson_port}")
            self.is_connected = False
            return False
    
    def _test_jetson_connection(self):
        """Test basic network connectivity to Jetson"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((self.jetson_ip, self.jetson_port))
            sock.close()
            
            if result != 0:
                raise ConnectionError(f"Cannot reach {self.jetson_ip}:{self.jetson_port}")
            
            print(f"✅ Network connectivity to Jetson confirmed")
            
        except Exception as e:
            print(f"❌ Network test failed: {e}")
            raise
    
    def connect_aria(self):
        """Conectar con las gafas Aria"""
        try:
            print("🔗 Connecting to Aria glasses...")
            
            # Device connection
            self.device_client = aria.DeviceClient()
            self.device = self.device_client.connect()
            
            print("✅ Aria device connected")
            
            # Streaming setup
            self.streaming_manager = self.device.streaming_manager
            
            streaming_config = aria.StreamingConfig()
            streaming_config.profile_name = "profile28"
            streaming_config.streaming_interface = aria.StreamingInterface.Usb
            streaming_config.security_options.use_ephemeral_certs = True
            
            self.streaming_manager.streaming_config = streaming_config
            self.streaming_manager.start_streaming()
            
            # Get calibration (optional for sending)
            try:
                sensors_calib_json = self.streaming_manager.sensors_calibration()
                sensors_calib = device_calibration_from_json_string(sensors_calib_json)
                rgb_calib = sensors_calib.get_camera_calib("camera-rgb")
                print("✅ RGB calibration obtained")
            except Exception as e:
                print(f"⚠️ Could not fetch RGB calibration: {e}")
            
            # Setup streaming client
            self.streaming_client = self.streaming_manager.streaming_client
            self.streaming_client.set_streaming_client_observer(self)
            self.streaming_client.subscribe()
            
            print("✅ Aria streaming active")
            return True
            
        except Exception as e:
            print(f"❌ Aria connection failed: {e}")
            return False
    
    def on_image_received(self, image: np.array, record: ImageDataRecord) -> None:
        """Aria SDK callback - solo procesar RGB"""
        camera_id = record.camera_id
        
        # Solo procesar cámara RGB (centro)
        if camera_id == aria.CameraId.Rgb:
            # Rotar imagen como en el Observer original
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            
            # Thread-safe storage
            with self._lock:
                self.current_frame = rotated
                self.last_frame_time = time.time()
    
    def on_streaming_client_failure(self, reason, message: str) -> None:
        """Callback para errores de streaming"""
        print(f"❌ Streaming failure: {reason}: {message}")
    
    def start_sending(self):
        """Iniciar envío de frames al Jetson"""
        if not self.is_connected:
            print("❌ ImageZMQ not connected")
            return False
        
        print("📡 Starting frame transmission to Jetson...")
        
        # Sender loop
        send_count = 0
        last_send_time = time.time()
        
        try:
            while not self._stop:
                # Get latest frame
                frame_to_send = None
                with self._lock:
                    if self.current_frame is not None:
                        frame_to_send = self.current_frame.copy()
                
                if frame_to_send is not None:
                    try:
                        # Send frame via ImageZMQ
                        reply = self.sender.send_image("mac_aria", frame_to_send)
                        
                        send_count += 1
                        self.frames_sent = send_count
                        
                        # Stats every 100 frames
                        if send_count % 100 == 0:
                            current_time = time.time()
                            elapsed = current_time - last_send_time
                            fps = 100 / elapsed if elapsed > 0 else 0
                            
                            print(f"📊 Sent {send_count} frames | FPS: {fps:.1f} | Reply: {reply}")
                            last_send_time = current_time
                        
                        # Frame rate control (30 FPS max)
                        time.sleep(0.033)
                        
                    except Exception as e:
                        print(f"⚠️ Send error: {e}")
                        time.sleep(0.1)
                else:
                    # No frame available yet
                    time.sleep(0.01)
                    
        except KeyboardInterrupt:
            print("\n🛑 Sending interrupted by user")
        except Exception as e:
            print(f"❌ Sending error: {e}")
        finally:
            self._stop = True
            print(f"📊 Total frames sent: {self.frames_sent}")
    
    def stop(self):
        """Detener sender y limpiar recursos"""
        print("🔄 Stopping Mac Aria Sender...")
        self._stop = True
        
        # Cleanup Aria
        if self.streaming_client:
            try:
                self.streaming_client.unsubscribe()
                print("✅ Aria unsubscribed")
            except Exception as e:
                print(f"⚠️ Unsubscribe error: {e}")
        
        if self.streaming_manager:
            try:
                self.streaming_manager.stop_streaming()
                print("✅ Aria streaming stopped")
            except Exception as e:
                print(f"⚠️ Stop streaming error: {e}")
        
        if self.device_client and self.device:
            try:
                self.device_client.disconnect(self.device)
                print("✅ Aria disconnected")
            except Exception as e:
                print(f"⚠️ Disconnect error: {e}")
        
        # Cleanup ImageZMQ
        if self.sender:
            try:
                self.sender.close()
                print("✅ ImageZMQ sender closed")
            except Exception as e:
                print(f"⚠️ ImageZMQ close error: {e}")
        
        print("✅ Mac Aria Sender stopped")


def main():
    """Función principal del Mac Sender"""
    print("=" * 60)
    print("📡 MAC ARIA IMAGEZMQ SENDER")
    print("🎯 TFM: Sistema Navegación Híbrido Mac ↔ Jetson")
    print("=" * 60)
    
    # Configuration
    JETSON_IP = "192.168.8.204"  # Update with your Jetson IP
    JETSON_PORT = 5555
    
    # User confirmation
    print(f"🎯 Target Jetson: {JETSON_IP}:{JETSON_PORT}")
    confirm = input("Is this correct? (y/n): ").lower()
    if confirm != 'y':
        new_ip = input("Enter Jetson IP: ").strip()
        if new_ip:
            JETSON_IP = new_ip
    
    sender = None
    
    try:
        # Initialize sender
        sender = MacSender(jetson_ip=JETSON_IP, jetson_port=JETSON_PORT)
        
        # Connect to Jetson
        print("\n🔗 Step 1: Connecting to Jetson...")
        if not sender.connect_imagezmq():
            print("❌ Cannot connect to Jetson. Make sure Jetson receiver is running.")
            return
        
        # Connect to Aria
        print("\n🔗 Step 2: Connecting to Aria glasses...")
        if not sender.connect_aria():
            print("❌ Cannot connect to Aria glasses.")
            return
        
        # Start sending
        print("\n📡 Step 3: Starting frame transmission...")
        print("💡 Frames from Aria will be sent to Jetson for processing")
        print("⌨️  Press Ctrl+C to stop")
        print("-" * 60)
        
        sender.start_sending()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if sender:
            sender.stop()
        print("👋 Mac Aria Sender finished")


if __name__ == "__main__":
    main()