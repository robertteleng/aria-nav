#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📡 Mac Sender 
Usa el Observer existente para capturar datos de Aria y los envía al Jetson via ImageZMQ

Autor: Roberto Rojas Sahuquillo
Fecha: TFM - Bloque 13 Mac Sender
"""

import cv2
import numpy as np
import time
import threading
import imagezmq
from utils.ctrl_handler import CtrlCHandler
from core.hardware.device_manager import DeviceManager
from core.observer import Observer


class MacSender:
    """
    Sender que usa Observer para capturar Aria y envía al Jetson
    """
    
    def __init__(self, jetson_ip="192.168.8.204", jetson_port=5555):
        self.jetson_ip = jetson_ip
        self.jetson_port = jetson_port
        
        # ImageZMQ sender
        self.sender = None
        self.is_connected = False
        
        # Observer components
        self.device_manager = None
        self.observer = None
        
        # Sending stats
        self.frames_sent = 0
        self.last_stats_time = time.time()
        
        # Control
        self._stop = False
        
        print(f"📡 Mac Sender initialized")
        print(f"🎯 Target: {jetson_ip}:{jetson_port}")
    
    def connect_jetson(self):
        """Conectar ImageZMQ al Jetson"""
        try:
            sender_address = f"tcp://{self.jetson_ip}:{self.jetson_port}"
            self.sender = imagezmq.ImageSender(connect_to=sender_address)
            self.is_connected = True
            
            print(f"✅ ImageZMQ connected: {sender_address}")
            return True
            
        except Exception as e:
            print(f"❌ Jetson connection failed: {e}")
            print(f"💡 Make sure Jetson receiver is running on {self.jetson_ip}:{self.jetson_port}")
            self.is_connected = False
            return False
    
    def setup_aria_observer(self):
        """Setup Observer para capturar datos de Aria"""
        try:
            print("🔗 Setting up Aria Observer...")
            
            # Device manager
            self.device_manager = DeviceManager()
            self.device_manager.connect()
            rgb_calib = self.device_manager.start_streaming()
            
            # Observer - SIN dashboard para este uso
            self.observer = Observer(rgb_calib=rgb_calib, enable_dashboard=False)
            self.device_manager.register_observer(self.observer)
            self.device_manager.subscribe()
            
            print("✅ Aria Observer ready")
            return True
            
        except Exception as e:
            print(f"❌ Aria Observer setup failed: {e}")
            return False
    
    def start_sending(self):
        """Enviar frames del Observer al Jetson"""
        if not self.is_connected:
            print("❌ Not connected to Jetson")
            return False
        
        print("📡 Starting frame transmission...")
        print("💡 Frames from Aria Observer → Jetson")
        print("⌨️  Press Ctrl+C to stop")
        print("-" * 50)
        
        try:
            while not self._stop:
                # Obtener frame más reciente del Observer
                current_frame = self.observer.get_latest_frame()
                
                if current_frame is not None:
                    try:
                        # Enviar frame al Jetson
                        reply = self.sender.send_image("mac_aria", current_frame)
                        self.frames_sent += 1
                        
                        # Stats periódicas
                        self._print_stats_if_needed()
                        
                        # Control de velocidad (~30 FPS)
                        time.sleep(0.033)
                        
                    except Exception as e:
                        print(f"⚠️ Send error: {e}")
                        time.sleep(0.1)
                else:
                    # No hay frame disponible aún
                    time.sleep(0.01)
                    
        except KeyboardInterrupt:
            print(f"\n🛑 Sending stopped by user")
        except Exception as e:
            print(f"❌ Send loop error: {e}")
        finally:
            self._stop = True
            print(f"📊 Total frames sent: {self.frames_sent}")
    
    def _print_stats_if_needed(self):
        """Imprimir estadísticas cada 5 segundos"""
        current_time = time.time()
        if current_time - self.last_stats_time >= 5.0:
            elapsed = current_time - self.last_stats_time
            fps = self.frames_sent / elapsed if elapsed > 0 else 0
            
            print(f"📊 Sent {self.frames_sent} frames | FPS: {fps:.1f}")
            
            # Reset counters
            self.frames_sent = 0
            self.last_stats_time = current_time
    
    def cleanup(self):
        """Limpiar todos los recursos"""
        print("🧹 Cleaning up Mac Sender...")
        self._stop = True
        
        # Cleanup Observer
        if self.observer:
            try:
                self.observer.stop()
                print("✅ Observer stopped")
            except Exception as e:
                print(f"⚠️ Observer cleanup error: {e}")
        
        # Cleanup Device Manager
        if self.device_manager:
            try:
                self.device_manager.cleanup()
                print("✅ Device Manager cleaned")
            except Exception as e:
                print(f"⚠️ Device Manager cleanup error: {e}")
        
        # Cleanup ImageZMQ
        if self.sender:
            try:
                self.sender.close()
                print("✅ ImageZMQ sender closed")
            except Exception as e:
                print(f"⚠️ ImageZMQ cleanup error: {e}")
        
        print("✅ Mac Sender cleanup complete")


def main():
    """Función principal del Mac Sender"""
    print("=" * 60)
    print("📡 MAC SENDER")
    print("🎯 TFM: Aria Observer → ImageZMQ → Jetson")
    print("=" * 60)
    
    # Configuration
    print("🔧 Configuration:")
    jetson_ip = input("Jetson IP [192.168.8.204]: ").strip() or "192.168.8.204"
    jetson_port = 5555
    
    print(f"🎯 Target: {jetson_ip}:{jetson_port}")
    confirm = input("¿Correcto? (y/n): ").lower()
    if confirm != 'y':
        print("❌ Aborted by user")
        return
    
    # Setup signal handler
    ctrl_handler = CtrlCHandler()
    sender = None
    
    try:
        # Initialize sender
        sender = MacSender(jetson_ip=jetson_ip, jetson_port=jetson_port)
        
        print("\n📡 Step 1: Connecting to Jetson...")
        if not sender.connect_jetson():
            print("❌ Cannot connect to Jetson")
            print("💡 Make sure Jetson receiver is running!")
            return
        
        print("\n🔗 Step 2: Setting up Aria Observer...")
        if not sender.setup_aria_observer():
            print("❌ Cannot setup Aria Observer")
            return
        
        print("\n📡 Step 3: Starting transmission...")
        
        # Give Observer time to start receiving frames
        print("⏳ Waiting for Observer to capture first frames...")
        time.sleep(3)
        
        # Start sending loop
        sender.start_sending()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if sender:
            sender.cleanup()
        print("👋 Mac Sender finished")


if __name__ == "__main__":
    main()