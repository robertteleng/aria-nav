#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mac Client - TFM Navigation System
Captura frames desde Aria SDK y los envía al Jetson para procesamiento distribuido

Arquitectura:
Mac: Aria SDK → MacClient → Socket → Jetson
Mac: Socket ← Dashboard ← Jetson
"""

import sys
import os
import socket
import threading
import time
import cv2
import numpy as np
from typing import Optional
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our modules
from communication.protocols import (
    FrameMessage, ProcessedMessage, MessageUtils, CommunicationConfig,
    CommunicationError, NetworkError, MessageValidationError
)
from core.hardware.device_manager import DeviceManager
from utils.ctrl_handler import CtrlCHandler


class MacClient:
    """
    Cliente Mac que envía frames al Jetson y recibe dashboard procesado
    
    Responsibilities:
    - Conectar a Aria SDK usando DeviceManager existente
    - Enviar frames al Jetson via socket
    - Recibir dashboard procesado de vuelta
    - Mostrar dashboard en ventana local
    """
    
    def __init__(self):
        self.config = CommunicationConfig
        
        # Networking
        self.frame_socket = None
        self.dashboard_socket = None
        self.connected = False
        
        # Aria SDK components (reutilizar código existente)
        self.device_manager = None
        self.rgb_calib = None
        
        # Frame handling
        self.frame_sequence = 0
        self.last_frame_sent = 0
        self.frames_sent = 0
        self.dashboards_received = 0
        
        # Threading
        self.running = False
        self.dashboard_thread = None
        self.stats_thread = None
        
        # Display
        self.current_dashboard = None
        self.display_window = "Aria Navigation - Distributed"
        
        print("[MAC CLIENT] Initialized - Ready for distributed processing")
    
    def connect_to_jetson(self) -> bool:
        """Establecer conexión con Jetson server"""
        try:
            print(f"[MAC CLIENT] Connecting to Jetson at {self.config.JETSON_IP}...")
            
            # Frame sending socket
            self.frame_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.frame_socket.settimeout(self.config.SOCKET_TIMEOUT)
            self.frame_socket.connect((self.config.JETSON_IP, self.config.FRAME_PORT))
            
            # Dashboard receiving socket
            self.dashboard_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.dashboard_socket.settimeout(self.config.SOCKET_TIMEOUT)
            self.dashboard_socket.connect((self.config.JETSON_IP, self.config.DASHBOARD_PORT))
            
            self.connected = True
            print("[MAC CLIENT] ✅ Connected to Jetson successfully")
            return True
            
        except Exception as e:
            print(f"[MAC CLIENT] ❌ Failed to connect to Jetson: {e}")
            self.cleanup_sockets()
            return False
    
    def setup_aria(self) -> bool:
        """Configurar conexión Aria usando DeviceManager existente"""
        try:
            print("[MAC CLIENT] Setting up Aria connection...")
            
            # Reutilizar DeviceManager existente
            self.device_manager = DeviceManager()
            self.device_manager.connect()
            self.rgb_calib = self.device_manager.start_streaming()
            
            # Registrar este cliente como observer
            self.device_manager.register_observer(self)
            
            print("[MAC CLIENT] ✅ Aria SDK configured successfully")
            return True
            
        except Exception as e:
            print(f"[MAC CLIENT] ❌ Failed to setup Aria: {e}")
            return False
    
    def on_image_received(self, image: np.array, record) -> None:
        """
        SDK callback - PUNTO DE SEPARACIÓN CRÍTICO
        
        ANTES: Procesaba localmente con coordinator
        AHORA: Envía frame al Jetson para procesamiento remoto
        """
        if not self.connected or not self.running:
            return
        
        try:
            # Procesar solo RGB camera (como en Observer original)
            if record.camera_id.name != "RGB":
                return
            
            # Rotar imagen (como en Observer original)
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            
            # Crear mensaje para Jetson
            frame_msg = MessageUtils.create_frame_message(
                rotated_image,
                camera_id="rgb",
                metadata={
                    "mac_timestamp": time.time(),
                    "aria_timestamp": record.capture_timestamp_ns
                }
            )
            
            # Enviar al Jetson
            self._send_frame_to_jetson(frame_msg)
            
        except Exception as e:
            print(f"[MAC CLIENT] Error processing frame: {e}")
    
    def _send_frame_to_jetson(self, frame_msg: FrameMessage) -> None:
        """Enviar frame al Jetson via socket"""
        try:
            # Serializar mensaje
            data = frame_msg.to_bytes()
            
            # Enviar tamaño del mensaje primero
            msg_size = len(data).to_bytes(8, byteorder='big')
            self.frame_socket.sendall(msg_size)
            
            # Enviar datos del mensaje
            self.frame_socket.sendall(data)
            
            # Actualizar estadísticas
            self.frames_sent += 1
            self.last_frame_sent = time.time()
            
            # Control de FPS
            if self.frames_sent % 30 == 0:
                print(f"[MAC CLIENT] Frames sent: {self.frames_sent}")
            
        except Exception as e:
            print(f"[MAC CLIENT] Error sending frame: {e}")
            if "Broken pipe" in str(e) or "Connection reset" in str(e):
                self.connected = False
    
    def _dashboard_receiver_thread(self) -> None:
        """Thread para recibir dashboard del Jetson"""
        print("[MAC CLIENT] Dashboard receiver thread started")
        
        while self.running and self.connected:
            try:
                # Recibir tamaño del mensaje
                size_data = self._receive_exact(self.dashboard_socket, 8)
                if not size_data:
                    break
                
                msg_size = int.from_bytes(size_data, byteorder='big')
                if msg_size > self.config.MAX_MESSAGE_SIZE:
                    print(f"[MAC CLIENT] Message too large: {msg_size} bytes")
                    break
                
                # Recibir datos del mensaje
                msg_data = self._receive_exact(self.dashboard_socket, msg_size)
                if not msg_data:
                    break
                
                # Deserializar mensaje
                processed_msg = ProcessedMessage.from_bytes(msg_data)
                
                # Validar mensaje
                if not MessageUtils.validate_processed_message(processed_msg):
                    print("[MAC CLIENT] Received invalid processed message")
                    continue
                
                # Actualizar dashboard actual
                self.current_dashboard = processed_msg.dashboard_frame
                self.dashboards_received += 1
                
                # Procesar comando de audio si existe
                if processed_msg.audio_command:
                    print(f"[MAC CLIENT] Audio command from Jetson: {processed_msg.audio_command}")
                
                # Debug de detecciones
                if processed_msg.detections:
                    det_names = [d.get('name', 'unknown') for d in processed_msg.detections]
                    print(f"[MAC CLIENT] Detections from Jetson: {det_names}")
                
            except Exception as e:
                print(f"[MAC CLIENT] Error in dashboard receiver: {e}")
                if self.running:
                    time.sleep(1)  # Retry delay
                break
    
    def _receive_exact(self, sock: socket.socket, size: int) -> Optional[bytes]:
        """Recibir exactamente 'size' bytes del socket"""
        data = b''
        while len(data) < size:
            try:
                chunk = sock.recv(size - len(data))
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                if not self.running:
                    return None
                continue
            except Exception as e:
                print(f"[MAC CLIENT] Socket receive error: {e}")
                return None
        return data
    
    def _stats_thread(self) -> None:
        """Thread para mostrar estadísticas periódicas"""
        while self.running:
            time.sleep(10)  # Stats cada 10 segundos
            
            if self.running:
                print(f"[MAC CLIENT STATS] "
                      f"Frames sent: {self.frames_sent}, "
                      f"Dashboards received: {self.dashboards_received}, "
                      f"Connected: {self.connected}")
    
    def start(self) -> bool:
        """Iniciar cliente Mac completo"""
        print("[MAC CLIENT] Starting distributed navigation system...")
        
        # 1. Conectar al Jetson
        if not self.connect_to_jetson():
            return False
        
        # 2. Configurar Aria SDK
        if not self.setup_aria():
            return False
        
        # 3. Iniciar threads
        self.running = True
        
        self.dashboard_thread = threading.Thread(
            target=self._dashboard_receiver_thread, daemon=True)
        self.dashboard_thread.start()
        
        self.stats_thread = threading.Thread(
            target=self._stats_thread, daemon=True)
        self.stats_thread.start()
        
        # 4. Suscribirse al stream Aria
        self.device_manager.subscribe()
        
        print("[MAC CLIENT] ✅ Distributed system started successfully")
        return True
    
    def display_loop(self) -> None:
        """Loop principal para mostrar dashboard"""
        cv2.namedWindow(self.display_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.display_window, 800, 600)
        
        print("[MAC CLIENT] Starting display loop - Press 'q' to quit")
        
        while self.running:
            if self.current_dashboard is not None:
                cv2.imshow(self.display_window, self.current_dashboard)
            else:
                # Mostrar placeholder mientras no hay dashboard
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Waiting for Jetson dashboard...", 
                           (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow(self.display_window, placeholder)
            
            # Check for quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[MAC CLIENT] 'q' pressed - stopping system")
                break
        
        self.stop()
    
    def stop(self) -> None:
        """Detener cliente Mac"""
        print("[MAC CLIENT] Stopping...")
        self.running = False
        
        # Cleanup Aria
        if self.device_manager:
            try:
                self.device_manager.cleanup()
            except Exception:
                pass
        
        # Cleanup sockets
        self.cleanup_sockets()
        
        # Cleanup display
        cv2.destroyAllWindows()
        
        print("[MAC CLIENT] ✅ Stopped successfully")
    
    def cleanup_sockets(self) -> None:
        """Limpiar conexiones de red"""
        self.connected = False
        
        for sock in [self.frame_socket, self.dashboard_socket]:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
        
        self.frame_socket = None
        self.dashboard_socket = None


def main():
    """Función principal del cliente Mac"""
    print("=" * 60)
    print("ARIA NAVIGATION - MAC CLIENT (Distributed Mode)")
    print("Sends frames to Jetson, displays processed dashboard")
    print("=" * 60)
    
    # Setup clean exit handler
    ctrl_handler = CtrlCHandler()
    client = None
    
    try:
        # Inicializar cliente
        client = MacClient()
        
        # Iniciar sistema distribuido
        if not client.start():
            print("[MAIN] Failed to start distributed system")
            return
        
        # Loop principal de display
        client.display_loop()
        
    except KeyboardInterrupt:
        print("\n[MAIN] Keyboard interrupt detected")
    except Exception as e:
        print(f"[MAIN] Error: {e}")
    finally:
        if client:
            client.stop()
        print("[MAIN] Mac client terminated")


if __name__ == "__main__":
    main()