#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson Server - TFM Navigation System
Recibe frames del Mac y ejecuta pipeline completo de procesamiento

Arquitectura:
Mac → Socket → JetsonServer → Coordinator → Dashboard → Socket → Mac
"""

import sys
import os
import socket
import threading
import time
import cv2
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import our modules
from communication.protocols import (
    FrameMessage, ProcessedMessage, MessageUtils, CommunicationConfig,
    CommunicationError, NetworkError, MessageValidationError
)
from core.navigation.builder import build_navigation_system
from utils.ctrl_handler import CtrlCHandler
from utils.config import Config


class JetsonServer:
    """
    Servidor Jetson que procesa frames del Mac usando pipeline existente
    
    Responsibilities:
    - Recibir frames del Mac via socket
    - Procesar usando Coordinator + Builder existentes
    - Generar dashboard con overlays y detecciones
    - Enviar dashboard procesado de vuelta al Mac
    - Ejecutar comandos de audio localmente
    """
    
    def __init__(self):
        self.config = CommunicationConfig
        
        # Networking
        self.frame_server_socket = None
        self.dashboard_server_socket = None
        self.frame_client_socket = None
        self.dashboard_client_socket = None
        self.running = False
        
        # Processing pipeline (reutilizar arquitectura existente)
        self.coordinator = None
        
        # Frame handling
        self.frames_received = 0
        self.frames_processed = 0
        self.dashboards_sent = 0
        self.processing_times = []
        
        # Threading
        self.frame_handler_thread = None
        self.stats_thread = None
        
        print("[JETSON SERVER] Initialized - Ready for distributed processing")
    
    def setup_processing_pipeline(self) -> bool:
        """Configurar pipeline de procesamiento usando Builder existente"""
        try:
            print("[JETSON SERVER] Setting up processing pipeline...")
            
            # Usar Builder existente para crear pipeline completo
            # No necesitamos dashboard interno porque generamos uno custom
            self.coordinator = build_navigation_system(enable_dashboard=False)
            
            print("[JETSON SERVER] ✅ Processing pipeline configured")
            print(f"[JETSON SERVER] Components ready:")
            print(f"  - YoloProcessor: {hasattr(self.coordinator, 'yolo_processor')}")
            print(f"  - AudioSystem: {hasattr(self.coordinator, 'audio_system')}")
            print(f"  - FrameRenderer: {hasattr(self.coordinator, 'frame_renderer')}")
            
            return True
            
        except Exception as e:
            print(f"[JETSON SERVER] ❌ Failed to setup processing pipeline: {e}")
            return False
    
    def setup_server_sockets(self) -> bool:
        """Configurar sockets del servidor"""
        try:
            print(f"[JETSON SERVER] Setting up server sockets...")
            
            # Socket para recibir frames del Mac
            self.frame_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.frame_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.frame_server_socket.bind(("0.0.0.0", self.config.FRAME_PORT))
            self.frame_server_socket.listen(1)
            
            # Socket para enviar dashboard al Mac
            self.dashboard_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.dashboard_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.dashboard_server_socket.bind(("0.0.0.0", self.config.DASHBOARD_PORT))
            self.dashboard_server_socket.listen(1)
            
            print(f"[JETSON SERVER] ✅ Server sockets configured:")
            print(f"  - Frame port: {self.config.FRAME_PORT}")
            print(f"  - Dashboard port: {self.config.DASHBOARD_PORT}")
            
            return True
            
        except Exception as e:
            print(f"[JETSON SERVER] ❌ Failed to setup server sockets: {e}")
            return False
    
    def wait_for_mac_connection(self) -> bool:
        """Esperar conexión del Mac"""
        try:
            print("[JETSON SERVER] Waiting for Mac client connection...")
            
            # Accept frame connection
            print(f"[JETSON SERVER] Waiting for frame connection on port {self.config.FRAME_PORT}...")
            self.frame_client_socket, frame_addr = self.frame_server_socket.accept()
            print(f"[JETSON SERVER] Frame connection from {frame_addr}")
            
            # Accept dashboard connection
            print(f"[JETSON SERVER] Waiting for dashboard connection on port {self.config.DASHBOARD_PORT}...")
            self.dashboard_client_socket, dashboard_addr = self.dashboard_server_socket.accept()
            print(f"[JETSON SERVER] Dashboard connection from {dashboard_addr}")
            
            # Configure socket timeouts
            self.frame_client_socket.settimeout(self.config.SOCKET_TIMEOUT)
            self.dashboard_client_socket.settimeout(self.config.SOCKET_TIMEOUT)
            
            print("[JETSON SERVER] ✅ Mac client connected successfully")
            return True
            
        except Exception as e:
            print(f"[JETSON SERVER] ❌ Failed to accept Mac connection: {e}")
            return False
    
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
                print(f"[JETSON SERVER] Socket receive error: {e}")
                return None
        return data
    
    def _frame_handler_thread(self) -> None:
        """Thread principal para recibir y procesar frames del Mac"""
        print("[JETSON SERVER] Frame handler thread started")
        
        while self.running:
            try:
                # Recibir tamaño del mensaje
                size_data = self._receive_exact(self.frame_client_socket, 8)
                if not size_data:
                    print("[JETSON SERVER] No size data received - Mac disconnected?")
                    break
                
                msg_size = int.from_bytes(size_data, byteorder='big')
                if msg_size > self.config.MAX_MESSAGE_SIZE:
                    print(f"[JETSON SERVER] Message too large: {msg_size} bytes")
                    continue
                
                # Recibir datos del mensaje
                msg_data = self._receive_exact(self.frame_client_socket, msg_size)
                if not msg_data:
                    print("[JETSON SERVER] No message data received")
                    break
                
                # Deserializar FrameMessage
                frame_msg = FrameMessage.from_bytes(msg_data)
                
                # Validar mensaje
                if not MessageUtils.validate_frame_message(frame_msg):
                    print("[JETSON SERVER] Received invalid frame message")
                    continue
                
                # Procesar frame usando pipeline existente
                self._process_frame(frame_msg)
                
                self.frames_received += 1
                
            except Exception as e:
                print(f"[JETSON SERVER] Error in frame handler: {e}")
                if self.running:
                    time.sleep(0.1)  # Brief delay before retry
                break
        
        print("[JETSON SERVER] Frame handler thread stopped")
    
    def _process_frame(self, frame_msg: FrameMessage) -> None:
        """
        Procesar frame usando pipeline existente y enviar resultado al Mac
        
        PUNTO CLAVE: Aquí reutilizamos tu Coordinator completo
        """
        processing_start = time.time()
        
        try:
            # Procesar frame usando Coordinator existente
            # Este es el MISMO proceso_frame de tu Observer original
            annotated_frame = self.coordinator.process_frame(frame_msg.frame_data)
            
            # Obtener detecciones del coordinator para dashboard
            detections = self.coordinator.current_detections
            
            # Obtener comando de audio si existe
            audio_command = None
            if hasattr(self.coordinator.audio_system, 'last_phrase'):
                if self.coordinator.audio_system.last_phrase:
                    audio_command = self.coordinator.audio_system.last_phrase
            
            # Crear dashboard mejorado con información adicional
            dashboard_frame = self._create_dashboard_frame(
                annotated_frame, detections, frame_msg
            )
            
            # Crear mensaje de respuesta
            processed_msg = MessageUtils.create_processed_message(
                dashboard_frame,
                detections,
                frame_msg.sequence_id,
                processing_start,
                audio_command
            )
            
            # Enviar al Mac
            self._send_dashboard_to_mac(processed_msg)
            
            # Actualizar estadísticas
            processing_time = time.time() - processing_start
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            
            self.frames_processed += 1
            
        except Exception as e:
            print(f"[JETSON SERVER] Error processing frame: {e}")
    
    def _create_dashboard_frame(self, annotated_frame: np.ndarray, 
                              detections: List[Dict], 
                              frame_msg: FrameMessage) -> np.ndarray:
        """
        Crear dashboard enriquecido con información del servidor
        """
        dashboard = annotated_frame.copy()
        
        # Añadir información del servidor en la parte superior
        info_y = 30
        line_height = 25
        
        # Stats del servidor
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        fps_estimate = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
        
        server_info = [
            f"JETSON SERVER - Frames: {self.frames_processed}",
            f"Processing: {avg_processing_time*1000:.1f}ms - FPS: {fps_estimate:.1f}",
            f"Detections: {len(detections)} objects"
        ]
        
        for i, info in enumerate(server_info):
            cv2.putText(dashboard, info, (10, info_y + i * line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Añadir timestamp del frame original
        if frame_msg.metadata and 'mac_timestamp' in frame_msg.metadata:
            mac_time = frame_msg.metadata['mac_timestamp']
            latency_ms = (time.time() - mac_time) * 1000
            cv2.putText(dashboard, f"Mac->Jetson Latency: {latency_ms:.1f}ms", 
                       (10, dashboard.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return dashboard
    
    def _send_dashboard_to_mac(self, processed_msg: ProcessedMessage) -> None:
        """Enviar dashboard procesado al Mac"""
        try:
            # Serializar mensaje
            data = processed_msg.to_bytes()
            
            # Enviar tamaño del mensaje
            msg_size = len(data).to_bytes(8, byteorder='big')
            self.dashboard_client_socket.sendall(msg_size)
            
            # Enviar datos
            self.dashboard_client_socket.sendall(data)
            
            self.dashboards_sent += 1
            
        except Exception as e:
            print(f"[JETSON SERVER] Error sending dashboard: {e}")
    
    def _stats_thread(self) -> None:
        """Thread para mostrar estadísticas del servidor"""
        while self.running:
            time.sleep(10)  # Stats cada 10 segundos
            
            if self.running and self.processing_times:
                avg_time = np.mean(self.processing_times) * 1000
                fps = 1000.0 / avg_time if avg_time > 0 else 0
                
                print(f"[JETSON SERVER STATS] "
                      f"Frames RX: {self.frames_received}, "
                      f"Processed: {self.frames_processed}, "
                      f"Dashboards TX: {self.dashboards_sent}, "
                      f"Avg processing: {avg_time:.1f}ms, "
                      f"FPS: {fps:.1f}")
    
    def start(self) -> bool:
        """Iniciar servidor Jetson completo"""
        print("[JETSON SERVER] Starting distributed processing server...")
        
        # 1. Setup processing pipeline
        if not self.setup_processing_pipeline():
            return False
        
        # 2. Setup server sockets
        if not self.setup_server_sockets():
            return False
        
        # 3. Wait for Mac connection
        if not self.wait_for_mac_connection():
            return False
        
        # 4. Start processing threads
        self.running = True
        
        self.frame_handler_thread = threading.Thread(
            target=self._frame_handler_thread, daemon=True)
        self.frame_handler_thread.start()
        
        self.stats_thread = threading.Thread(
            target=self._stats_thread, daemon=True)
        self.stats_thread.start()
        
        print("[JETSON SERVER] ✅ Server started successfully")
        print("[JETSON SERVER] Processing frames from Mac...")
        return True
    
    def run(self) -> None:
        """Loop principal del servidor"""
        try:
            print("[JETSON SERVER] Running - Press Ctrl+C to stop")
            
            while self.running:
                time.sleep(1)
                
                # Check if threads are still alive
                if not self.frame_handler_thread.is_alive():
                    print("[JETSON SERVER] Frame handler thread died - stopping")
                    break
        
        except KeyboardInterrupt:
            print("\n[JETSON SERVER] Keyboard interrupt received")
        
        self.stop()
    
    def stop(self) -> None:
        """Detener servidor Jetson"""
        print("[JETSON SERVER] Stopping...")
        self.running = False
        
        # Cleanup sockets
        for sock in [self.frame_client_socket, self.dashboard_client_socket,
                    self.frame_server_socket, self.dashboard_server_socket]:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
        
        # Cleanup coordinator
        if self.coordinator:
            try:
                if hasattr(self.coordinator, 'cleanup'):
                    self.coordinator.cleanup()
            except Exception:
                pass
        
        print("[JETSON SERVER] ✅ Stopped successfully")


def main():
    """Función principal del servidor Jetson"""
    print("=" * 60)
    print("ARIA NAVIGATION - JETSON SERVER (Distributed Mode)")
    print("Receives frames from Mac, processes with full pipeline")
    print("=" * 60)
    
    # Setup clean exit handler
    ctrl_handler = CtrlCHandler()
    server = None
    
    try:
        # Inicializar servidor
        server = JetsonServer()
        
        # Iniciar servidor
        if not server.start():
            print("[MAIN] Failed to start Jetson server")
            return
        
        # Run main loop
        server.run()
        
    except Exception as e:
        print(f"[MAIN] Error: {e}")
    finally:
        if server:
            server.stop()
        print("[MAIN] Jetson server terminated")


if __name__ == "__main__":
    main()