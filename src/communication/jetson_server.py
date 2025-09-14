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

# Add current directory to path for imports
sys.path.insert(0, '/workspace')

# Import protocols from same directory
from protocols import (
    FrameMessage, ProcessedMessage, MessageUtils, CommunicationConfig,
    CommunicationError, NetworkError, MessageValidationError
)

# Mock imports for missing modules - we'll implement simplified versions
class Config:
    """Simplified config for Jetson"""
    YOLO_MODEL = "yolo11n.pt"
    YOLO_CONFIDENCE = 0.5
    TARGET_FPS = 30
    TTS_RATE = 150
    AUDIO_COOLDOWN = 2.0

class CtrlCHandler:
    """Simple Ctrl+C handler"""
    def __init__(self):
        self.should_stop = False
        import signal
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        print("\n[INFO] Interrupt signal detected, closing cleanly...")
        self.should_stop = True

# Simplified coordinator that we'll implement here
class SimpleCoordinator:
    """Simplified coordinator for Jetson"""
    def __init__(self):
        self.current_detections = []
        self.audio_system = SimpleAudioSystem()
        
        # Initialize YOLO
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(Config.YOLO_MODEL)
            print("[COORDINATOR] YOLO model loaded")
        except Exception as e:
            print(f"[COORDINATOR] Error loading YOLO: {e}")
            self.yolo_model = None
    
    def process_frame(self, frame):
        """Process frame with YOLO"""
        if self.yolo_model is None:
            return frame
        
        try:
            # YOLO inference
            results = self.yolo_model(
                frame,
                conf=Config.YOLO_CONFIDENCE,
                verbose=False,
                show=False,
                save=False
            )
            
            # Process detections
            detections = []
            annotated_frame = frame.copy()
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for i in range(len(boxes.xyxy)):
                    bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                    confidence = float(boxes.conf[i].cpu().numpy())
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = self.yolo_model.names[class_id]
                    
                    # Simple detection dict
                    detection = {
                        'bbox': tuple(bbox),
                        'name': class_name,
                        'confidence': confidence,
                        'zone': self._get_zone(bbox, frame.shape[1]),
                        'priority': confidence * 10  # Simple priority
                    }
                    detections.append(detection)
                    
                    # Draw annotation
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            self.current_detections = detections
            
            # Process audio
            self.audio_system.process_detections(detections)
            
            return annotated_frame
            
        except Exception as e:
            print(f"[COORDINATOR] Error processing frame: {e}")
            return frame
    
    def _get_zone(self, bbox, frame_width):
        """Simple zone classification"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        
        if center_x < frame_width // 3:
            return 'left'
        elif center_x < 2 * frame_width // 3:
            return 'center'
        else:
            return 'right'
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self.audio_system, 'cleanup'):
            self.audio_system.cleanup()

class SimpleAudioSystem:
    """Simplified audio system for Jetson"""
    def __init__(self):
        self.last_announcement = 0
        self.last_phrase = None
        
        # Spanish translations
        self.spanish_names = {
            'person': 'persona', 'car': 'coche', 'truck': 'camión',
            'bus': 'autobús', 'bicycle': 'bicicleta'
        }
        
        self.zone_names = {
            'left': 'izquierda', 'center': 'centro', 'right': 'derecha'
        }
    
    def process_detections(self, detections):
        """Process detections and generate audio commands"""
        current_time = time.time()
        
        if current_time - self.last_announcement < Config.AUDIO_COOLDOWN:
            return
        
        if not detections:
            return
        
        # Get highest priority detection
        top_detection = max(detections, key=lambda x: x['priority'])
        
        if top_detection['priority'] >= 5.0:  # Threshold
            message = self._generate_message(top_detection)
            self._speak_async(message)
            self.last_announcement = current_time
            self.last_phrase = message
    
    def _generate_message(self, detection):
        """Generate Spanish audio message"""
        obj_name = self.spanish_names.get(detection['name'], detection['name'])
        zone_name = self.zone_names.get(detection['zone'], detection['zone'])
        
        return f"{obj_name} a la {zone_name}"
    
    def _speak_async(self, message):
        """Speak message using espeak"""
        try:
            import subprocess
            subprocess.Popen(['espeak', '-v', 'es', '-s', str(Config.TTS_RATE), message], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[AUDIO] {message}")
        except Exception as e:
            print(f"[AUDIO] TTS error: {e}")
    
    def cleanup(self):
        pass

def build_navigation_system(enable_dashboard=False):
    """Simple builder function"""
    return SimpleCoordinator()


class JetsonServer:
    """
    Servidor Jetson que procesa frames del Mac usando pipeline simplificado
    """
    
    def __init__(self):
        self.config = CommunicationConfig
        
        # Networking
        self.frame_server_socket = None
        self.dashboard_server_socket = None
        self.frame_client_socket = None
        self.dashboard_client_socket = None
        self.running = False
        
        # Processing pipeline
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
        """Configurar pipeline de procesamiento"""
        try:
            print("[JETSON SERVER] Setting up processing pipeline...")
            
            self.coordinator = build_navigation_system(enable_dashboard=False)
            
            print("[JETSON SERVER] ✅ Processing pipeline configured")
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
                
                # Procesar frame
                self._process_frame(frame_msg)
                
                self.frames_received += 1
                
            except Exception as e:
                print(f"[JETSON SERVER] Error in frame handler: {e}")
                if self.running:
                    time.sleep(0.1)
                break
        
        print("[JETSON SERVER] Frame handler thread stopped")
    
    def _process_frame(self, frame_msg: FrameMessage) -> None:
        """Procesar frame usando pipeline y enviar resultado al Mac"""
        processing_start = time.time()
        
        try:
            # Procesar frame usando Coordinator
            annotated_frame = self.coordinator.process_frame(frame_msg.frame_data)
            
            # Obtener detecciones
            detections = self.coordinator.current_detections
            
            # Obtener comando de audio
            audio_command = getattr(self.coordinator.audio_system, 'last_phrase', None)
            
            # Crear dashboard con información adicional
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
        """Crear dashboard con información del servidor"""
        dashboard = annotated_frame.copy()
        
        # Añadir información del servidor
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
        
        # Añadir latency info
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
            time.sleep(10)
            
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
                self.coordinator.cleanup()
            except Exception:
                pass
        
        print("[JETSON SERVER] ✅ Stopped successfully")


def test_components():
    """Test de componentes básicos"""
    print("🧪 Testing components...")
    
    results = {}
    
    # Test YOLO
    try:
        from ultralytics import YOLO
        model = YOLO('yolo11n.pt')
        print("✅ YOLO: Available")
        results['yolo'] = True
    except Exception as e:
        print(f"❌ YOLO: {e}")
        results['yolo'] = False
    
    # Test OpenCV
    try:
        import cv2
        print(f"✅ OpenCV: {cv2.__version__}")
        results['opencv'] = True
    except Exception as e:
        print(f"❌ OpenCV: {e}")
        results['opencv'] = False
    
    # Test Audio
    try:
        import subprocess
        subprocess.run(['espeak', '--version'], capture_output=True, timeout=5)
        print("✅ Audio: espeak available")
        results['audio'] = True
    except Exception as e:
        print(f"❌ Audio: {e}")
        results['audio'] = False
    
    return results


def main():
    """Función principal del servidor Jetson"""
    print("=" * 60)
    print("ARIA NAVIGATION - JETSON SERVER (Distributed Mode)")
    print("Receives frames from Mac, processes with simplified pipeline")
    print("=" * 60)
    
    # Parse command line arguments
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    
    if command == "test":
        print("🧪 Running component tests...")
        results = test_components()
        
        print("\n📊 Test Results:")
        for component, result in results.items():
            status = "✅" if result else "❌"
            print(f"{status} {component}")
        
        all_passed = all(results.values())
        print(f"\n{'✅ All tests passed!' if all_passed else '❌ Some tests failed'}")
        return
    
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