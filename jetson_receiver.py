#!/usr/bin/env python3
"""
🎯 JetsonObserver - Adaptación específica de tu Observer existente

CAMBIOS MÍNIMOS:
1. En lugar de on_image_received() del SDK → recv_image() de ImageZMQ
2. Todo lo demás IGUAL: Builder, Coordinator, AudioSystem, etc.
"""

import sys
import imagezmq
import numpy as np
from pathlib import Path

# Importar TU arquitectura existente
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.ctrl_handler import CtrlCHandler
from core.navigation.builder import build_navigation_system


class JetsonObserver:
    """
    Versión adaptada de tu Observer que recibe frames por ImageZMQ
    en lugar del SDK Aria, pero usa TODA tu arquitectura existente.
    """
    
    def __init__(self, port=5555, enable_dashboard=False):
        print(f"[JETSON] Inicializando Observer adaptado...")
        
        # 🎯 ESTO ES IGUAL A TU OBSERVER ORIGINAL
        # Usar tu Builder pattern existente
        self.coordinator = build_navigation_system(enable_dashboard=enable_dashboard)
        
        # Variables de estado (igual que tu Observer)
        self.current_frame = None
        self.frames_processed = 0
        
        # 🔄 ESTO ES LO ÚNICO DIFERENTE
        # En lugar de device_manager.register_observer(self)
        # Usar ImageZMQ para recibir frames
        self.image_hub = imagezmq.ImageHub(open_port=f"tcp://*:{port}")
        
        print(f"[JETSON] ✅ Observer listo - puerto {port}")
    
    def run(self):
        """
        Main loop que reemplaza el callback on_image_received()
        """
        ctrl_handler = CtrlCHandler()
        
        while not ctrl_handler.should_stop:
            try:
                # 📥 RECIBIR frame (en lugar de on_image_received callback)
                rpi_name, frame = self.image_hub.recv_image()
                
                # 🔄 PROCESAR exactamente igual que tu Observer original
                self._process_frame(frame)
                
                # 📤 CONFIRMAR recepción (protocolo ImageZMQ)
                self.image_hub.send_reply(b'OK')
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[ERROR] {e}")
    
    def _process_frame(self, frame):
        """
        Procesamiento de frame - EXACTAMENTE igual que tu Observer original
        """
        # Rotar frame (igual que en tu Observer)
        rotated_frame = self._rotate_frame(frame)
        
        # 🎯 USAR TU COORDINATOR EXISTENTE
        # Esto llama a tu arquitectura completa:
        # ImageEnhancer → YoloProcessor → AudioSystem → FrameRenderer
        annotated_frame = self.coordinator.process_frame(rotated_frame)
        
        # Actualizar estado (igual que tu Observer)
        self.current_frame = annotated_frame
        self.frames_processed += 1
        
        # Stats cada 100 frames (igual que tu Observer)
        if self.frames_processed % 100 == 0:
            print(f"[JETSON] Frames procesados: {self.frames_processed}")
    
    def _rotate_frame(self, frame):
        """Rotación de frame - igual que tu Observer original"""
        import cv2
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    
    # 🎯 MÉTODOS COMPATIBLES con tu Observer original
    def get_latest_frame(self):
        """API compatibility con tu Observer"""
        return self.current_frame
    
    def test_audio(self):
        """Test audio - igual que tu Observer"""
        try:
            self.coordinator.audio_system.speak_force("test audio")
        except Exception as e:
            print(f"[WARN] Audio test failed: {e}")
    
    def print_stats(self):
        """Print stats - igual que tu Observer"""
        print(f"[STATS] Frames procesados: {self.frames_processed}")
        
        # Stats del coordinator (si tiene)
        if hasattr(self.coordinator, 'get_status'):
            status = self.coordinator.get_status()
            print(f"[STATS] Detecciones: {status.get('current_detections_count', 0)}")


# 🎯 MAIN que usa la adaptación
def main():
    print("🎯 JETSON NAVIGATION - Usando arquitectura existente")
    print("=" * 60)
    
    try:
        # Crear Observer adaptado
        observer = JetsonObserver(port=5555, enable_dashboard=False)
        
        print("[INFO] Sistema listo - esperando frames del Mac...")
        print("[INFO] Tu AudioSystem, YOLO, etc. funcionando igual que en Mac")
        
        # Ejecutar loop principal
        observer.run()
        
    except Exception as e:
        print(f"[ERROR] {e}")
    
    print("[INFO] Sistema terminado")


if __name__ == "__main__":
    main()