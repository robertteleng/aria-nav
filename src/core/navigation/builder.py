#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏗️ Simple Builder Pattern - TFM Navigation System
"""

from core.vision.yolo_processor import YoloProcessor
from core.audio.audio_system import AudioSystem
from presentation.renderers.frame_renderer import FrameRenderer
from vision.image_enhancer import ImageEnhancer
from navigation.coordinator import Coordinator

class Builder:
    """Builder que crea todas las dependencias del sistema"""
    
    def __init__(self):
        pass  # Las clases leen Config internamente
    
    def build_yolo_processor(self):
        print("  📦 Creando YOLO Processor...")
        return YoloProcessor()  # Sin parámetros, lee Config internamente
    
    def build_audio_system(self):
        print("  📦 Creando Audio System...")
        return AudioSystem()  # Sin parámetros, lee Config internamente
    
    def build_frame_renderer(self):
        print("  📦 Creando Frame Renderer...")
        return FrameRenderer()  # Sin parámetros, lee Config internamente
    
    def build_image_enhancer(self):
        print("  📦 Creando Image Enhancer...")
        return ImageEnhancer()  # Sin parámetros, lee Config internamente
    
    def build_coordinator(self, yolo_processor, audio_system, frame_renderer, image_enhancer, dashboard=None):
        print("  📦 Creando Coordinator...")
        return Coordinator(
            yolo_processor=yolo_processor,
            audio_system=audio_system,
            frame_renderer=frame_renderer,
            image_enhancer=image_enhancer,
            dashboard=dashboard
        )
    
    # def build_coordinator(self, yolo_processor, audio_system, frame_renderer, image_enhancer):
    #     """Coordinator SIN dashboard - el Observer maneja su propio dashboard"""
    #     print("  📦 Creando Coordinator...")
    #     return Coordinator(
    #         yolo_processor=yolo_processor,
    #         audio_system=audio_system,
    #         frame_renderer=frame_renderer,
    #         image_enhancer=image_enhancer,
    #         dashboard=None  # Sin dashboard interno
    #     )

    def build_full_system(self, enable_dashboard=False):  # False por defecto
        print("🏗️ Construyendo sistema completo...")
        
        # Crear componentes SIN dashboard
        yolo_processor = self.build_yolo_processor()
        audio_system = self.build_audio_system()
        frame_renderer = self.build_frame_renderer()
        image_enhancer = self.build_image_enhancer()
        
        # Coordinator sin dashboard - Observer maneja el suyo
        coordinator = self.build_coordinator(
            yolo_processor, audio_system, frame_renderer, image_enhancer
        )
        
        print("✅ Sistema completo construido!")
        return coordinator

# 🔧 FUNCIÓN FUERA DE LA CLASE
def build_navigation_system(enable_dashboard=True):
    """Función de conveniencia para crear sistema completo"""
    builder = Builder()
    return builder.build_full_system(enable_dashboard=enable_dashboard)

# Testing
if __name__ == "__main__":
    print("🧪 Testing Builder...")
    try:
        coordinator = build_navigation_system(enable_dashboard=False)
        print("✅ Test pasado!")
    except Exception as e:
        print(f"❌ Error: {e}")
