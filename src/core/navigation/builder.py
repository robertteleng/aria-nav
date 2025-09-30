"""
🏗️ Simple Builder Pattern - TFM Navigation System
"""

from typing import Optional

from core.vision.yolo_processor import YoloProcessor
from core.audio.audio_system import AudioSystem
from core.audio.navigation_audio_router import NavigationAudioRouter
from core.vision.slam_detection_worker import SlamDetectionWorker, CameraSource
from presentation.renderers.frame_renderer import FrameRenderer
from core.vision.image_enhancer import ImageEnhancer
from core.navigation.coordinator import Coordinator
from core.navigation.navigation_pipeline import NavigationPipeline
from core.navigation.navigation_decision_engine import NavigationDecisionEngine
from utils.config import Config

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

    def build_audio_router(self, audio_system: AudioSystem) -> NavigationAudioRouter:
        print("  📦 Creando NavigationAudioRouter...")
        return NavigationAudioRouter(audio_system)
    
    def build_frame_renderer(self):
        print("  📦 Creando Frame Renderer...")
        return FrameRenderer()  # Sin parámetros, lee Config internamente
    
    def build_image_enhancer(self):
        print("  📦 Creando Image Enhancer...")
        return ImageEnhancer()  # Sin parámetros, lee Config internamente

    def build_navigation_pipeline(self, yolo_processor, image_enhancer) -> NavigationPipeline:
        print("  📦 Creando NavigationPipeline...")
        return NavigationPipeline(
            yolo_processor=yolo_processor,
            image_enhancer=image_enhancer,
        )

    def build_decision_engine(self) -> NavigationDecisionEngine:
        print("  📦 Creando NavigationDecisionEngine...")
        return NavigationDecisionEngine()

    def build_coordinator(
        self,
        yolo_processor,
        audio_system,
        frame_renderer,
        image_enhancer,
        dashboard=None,
        audio_router=None,
        navigation_pipeline: Optional[NavigationPipeline] = None,
        decision_engine: Optional[NavigationDecisionEngine] = None,
    ):
        print("  📦 Creando Coordinator...")
        return Coordinator(
            yolo_processor=yolo_processor,
            audio_system=audio_system,
            frame_renderer=frame_renderer,
            image_enhancer=image_enhancer,
            dashboard=dashboard,
            audio_router=audio_router,
            navigation_pipeline=navigation_pipeline,
            decision_engine=decision_engine,
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
        audio_router = self.build_audio_router(audio_system)
        navigation_pipeline = self.build_navigation_pipeline(yolo_processor, image_enhancer)
        decision_engine = self.build_decision_engine()

        # Coordinator sin dashboard - Observer maneja el suyo
        coordinator = self.build_coordinator(
            yolo_processor,
            audio_system,
            frame_renderer,
            image_enhancer,
            audio_router=audio_router,
            navigation_pipeline=navigation_pipeline,
            decision_engine=decision_engine,
        )

        if getattr(Config, "PERIPHERAL_VISION_ENABLED", False) and CameraSource is not None:
            print("  🔁 Configurando visión periférica (SLAM)...")
            slam_workers = {
                CameraSource.SLAM1: SlamDetectionWorker(
                    CameraSource.SLAM1,
                    target_fps=getattr(Config, "SLAM_TARGET_FPS", 8),
                ),
                CameraSource.SLAM2: SlamDetectionWorker(
                    CameraSource.SLAM2,
                    target_fps=getattr(Config, "SLAM_TARGET_FPS", 8),
                ),
            }
            coordinator.attach_peripheral_system(slam_workers, audio_router)

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
