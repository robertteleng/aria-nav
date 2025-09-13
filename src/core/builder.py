#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏗️ Simple Builder Pattern - TFM Navigation System
Crea todas las dependencias del sistema de navegación de forma centralizada

Fecha: Día 2 - Sistema modular
Versión: 1.0 - Builder pattern
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.yolo_processor import YoloProcessor
from audio.audio_system import AudioSystem
from vision.dashboard import Dashboard
from core.coordinator import Coordinator


class Builder:
    """
    🏗️ Builder que crea todas las dependencias del sistema
    
    Responsabilidades:
    - Crear instancias de todos los módulos
    - Configurar parámetros por defecto
    - Inyectar dependencias en el coordinator
    - Proporcionar punto único de creación
    """
    
    def __init__(self):
        # Configuraciones por defecto
        self.yolo_config = {
            'model_path': 'yolo11n.pt',
            'device': 'cpu',  # Usar CPU para evitar bug MPS
            'confidence_threshold': 0.5,
            'verbose': False
        }
        
        self.audio_config = {
            'speech_rate': 150,
            'volume': 0.9,
            'voice_language': 'es',
            'queue_max_size': 3,
            'announcement_cooldown': 2.0
        }
        
        self.dashboard_config = {
            'window_name': 'Aria Navigation System',
            'window_size': (800, 600),
            'show_debug_info': True
        }
    
    def build_yolo_processor(self):
        """
        🤖 Crear y configurar YoloProcessor
        
        Returns:
            YoloProcessor: Instancia configurada para detección
        """
        print("  📦 Creando YOLO Processor...")
        yolo_processor = YoloProcessor(
            model_path=self.yolo_config['model_path'],
            device=self.yolo_config['device'],
            confidence_threshold=self.yolo_config['confidence_threshold'],
            verbose=self.yolo_config['verbose']
        )
        return yolo_processor
    
    def build_audio_system(self):
        """
        🔊 Crear y configurar AudioSystem
        
        Returns:
            AudioSystem: Instancia configurada para TTS
        """
        print("  📦 Creando Audio System...")
        audio_system = AudioSystem(
            speech_rate=self.audio_config['speech_rate'],
            volume=self.audio_config['volume'],
            voice_language=self.audio_config['voice_language'],
            queue_max_size=self.audio_config['queue_max_size'],
            announcement_cooldown=self.audio_config['announcement_cooldown']
        )
        return audio_system
    
    def build_dashboard(self):
        """
        📊 Crear y configurar Dashboard
        
        Returns:
            Dashboard: Instancia configurada para visualización
        """
        print("  📦 Creando Dashboard...")
        dashboard = Dashboard(
            window_name=self.dashboard_config['window_name'],
            window_size=self.dashboard_config['window_size'],
            show_debug_info=self.dashboard_config['show_debug_info']
        )
        return dashboard
    
    def build_coordinator(self, yolo_processor, audio_system, dashboard=None):
        """
        🎯 Crear coordinator con dependencias inyectadas
        
        Args:
            yolo_processor: Instancia de YoloProcessor
            audio_system: Instancia de AudioSystem
            dashboard: Instancia de Dashboard (opcional)
            
        Returns:
            SimpleCoordinatorWithDeps: Coordinator con dependencias
        """
        print("  📦 Creando Coordinator...")
        coordinator = SimpleCoordinatorWithDeps(
            yolo_processor=yolo_processor,
            audio_system=audio_system,
            dashboard=dashboard
        )
        return coordinator
    
    def build_full_system(self, enable_dashboard=True):
        """
        🏗️ Construir sistema completo con todas las dependencias
        
        Args:
            enable_dashboard: Si crear dashboard para visualización
            
        Returns:
            SimpleCoordinatorWithDeps: Sistema completo configurado
        """
        print("🏗️ Construyendo sistema completo...")
        
        # 1. Crear todos los componentes
        yolo_processor = self.build_yolo_processor()
        audio_system = self.build_audio_system()
        
        # 2. Dashboard opcional
        dashboard = None
        if enable_dashboard:
            dashboard = self.build_dashboard()
        
        # 3. Crear coordinator con dependencias inyectadas
        coordinator = self.build_coordinator(
            yolo_processor=yolo_processor,
            audio_system=audio_system,
            dashboard=dashboard
        )
        
        print("✓ YOLO, Audio, Dashboard creados")
        print("✓ Coordinator creado con dependencias inyectadas")
        print("✅ Sistema completo construido!")
        
        return coordinator


def build_navigation_system(enable_dashboard=True):
    """
    🎯 Función de conveniencia para crear sistema completo
    
    Esta es la función que usará el Observer para obtener
    un coordinator completamente configurado.
    
    Args:
        enable_dashboard: Si habilitar visualización
        
    Returns:
        SimpleCoordinatorWithDeps: Sistema listo para usar
    """
    builder = SimpleBuilder()
    return builder.build_full_system(enable_dashboard=enable_dashboard)


# 🧪 TESTING DEL BUILDER
if __name__ == "__main__":
    """
    Test básico del builder para verificar que todo se crea correctamente
    """
    print("🧪 Testing Simple Builder...")
    print("=" * 50)
    
    try:
        # Test 1: Crear sistema completo con dashboard
        print("\n🧪 Test 1: Sistema completo con dashboard")
        coordinator_with_dashboard = build_navigation_system(enable_dashboard=True)
        print(f"✓ Coordinator creado: {type(coordinator_with_dashboard).__name__}")
        print(f"✓ Tiene YOLO: {coordinator_with_dashboard.yolo_processor is not None}")
        print(f"✓ Tiene Audio: {coordinator_with_dashboard.audio_system is not None}")
        print(f"✓ Tiene Dashboard: {coordinator_with_dashboard.dashboard is not None}")
        
        # Test 2: Crear sistema sin dashboard
        print("\n🧪 Test 2: Sistema sin dashboard")
        coordinator_no_dashboard = build_navigation_system(enable_dashboard=False)
        print(f"✓ Coordinator creado: {type(coordinator_no_dashboard).__name__}")
        print(f"✓ Tiene YOLO: {coordinator_no_dashboard.yolo_processor is not None}")
        print(f"✓ Tiene Audio: {coordinator_no_dashboard.audio_system is not None}")
        print(f"✓ Sin Dashboard: {coordinator_no_dashboard.dashboard is None}")
        
        # Test 3: Verificar configuraciones
        print("\n🧪 Test 3: Verificar configuraciones")
        yolo = coordinator_with_dashboard.yolo_processor
        print(f"✓ YOLO device: {yolo.device}")
        print(f"✓ YOLO model: {yolo.model_path}")
        
        audio = coordinator_with_dashboard.audio_system
        print(f"✓ Audio rate: {audio.speech_rate} WPM")
        print(f"✓ Audio volume: {audio.volume}")
        
        print("\n✅ Todos los tests pasaron!")
        print("🎯 Builder listo para integración con Observer")
        
    except Exception as e:
        print(f"\n❌ Error en testing: {e}")
        print("🔧 Revisa las dependencias de los módulos")
        import traceback
        traceback.print_exc()