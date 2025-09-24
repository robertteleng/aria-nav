#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Navigation System for Blind Users - Refactored Architecture
Sistema de navegación para personas con discapacidad visual usando Meta Aria glasses

Arquitectura Separada:
- AriaObserver: Solo manejo del SDK de Aria
- Coordinator: Solo pipeline de navegación (YOLO + Audio)  
- PresentationManager: Solo UI/Dashboard/Display

Author: Roberto Rojas Sahuquillo
Date: Septiembre 2025  
Version: 2.0 - Clean Separated Architecture
"""

import cv2
import time
from utils.ctrl_handler import CtrlCHandler
from core.hardware.device_manager import DeviceManager

# Componentes separados de la nueva arquitectura
from core.observer import Observer  # Solo SDK
from core.navigation.coordinator import Coordinator  # Solo Pipeline
from core.navigation.builder import Builder  # Factory
from presentation.presentation_manager import PresentationManager  # Solo UI


def main():
    """
    🎯 Punto de entrada principal con arquitectura limpia separada
    
    Flujo:
    1. Inicialización de componentes separados
    2. Setup de Aria SDK y streaming
    3. Loop principal de procesamiento
    4. Cleanup ordenado
    """
    print("=" * 60)
    print("🚀 ARIA Navigation System - Arquitectura Separada")
    print("Sistema de navegación para personas con discapacidad visual")
    print("=" * 60)
    
    # Setup clean exit handler
    ctrl_handler = CtrlCHandler()

    # UI Configuration
    enable_dashboard = input("¿Habilitar dashboard? (y/n): ").lower() == 'y'
    dashboard_type = "opencv"  # Default
    
    if enable_dashboard:
        allowed_dashboards = ["opencv", "rerun", "web"]
        dashboard_choice = input("Dashboard type (opencv/rerun/web) [opencv]: ").lower() or "opencv"
        if dashboard_choice not in allowed_dashboards:
            print(f"[MAIN] Dashboard '{dashboard_choice}' no reconocido, usando 'opencv'")
            dashboard_choice = "opencv"
        dashboard_type = dashboard_choice
        print(f"[MAIN] Dashboard {dashboard_type} habilitado")
    else:
        print("[MAIN] Display OpenCV simple habilitado")
    
    # Component initialization
    device_manager = None
    observer = None
    coordinator = None
    presentation = None
    
    try:
        print("\n🔧 Inicializando componentes...")
        
        # 1. Aria Device Setup
        print("  📱 Conectando con Aria glasses...")
        device_manager = DeviceManager()
        device_manager.connect()
        rgb_calib = device_manager.start_streaming()
        
        # 2. Observer Setup (Solo SDK)
        print("  👁️ Inicializando AriaObserver (SDK only)...")
        observer = Observer(rgb_calib=rgb_calib)
        device_manager.register_observer(observer)
        device_manager.subscribe()
        
        # 3. Navigation Coordinator Setup (Solo Pipeline)
        print("  🧭 Inicializando Coordinator (Navigation Pipeline)...")
        builder = Builder()
        coordinator = builder.build_full_system(enable_dashboard=False)  # Sin dashboard propio
        
        # 4. Presentation Manager Setup (Solo UI)
        print("  🎨 Inicializando PresentationManager (UI Layer)...")
        presentation = PresentationManager(
            enable_dashboard=enable_dashboard,
            dashboard_type=dashboard_type
        )
        
        print("✅ Todos los componentes inicializados correctamente")
        print("\n🎮 Controles:")
        print("  - 'q': Salir del sistema")
        print("  - 't': Test del sistema de audio")
        print("  - Ctrl+C: Salida limpia")
        print("\n🔄 Sistema activo - procesando frames...")
        
        # 5. Main Processing Loop
        frames_processed = 0
        last_stats_print = time.time()
        
        while not ctrl_handler.should_stop:
            try:
                # Obtener datos del Observer (Solo SDK)
                frame = observer.get_latest_frame('rgb')
                slam1_frame = observer.get_latest_frame('slam1')
                slam2_frame = observer.get_latest_frame('slam2')
                motion_data = observer.get_motion_state()
                motion_state = motion_data.get('state', 'unknown') if motion_data else 'unknown'
                
                if frame is not None:
                    frames_processed += 1
                    
                    # Procesar con Coordinator (Solo Pipeline)
                    processed_frame = coordinator.process_frame(frame, motion_state)
                    if hasattr(coordinator, 'handle_slam_frames'):
                        coordinator.handle_slam_frames(slam1_frame, slam2_frame)
                    depth_map = coordinator.get_latest_depth_map()
                    slam_events = coordinator.get_slam_events() if hasattr(coordinator, 'get_slam_events') else None
                    
                    # Actualizar UI con PresentationManager (Solo UI)
                    key = presentation.update_display(
                        frame=processed_frame,
                        detections=coordinator.get_current_detections(),
                        motion_state=motion_state,
                        coordinator_stats=coordinator.get_status(),
                        depth_map=depth_map,
                        slam1_frame=slam1_frame,
                        slam2_frame=slam2_frame,
                        slam_events=slam_events
                    )
                    
                    # Handle UI Events
                    if key == 'q':
                        print("\n[INFO] 'q' detected, closing application...")
                        break
                    elif key == 't':
                        print("[INFO] Testing audio system...")
                        coordinator.test_audio()
                        presentation.log_audio_command("Test del sistema", 5)
                
                # Estadísticas periódicas
                current_time = time.time()
                if current_time - last_stats_print > 10.0:  # Cada 10 segundos
                    print(f"[STATUS] Frames: {frames_processed}, Motion: {motion_state}")
                    last_stats_print = current_time
                
            except Exception as e:
                print(f"[WARN] Error en processing loop: {e}")
                time.sleep(0.1)  # Evitar spam de errores
        
        print(f"\n📊 Sesión completada: {frames_processed} frames procesados")
        
        # Final statistics
        print("\n📈 Estadísticas finales:")
        if observer:
            observer.print_stats()
        if coordinator:
            coordinator.print_stats()
        if presentation:
            presentation.print_ui_stats()
        
    except KeyboardInterrupt:
        print("\n[INFO] ⌨️ Interrupción de teclado detectada")
        
    except Exception as e:
        print(f"\n[ERROR] ❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup ordenado de todos los componentes
        print("\n🧹 Iniciando limpieza de recursos...")
        
        try:
            if coordinator:
                coordinator.cleanup()
                print("  ✅ Coordinator cleanup")
        except Exception as e:
            print(f"  ⚠️ Coordinator cleanup error: {e}")
        
        try:
            if presentation:
                presentation.cleanup()
                print("  ✅ PresentationManager cleanup")
        except Exception as e:
            print(f"  ⚠️ PresentationManager cleanup error: {e}")
        
        try:
            if observer:
                observer.stop()
                print("  ✅ AriaObserver cleanup")
        except Exception as e:
            print(f"  ⚠️ AriaObserver cleanup error: {e}")
        
        try:
            if device_manager:
                device_manager.cleanup()
                print("  ✅ DeviceManager cleanup")
        except Exception as e:
            print(f"  ⚠️ DeviceManager cleanup error: {e}")
        
        # Final OpenCV cleanup
        try:
            cv2.destroyAllWindows()
            print("  ✅ OpenCV cleanup")
        except Exception as e:
            print(f"  ⚠️ OpenCV cleanup error: {e}")
        
        print("✅ Programa terminado exitosamente")
        print("=" * 60)


def main_debug():
    """
    🐛 Versión de debug con componentes mock para testing sin hardware
    """
    print("🧪 DEBUG MODE - Testing sin hardware Aria")
    
    import numpy as np
    from time import sleep
    
    # Mock observer para testing
    class MockObserver:
        def __init__(self):
            self.frame_count = 0
        
        def get_latest_frame(self, camera='rgb'):
            # Generar frame sintético
            self.frame_count += 1
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Añadir texto indicando que es mock
            cv2.putText(frame, f"MOCK FRAME {self.frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return frame
        
        def get_motion_state(self):
            # Alternar entre stationary y walking
            state = "walking" if (self.frame_count // 30) % 2 else "stationary"
            return {'state': state, 'magnitude': 9.8}
        
        def print_stats(self):
            print(f"  MockObserver: {self.frame_count} frames generados")
        
        def stop(self):
            pass
    
    try:
        print("🔧 Inicializando componentes mock...")
        
        # Solo coordinator y presentation para testing
        builder = Builder()
        coordinator = builder.build_full_system(enable_dashboard=False)
        presentation = PresentationManager(enable_dashboard=False)
        observer = MockObserver()
        
        ctrl_handler = CtrlCHandler()
        
        print("✅ Componentes mock inicializados")
        print("🔄 Loop de testing activo...")
        
        frames_processed = 0
        
        while not ctrl_handler.should_stop and frames_processed < 300:  # Límite para testing
            frame = observer.get_latest_frame()
            slam1_frame = observer.get_latest_frame('slam1')
            slam2_frame = observer.get_latest_frame('slam2')
            motion_data = observer.get_motion_state()
            
            if frame is not None:
                processed_frame = coordinator.process_frame(frame, motion_data['state'])
                depth_map = coordinator.get_latest_depth_map()
                
                key = presentation.update_display(
                    frame=processed_frame,
                    detections=coordinator.get_current_detections(),
                    motion_state=motion_data['state'],
                    coordinator_stats=coordinator.get_status(),
                    depth_map=depth_map,
                    slam1_frame=slam1_frame,
                    slam2_frame=slam2_frame
                )
                
                if key == 'q':
                    break
                
                frames_processed += 1
                
                # Simular 30fps
                sleep(1/30)
        
        print(f"🧪 Testing completado: {frames_processed} frames")
        
        # Stats
        observer.print_stats()
        coordinator.print_stats()
        presentation.print_ui_stats()
        
    except KeyboardInterrupt:
        print("\n🧪 Testing interrumpido")
    finally:
        try:
            coordinator.cleanup()
            presentation.cleanup()
        except:
            pass
        cv2.destroyAllWindows()


def main_hybrid_mac():
    """
    🌉 Versión híbrida para Mac - Solo ImageZMQ Sender
    Para usar con Jetson processing remoto
    """
    print("🌉 HYBRID MODE - Mac ImageZMQ Sender")
    print("Enviando frames al Jetson para procesamiento...")
    
    # Esta función se implementará cuando creemos el ImageZMQ sender
    # Por ahora, placeholder para la arquitectura híbrida
    print("⚠️ Función híbrida aún no implementada")
    print("💡 Usar main() normal para sistema local completo")


if __name__ == "__main__":
    import sys
    
    # Permitir diferentes modos de ejecución
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "debug":
            main_debug()
        elif mode == "hybrid":
            main_hybrid_mac()
        else:
            print(f"❌ Modo '{mode}' no reconocido")
            print("💡 Modos disponibles: debug, hybrid")
            print("💡 Sin argumentos = modo normal")
            sys.exit(1)
    else:
        # Modo normal
        main()
