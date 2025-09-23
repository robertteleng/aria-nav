#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Jetson Navigation System - Refactored Architecture
Sistema de navegación para personas con discapacidad visual usando datos del Mac

Arquitectura Separada:
- JetsonObserver: Solo manejo de ImageZMQ receiver
- Coordinator: Solo pipeline de navegación (YOLO + Audio)  
- PresentationManager: Solo UI/Dashboard/Display

Author: Roberto Rojas Sahuquillo
Date: Septiembre 2025  
Version: 2.0 - Clean Separated Architecture for Jetson
"""

import cv2
import time
from utils.ctrl_handler import CtrlCHandler

# Componentes separados de la nueva arquitectura
from jetson_observer import JetsonObserver  # Solo ImageZMQ receiver
from core.navigation.coordinator import Coordinator  # Solo Pipeline
from core.navigation.builder import Builder  # Factory
from presentation.presentation_manager import PresentationManager  # Solo UI


def main():
    """
    🎯 Punto de entrada principal con arquitectura limpia separada para Jetson
    
    Flujo:
    1. Inicialización de componentes separados
    2. Setup de ImageZMQ receiver en lugar de Aria SDK
    3. Loop principal de procesamiento
    4. Cleanup ordenado
    """
    print("=" * 60)
    print("🚀 JETSON Navigation System - Arquitectura Separada")
    print("Sistema de navegación para personas con discapacidad visual")
    print("Recibiendo datos del Mac via ImageZMQ")
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
    observer = None
    coordinator = None
    presentation = None
    
    try:
        print("\n🔧 Inicializando componentes...")
        
        # 1. Observer Setup (reemplaza DeviceManager + Observer)
        print("  👁️ Inicializando JetsonObserver (ImageZMQ + Pipeline)...")
        observer = JetsonObserver(enable_dashboard=False)  # Sin dashboard propio
        
        if not observer.start():
            print("❌ Error iniciando observer")
            return
        
        # 2. Navigation Coordinator Setup (Solo Pipeline)
        print("  🧭 Inicializando Coordinator (Navigation Pipeline)...")
        builder = Builder()
        coordinator = builder.build_full_system(enable_dashboard=False)  # Sin dashboard propio
        
        # 3. Presentation Manager Setup (Solo UI)
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
        print("\n🔄 Sistema activo - esperando datos del Mac...")
        
        # 4. Main Processing Loop
        frames_processed = 0
        last_stats_print = time.time()
        
        while not ctrl_handler.should_stop:
            try:
                # Obtener datos del Observer (Solo ImageZMQ receiver adaptado)
                frame = observer.get_latest_frame('rgb')
                slam1_frame = observer.get_latest_frame('slam1')
                slam2_frame = observer.get_latest_frame('slam2')
                motion_data = observer.get_motion_state()
                motion_state = motion_data.get('state', 'unknown') if motion_data else 'unknown'
                
                if frame is not None:
                    frames_processed += 1
                    
                    # Procesar con Coordinator (Solo Pipeline)
                    processed_frame = coordinator.process_frame(frame, motion_state)
                    depth_map = coordinator.get_latest_depth_map()
                    
                    # Actualizar UI con PresentationManager (Solo UI)
                    key = presentation.update_display(
                        frame=processed_frame,
                        detections=coordinator.get_current_detections(),
                        motion_state=motion_state,
                        coordinator_stats=coordinator.get_status(),
                        depth_map=depth_map,
                        slam1_frame=slam1_frame,
                        slam2_frame=slam2_frame
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
                    if observer and hasattr(observer, 'receiver'):
                        receiver_stats = observer.receiver.get_stats()
                        print(f"[STATUS] Receiver: {receiver_stats['connection_status']}, FPS: {receiver_stats['fps']:.1f}")
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
                print("  ✅ JetsonObserver cleanup")
        except Exception as e:
            print(f"  ⚠️ JetsonObserver cleanup error: {e}")
        
        # Final OpenCV cleanup
        try:
            cv2.destroyAllWindows()
            print("  ✅ OpenCV cleanup")
        except Exception as e:
            print(f"  ⚠️ OpenCV cleanup error: {e}")
        
        print("✅ Programa terminado exitosamente")
        print("=" * 60)


if __name__ == "__main__":
    main()