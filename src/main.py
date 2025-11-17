#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Navigation System for Blind Users - Refactored Architecture
Sistema de navegación para personas con discapacidad visual usando Meta Aria glasses

Arquitectura Separada:
- AriaObserver: Solo manejo del SDK de Aria
- Coordinator: Solo pipeline de navegación (YOLO + Audio)  
- PresentationManager: Solo UI/Dashboard/Display

MODO MOCK: Soporta desarrollo sin gafas usando MockObserver

Author: Roberto Rojas Sahuquillo
Date: Septiembre 2025  
Version: 2.0 - Clean Separated Architecture + Mock Support
"""

# NOTE: Multiprocessing spawn is configured in run.py wrapper
# Do NOT call set_start_method here - it must be done before importing this module

import cv2
import time
from utils.ctrl_handler import CtrlCHandler
from utils.config import Config
from core.hardware.device_manager import DeviceManager


# Componentes separados de la nueva arquitectura
from core.observer import Observer  # Solo SDK
from core.mock_observer import MockObserver  # Mock para desarrollo
from core.navigation.coordinator import Coordinator  # Solo Pipeline
from core.navigation.builder import Builder  # Factory
from presentation.presentation_manager import PresentationManager  # Solo UI

# Telemetría centralizada
from core.telemetry.telemetry_logger import TelemetryLogger

# Depth logger
from utils.depth_logger import init_depth_logger

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

    # 🆕 MOCK MODE: Preguntar si usar gafas reales o mock
    print("\n📱 Modo de operación:")
    print("  1. Gafas Aria reales (requiere hardware)")
    print("  2. Mock sintético (desarrollo sin hardware)")
    print("  3. Mock con video (replay de sesión grabada)")
    print("  4. Mock con imagen estática")
    
    mode_choice = input("\nSelecciona modo [1]: ").strip() or "1"
    use_mock = mode_choice != "1"
    mock_mode = None
    mock_source = None
    
    if use_mock:
        if mode_choice == "2":
            mock_mode = "synthetic"
            print("🤖 Modo MOCK: Sintético (frames generados)")
        elif mode_choice == "3":
            mock_mode = "video"
            mock_source = input("Ruta al video [data/session.mp4]: ").strip() or "data/session.mp4"
            print(f"🎥 Modo MOCK: Video replay ({mock_source})")
        elif mode_choice == "4":
            mock_mode = "static"
            mock_source = input("Ruta a la imagen [data/frame.jpg]: ").strip() or "data/frame.jpg"
            print(f"🖼️  Modo MOCK: Imagen estática ({mock_source})")
        else:
            print("⚠️  Opción no válida, usando modo sintético")
            mock_mode = "synthetic"
    else:
        print("📱 Modo: Gafas Aria reales")

    # UI Configuration
    enable_dashboard = input("\n¿Habilitar dashboard? (y/n): ").lower() == 'y'
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
    telemetry = None
    
    try:
        print("\n🔧 Inicializando componentes...")

        # 0. Inicializar telemetría PRIMERO
        print("  📊 Inicializando TelemetryLogger...")
        telemetry = TelemetryLogger()
        
        # 0b. Inicializar depth logger con la misma sesión
        print("  🔍 Inicializando DepthLogger...")
        init_depth_logger(session_dir=telemetry.output_dir)
        
        # 1. Observer Setup - Real o Mock
        if use_mock:
            print(f"  🤖 Inicializando MockObserver (modo: {mock_mode})...")
            observer_kwargs = {
                'mode': mock_mode,
                'fps': 60,
                'resolution': (1408, 1408),
            }
            if mock_mode == 'video':
                observer_kwargs['video_path'] = mock_source
            elif mock_mode == 'static':
                observer_kwargs['image_path'] = mock_source
            
            observer = MockObserver(**observer_kwargs)
            observer.start()
            print("  ✅ MockObserver iniciado - desarrollo sin hardware")
        else:
            # Modo real con Aria glasses
            print("  📱 Conectando con Aria glasses...")
            device_manager = DeviceManager()
            device_manager.connect()
            rgb_calib = device_manager.start_streaming()
            
            print("  👁️ Inicializando AriaObserver (SDK only)...")
            observer = Observer(rgb_calib=rgb_calib)
            device_manager.register_observer(observer)
            device_manager.subscribe()
            print("  ✅ AriaObserver conectado al hardware real")
        
        # 3. Navigation Coordinator Setup (Solo Pipeline)
        print("  🧭 Inicializando Coordinator (Navigation Pipeline)...")
        builder = Builder()
        coordinator = builder.build_full_system(
            enable_dashboard=False,
            telemetry=telemetry,
        )  # Sin dashboard propio
        
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
        fps_start_time = time.time() 

        
        while not ctrl_handler.should_stop:
            try:

                 # Timestamp inicio del frame
                frame_start_time = time.time()

                # Obtener datos del Observer (Solo SDK)
                frame = observer.get_latest_frame('rgb')
                slam1_frame = observer.get_latest_frame('slam1')
                slam2_frame = observer.get_latest_frame('slam2')
                motion_data = observer.get_motion_state()
                motion_state = motion_data.get('state', 'unknown') if motion_data else 'unknown'
                
                if frame is not None:
                    frames_processed += 1
                    
                    # Construir frames_dict para Phase 2 multiproc
                    frames_dict = {
                        'central': frame,
                        'slam1': slam1_frame,
                        'slam2': slam2_frame,
                    }
                    
                    # Procesar con Coordinator (Solo Pipeline)
                    processed_frame = coordinator.process_frame(frame, motion_state, frames_dict=frames_dict)
                    # if hasattr(coordinator, 'handle_slam_frames'):
                    #     coordinator.handle_slam_frames(slam1_frame, slam2_frame)

                    slam_skip = getattr(Config, 'SLAM_FRAME_SKIP', 3)
                    if frames_processed % slam_skip == 0:
                        if hasattr(coordinator, 'handle_slam_frames'):
                            coordinator.handle_slam_frames(slam1_frame, slam2_frame)   
                    
                    # Only get depth map if enabled
                    depth_map = coordinator.get_latest_depth_map() if Config.DEPTH_ENABLED else None
                    slam_events = coordinator.get_slam_events() if hasattr(coordinator, 'get_slam_events') else None
                    
                     # ✅ NUEVO: Calcular métricas de performance
                    frame_end_time = time.time()
                    frame_latency_ms = (frame_end_time - frame_start_time) * 1000
                    elapsed_total = frame_end_time - fps_start_time
                    current_fps = frames_processed / elapsed_total if elapsed_total > 0 else 0
                    
                    # ✅ NUEVO: Log performance
                    telemetry.log_frame_performance(
                        frame_number=frames_processed,
                        fps=current_fps,
                        latency_ms=frame_latency_ms
                    )
                    
                    # ✅ NUEVO: Log detecciones RGB
                    current_detections = coordinator.get_current_detections()
                    if current_detections:
                        telemetry.log_detections_batch(
                            frame_number=frames_processed,
                            source="rgb",
                            detections=current_detections
                        )

                    # Actualizar UI con PresentationManager (Solo UI)
                    key = presentation.update_display(
                        frame=processed_frame,
                        # detections=coordinator.get_current_detections(),
                        detections=current_detections,
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
                    # ✅ NUEVO: Stats de performance en consola
                    if telemetry:
                        perf = telemetry.get_performance_summary()
                        print(f"[PERF] FPS: {perf.get('avg_fps', 0):.1f}, Latencia: {perf.get('avg_latency_ms', 0):.0f}ms")
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
        
        # Finalizar telemetría PRIMERO
        try:
            if telemetry:
                print("  📊 Finalizando telemetría...")
                summary = telemetry.finalize_session()
                print("\n" + "="*60)
                print("📊 RESUMEN DE SESIÓN")
                print("="*60)
                print(f"  ⏱️  Duración: {summary['duration_seconds']:.1f}s")
                print(f"  🎞️  Frames totales: {summary['total_frames']}")
                print(f"  📹 FPS promedio: {summary['avg_fps']:.2f}")
                print(f"  ⚡ Latencia promedio: {summary['avg_latency_ms']:.1f}ms")
                print(f"  🎯 Detecciones totales: {summary['total_detections']}")
                if summary.get('detections_by_class'):
                    print(f"  📦 Por clase: {summary['detections_by_class']}")
                print(f"\n📁 Logs guardados en: {telemetry.output_dir}")
                print("="*60 + "\n")
        except Exception as e:
            print(f"  ⚠️ Telemetry finalize error: {e}")

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
    
    import os
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
        coordinator = builder.build_full_system(
            enable_dashboard=False,
            telemetry=None,
        )
        
        # Skip UI in multiprocessing mode to avoid Qt/OpenCV conflicts
        from utils.config import Config
        multiproc_enabled = getattr(Config, "PHASE2_MULTIPROC_ENABLED", False)
        if multiproc_enabled:
            print("🔄 Multiprocessing mode - UI disabled")
            presentation = None
        else:
            presentation = PresentationManager(enable_dashboard=False)
        
        observer = MockObserver()
        
        ctrl_handler = CtrlCHandler()
        
        print("✅ Componentes mock inicializados")
        print("🔄 Loop de testing activo...")
        
        max_frames = int(os.environ.get("DEBUG_MAX_FRAMES", "300"))
        print(f"📊 Processing up to {max_frames} frames...")
        frames_processed = 0
        frame_times = []  # Track frame processing times
        start_time = time.time()
        
        while not ctrl_handler.should_stop and frames_processed < max_frames:
            frame = observer.get_latest_frame()
            slam1_frame = observer.get_latest_frame('slam1')
            slam2_frame = observer.get_latest_frame('slam2')
            motion_data = observer.get_motion_state()
            
            if frame is not None:
                # Build frames_dict for multiprocessing
                frames_dict = {
                    'central': frame,
                    'slam1': slam1_frame,
                    'slam2': slam2_frame,
                }
                
                frame_start = time.time()
                processed_frame = coordinator.process_frame(frame, motion_data['state'], frames_dict=frames_dict)
                frame_end = time.time()
                frame_times.append(frame_end - frame_start)
                
                depth_map = coordinator.get_latest_depth_map()
                
                if presentation is not None:
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
                else:
                    # No UI mode - just print progress
                    if frames_processed % 10 == 0:
                        elapsed = time.time() - start_time
                        current_fps = frames_processed / elapsed if elapsed > 0 else 0
                        avg_latency = sum(frame_times) / len(frame_times) * 1000 if frame_times else 0
                        print(f"📊 Frame {frames_processed}/{max_frames} | FPS: {current_fps:.1f} | Latency: {avg_latency:.1f}ms")
                
                frames_processed += 1
                
                # Simular 30fps
                sleep(1/30)
        
        # Final metrics
        end_time = time.time()
        total_time = end_time - start_time
        avg_fps = frames_processed / total_time if total_time > 0 else 0
        avg_latency = sum(frame_times) / len(frame_times) * 1000 if frame_times else 0
        
        print(f"\n🎯 RESULTADOS FINALES:")
        print(f"  Frames procesados: {frames_processed}")
        print(f"  Tiempo total: {total_time:.2f}s")
        print(f"  FPS promedio: {avg_fps:.2f}")
        print(f"  Latency promedio: {avg_latency:.2f}ms")
        if frame_times:
            import numpy as np
            print(f"  Latency p50: {np.percentile(frame_times, 50)*1000:.2f}ms")
            print(f"  Latency p95: {np.percentile(frame_times, 95)*1000:.2f}ms")
            print(f"  Latency p99: {np.percentile(frame_times, 99)*1000:.2f}ms")
        
        print(f"🧪 Testing completado: {frames_processed} frames")
        
        # Stats
        observer.print_stats()
        coordinator.print_stats()
        if presentation is not None:
            presentation.print_ui_stats()
        
    except KeyboardInterrupt:
        print("\n🧪 Testing interrumpido")
    finally:
        try:
            coordinator.cleanup()
            if presentation is not None:
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
