#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Navigation System for Blind Users - Refactored Architecture.

Assisted navigation system for visually impaired users using Meta Aria glasses.

Separated Architecture:
- AriaObserver: Handles Aria SDK only
- Coordinator: Navigation pipeline (YOLO + Audio) only
- PresentationManager: UI/Dashboard/Display only

MOCK MODE: Supports development without glasses using MockObserver

Author: Roberto Rojas Sahuquillo
Date: September 2025
Version: 2.0 - Clean Separated Architecture + Mock Support

Usage:
    python src/main.py           # Normal mode with Aria glasses
    python src/main.py debug     # Debug mode with mock frames
    python src/main.py hybrid    # Hybrid mode (Mac ImageZMQ sender)
"""

# NOTE: Multiprocessing spawn is configured in run.py wrapper
# Do NOT call set_start_method here - it must be done before importing this module

# IMPORTANT: Set headless mode BEFORE importing cv2 to avoid Qt plugin errors
import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Use xcb backend (available on this system)

import cv2
import time
import gc
from utils.ctrl_handler import CtrlCHandler
from utils.config import Config
from utils.config_sections import (
    PipelineConfig,
    load_pipeline_config,
    PeripheralVisionConfig,
    load_peripheral_vision_config,
)
from core.hardware.device_manager import DeviceManager


# Separated components from new architecture
from core.observer import Observer  # SDK only
from core.mock_observer import MockObserver  # Mock for development
from core.navigation.coordinator import Coordinator  # Pipeline only
from core.navigation.builder import Builder  # Factory
from presentation.presentation_manager import PresentationManager  # UI only

# Centralized telemetry
from core.telemetry.loggers.telemetry_logger import AsyncTelemetryLogger

# Depth logger
from core.telemetry.loggers.depth_logger import init_depth_logger

def _print_welcome():
    """Print welcome banner."""
    print("=" * 60)
    print("🚀 ARIA Navigation System - Arquitectura Separada")
    print("Sistema de navegación para personas con discapacidad visual")
    print("=" * 60)


def _select_operation_mode():
    """
    Select operation mode (real glasses vs mock).

    Returns:
        tuple: (use_mock, mock_mode, mock_source)
    """
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

    return use_mock, mock_mode, mock_source


def _select_dashboard_config():
    """
    Select dashboard configuration.

    Web dashboard is enabled by default for better compatibility.

    Returns:
        tuple: (enable_dashboard, dashboard_type)
    """
    # Web dashboard enabled by default - no prompts needed
    enable_dashboard = True
    dashboard_type = "web"
    print("[MAIN] Web dashboard habilitado por defecto (http://localhost:5000)")

    return enable_dashboard, dashboard_type


def _initialize_components(use_mock, mock_mode, mock_source, enable_dashboard, dashboard_type):
    """
    Initialize all system components (telemetry, observer, coordinator, presentation).

    Args:
        use_mock: Whether to use mock observer instead of real hardware
        mock_mode: Mock mode type ('synthetic', 'video', 'static')
        mock_source: Source path for video/static mock modes
        enable_dashboard: Whether to enable dashboard UI
        dashboard_type: Type of dashboard ('opencv', 'rerun', 'web')

    Returns:
        dict: Dictionary containing all initialized components
    """
    print("\n🔧 Inicializando componentes...")

    # 0. Initialize telemetry FIRST
    print("  📊 Inicializando AsyncTelemetryLogger...")
    telemetry = AsyncTelemetryLogger()

    # 0a. Initialize ResourceMonitor for telemetry
    print("  💻 Inicializando ResourceMonitor...")
    from utils.resource_monitor import ResourceMonitor
    resource_monitor = ResourceMonitor(
        interval=2.0,  # Sample every 2 seconds
        callback=lambda data: telemetry.log_resources(data)
    )
    resource_monitor.start()
    print("  ✅ ResourceMonitor started (logging to telemetry)")

    # 0a2. Initialize MemoryProfiler
    print("  🧠 Inicializando MemoryProfiler...")
    from utils.memory_profiler import MemoryProfiler
    memory_profiler = MemoryProfiler(
        enabled=True,
        snapshot_interval=30.0  # Snapshot every 30s
    )
    print("  ✅ MemoryProfiler started (snapshot every 30s)")

    # 0b. Initialize depth logger with same session
    print("  🔍 Inicializando DepthLogger...")
    init_depth_logger(session_dir=str(telemetry.output_dir))

    # 1. Observer Setup - Real or Mock
    device_manager = None
    if use_mock:
        print(f"  🤖 Inicializando MockObserver (modo: {mock_mode})...")
        observer_kwargs = {
            'mode': mock_mode,
            'fps': 60,
            'resolution': (Config.ARIA_RGB_WIDTH, Config.ARIA_RGB_HEIGHT),
        }
        if mock_mode == 'video':
            observer_kwargs['video_path'] = mock_source
        elif mock_mode == 'static':
            observer_kwargs['image_path'] = mock_source

        observer = MockObserver(**observer_kwargs)
        observer.start()
        print("  ✅ MockObserver iniciado - desarrollo sin hardware")
    else:
        # Real mode with Aria glasses
        print("  📱 Conectando con Aria glasses...")
        device_manager = DeviceManager()
        device_manager.connect()
        rgb_calib, slam1_calib, slam2_calib = device_manager.start_streaming()

        print("  👁️ Inicializando AriaObserver (SDK only)...")
        observer = Observer(
            rgb_calib=rgb_calib,
            slam1_calib=slam1_calib,
            slam2_calib=slam2_calib
        )
        device_manager.register_observer(observer)
        device_manager.subscribe()
        print("  ✅ AriaObserver conectado con calibraciones RGB + SLAM")

    # 0b. Initialize NavigationLogger BEFORE coordinator (with same session_dir)
    print("  📝 Inicializando NavigationLogger...")
    from core.telemetry.loggers.navigation_logger import get_navigation_logger
    nav_logger = get_navigation_logger(session_dir=telemetry.get_session_dir())
    print(f"  ✅ NavigationLogger ready (logs → {nav_logger.log_dir})")

    # 3. Navigation Coordinator Setup (Pipeline only)
    print("  🧭 Inicializando Coordinator (Navigation Pipeline)...")
    builder = Builder()
    coordinator = builder.build_full_system(
        enable_dashboard=False,
        telemetry=telemetry,
    )  # Without own dashboard

    # 4. Presentation Manager Setup (UI only)
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

    return {
        'device_manager': device_manager,
        'observer': observer,
        'coordinator': coordinator,
        'presentation': presentation,
        'telemetry': telemetry,
        'resource_monitor': resource_monitor,
        'memory_profiler': memory_profiler,
    }


def _run_processing_loop(ctrl_handler, observer, coordinator, presentation, telemetry, memory_profiler):
    """
    Main processing loop that handles frame acquisition and processing.

    Args:
        ctrl_handler: Clean exit handler
        observer: Observer component (real or mock)
        coordinator: Navigation coordinator
        presentation: Presentation manager
        telemetry: Telemetry logger
        memory_profiler: Memory profiler

    Returns:
        int: Number of frames processed
    """
    frames_processed = 0
    last_stats_print = time.time()
    fps_start_time = time.time()
    latency_history = []

    while not ctrl_handler.should_stop:
        try:
            # Frame start timestamp
            frame_start_time = time.time()
            t0 = frame_start_time

            # Get data from Observer (SDK only)
            frame = observer.get_latest_frame('rgb')
            t1 = time.time()
            # Get SLAM frames and events
            slam1_frame = observer.get_latest_frame('slam1')
            slam2_frame = observer.get_latest_frame('slam2')
            t2 = time.time()
            slam_events = coordinator.get_slam_events()
            motion_data = observer.get_motion_state()
            motion_state = motion_data.get('state', 'unknown') if motion_data else 'unknown'
            t3 = time.time()

        except KeyboardInterrupt:
            print("\n[INFO] ⌨️ Ctrl+C detected in main loop, stopping...")
            break
        except Exception as e:
            print(f"[WARN] Error obtaining frames: {e}")
            time.sleep(0.1)
            continue

        try:
            if frame is not None:
                frames_processed += 1

                # Build frames_dict for Phase 2 multiproc
                frames_dict = {
                    'rgb': frame,
                    'slam1': slam1_frame,
                    'slam2': slam2_frame,
                }
                t4 = time.time()

                # Process with Coordinator (Pipeline only)
                processed_frame = coordinator.process_frame(frame, motion_state, frames_dict=frames_dict)
                t5 = time.time()

                peripheral_config = load_peripheral_vision_config()
                if frames_processed % peripheral_config.frame_skip == 0:
                    if hasattr(coordinator, 'handle_slam_frames'):
                        coordinator.handle_slam_frames(slam1_frame, slam2_frame)
                t6 = time.time()

                # Only get depth map if enabled
                depth_map = coordinator.get_latest_depth_map() if Config.DEPTH_ENABLED else None
                slam_events = coordinator.get_slam_events() if hasattr(coordinator, 'get_slam_events') else None
                t7 = time.time()

                # Log RGB detections
                current_detections = coordinator.get_current_detections()
                if current_detections:
                    telemetry.log_detections_batch(
                        frame_number=frames_processed,
                        source="rgb",
                        detections=current_detections
                    )

                # RENDER: Draw SLAM detections on their frames
                slam1_rendered = slam1_frame
                slam2_rendered = slam2_frame
                if slam_events and coordinator.frame_renderer:
                    if slam_events.get('slam1') and slam1_frame is not None:
                        slam1_rendered = coordinator.frame_renderer.draw_slam_detections(
                            slam1_frame, slam_events['slam1'], color=(0, 165, 255))  # Orange
                    if slam_events.get('slam2') and slam2_frame is not None:
                        slam2_rendered = coordinator.frame_renderer.draw_slam_detections(
                            slam2_frame, slam_events['slam2'], color=(255, 128, 0))  # Yellow
                t8 = time.time()

                # Update UI with PresentationManager (UI only)
                key = presentation.update_display(
                    frame=processed_frame,
                    detections=current_detections,
                    motion_state=motion_state,
                    coordinator_stats=coordinator.get_status(),
                    depth_map=depth_map,
                    slam1_frame=slam1_rendered,
                    slam2_frame=slam2_rendered,
                    slam_events=slam_events
                )
                t9 = time.time()

                # Handle UI Events
                if key == 'q':
                    print("\n[INFO] 'q' detected, closing application...")
                    break
                elif key == 't':
                    print("[INFO] Testing audio system...")
                    coordinator.test_audio()
                    presentation.log_audio_command("Test del sistema", 5)

                # Log detailed timing breakdown
                timing_ms = {
                    'get_rgb': (t1 - t0) * 1000,
                    'get_slam': (t2 - t1) * 1000,
                    'get_motion': (t3 - t2) * 1000,
                    'build_dict': (t4 - t3) * 1000,
                    'process_frame': (t5 - t4) * 1000,
                    'slam_handling': (t6 - t5) * 1000,
                    'get_results': (t7 - t6) * 1000,
                    'render_slam': (t8 - t7) * 1000,
                    'update_display': (t9 - t8) * 1000,
                    'total': (t9 - t0) * 1000
                }

                # Update telemetry with timing breakdown
                frame_end_time = t9
                frame_latency_ms = (frame_end_time - frame_start_time) * 1000
                elapsed_total = frame_end_time - fps_start_time
                current_fps = frames_processed / elapsed_total if elapsed_total > 0 else 0

                telemetry.log_frame_performance(
                    frame_number=frames_processed,
                    fps=current_fps,
                    latency_ms=frame_latency_ms,
                    timing_breakdown=timing_ms
                )

                # Log every 30 frames
                if frames_processed % 30 == 0:
                    avg_latency = sum(latency_history) / len(latency_history) if latency_history else 0
                    print(f"[TIMING] Frame {frames_processed} | Avg latency: {avg_latency:.1f}ms")
                    print(f"  Observer: RGB={timing_ms['get_rgb']:.2f}ms SLAM={timing_ms['get_slam']:.2f}ms Motion={timing_ms['get_motion']:.2f}ms")
                    print(f"  Pipeline: process={timing_ms['process_frame']:.2f}ms")
                    print(f"  Post: slam_handle={timing_ms['slam_handling']:.2f}ms results={timing_ms['get_results']:.2f}ms")
                    print(f"  Render: slam={timing_ms['render_slam']:.2f}ms display={timing_ms['update_display']:.2f}ms")
                    print(f"  TOTAL: {timing_ms['total']:.2f}ms ({1000/timing_ms['total']:.1f} FPS possible)")

                # MEMORY: Check if should take snapshot
                if memory_profiler.maybe_take_snapshot():
                    print(f"[MEMORY] Snapshot taken at frame {frames_processed}")

                # MEMORY: Release frame references to help GC
                del frame, processed_frame, depth_map, slam1_frame, slam2_frame
                frame = processed_frame = depth_map = slam1_frame = slam2_frame = None

            # Periodic statistics
            current_time = time.time()
            if current_time - last_stats_print > 10.0:  # Every 10 seconds
                print(f"[STATUS] Frames: {frames_processed}, Motion: {motion_state}")
                # Performance stats in console
                if telemetry:
                    perf = telemetry.get_performance_summary()
                    print(f"[PERF] FPS: {perf.get('avg_fps', 0):.1f}, Latencia: {perf.get('avg_latency_ms', 0):.0f}ms")

                # MEMORY: Force garbage collection periodically
                gc.collect()
                print(f"[MEMORY] GC executed (freed memory)")

                last_stats_print = current_time

        except KeyboardInterrupt:
            print("\n[INFO] ⌨️ Ctrl+C detected in processing loop, stopping...")
            break
        except Exception as e:
            print(f"[WARN] Error in processing loop: {e}")
            time.sleep(0.1)  # Avoid error spam

    return frames_processed


def _print_final_stats(frames_processed, observer, coordinator, presentation):
    """Print final session statistics."""
    print(f"\n📊 Sesión completada: {frames_processed} frames procesados")
    print("\n📈 Estadísticas finales:")
    if observer:
        observer.print_stats()
    if coordinator:
        coordinator.print_stats()
    if presentation:
        presentation.print_ui_stats()


def _cleanup_resources(resource_monitor, memory_profiler, telemetry,
                       coordinator, presentation, observer, device_manager):
    """
    Clean up all system resources in proper order.

    Args:
        resource_monitor: Resource monitor instance
        memory_profiler: Memory profiler instance
        telemetry: Telemetry logger instance
        coordinator: Coordinator instance
        presentation: Presentation manager instance
        observer: Observer instance
        device_manager: Device manager instance
    """
    print("\n🧹 Iniciando limpieza de recursos...")

    # Stop ResourceMonitor
    try:
        if resource_monitor:
            print("  💻 Deteniendo ResourceMonitor...")
            resource_monitor.stop()
    except Exception as e:
        print(f"  ⚠️ Error deteniendo ResourceMonitor: {e}")

    # Finalize MemoryProfiler
    try:
        if memory_profiler:
            print("  🧠 Finalizando MemoryProfiler...")
            memory_profiler.print_summary()
            if telemetry:
                memory_profiler.log_to_file(telemetry.output_dir)
            memory_profiler.stop()
    except Exception as e:
        print(f"  ⚠️ Error finalizando MemoryProfiler: {e}")

    # Finalize telemetry FIRST
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


def main():
    """
    Main entry point with clean separated architecture.

    Flow:
    1. Initialization of separated components
    2. Aria SDK and streaming setup
    3. Main processing loop
    4. Ordered cleanup
    """
    _print_welcome()

    # Setup clean exit handler
    ctrl_handler = CtrlCHandler()

    # Select operation mode
    use_mock, mock_mode, mock_source = _select_operation_mode()

    # Select dashboard configuration
    enable_dashboard, dashboard_type = _select_dashboard_config()
    
    # Component initialization
    device_manager = None
    observer = None
    coordinator = None
    presentation = None
    telemetry = None
    resource_monitor = None
    memory_profiler = None

    try:
        components = _initialize_components(
            use_mock, mock_mode, mock_source,
            enable_dashboard, dashboard_type
        )
        device_manager = components['device_manager']
        observer = components['observer']
        coordinator = components['coordinator']
        presentation = components['presentation']
        telemetry = components['telemetry']
        resource_monitor = components['resource_monitor']
        memory_profiler = components['memory_profiler']

        # Run main processing loop
        frames_processed = _run_processing_loop(
            ctrl_handler, observer, coordinator,
            presentation, telemetry, memory_profiler
        )

        # Print final statistics
        _print_final_stats(frames_processed, observer, coordinator, presentation)
        
    except KeyboardInterrupt:
        print("\n[INFO] ⌨️ Interrupción de teclado detectada")

    except Exception as e:
        print(f"\n[ERROR] ❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

    finally:
        _cleanup_resources(
            resource_monitor, memory_profiler, telemetry,
            coordinator, presentation, observer, device_manager
        )


def main_debug():
    """
    Debug version with mock components for testing without hardware.
    """
    print("🧪 DEBUG MODE - Testing sin hardware Aria")

    import os
    import numpy as np
    from time import sleep

    # Mock observer for testing
    class MockObserver:
        def __init__(self):
            self.frame_count = 0
        
        def get_latest_frame(self, camera='rgb'):
            # Generate synthetic frame
            self.frame_count += 1
            frame = np.random.randint(0, 255, (Config.TEST_FRAME_HEIGHT, Config.TEST_FRAME_WIDTH, 3), dtype=np.uint8)

            # Add text indicating it's mock
            cv2.putText(frame, f"MOCK FRAME {self.frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return frame

        def get_motion_state(self):
            # Alternate between stationary and walking
            state = "walking" if (self.frame_count // 30) % 2 else "stationary"
            return {'state': state, 'magnitude': 9.8}

        def print_stats(self):
            print(f"  MockObserver: {self.frame_count} frames generados")
        
        def stop(self):
            pass


    try:
        print("🔧 Inicializando componentes mock...")

        # Coordinator and presentation only for testing
        builder = Builder()
        coordinator = builder.build_full_system(
            enable_dashboard=False,
            telemetry=None,
        )
        
        # Skip UI in multiprocessing mode to avoid Qt/OpenCV conflicts
        pipeline_config = load_pipeline_config()
        if pipeline_config.multiproc_enabled:
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

                # Simulate 30fps
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
    Hybrid version for Mac - ImageZMQ Sender only.
    For use with remote Jetson processing.
    """
    print("🌉 HYBRID MODE - Mac ImageZMQ Sender")
    print("Enviando frames al Jetson para procesamiento...")

    # This function will be implemented when we create the ImageZMQ sender
    # For now, placeholder for hybrid architecture
    print("⚠️ Función híbrida aún no implementada")
    print("💡 Usar main() normal para sistema local completo")


if __name__ == "__main__":
    import sys

    # Allow different execution modes
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
        # Normal mode
        main()
