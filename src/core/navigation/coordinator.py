#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
🎯 Coordinator Mejorado - TFM Navigation System
Basado en la versión original, con mejoras para soporte de motion detection
y compatibilidad con arquitectura híbrida

Fecha: Septiembre 2025
Versión: 1.1 - Enhanced with Motion Support + Hybrid Architecture Ready
"""

import numpy as np
import time
from enum import Enum
from typing import Optional, Dict, List, Any

from utils.config import Config

from core.navigation.navigation_decision_engine import (
    NavigationDecisionEngine,
)
from core.navigation.rgb_audio_router import RgbAudioRouter
from core.navigation.navigation_pipeline import NavigationPipeline
from core.navigation.slam_audio_router import SlamAudioRouter, SlamRoutingState

try:
    from core.vision.slam_detection_worker import (
        CameraSource,
        SlamDetectionEvent,
        SlamDetectionWorker,
    )
    from core.audio.navigation_audio_router import (
        NavigationAudioRouter,
        EventPriority,
    )
except Exception:
    CameraSource = Any  # type: ignore[assignment]
    SlamDetectionEvent = Any  # type: ignore[assignment]
    SlamDetectionWorker = Any  # type: ignore[assignment]
    NavigationAudioRouter = Any  # type: ignore[assignment]

    class _FallbackPriority(Enum):
        CRITICAL = 1
        HIGH = 2
        MEDIUM = 3
        LOW = 4

    EventPriority = _FallbackPriority  # type: ignore[assignment]


class Coordinator:
    """
    🎯 Coordinator que orquesta el flujo de datos entre módulos
    
    Este coordinator RECIBE dependencias ya creadas (no las crea él).
    Se enfoca únicamente en coordinar el flujo de procesamiento.
    
    Pipeline:
    Image → Enhancement → YOLO → Navigation → Audio → Rendering
    """
    
    def __init__(
        self,
        yolo_processor,
        audio_system,
        frame_renderer=None,
        image_enhancer=None,
        dashboard=None,
        audio_router: Optional[Any] = None,
        navigation_pipeline: Optional[NavigationPipeline] = None,
        decision_engine: Optional[NavigationDecisionEngine] = None,
        telemetry=None,
    ):
        """
        Inicializar coordinator con dependencias inyectadas
        
        Args:
            yolo_processor: Instancia ya configurada de YoloProcessor
            audio_system: Instancia ya configurada de AudioSystem
            frame_renderer: Instancia opcional de FrameRenderer
            image_enhancer: Instancia opcional de ImageEnhancer
            dashboard: Instancia opcional de Dashboard
        """
        # Dependencias inyectadas
        self.audio_system = audio_system
        self.frame_renderer = frame_renderer
        self.dashboard = dashboard
        self.audio_router: Optional[Any] = audio_router
        self.telemetry = telemetry

        self.pipeline = navigation_pipeline or NavigationPipeline(
            yolo_processor=yolo_processor,
            image_enhancer=image_enhancer,
        )
        self.decision_engine = decision_engine or NavigationDecisionEngine()

        # Compatibilidad hacia atrás
        self.yolo_processor = self.pipeline.yolo_processor
        self.image_enhancer = self.pipeline.image_enhancer
        self.depth_estimator = self.pipeline.depth_estimator

        # Estado interno
        self.frames_processed = 0
        self.last_announcement_time = 0.0
        self.current_detections: List[Dict[str, Any]] = []

        self.profile_enabled = getattr(Config, 'PROFILE_PIPELINE', False)
        self.profile_window = max(1, getattr(Config, 'PROFILE_WINDOW_FRAMES', 30))
        self._profile_acc = {
            'enhance': 0.0,
            'depth': 0.0,
            'yolo': 0.0,
            'nav_audio': 0.0,
            'render': 0.0,
            'total': 0.0,
        }
        self._profile_frames = 0

        # Peripheral SLAM support
        self.peripheral_enabled = False
        self.slam_state = SlamRoutingState(
            workers={},
            frame_counters={},
            last_indices={},
            latest_events={},
        )
        # Capa simétrica a SlamAudioRouter: formatea eventos RGB antes de encolarlos.
        self.slam_router = SlamAudioRouter(self.audio_router)
        self.rgb_router = RgbAudioRouter(audio_system, self.audio_router, self.slam_router)
        if self.audio_router and not getattr(self.audio_router, "_running", False):
            self.audio_router.start()

        print(f"✅ Coordinator inicializado")
        print(f"  - YOLO: {type(self.yolo_processor).__name__}")
        print(f"  - Audio: {type(self.audio_system).__name__}")
        print(f"  - Frame Renderer: {type(self.frame_renderer).__name__ if self.frame_renderer else 'None'}")
        print(f"  - Image Enhancer: {type(self.image_enhancer).__name__ if self.image_enhancer else 'None'}")
        print(f"  - Dashboard: {type(self.dashboard).__name__ if self.dashboard else 'None'}")
    
    def process_frame(self, frame: np.ndarray, motion_state: str = "stationary") -> np.ndarray:
        """
        🔄 Procesar frame completo a través del pipeline
        
        Args:
            frame: Frame BGR de entrada
            motion_state: Estado de movimiento ("stationary", "walking")
            
        Returns:
            np.ndarray: Frame procesado con anotaciones
        """
        self.frames_processed += 1

        total_start = time.perf_counter() if self.profile_enabled else 0.0

        pipeline_result = self.pipeline.process(frame, profile=self.profile_enabled)
        processed_frame = pipeline_result.frame
        detections = pipeline_result.detections
        depth_map = pipeline_result.depth_map

        if self.profile_enabled:
            timings = pipeline_result.timings
            self._profile_acc['enhance'] += timings.get('enhance', 0.0)
            self._profile_acc['depth'] += timings.get('depth', 0.0)
            self._profile_acc['yolo'] += timings.get('yolo', 0.0)

        # 3. Navigation Analysis
        nav_start = time.perf_counter() if self.profile_enabled else 0.0
        navigation_objects = self.decision_engine.analyze(detections)

        # 4. Audio Commands (con motion-aware cooldown)
        decision_candidate = self.decision_engine.evaluate(navigation_objects, motion_state)
        if decision_candidate is not None:
            self.rgb_router.route(decision_candidate)
        if self.profile_enabled:
            self._profile_acc['nav_audio'] += time.perf_counter() - nav_start

        self.last_announcement_time = self.decision_engine.last_announcement_time

        # 5. Frame Rendering (si está disponible)
        render_start = time.perf_counter() if self.profile_enabled else 0.0
        annotated_frame = processed_frame
        if self.frame_renderer is not None:
            try:
                annotated_frame = self.frame_renderer.draw_navigation_overlay(
                    processed_frame, detections, self.audio_system, depth_map
                )
            except Exception as err:
                print(f"[WARN] Frame rendering skipped: {err}")
                annotated_frame = processed_frame
        if self.profile_enabled:
            self._profile_acc['render'] += time.perf_counter() - render_start

        # 6. Dashboard Update (si está disponible)
        if self.dashboard:
            try:
                self.dashboard.update_detections(detections)
                self.dashboard.update_navigation_status(navigation_objects)
            except Exception as err:
                print(f"[WARN] Dashboard update skipped: {err}")

        # Guardar estado actual
        self.current_detections = detections

        if self.profile_enabled:
            self._profile_acc['total'] += time.perf_counter() - total_start
            self._profile_frames += 1
            if self._profile_frames >= self.profile_window:
                self._log_profile_metrics()

        return annotated_frame

    # ------------------------------------------------------------------
    # Peripheral vision (SLAM) integration
    # ------------------------------------------------------------------

    def attach_peripheral_system(
        self,
        slam_workers: Dict[Any, Any],
        audio_router: Optional[Any] = None,
    ) -> None:
        if CameraSource is None:
            print("[WARN] Peripheral system not available (missing dependencies)")
            return

        self.peripheral_enabled = True
        self.slam_state.workers = slam_workers
        if audio_router is not None:
            self.audio_router = audio_router
            self.slam_router = SlamAudioRouter(audio_router)
            self.rgb_router.set_audio_router(self.audio_router)
            self.rgb_router.set_slam_router(self.slam_router)

        for source, worker in slam_workers.items():
            worker.start()
            self.slam_state.frame_counters[source] = 0
            self.slam_state.last_indices[source] = -1
            self.slam_state.latest_events[source] = []

        if self.audio_router and not getattr(self.audio_router, "_running", False):
            self.audio_router.start()

        print("[PERIPHERAL] SLAM workers attached")

    def handle_slam_frames(
        self,
        slam1_frame: Optional[np.ndarray] = None,
        slam2_frame: Optional[np.ndarray] = None,
    ) -> None:
        if not self.peripheral_enabled or CameraSource is None:
            return

        slam1_source = getattr(CameraSource, 'SLAM1', None)
        slam2_source = getattr(CameraSource, 'SLAM2', None)

        if slam1_frame is not None and slam1_source in self.slam_state.workers:
            self.slam_router.submit_and_route(self.slam_state, slam1_source, slam1_frame)

        if slam2_frame is not None and slam2_source in self.slam_state.workers:
            self.slam_router.submit_and_route(self.slam_state, slam2_source, slam2_frame)

    def get_slam_events(self) -> Dict[str, List[dict]]:
        if not self.peripheral_enabled:
            return {}

        event_dict: Dict[str, List[dict]] = {}
        for source, events in self.slam_state.latest_events.items():
            simplified = []
            for event in events:
                simplified.append(
                    {
                        'bbox': event.bbox,
                        'name': event.object_name,
                        'confidence': event.confidence,
                        'zone': event.zone,
                        'distance': event.distance,
                        'message': self.slam_router.describe_event(event),
                    }
                )
            event_dict[source.value] = simplified
        return event_dict

    def _log_profile_metrics(self) -> None:
        frame_count = max(1, self._profile_frames)
        averaged = {
            key: (value / frame_count) * 1000.0 for key, value in self._profile_acc.items()
        }
        msg = "[PROFILE] enhance={enhance:.1f}ms | depth={depth:.1f}ms | yolo={yolo:.1f}ms | nav+audio={nav_audio:.1f}ms | render={render:.1f}ms | total={total:.1f}ms".format(**averaged)
        print(msg)
        
        # Log to telemetry if available
        if hasattr(self, 'telemetry') and self.telemetry:
            try:
                self.telemetry.log_system_event({
                    'event_type': 'profile_metrics',
                    'enhance_ms': averaged['enhance'],
                    'depth_ms': averaged['depth'],
                    'yolo_ms': averaged['yolo'],
                    'nav_audio_ms': averaged['nav_audio'],
                    'render_ms': averaged['render'],
                    'total_ms': averaged['total'],
                    'frames_averaged': frame_count
                })
            except Exception:
                pass
        
        for key in self._profile_acc:
            self._profile_acc[key] = 0.0
        self._profile_frames = 0

    def get_latest_depth_map(self) -> Optional[np.ndarray]:
        """Obtener el último depth map estimado"""
        return self.pipeline.get_latest_depth_map()
    
    def get_status(self):
        """
        📊 Obtener estado actual del coordinator
        
        Returns:
            dict: Estado con métricas y información
        """
        audio_queue_size = 0
        beep_stats = {}
        try:
            if hasattr(self.audio_system, 'get_queue_size'):
                audio_queue_size = self.audio_system.get_queue_size()
            elif hasattr(self.audio_system, 'audio_queue'):
                audio_queue_size = len(self.audio_system.audio_queue)
            
            # Get beep statistics if available
            if hasattr(self.audio_system, 'get_beep_stats'):
                beep_stats = self.audio_system.get_beep_stats()
        except:
            pass
        
        status = {
            'frames_processed': self.frames_processed,
            'current_detections_count': len(self.current_detections),
            'audio_queue_size': audio_queue_size,
            'has_dashboard': self.dashboard is not None,
            'slam_events': (
                sum(len(events) for events in self.slam_state.latest_events.values())
                if self.peripheral_enabled
                else 0
            ),
            'has_frame_renderer': self.frame_renderer is not None,
            'has_image_enhancer': self.image_enhancer is not None,
            'last_announcement': self.last_announcement_time
        }
        
        # Add beep statistics
        status.update(beep_stats)
        
        return status
    
    def get_current_detections(self):
        """
        🎯 Obtener detecciones actuales (para compatibilidad con Observer)
        
        Returns:
            list: Lista de detecciones actuales
        """
        return self.current_detections.copy()
    
    def test_audio(self):
        """
        🔊 Test del sistema de audio
        """
        try:
            if hasattr(self.audio_system, 'speak_force'):
                self.audio_system.speak_force("Navigation system test")
            else:
                self.audio_system.speak_async("Navigation system test")
            print("[COORDINATOR] Audio test emitted")
        except Exception as e:
            print(f"[COORDINATOR] Audio test failed: {e}")
    
    def print_stats(self):
        """
        📈 Imprimir estadísticas del coordinator
        """
        status = self.get_status()
        
        print(f"\n[COORDINATOR STATS]")
        print(f"  Frames procesados: {status['frames_processed']}")
        print(f"  Detecciones actuales: {status['current_detections_count']}")
        print(f"  Audio queue size: {status['audio_queue_size']}")
        print(f"  Dashboard: {'✅' if status['has_dashboard'] else '❌'}")
        print(f"  Frame Renderer: {'✅' if status['has_frame_renderer'] else '❌'}")
        print(f"  Image Enhancer: {'✅' if status['has_image_enhancer'] else '❌'}")
        
        # Mostrar últimas detecciones
        if self.current_detections:
            print(f"  Últimas detecciones:")
            for det in self.current_detections[:3]:  # Top 3
                name = det.get('name', 'unknown')
                zone = det.get('zone', 'unknown')
                conf = det.get('confidence', 0)
                print(f"    - {name} en {zone} (conf: {conf:.2f})")
    
    def cleanup(self):
        """
        🧹 Limpieza de recursos
        """
        print("🧹 Limpiando Coordinator...")

        if self.peripheral_enabled:
            try:
                if self.audio_router:
                    self.audio_router.stop()
            except Exception as err:
                print(f"  ⚠️ Peripheral audio router cleanup error: {err}")

            for source, worker in self.slam_state.workers.items():
                try:
                    worker.stop()
                except Exception as err:
                    print(f"  ⚠️ SLAM worker {source.value} cleanup error: {err}")
            self.slam_state = SlamRoutingState(
                workers={},
                frame_counters={},
                last_indices={},
                latest_events={},
            )

        try:
            if self.audio_system:
                if hasattr(self.audio_system, 'cleanup'):
                    self.audio_system.cleanup()
                elif hasattr(self.audio_system, 'close'):
                    self.audio_system.close()
                print("  ✅ Audio system cleanup")
        except Exception as e:
            print(f"  ⚠️ Audio cleanup error: {e}")
        
        try:
            if self.dashboard:
                if hasattr(self.dashboard, 'cleanup'):
                    self.dashboard.cleanup()
                elif hasattr(self.dashboard, 'shutdown'):
                    self.dashboard.shutdown()
                print("  ✅ Dashboard cleanup")
        except Exception as e:
            print(f"  ⚠️ Dashboard cleanup error: {e}")
        
        # Frame renderer y image enhancer normalmente no necesitan cleanup
        
        # Reset estado
        self.current_detections = []
        
        print("✅ Coordinator limpiado")
