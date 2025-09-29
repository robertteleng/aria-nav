#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Coordinator Mejorado - TFM Navigation System
Basado en la versión original, con mejoras para soporte de motion detection
y compatibilidad con arquitectura híbrida

Fecha: Septiembre 2025
Versión: 1.1 - Enhanced with Motion Support + Hybrid Architecture Ready
"""

import numpy as np
import cv2
import time
from enum import Enum
from typing import Optional, Dict, List, Any

from utils.config import Config

try:
    from core.vision.depth_estimator import DepthEstimator
except Exception:
    DepthEstimator = None

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
    CameraSource = None  # type: ignore[assignment]
    SlamDetectionWorker = None  # type: ignore[assignment]
    NavigationAudioRouter = None  # type: ignore[assignment]

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
        audio_router: Optional[NavigationAudioRouter] = None,
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
        self.yolo_processor = yolo_processor
        self.audio_system = audio_system
        self.frame_renderer = frame_renderer
        self.image_enhancer = image_enhancer  
        self.dashboard = dashboard
        self.audio_router: Optional[NavigationAudioRouter] = audio_router
        
        # Estado interno del coordinator
        self.frames_processed = 0
        self.last_announcement_time = 0
        self.current_detections = []
        
        # Configuración de zonas espaciales (pixels)
        self.zones = {
            'left': (0, 213),
            'center': (213, 426), 
            'right': (426, 640)
        }
        
        # Prioridades de objetos para navegación
        self.object_priorities = {
            'person': {'priority': 10, 'spanish': 'persona'},
            'car': {'priority': 8, 'spanish': 'coche'},
            'truck': {'priority': 8, 'spanish': 'camión'},
            'bus': {'priority': 8, 'spanish': 'autobús'},
            'bicycle': {'priority': 7, 'spanish': 'bicicleta'},
            'motorcycle': {'priority': 7, 'spanish': 'motocicleta'},
            'stop sign': {'priority': 9, 'spanish': 'señal de stop'},
            'traffic light': {'priority': 6, 'spanish': 'semáforo'},
            'chair': {'priority': 3, 'spanish': 'silla'},
            'door': {'priority': 4, 'spanish': 'puerta'},
            'stairs': {'priority': 5, 'spanish': 'escaleras'}
        }
        
        self.depth_estimator = None
        self.depth_frame_skip = max(1, getattr(Config, 'DEPTH_FRAME_SKIP', 1))
        self.latest_depth_map = None
        if getattr(Config, 'DEPTH_ENABLED', False) and DepthEstimator is not None:
            try:
                self.depth_estimator = DepthEstimator()
                # Si el modelo no está disponible, mantener referencia pero avisar
                if getattr(self.depth_estimator, 'model', None) is None:
                    print("[WARN] Depth estimator initialized without model (disabled)")
            except Exception as err:
                print(f"[WARN] Depth estimator init failed: {err}")
                self.depth_estimator = None

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
        self.slam_workers: Dict[CameraSource, SlamDetectionWorker] = {}
        self.slam_frame_counters: Dict[CameraSource, int] = {}
        self._last_slam_indices: Dict[CameraSource, int] = {}
        self.latest_slam_events: Dict[CameraSource, List[SlamDetectionEvent]] = {}
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

        # 0. Optional low-light enhancement
        enhance_start = time.perf_counter() if self.profile_enabled else 0.0
        processed_frame = frame
        if self.image_enhancer is not None:
            try:
                processed_frame = self.image_enhancer.enhance_frame(frame)
            except Exception as err:
                print(f"[WARN] Image enhancement skipped: {err}")
                processed_frame = frame
        if self.profile_enabled:
            self._profile_acc['enhance'] += time.perf_counter() - enhance_start

        # 1. Depth estimation (opcional)
        depth_start = time.perf_counter() if self.profile_enabled else 0.0
        depth_map = None
        if self.depth_estimator is not None and getattr(self.depth_estimator, 'model', None) is not None:
            try:
                if self.frames_processed % self.depth_frame_skip == 0:
                    depth_candidate = self.depth_estimator.estimate_depth(processed_frame)
                    if depth_candidate is not None:
                        self.latest_depth_map = depth_candidate
                depth_map = self.latest_depth_map
            except Exception as err:
                print(f"[WARN] Depth estimation skipped: {err}")
        if self.profile_enabled:
            self._profile_acc['depth'] += time.perf_counter() - depth_start

        # 2. YOLO Detection
        yolo_start = time.perf_counter() if self.profile_enabled else 0.0
        detections = self.yolo_processor.process_frame(processed_frame, depth_map)
        if self.profile_enabled:
            self._profile_acc['yolo'] += time.perf_counter() - yolo_start

        # 3. Navigation Analysis
        nav_start = time.perf_counter() if self.profile_enabled else 0.0
        navigation_objects = self._analyze_navigation_objects(detections)

        # 4. Audio Commands (con motion-aware cooldown)
        self._generate_audio_commands(navigation_objects, motion_state)
        if self.profile_enabled:
            self._profile_acc['nav_audio'] += time.perf_counter() - nav_start

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
        slam_workers: Dict[CameraSource, SlamDetectionWorker],
        audio_router: Optional[NavigationAudioRouter] = None,
    ) -> None:
        if CameraSource is None:
            print("[WARN] Peripheral system not available (missing dependencies)")
            return

        self.peripheral_enabled = True
        self.slam_workers = slam_workers
        if audio_router is not None:
            self.audio_router = audio_router

        for source, worker in slam_workers.items():
            worker.start()
            self.slam_frame_counters[source] = 0
            self._last_slam_indices[source] = -1
            self.latest_slam_events[source] = []

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

        if slam1_frame is not None and CameraSource.SLAM1 in self.slam_workers:
            self._submit_and_route(CameraSource.SLAM1, slam1_frame)

        if slam2_frame is not None and CameraSource.SLAM2 in self.slam_workers:
            self._submit_and_route(CameraSource.SLAM2, slam2_frame)

    def _submit_and_route(self, source: CameraSource, frame: np.ndarray) -> None:
        worker = self.slam_workers[source]
        self.slam_frame_counters[source] += 1
        worker.submit(frame, self.slam_frame_counters[source])

        events = worker.latest_events()
        if not events:
            # Limpiar eventos previos cuando no hay nuevas detecciones para evitar overlays persistentes
            self.latest_slam_events[source] = []
            return

        latest_index = events[0].frame_index
        if latest_index == self._last_slam_indices.get(source, -1):
            return

        self._last_slam_indices[source] = latest_index
        self.latest_slam_events[source] = events

        if self.audio_router:
            for event in events:
                priority = self._determine_slam_priority(event)
                message = self._build_slam_message(event)
                self.audio_router.enqueue_from_slam(event, message, priority)

    def _determine_slam_priority(self, event: SlamDetectionEvent) -> EventPriority:
        if EventPriority is None:
            return EventPriority.HIGH  # type: ignore[return-value]

        distance = event.distance
        name = event.object_name

        if name in {"car", "truck", "bus", "motorcycle"} and distance in {"close", "very_close"}:
            return EventPriority.CRITICAL
        if name == "person" and distance in {"close", "very_close"}:
            return EventPriority.HIGH
        if name in {"bicycle", "motorbike"}:
            return EventPriority.HIGH
        return EventPriority.MEDIUM

    def _build_slam_message(self, event: SlamDetectionEvent) -> str:
        zone_map = {
            "far_left": "extrema izquierda",
            "left": "izquierda lateral",
            "right": "derecha lateral",
            "far_right": "extrema derecha",
        }
        object_map = {
            "person": "persona",
            "car": "coche",
            "truck": "camión",
            "bus": "autobús",
            "bicycle": "bicicleta",
            "motorcycle": "moto",
        }

        zone_text = zone_map.get(event.zone, event.zone)
        name = object_map.get(event.object_name, event.object_name)
        distance = event.distance

        if distance in {"close", "very_close"} and event.object_name in {"car", "truck", "bus"}:
            return f"Cuidado: {name} acercándose por la {zone_text}"
        if event.object_name == "person" and distance in {"close", "very_close"}:
            return f"Persona cerca en la {zone_text}"
        if distance not in {"", "unknown"}:
            return f"{name} {distance} en la {zone_text}"
        return f"{name} en la {zone_text}"

    def get_slam_events(self) -> Dict[str, List[dict]]:
        if not self.peripheral_enabled:
            return {}

        event_dict: Dict[str, List[dict]] = {}
        for source, events in self.latest_slam_events.items():
            simplified = []
            for event in events:
                simplified.append(
                    {
                        'bbox': event.bbox,
                        'name': event.object_name,
                        'confidence': event.confidence,
                        'zone': event.zone,
                        'distance': event.distance,
                        'message': self._build_slam_message(event),
                    }
                )
            event_dict[source.value] = simplified
        return event_dict

    def _log_profile_metrics(self) -> None:
        frame_count = max(1, self._profile_frames)
        averaged = {
            key: (value / frame_count) * 1000.0 for key, value in self._profile_acc.items()
        }
        print(
            "[PROFILE] enhance={enhance:.1f}ms | depth={depth:.1f}ms | yolo={yolo:.1f}ms | "
            "nav+audio={nav_audio:.1f}ms | render={render:.1f}ms | total={total:.1f}ms".format(**averaged)
        )
        for key in self._profile_acc:
            self._profile_acc[key] = 0.0
        self._profile_frames = 0

    def get_latest_depth_map(self) -> Optional[np.ndarray]:
        """Obtener el último depth map estimado"""
        return self.latest_depth_map
    
    def _analyze_navigation_objects(self, detections):
        """
        🧠 Analizar detecciones para navegación
        
        Args:
            detections: Lista de objetos detectados por YOLO
            
        Returns:
            list: Objetos relevantes con metadatos de navegación
        """
        navigation_objects = []
        
        for detection in detections:
            class_name = detection['name']
            
            # Solo procesar objetos relevantes para navegación
            if class_name not in self.object_priorities:
                continue
            
            # Calcular zona espacial
            bbox = detection['bbox']
            x_center = bbox[0] + bbox[2] / 2
            zone = self._calculate_zone(x_center)
            
            # Estimar distancia relativa
            distance_category = self._estimate_distance(bbox, class_name)
            
            # Calcular prioridad final
            base_priority = self.object_priorities[class_name]['priority']
            final_priority = self._calculate_final_priority(
                base_priority, zone, distance_category
            )
            
            navigation_obj = {
                'class': class_name,
                'spanish_name': self.object_priorities[class_name]['spanish'],
                'bbox': bbox,
                'confidence': detection['confidence'],
                'zone': zone,
                'distance': distance_category,
                'priority': final_priority,
                'original_priority': base_priority
            }
            
            navigation_objects.append(navigation_obj)
        
        # Ordenar por prioridad descendente
        navigation_objects.sort(key=lambda x: x['priority'], reverse=True)
        
        return navigation_objects
    
    def _calculate_zone(self, x_center):
        """
        📍 Calcular zona espacial de un objeto
        
        Args:
            x_center: Coordenada X del centro del objeto
            
        Returns:
            str: 'left', 'center', o 'right'
        """
        if x_center < self.zones['left'][1]:
            return 'left'
        elif x_center < self.zones['center'][1]:
            return 'center'
        else:
            return 'right'
    
    def _estimate_distance(self, bbox, class_name):
        """
        📏 Estimar distancia relativa basada en tamaño del bbox
        
        Args:
            bbox: Bounding box [x, y, w, h]
            class_name: Tipo de objeto
            
        Returns:
            str: 'cerca', 'medio', 'lejos'
        """
        height = bbox[3]
        
        # Umbrales específicos por tipo de objeto
        if class_name == 'person':
            if height > 200:
                return 'cerca'
            elif height > 100:
                return 'medio'
            else:
                return 'lejos'
        elif class_name in ['car', 'truck', 'bus']:
            if height > 150:
                return 'cerca'
            elif height > 75:
                return 'medio'
            else:
                return 'lejos'
        else:
            # Default para otros objetos
            if height > 100:
                return 'cerca'
            elif height > 50:
                return 'medio'
            else:
                return 'lejos'
    
    def _calculate_final_priority(self, base_priority, zone, distance):
        """
        ⚖️ Calcular prioridad final con modificadores
        
        Args:
            base_priority: Prioridad base del objeto
            zone: Zona espacial del objeto
            distance: Categoría de distancia
            
        Returns:
            float: Prioridad final modificada
        """
        priority = float(base_priority)
        
        # Modificador por distancia
        distance_multipliers = {
            'cerca': 2.0,
            'medio': 1.5,
            'lejos': 1.0
        }
        priority *= distance_multipliers.get(distance, 1.0)
        
        # Modificador por zona (centro es más importante)
        zone_multipliers = {
            'center': 1.3,
            'left': 1.0,
            'right': 1.0
        }
        priority *= zone_multipliers.get(zone, 1.0)
        
        return priority
    
    def _generate_audio_commands(self, navigation_objects, motion_state="stationary"):
        """
        🔊 Generar comandos de audio basados en objetos detectados
        MEJORADO: Ahora incluye motion-aware cooldown
        
        Args:
            navigation_objects: Lista de objetos ordenados por prioridad
            motion_state: Estado de movimiento del usuario
        """
        current_time = time.time()

        # Cooldown adaptativo según movimiento
        if motion_state == "walking":
            cooldown = 1.5
        else:
            cooldown = 3.0

        if current_time - self.last_announcement_time < cooldown:
            return

        if not navigation_objects:
            return

        top_object = navigation_objects[0]

        if top_object['priority'] < 8.0:
            return

        message = self._create_audio_message(top_object)
        metadata: Dict[str, Any] = {
            'class': top_object.get('name'),
            'spanish_name': top_object.get('spanish_name'),
            'priority': top_object.get('priority'),
            'zone': top_object.get('zone'),
            'distance': top_object.get('distance'),
            'motion_state': motion_state,
        }

        event_priority = self._map_priority_for_audio(top_object)

        self.audio_system.set_repeat_cooldown(cooldown)
        if hasattr(self.audio_system, 'set_announcement_cooldown'):
            self.audio_system.set_announcement_cooldown(max(0.0, cooldown * 0.5))

        if self.audio_router and hasattr(self.audio_router, 'enqueue_from_rgb'):
            if hasattr(self.audio_router, 'set_source_cooldown'):
                self.audio_router.set_source_cooldown('rgb', cooldown)
            self.audio_router.enqueue_from_rgb(
                message=message,
                priority=event_priority,
                metadata=metadata,
            )
        else:
            # Fallback directo al sistema TTS sin router
            self.audio_system.queue_message(message)

        self.last_announcement_time = current_time
    
    def _create_audio_message(self, nav_object):
        """
        📢 Crear mensaje de audio descriptivo
        
        Args:
            nav_object: Objeto de navegación con metadatos
            
        Returns:
            str: Mensaje en español para TTS
        """
        name = nav_object['spanish_name']
        zone = nav_object['zone']
        distance = nav_object['distance']
        
        # Traducir zona a español
        zone_spanish = {
            'left': 'izquierda',
            'center': 'centro',
            'right': 'derecha'
        }
        zone_text = zone_spanish.get(zone, zone)
        
        # Traducir distancia
        distance_spanish = {
            'cerca': 'cerca',
            'medio': 'a distancia media',
            'lejos': 'lejos'
        }
        distance_text = distance_spanish.get(distance, distance)
        
        # Construir mensaje contextual
        if distance == 'cerca' and nav_object['priority'] >= 9:
            message = f"Cuidado, {name} muy {distance_text} a la {zone_text}"
        else:
            message = f"{name} a la {zone_text}, {distance_text}"
        
        return message

    def _map_priority_for_audio(self, nav_object: Dict[str, Any]) -> EventPriority:
        """Mapear prioridad numérica a EventPriority"""
        priority_value = float(nav_object.get('priority', 0.0) or 0.0)
        distance = (nav_object.get('distance') or '').lower()

        if priority_value >= 10.0 or distance in {'muy_cerca', 'very_close'}:
            return EventPriority.CRITICAL
        if priority_value >= 9.0 or distance in {'cerca', 'close'}:
            return EventPriority.HIGH
        if priority_value >= 8.0:
            return EventPriority.MEDIUM
        return EventPriority.LOW
    
    def get_status(self):
        """
        📊 Obtener estado actual del coordinator
        
        Returns:
            dict: Estado con métricas y información
        """
        audio_queue_size = 0
        try:
            if hasattr(self.audio_system, 'get_queue_size'):
                audio_queue_size = self.audio_system.get_queue_size()
            elif hasattr(self.audio_system, 'audio_queue'):
                audio_queue_size = len(self.audio_system.audio_queue)
        except:
            pass
        
        return {
            'frames_processed': self.frames_processed,
            'current_detections_count': len(self.current_detections),
            'audio_queue_size': audio_queue_size,
            'has_dashboard': self.dashboard is not None,
            'slam_events': sum(len(events) for events in self.latest_slam_events.values()) if self.peripheral_enabled else 0,
            'has_frame_renderer': self.frame_renderer is not None,
            'has_image_enhancer': self.image_enhancer is not None,
            'last_announcement': self.last_announcement_time
        }
    
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
                self.audio_system.speak_force("Test del sistema de navegación")
            else:
                self.audio_system.speak_async("Test del sistema de navegación")
            print("[COORDINATOR] Audio test enviado")
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

            for source, worker in self.slam_workers.items():
                try:
                    worker.stop()
                except Exception as err:
                    print(f"  ⚠️ SLAM worker {source.value} cleanup error: {err}")

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
