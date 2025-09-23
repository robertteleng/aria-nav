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
from typing import Optional

from utils.config import Config

try:
    from core.vision.depth_estimator import DepthEstimator
except Exception:
    DepthEstimator = None


class Coordinator:
    """
    🎯 Coordinator que orquesta el flujo de datos entre módulos
    
    Este coordinator RECIBE dependencias ya creadas (no las crea él).
    Se enfoca únicamente en coordinar el flujo de procesamiento.
    
    Pipeline:
    Image → Enhancement → YOLO → Navigation → Audio → Rendering
    """
    
    def __init__(self, yolo_processor, audio_system, frame_renderer=None, image_enhancer=None, dashboard=None):
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
        
        # Cooldown adaptativo según movimiento (como en tu AudioSystem original)
        if motion_state == "walking":
            cooldown = 1.5  # Más frecuente cuando camina
        else:
            cooldown = 3.0  # Menos frecuente cuando está parado
        
        # Verificar cooldown
        if current_time - self.last_announcement_time < cooldown:
            return
        
        # Solo anunciar el objeto de mayor prioridad
        if navigation_objects:
            top_object = navigation_objects[0]
            
            # Filtrar solo objetos de alta prioridad
            if top_object['priority'] >= 8.0:
                message = self._create_audio_message(top_object)
                
                # Pasar motion_state al audio system si lo soporta
                try:
                    # Intentar llamada con motion_state (nueva versión)
                    if hasattr(self.audio_system, 'process_detections'):
                        # Si audio_system tiene process_detections, usarlo
                        self.audio_system.process_detections([top_object], motion_state)
                    else:
                        # Fallback al método original
                        self.audio_system.speak_async(message)
                except Exception as e:
                    # Fallback seguro
                    print(f"[WARN] Audio command fallback: {e}")
                    self.audio_system.speak_async(message)
                
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


# ============================================================================
# TESTING DEL COORDINATOR MEJORADO
# ============================================================================

def test_coordinator():
    """Test básico del Coordinator mejorado"""
    print("🧪 Testing Coordinator mejorado...")
    
    # Mock classes para testing
    class MockYolo:
        def process_frame(self, frame):
            return [
                {'name': 'person', 'confidence': 0.9, 'bbox': [100, 100, 50, 200]},
                {'name': 'car', 'confidence': 0.8, 'bbox': [300, 150, 100, 80]}
            ]
    
    class MockAudio:
        def __init__(self):
            self.announcement_cooldown = 2.0
            self.audio_queue = []
        
        def speak_async(self, message):
            self.audio_queue.append(message)
            print(f"🔊 Mock Audio: {message}")
        
        def process_detections(self, detections, motion_state):
            if detections:
                message = f"[{motion_state.upper()}] {detections[0]['name']}"
                self.speak_async(message)
        
        def get_queue_size(self):
            return len(self.audio_queue)
        
        def cleanup(self):
            pass
    
    class MockRenderer:
        def draw_navigation_overlay(self, frame, detections, audio_system, depth_map):
            # Simular rendering añadiendo texto
            import cv2
            annotated = frame.copy()
            cv2.putText(annotated, f"Detections: {len(detections)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return annotated
    
    # Crear mocks
    mock_yolo = MockYolo()
    mock_audio = MockAudio()
    mock_renderer = MockRenderer()
    
    # Crear coordinator
    coordinator = Coordinator(
        yolo_processor=mock_yolo,
        audio_system=mock_audio,
        frame_renderer=mock_renderer
    )
    
    # Test con frame sintético
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("  🔄 Procesando frame de test...")
    
    # Test con diferentes motion states
    result_stationary = coordinator.process_frame(test_frame, "stationary")
    print(f"  ✅ Frame estationary procesado: {result_stationary.shape}")
    
    result_walking = coordinator.process_frame(test_frame, "walking")
    print(f"  ✅ Frame walking procesado: {result_walking.shape}")
    
    # Test status
    status = coordinator.get_status()
    print(f"  ✅ Status: {status}")
    
    # Test detections
    detections = coordinator.get_current_detections()
    print(f"  ✅ Detecciones: {len(detections)}")
    
    # Test audio
    coordinator.test_audio()
    
    # Test stats
    coordinator.print_stats()
    
    # Test cleanup
    coordinator.cleanup()
    
    print("✅ Coordinator mejorado test completado!")


if __name__ == "__main__":
    test_coordinator()
