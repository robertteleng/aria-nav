#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Simple Coordinator With Dependencies - TFM Navigation System
Coordinator que recibe dependencias ya creadas por el Builder

Fecha: Día 2 - Sistema modular
Versión: 1.0 - Dependency Injection
"""

import numpy as np
import cv2
import time
from typing import Optional


class Coordinator:
    """
    🎯 Coordinator que orquesta el flujo de datos entre módulos
    
    Este coordinator RECIBE dependencias ya creadas (no las crea él).
    Se enfoca únicamente en coordinar el flujo de procesamiento.
    
    Pipeline:
    Image → YOLO → Navigation → Audio → Dashboard
    """
    
    def __init__(self, yolo_processor, audio_system, frame_renderer=None, image_enhancer=None, dashboard=None):
        """
        Inicializar coordinator con dependencias inyectadas
        
        Args:
            yolo_processor: Instancia ya configurada de YoloProcessor
            audio_system: Instancia ya configurada de AudioSystem
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
        
        print(f"✓ Coordinator inicializado")
        print(f"  - YOLO: {type(self.yolo_processor).__name__}")
        print(f"  - Audio: {type(self.audio_system).__name__}")
        print(f"  - Dashboard: {type(self.dashboard).__name__ if self.dashboard else 'None'}")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        🔄 Procesar frame completo a través del pipeline
        
        Args:
            frame: Frame BGR de entrada
            
        Returns:
            np.ndarray: Frame procesado con anotaciones
        """
        self.frames_processed += 1

        # 0. Optional low-light enhancement
        processed_frame = frame
        if self.image_enhancer is not None:
            try:
                processed_frame = self.image_enhancer.enhance_frame(frame)
            except Exception as err:
                print(f"[WARN] Image enhancement skipped: {err}")
                processed_frame = frame

        # 1. YOLO Detection
        detections = self.yolo_processor.process_frame(processed_frame)
        annotated_frame = processed_frame
        
        # 2. Navigation Analysis
        navigation_objects = self._analyze_navigation_objects(detections)
        
        # 3. Audio Commands (con cooldown)
        self._generate_audio_commands(navigation_objects)
        
        # 4. Dashboard Update (si está disponible)
        if self.dashboard:
            self.dashboard.update_detections(detections)
            self.dashboard.update_navigation_status(navigation_objects)
        
        # Guardar estado actual
        self.current_detections = detections
        
        return annotated_frame
    
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
    
    def _generate_audio_commands(self, navigation_objects):
        """
        🔊 Generar comandos de audio basados en objetos detectados
        
        Args:
            navigation_objects: Lista de objetos ordenados por prioridad
        """
        current_time = time.time()
        
        # Cooldown para evitar spam de mensajes
        if current_time - self.last_announcement_time < self.audio_system.announcement_cooldown:
            return
        
        # Solo anunciar el objeto de mayor prioridad
        if navigation_objects:
            top_object = navigation_objects[0]
            
            # Filtrar solo objetos de alta prioridad
            if top_object['priority'] >= 8.0:
                message = self._create_audio_message(top_object)
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
        return {
            'frames_processed': self.frames_processed,
            'current_detections_count': len(self.current_detections),
            'audio_queue_size': self.audio_system.get_queue_size(),
            'has_dashboard': self.dashboard is not None,
            'last_announcement': self.last_announcement_time
        }
    
    def cleanup(self):
        """
        🧹 Limpieza de recursos
        """
        print("🧹 Limpiando Coordinator...")
        
        if self.audio_system:
            self.audio_system.cleanup()
        
        if self.dashboard:
            self.dashboard.cleanup()
            
        if self.frame_renderer:
            # FrameRenderer normalmente no necesita cleanup específico
            pass
            
        if self.image_enhancer:
            # ImageEnhancer normalmente no necesita cleanup específico  
            pass
        
        print("✓ Coordinator limpiado")
