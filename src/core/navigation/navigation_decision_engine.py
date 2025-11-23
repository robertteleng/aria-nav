"""Decision engine for navigation audio events."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from collections import defaultdict

from utils.config import Config

try:
    from core.audio.navigation_audio_router import EventPriority
except Exception:
    from enum import Enum

    class EventPriority(Enum):
        CRITICAL = 1
        HIGH = 2
        MEDIUM = 3
        LOW = 4


@dataclass
class DecisionCandidate:
    """Navigation decision that still needs audio formatting."""

    nav_object: Dict[str, Any]
    metadata: Dict[str, Any]
    priority: EventPriority


class NavigationDecisionEngine:
    """Analyzes detections and produces prioritized navigation candidates."""

    def __init__(
        self,
        *,
        zones: Optional[Dict[str, tuple]] = None,
        object_priorities: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.zones = zones or {
            "left": (0, 213),
            "center": (213, 426),
            "right": (426, 640),
        }

        self.object_priorities = object_priorities or {
            "person": {"priority": 10, "spanish": "persona"},
            "car": {"priority": 8, "spanish": "coche"},
            "truck": {"priority": 8, "spanish": "camión"},
            "bus": {"priority": 8, "spanish": "autobús"},
            "bicycle": {"priority": 7, "spanish": "bicicleta"},
            "motorcycle": {"priority": 7, "spanish": "motocicleta"},
            "stop sign": {"priority": 9, "spanish": "señal de stop"},
            "traffic light": {"priority": 6, "spanish": "semáforo"},
            "chair": {"priority": 3, "spanish": "silla"},
            "table": {"priority": 3, "spanish": "mesa"},
            "door": {"priority": 4, "spanish": "puerta"},
            "bottle": {"priority": 2, "spanish": "botella"},
            "laptop": {"priority": 2, "spanish": "laptop"},
            "stairs": {"priority": 5, "spanish": "escaleras"},
        }

        self.last_announcement_time = 0.0
        self.last_critical_time = 0.0
        self.last_critical_class = None
        
        # Tracking de persistencia para objetos normales
        self.detection_history: Dict[str, int] = defaultdict(int)  # class -> frame_count
        self.last_normal_announcement: Dict[str, float] = {}  # class -> timestamp

    # ------------------------------------------------------------------
    # analysis
    # ------------------------------------------------------------------

    def analyze(self, detections: List[Dict[str, Any]], depth_map: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        navigation_objects: List[Dict[str, Any]] = []

        for detection in detections:
            class_name = detection.get("name")
            if class_name not in self.object_priorities:
                continue

            bbox = detection.get("bbox")
            if not bbox:
                continue

            zone = self._calculate_zone(bbox[0] + bbox[2] / 2)
            distance_category = self._estimate_distance(bbox, class_name, detection, depth_map)
            base_priority = self.object_priorities[class_name]["priority"]
            final_priority = self._calculate_final_priority(base_priority, zone, distance_category)

            navigation_obj = {
                "class": class_name,
                "spanish_name": self.object_priorities[class_name]["spanish"],
                "bbox": bbox,
                "confidence": detection.get("confidence"),
                "zone": zone,
                "distance": distance_category,
                "priority": final_priority,
                "original_priority": base_priority,
            }
            navigation_objects.append(navigation_obj)

        navigation_objects.sort(key=lambda item: item["priority"], reverse=True)
        return navigation_objects

    def evaluate(
        self,
        navigation_objects: List[Dict[str, Any]],
        motion_state: str = "stationary",
    ) -> Optional[DecisionCandidate]:
        """Evaluate detections and return candidate if announcement criteria are met.
        
        Now supports two priority levels:
        - CRITICAL: immediate risks (person, vehicles) at critical distances
        - NORMAL: obstacles (chair, table, bottle) with persistence and yellow zone
        """
        if not navigation_objects:
            # Decay detection history when no objects present
            self.detection_history.clear()
            return None

        now = time.time()
        
        # Update detection history for persistence tracking
        current_classes = {obj.get("class") for obj in navigation_objects}
        for class_name in list(self.detection_history.keys()):
            if class_name not in current_classes:
                self.detection_history[class_name] = 0
        
        # Evaluate CRITICAL candidates first
        critical_candidate = self._evaluate_critical(navigation_objects, motion_state, now)
        if critical_candidate is not None:
            return critical_candidate
        
        # Then evaluate NORMAL candidates (if critical didn't trigger)
        normal_candidate = self._evaluate_normal(navigation_objects, motion_state, now)
        return normal_candidate
    
    def _evaluate_critical(
        self,
        navigation_objects: List[Dict[str, Any]],
        motion_state: str,
        now: float,
    ) -> Optional[DecisionCandidate]:
        """Evaluate critical-priority objects (immediate risks)."""
        critical_classes = getattr(Config, "CRITICAL_ALLOWED_CLASSES", {"person", "car", "truck", "bus", "bicycle", "motorcycle"})
        critical_distances_walking = getattr(Config, "CRITICAL_DISTANCE_WALKING", {"very_close", "close"})
        critical_distances_stationary = getattr(Config, "CRITICAL_DISTANCE_STATIONARY", {"very_close"})
        center_tolerance = getattr(Config, "CRITICAL_CENTER_TOLERANCE", 0.30)
        bbox_coverage_threshold = getattr(Config, "CRITICAL_BBOX_COVERAGE_THRESHOLD", 0.35)
        repeat_grace = getattr(Config, "CRITICAL_REPEAT_GRACE", 1.5)
        require_yellow_zone = getattr(Config, "CRITICAL_REQUIRE_YELLOW_ZONE", False)  # NEW: optional filter
        
        critical_distances = critical_distances_walking if motion_state == "walking" else critical_distances_stationary
        
        for obj in navigation_objects:
            class_name = obj.get("class", "").lower()
            if class_name not in critical_classes:
                continue
            
            distance = obj.get("distance", "").lower()
            bbox = obj.get("bbox")
            
            # Check if in critical distance
            if distance not in critical_distances:
                # Exception: allow "close" if bbox coverage is high (large object blocking)
                if distance == "close" and bbox:
                    frame_width = 640  # TODO: get from config or frame
                    center_x = bbox[0] + bbox[2] / 2
                    zone_width = frame_width * center_tolerance * 2
                    bbox_coverage = bbox[2] / zone_width
                    if bbox_coverage < bbox_coverage_threshold:
                        continue
                else:
                    continue
            
            # Check if in yellow zone (center ±tolerance) - OPTIONAL
            if require_yellow_zone and not self._in_yellow_zone(bbox, center_tolerance):
                continue
            
            # Check repeat grace for same class
            if (self.last_critical_class == class_name and 
                now - self.last_critical_time < repeat_grace):
                continue
            
            # Critical candidate found - SUCCESS!
            cooldown = getattr(Config, "CRITICAL_COOLDOWN_WALKING", 1.0) if motion_state == "walking" else getattr(Config, "CRITICAL_COOLDOWN_STATIONARY", 2.0)
            
            metadata: Dict[str, Any] = {
                "class": class_name,
                "spanish_name": self.object_priorities.get(class_name, {}).get("spanish", class_name),
                "priority": obj.get("priority"),
                "zone": obj.get("zone"),
                "distance": distance,
                "motion_state": motion_state,
                "cooldown": cooldown,
                "level": "critical",
            }
            
            self.last_announcement_time = now
            self.last_critical_time = now
            self.last_critical_class = class_name
            
            return DecisionCandidate(
                nav_object=obj,
                metadata=metadata,
                priority=EventPriority.CRITICAL,
            )
        
        return None
    
    def _evaluate_normal(
        self,
        navigation_objects: List[Dict[str, Any]],
        motion_state: str,
        now: float,
    ) -> Optional[DecisionCandidate]:
        """Evaluate normal-priority objects (obstacles, furniture)."""
        normal_classes = getattr(Config, "NORMAL_ALLOWED_CLASSES", {"chair", "table", "bottle", "door", "laptop"})
        normal_distances = getattr(Config, "NORMAL_DISTANCE", {"close", "medium"})
        center_tolerance = getattr(Config, "NORMAL_CENTER_TOLERANCE", 0.30)
        require_yellow_zone = getattr(Config, "NORMAL_REQUIRE_YELLOW_ZONE", True)
        persistence_threshold = getattr(Config, "NORMAL_PERSISTENCE_FRAMES", 2)
        normal_cooldown = getattr(Config, "NORMAL_COOLDOWN", 2.5)
        
        for obj in navigation_objects:
            class_name = obj.get("class", "").lower()
            if class_name not in normal_classes:
                continue
            
            distance = obj.get("distance", "").lower()
            if distance not in normal_distances:
                continue
            
            bbox = obj.get("bbox")
            if require_yellow_zone and not self._in_yellow_zone(bbox, center_tolerance):
                continue
            
            # Update persistence counter
            self.detection_history[class_name] += 1
            
            # Check persistence threshold
            if self.detection_history[class_name] < persistence_threshold:
                continue
            
            # Check cooldown for this specific class
            last_time = self.last_normal_announcement.get(class_name, 0.0)
            if now - last_time < normal_cooldown:
                continue
            
            # Normal candidate found
            metadata: Dict[str, Any] = {
                "class": class_name,
                "spanish_name": self.object_priorities.get(class_name, {}).get("spanish", class_name),
                "priority": obj.get("priority"),
                "zone": obj.get("zone"),
                "distance": distance,
                "motion_state": motion_state,
                "cooldown": normal_cooldown,
                "level": "normal",
            }
            
            self.last_announcement_time = now
            self.last_normal_announcement[class_name] = now
            
            return DecisionCandidate(
                nav_object=obj,
                metadata=metadata,
                priority=EventPriority.MEDIUM,  # NORMAL uses MEDIUM priority
            )
        
        return None
    
    def _in_yellow_zone(self, bbox, center_tolerance: float) -> bool:
        """Check if bbox center is within yellow zone (center ±tolerance)."""
        if not bbox:
            return False
        
        frame_width = 640  # TODO: get from config or frame dimensions
        center_x = bbox[0] + bbox[2] / 2
        frame_center = frame_width / 2
        
        zone_half_width = frame_width * center_tolerance
        return abs(center_x - frame_center) <= zone_half_width

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _calculate_zone(self, x_center: float) -> str:
        if x_center < self.zones["left"][1]:
            return "left"
        if x_center < self.zones["center"][1]:
            return "center"
        return "right"


    def _estimate_distance(self, bbox, class_name: str, detection: Optional[Dict[str, Any]] = None, depth_map: Optional[np.ndarray] = None) -> str:
        # Prioridad 1: Usar mapa de profundidad si está disponible
        if depth_map is not None and bbox:
            x1, y1, w, h = [int(v) for v in bbox]
            x2, y2 = x1 + w, y1 + h
            
            # Asegurarse de que las coordenadas están dentro de los límites del mapa de profundidad
            h_depth, w_depth = depth_map.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_depth, x2), min(h_depth, y2)
            
            if x1 < x2 and y1 < y2:
                depth_roi = depth_map[y1:y2, x1:x2]
                # Usar la mediana para ser robusto a outliers
                median_depth = np.median(depth_roi[depth_roi > 0])
                
                if median_depth > 0:
                    # Definir umbrales de distancia en metros
                    if median_depth < 1.5: return "very_close"
                    if median_depth < 3.0: return "close"
                    if median_depth < 5.0: return "medium"
                    return "far"

        # Prioridad 2: Fallback a la estimación por altura del bbox si no hay mapa de profundidad
        height = bbox[3]
        if class_name == "person":
            if height > 200: return "very_close"
            if height > 100: return "close"
            return "far"
        if class_name in {"car", "truck", "bus"}:
            if height > 150: return "very_close"
            if height > 75: return "close"
            return "far"
        if height > 100: return "close"
        if height > 50: return "medium"
        return "far"

    def _calculate_final_priority(self, base_priority: float, zone: str, distance: str) -> float:
        priority = float(base_priority)
        distance_multipliers = {
            "very_close": 2.5,
            "close": 2.0,
            "medium": 1.5,
            "far": 1.0,
        }
        priority *= distance_multipliers.get(distance, 1.0)

        zone_multipliers = {
            "center": 1.3,
            "left": 1.0,
            "right": 1.0,
        }
        priority *= zone_multipliers.get(zone, 1.0)
        return priority


__all__ = [
    "DecisionCandidate",
    "NavigationDecisionEngine",
]
