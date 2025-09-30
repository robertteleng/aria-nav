"""Decision engine for navigation audio events."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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
            "door": {"priority": 4, "spanish": "puerta"},
            "stairs": {"priority": 5, "spanish": "escaleras"},
        }

        self.last_announcement_time = 0.0

    # ------------------------------------------------------------------
    # analysis
    # ------------------------------------------------------------------

    def analyze(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        navigation_objects: List[Dict[str, Any]] = []

        for detection in detections:
            class_name = detection.get("name")
            if class_name not in self.object_priorities:
                continue

            bbox = detection.get("bbox")
            if not bbox:
                continue

            zone = self._calculate_zone(bbox[0] + bbox[2] / 2)
            distance_category = self._estimate_distance(bbox, class_name)
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
        if not navigation_objects:
            return None

        top_object = navigation_objects[0]
        if top_object.get("priority", 0.0) < 8.0:
            return None

        cooldown = 1.5 if motion_state == "walking" else 3.0
        now = time.time()
        if now - self.last_announcement_time < cooldown:
            return None

        # Datos mínimos que necesita la capa de audio para formatear el mensaje.
        metadata: Dict[str, Any] = {
            "class": top_object.get("class"),
            "spanish_name": top_object.get("spanish_name"),
            "priority": top_object.get("priority"),
            "zone": top_object.get("zone"),
            "distance": top_object.get("distance"),
            "motion_state": motion_state,
            "cooldown": cooldown,
        }

        priority_enum = self._map_priority_for_audio(top_object)
        self.last_announcement_time = now

        return DecisionCandidate(
            nav_object=top_object,
            metadata=metadata,
            priority=priority_enum,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _calculate_zone(self, x_center: float) -> str:
        if x_center < self.zones["left"][1]:
            return "left"
        if x_center < self.zones["center"][1]:
            return "center"
        return "right"

    def _estimate_distance(self, bbox, class_name: str) -> str:
        height = bbox[3]

        if class_name == "person":
            if height > 200:
                return "cerca"
            if height > 100:
                return "medio"
            return "lejos"
        if class_name in {"car", "truck", "bus"}:
            if height > 150:
                return "cerca"
            if height > 75:
                return "medio"
            return "lejos"
        if height > 100:
            return "cerca"
        if height > 50:
            return "medio"
        return "lejos"

    def _calculate_final_priority(self, base_priority: float, zone: str, distance: str) -> float:
        priority = float(base_priority)
        distance_multipliers = {
            "cerca": 2.0,
            "medio": 1.5,
            "lejos": 1.0,
        }
        priority *= distance_multipliers.get(distance, 1.0)

        zone_multipliers = {
            "center": 1.3,
            "left": 1.0,
            "right": 1.0,
        }
        priority *= zone_multipliers.get(zone, 1.0)
        return priority

    def _map_priority_for_audio(self, nav_object: Dict[str, Any]) -> EventPriority:
        priority_value = float(nav_object.get("priority", 0.0) or 0.0)
        distance = (nav_object.get("distance") or "").lower()

        if priority_value >= 10.0 or distance in {"muy_cerca", "very_close"}:
            return EventPriority.CRITICAL
        if priority_value >= 9.0 or distance in {"cerca", "close"}:
            return EventPriority.HIGH
        if priority_value >= 8.0:
            return EventPriority.MEDIUM
        return EventPriority.LOW


__all__ = [
    "DecisionCandidate",
    "NavigationDecisionEngine",
]
