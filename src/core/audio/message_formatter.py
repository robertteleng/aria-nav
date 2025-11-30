"""
Message formatter service for navigation audio messages.

Centralizes message formatting logic used across RGB and SLAM audio routers,
eliminating code duplication and providing consistent message generation.
"""

from typing import Dict, Any, Optional
from utils.config import Config

import logging
log = logging.getLogger(__name__)


class MessageFormatter:
    """
    Centralized service for formatting navigation audio messages.

    Handles:
    - Object name translation (technical -> user-friendly)
    - Zone name translation
    - Distance formatting
    - Simple and detailed message construction
    """

    def __init__(self):
        """Initialize message formatter with config-based label mappings."""
        self.object_labels = Config.AUDIO_OBJECT_LABELS
        self.zone_labels = Config.AUDIO_ZONE_LABELS

    def format_object_name(self, class_name: str) -> str:
        """
        Format object class name to user-friendly label.

        Args:
            class_name: Technical class name (e.g., "person", "car")

        Returns:
            User-friendly label from config, or capitalized class name if not found

        Examples:
            >>> formatter.format_object_name("person")
            "Persona"  # From Config.AUDIO_OBJECT_LABELS
            >>> formatter.format_object_name("unknown_object")
            "Unknown object"  # Fallback capitalization
        """
        class_name = str(class_name or "").strip()
        if not class_name:
            return "Object"

        return self.object_labels.get(class_name, class_name.capitalize())

    def format_zone(self, zone: str) -> str:
        """
        Format zone identifier to user-friendly label.

        Args:
            zone: Zone identifier (e.g., "center", "left", "right")

        Returns:
            User-friendly zone label from config, or original zone if not found

        Examples:
            >>> formatter.format_zone("center")
            "Centro"  # From Config.AUDIO_ZONE_LABELS
        """
        zone = str(zone or "").strip()
        if not zone:
            return "unknown zone"

        return self.zone_labels.get(zone, zone)

    def format_distance(self, distance: str) -> str:
        """
        Format distance to user-friendly label.

        Args:
            distance: Distance identifier (e.g., "very_close", "close", "far")

        Returns:
            User-friendly distance label

        Examples:
            >>> formatter.format_distance("very_close")
            "muy cerca"
            >>> formatter.format_distance("close")
            "cerca"
        """
        distance_labels = {
            "very_close": "muy cerca",
            "close": "cerca",
            "medium": "medio",
            "far": "lejos",
        }
        return distance_labels.get(distance, distance)

    def build_simple_message(self, nav_object: Dict[str, Any]) -> str:
        """
        Build simple message from navigation object (RGB detections).

        Args:
            nav_object: Detection dictionary with 'class' key

        Returns:
            Simple message with just the object name

        Examples:
            >>> formatter.build_simple_message({"class": "person"})
            "Persona"
        """
        class_name = str(nav_object.get("class", "")).strip()
        return self.format_object_name(class_name)

    def build_detailed_message(
        self,
        object_name: str,
        zone: str,
        distance: Optional[str] = None,
        camera_source: Optional[str] = None,
    ) -> str:
        """
        Build detailed message with zone and distance information (SLAM detections).

        Args:
            object_name: Object class name
            zone: Zone identifier (center, left, right)
            distance: Optional distance identifier
            camera_source: Optional camera source for side detection

        Returns:
            Detailed message with object, zone, and distance

        Examples:
            >>> formatter.build_detailed_message("person", "center", "close")
            "Persona centro cerca"

            >>> formatter.build_detailed_message("car", "left", camera_source="slam1")
            "Coche lado izquierdo"
        """
        # Format object name
        name = self.format_object_name(object_name)

        # Format zone
        zone_text = self.format_zone(zone)

        # Special handling for side zones based on camera
        if zone in ["left", "right"] and camera_source:
            # Simplify to "lado izquierdo" or "lado derecho"
            if zone == "left":
                zone_text = "lado izquierdo"
            elif zone == "right":
                zone_text = "lado derecho"

        # Build message parts
        parts = [name, zone_text]

        # Add distance if provided
        if distance:
            parts.append(self.format_distance(distance))

        return " ".join(parts)

    def build_slam_event_message(self, event: Any) -> str:
        """
        Build message from SLAM detection event.

        Args:
            event: SlamDetectionEvent with object_name, zone, distance, camera_source

        Returns:
            Formatted message for the SLAM event

        Examples:
            >>> event = SlamDetectionEvent(object_name="person", zone="center", distance="close")
            >>> formatter.build_slam_event_message(event)
            "Persona centro cerca"
        """
        return self.build_detailed_message(
            object_name=event.object_name,
            zone=event.zone,
            distance=getattr(event, 'distance', None),
            camera_source=getattr(event, 'camera_source', None),
        )
