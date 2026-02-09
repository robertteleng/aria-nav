"""
Navigation decision engine for prioritized object announcements.

This module analyzes detected objects and produces prioritized navigation candidates
for audio feedback. It implements a two-tier priority system with per-instance tracking
to prevent redundant announcements while ensuring critical safety warnings are timely.

Features:
- Two-tier priority system (CRITICAL for immediate risks, NORMAL for obstacles)
- Per-instance tracking with GlobalObjectTracker (track_id based cooldowns)
- Cross-camera tracking integration (RGB + SLAM1 + SLAM2)
- Motion-aware cooldowns (different timings for walking vs. stationary)
- Yellow zone filtering (center ±tolerance) for directional relevance
- Persistence-based filtering for normal objects (must be seen N frames)
- Distance-based priority adjustment
- Configurable detection criteria per priority level

Priority Levels:
- CRITICAL: Immediate safety risks (person, vehicles) at close distances
  - Motion-aware cooldowns (1.0s walking, 2.0s stationary for person)
  - Optional yellow zone requirement (Config.CRITICAL_REQUIRE_YELLOW_ZONE)
  - Distance thresholds: very_close/close for walking, very_close for stationary

- NORMAL: Obstacles and furniture (chair, table, door, bottle)
  - Persistence requirement (must appear N consecutive frames)
  - Yellow zone requirement (Config.NORMAL_REQUIRE_YELLOW_ZONE)
  - Distance thresholds: close/medium
  - Longer cooldown (2.5s default)

Architecture:
    Detections → analyze() → Navigation Objects
                               ↓
                         evaluate() → Decision Candidate
                               ↓
                    GlobalObjectTracker (track_id enrichment + cooldown)
                               ↓
                         Audio Router

Usage:
    engine = NavigationDecisionEngine()
    nav_objects = engine.analyze(detections, depth_map)
    candidate = engine.evaluate(nav_objects, motion_state="walking")
    if candidate:
        audio_router.route(candidate)
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from collections import defaultdict

from utils.config import Config
from utils.config_sections import (
    TrackerConfig,
    load_tracker_config,
    CriticalDetectionConfig,
    load_critical_detection_config,
    NormalDetectionConfig,
    load_normal_detection_config,
    SlamAudioConfig,
    load_slam_audio_config,
)
from core.telemetry.loggers.navigation_logger import get_navigation_logger
from core.vision.global_object_tracker import GlobalObjectTracker

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

        # Global announcement tracking
        self.last_announcement_time = 0.0

        # Persistence tracking for normal objects (frame count)
        self.detection_history: Dict[str, int] = defaultdict(int)  # class -> frame_count

        # Load typed configs
        tracker_config = load_tracker_config()
        self._critical_config = load_critical_detection_config()
        self._normal_config = load_normal_detection_config()
        self._slam_audio_config = load_slam_audio_config()

        # Global Object Tracker for cross-camera tracking (RGB + SLAM1 + SLAM2)
        self.global_tracker = GlobalObjectTracker(
            iou_threshold=tracker_config.iou_threshold,
            max_age=tracker_config.max_age,
            handoff_timeout=tracker_config.handoff_timeout,
        )

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

            # Pre-compute yellow zone to avoid redundant calculations
            in_yellow_zone = self._in_yellow_zone(bbox, self._critical_config.center_tolerance)

            navigation_obj = {
                "name": class_name,  # Standard field name for frontend compatibility
                "class": class_name,  # Keep for backwards compatibility
                "spanish_name": self.object_priorities[class_name]["spanish"],
                "bbox": bbox,
                "confidence": detection.get("confidence"),
                "zone": zone,
                "distance": distance_category,
                "distance_bucket": distance_category,  # Alias for frontend
                "priority": final_priority,
                "original_priority": base_priority,
                "in_yellow_zone": in_yellow_zone,  # Pre-computed
                "camera_source": detection.get("camera_source", "rgb"),  # Preserve source for rendering filter
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

        With per-instance tracking for granular cooldowns.
        """
        if not navigation_objects:
            # Decay detection history when no objects present
            self.detection_history.clear()
            return None

        now = time.time()

        # DEBUG: Log what objects we're evaluating
        logger = get_navigation_logger().decision
        if navigation_objects:
            logger.info(f"Evaluating {len(navigation_objects)} objects")
            for obj in navigation_objects[:3]:
                logger.debug(f"  - {obj.get('class')}: zone={obj.get('zone')}, distance={obj.get('distance')}, priority={obj.get('priority'):.2f}")

        # Update global tracker with new detections (default camera: rgb)
        person_cooldown = (
            self._critical_config.critical_cooldown_walking
            if motion_state == "walking"
            else self._critical_config.critical_cooldown_stationary
        )
        normal_cooldown = self._normal_config.normal_cooldown
        cooldown_per_class = {
            "person": person_cooldown,
            "car": 1.5,
            "truck": 1.5,
            "bus": 1.5,
            "bicycle": 1.5,
            "motorcycle": 1.5,
            "chair": normal_cooldown,
            "table": normal_cooldown,
            "bottle": normal_cooldown,
            "door": normal_cooldown,
            "laptop": normal_cooldown,
        }
        tracking_results = self.global_tracker.update_and_check(
            navigation_objects, cooldown_per_class, camera_source="rgb"
        )

        # Enrich navigation_objects with track_id and should_announce
        for i, (detection, track_id, should_announce) in enumerate(tracking_results):
            if i < len(navigation_objects):
                navigation_objects[i]["track_id"] = track_id
                navigation_objects[i]["tracker_allows"] = should_announce

        # Update detection history for persistence tracking
        current_classes = {obj.get("class") for obj in navigation_objects}
        for class_name in list(self.detection_history.keys()):
            if class_name not in current_classes:
                self.detection_history[class_name] = 0

        # Evaluate CRITICAL candidates first
        critical_candidate = self._evaluate_critical(navigation_objects, motion_state, now)
        if critical_candidate is not None:
            logger.info(f"✓ CRITICAL candidate: {critical_candidate.nav_object.get('class')} (track_id={critical_candidate.nav_object.get('track_id')})")
            # Mark as announced in tracker
            track_id = critical_candidate.nav_object.get("track_id")
            if track_id is not None:
                self.global_tracker.mark_announced(track_id)
            return critical_candidate

        # Then evaluate NORMAL candidates (if critical didn't trigger)
        normal_candidate = self._evaluate_normal(navigation_objects, motion_state, now)
        if normal_candidate is not None:
            logger.info(f"✓ NORMAL candidate: {normal_candidate.nav_object.get('class')} (track_id={normal_candidate.nav_object.get('track_id')})")
            # Mark as announced in tracker
            track_id = normal_candidate.nav_object.get("track_id")
            if track_id is not None:
                self.global_tracker.mark_announced(track_id)
        else:
            logger.debug(f"✗ No candidate selected")
        return normal_candidate
    
    def _evaluate_critical(
        self,
        navigation_objects: List[Dict[str, Any]],
        motion_state: str,
        now: float,
    ) -> Optional[DecisionCandidate]:
        """Evaluate critical-priority objects (immediate risks)."""
        logger = get_navigation_logger().decision
        cfg = self._critical_config
        critical_classes = cfg.critical_allowed_classes
        critical_distances_walking = self._slam_audio_config.critical_distances_walking
        critical_distances_stationary = self._slam_audio_config.critical_distances_stationary
        center_tolerance = cfg.center_tolerance
        bbox_coverage_threshold = getattr(Config, "CRITICAL_BBOX_COVERAGE_THRESHOLD", 0.35)
        require_yellow_zone = cfg.critical_require_yellow_zone

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
                    frame_width = Config.ARIA_RGB_WIDTH
                    center_x = bbox[0] + bbox[2] / 2
                    zone_width = frame_width * center_tolerance * 2
                    bbox_coverage = bbox[2] / zone_width
                    if bbox_coverage < bbox_coverage_threshold:
                        continue
                else:
                    continue
            
            # Check if in yellow zone (center ±tolerance) - OPTIONAL (use pre-computed value)
            if require_yellow_zone and not obj.get("in_yellow_zone", False):
                continue

            # Check per-instance tracker cooldown
            if not obj.get("tracker_allows", True):
                logger.debug(f"CRITICAL {class_name}: blocked by tracker (track_id={obj.get('track_id')})")
                continue

            # Critical candidate found - SUCCESS!
            cooldown = cfg.critical_cooldown_walking if motion_state == "walking" else cfg.critical_cooldown_stationary

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
        cfg = self._normal_config
        normal_classes = cfg.normal_allowed_classes
        normal_distances = getattr(Config, "NORMAL_DISTANCE", {"close", "medium"})
        require_yellow_zone = cfg.normal_require_yellow_zone
        persistence_threshold = cfg.normal_min_frames
        normal_cooldown = cfg.normal_cooldown
        
        logger = get_navigation_logger().decision
        for obj in navigation_objects:
            class_name = obj.get("class", "").lower()
            if class_name not in normal_classes:
                continue
            
            distance = obj.get("distance", "").lower()
            if distance not in normal_distances:
                logger.debug(f"NORMAL {class_name}: distance {distance} not in {normal_distances}")
                continue
            
            # Use pre-computed yellow zone value
            in_yellow = obj.get("in_yellow_zone", False)
            if require_yellow_zone and not in_yellow:
                logger.debug(f"NORMAL {class_name}: not in yellow zone (require_yellow_zone={require_yellow_zone})")
                continue
            
            # Update persistence counter
            self.detection_history[class_name] += 1
            current_persistence = self.detection_history[class_name]

            # Check persistence threshold
            if current_persistence < persistence_threshold:
                logger.debug(f"NORMAL {class_name}: persistence {current_persistence}/{persistence_threshold}")
                continue

            # Check per-instance tracker cooldown
            if not obj.get("tracker_allows", True):
                logger.debug(f"NORMAL {class_name}: blocked by tracker (track_id={obj.get('track_id')})")
                continue

            # Normal candidate found
            logger.info(f"NORMAL ✓ {class_name}: PASSED all checks (persistence={current_persistence}, yellow={in_yellow})")
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

        frame_width = Config.ARIA_RGB_WIDTH
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
        # bbox format is [x1, y1, x2, y2], NOT [x1, y1, w, h]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        height = y2 - y1
        width = x2 - x1

        # Prioridad 1: Usar mapa de profundidad si está disponible
        if depth_map is not None:
            # Asegurarse de que las coordenadas están dentro de los límites del mapa de profundidad
            h_depth, w_depth = depth_map.shape
            x1_clip, y1_clip = max(0, x1), max(0, y1)
            x2_clip, y2_clip = min(w_depth, x2), min(h_depth, y2)

            if x1_clip < x2_clip and y1_clip < y2_clip:
                depth_roi = depth_map[y1_clip:y2_clip, x1_clip:x2_clip]
                # Usar la mediana para ser robusto a outliers
                valid_depths = depth_roi[depth_roi > 0]
                if len(valid_depths) > 0:
                    median_depth = np.median(valid_depths)

                    if median_depth > 0:
                        # Definir umbrales de distancia en metros
                        if median_depth < 1.5: return "very_close"
                        if median_depth < 3.0: return "close"
                        if median_depth < 5.0: return "medium"
                        return "far"

        # Prioridad 2: Fallback a la estimación por altura del bbox
        if class_name == "person":
            if height > 200: return "very_close"
            if height > 100: return "close"
            if height > 50: return "medium"
            return "far"
        if class_name in {"car", "truck", "bus"}:
            if height > 150: return "very_close"
            if height > 75: return "close"
            if height > 40: return "medium"
            return "far"
        # Generic objects
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
