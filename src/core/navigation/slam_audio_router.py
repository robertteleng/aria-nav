"""Helper utilities for SLAM audio routing logic."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from core.audio.navigation_audio_router import EventPriority, NavigationAudioRouter
    from core.vision.slam_detection_worker import CameraSource, SlamDetectionEvent
    from utils.config import Config
except Exception:  # pragma: no cover - fallback for import issues in docs
    NavigationAudioRouter = None  # type: ignore[assignment]
    CameraSource = None  # type: ignore[assignment]
    Config = None  # type: ignore[assignment]

    from enum import Enum

    class EventPriority(Enum):  # type: ignore[override]
        CRITICAL = 1
        HIGH = 2
        MEDIUM = 3
        LOW = 4


@dataclass
class SlamRoutingState:
    workers: Dict[CameraSource, "SlamDetectionWorker"]
    frame_counters: Dict[CameraSource, int]
    last_indices: Dict[CameraSource, int]
    latest_events: Dict[CameraSource, List[SlamDetectionEvent]]


class SlamAudioRouter:
    """Encapsulates SLAM submission and audio routing logic with critical-only filtering."""

    def __init__(self, audio_router: Optional[NavigationAudioRouter]) -> None:
        self.audio_router = audio_router
        # Track RGB frontal announcements to avoid SLAM duplicates
        self._rgb_class_history: Dict[str, float] = {}
        self.duplicate_grace = getattr(Config, "SLAM_AUDIO_DUPLICATE_GRACE", 1.0)
        self.critical_only = getattr(Config, "SLAM_CRITICAL_ONLY", True)
        self.critical_distances_walking = getattr(Config, "CRITICAL_DISTANCE_WALKING", {"very_close", "close"})
        self.critical_distances_stationary = getattr(Config, "CRITICAL_DISTANCE_STATIONARY", {"very_close"})
    
    def register_rgb_announcement(self, class_name: str) -> None:
        """Register an RGB frontal announcement to avoid SLAM duplicates."""
        self._rgb_class_history[class_name] = time.time()
    
    def _is_duplicate_with_rgb(self, class_name: str) -> bool:
        """Check if this class was recently announced from RGB frontal."""
        if class_name not in self._rgb_class_history:
            return False
        elapsed = time.time() - self._rgb_class_history[class_name]
        return elapsed < self.duplicate_grace

    def submit_and_route(
        self,
        state: SlamRoutingState,
        source: CameraSource,
        frame,
    ) -> None:
        worker = state.workers[source]
        state.frame_counters[source] += 1
        worker.submit(frame, state.frame_counters[source])

        events = worker.latest_events()
        if not events:
            state.latest_events[source] = []
            return

        latest_index = events[0].frame_index
        if latest_index == state.last_indices.get(source, -1):
            return

        state.last_indices[source] = latest_index
        state.latest_events[source] = events

        if self.audio_router:
            for event in events:
                # Filter: skip if not critical distance when SLAM_CRITICAL_ONLY is enabled
                if self.critical_only:
                    distance = (event.distance or "").lower()
                    is_critical_distance = (
                        distance in self.critical_distances_walking
                        or distance in self.critical_distances_stationary
                    )
                    if not is_critical_distance:
                        continue
                
                # Filter: skip if recently announced from RGB frontal
                if self._is_duplicate_with_rgb(event.object_name):
                    continue
                
                priority = self._determine_slam_priority(event)
                message = self._build_slam_message(event)
                self.audio_router.enqueue_from_slam(event, message, priority)

    def describe_event(self, event: "SlamDetectionEvent") -> str:
        """Return the spoken message that would be generated for an event."""
        return self._build_slam_message(event)

    @staticmethod
    def _determine_slam_priority(event: "SlamDetectionEvent") -> EventPriority:
        distance = event.distance
        name = event.object_name

        if name in {"car", "truck", "bus", "motorcycle"} and distance in {"close", "very_close"}:
            return EventPriority.CRITICAL
        if name == "person" and distance in {"close", "very_close"}:
            return EventPriority.HIGH
        if name in {"bicycle", "motorbike"}:
            return EventPriority.HIGH
        return EventPriority.MEDIUM

    @staticmethod
    def _build_slam_message(event: "SlamDetectionEvent") -> str:
        zone_map = {
            "far_left": "far left side",
            "left": "left side",
            "right": "right side",
            "far_right": "far right side",
        }
        object_map = {
            "person": "person",
            "car": "car",
            "truck": "truck",
            "bus": "bus",
            "bicycle": "bicycle",
            "motorcycle": "motorcycle",
            "motorbike": "motorbike",
        }

        zone_text = zone_map.get(event.zone, event.zone)
        name = object_map.get(event.object_name, event.object_name)
        distance = (event.distance or "").lower()

        if distance in {"close", "very_close"} and event.object_name in {"car", "truck", "bus"}:
            return f"Warning, {name} approaching on the {zone_text}"
        if event.object_name == "person" and distance in {"close", "very_close"}:
            return f"Person close on the {zone_text}"
        if distance and distance != "unknown":
            if distance == "medium":
                distance_text = "at medium distance"
            elif distance == "far":
                distance_text = "far"
            else:
                distance_text = distance.replace("_", " ")
            return f"{name.capitalize()} {distance_text} on the {zone_text}"
        return f"{name.capitalize()} on the {zone_text}"


__all__ = [
    "SlamAudioRouter",
    "SlamRoutingState",
]
