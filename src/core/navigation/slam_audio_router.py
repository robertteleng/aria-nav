"""Helper utilities for SLAM audio routing logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from core.audio.navigation_audio_router import EventPriority, NavigationAudioRouter
    from core.vision.slam_detection_worker import CameraSource, SlamDetectionEvent
except Exception:  # pragma: no cover - fallback for import issues in docs
    NavigationAudioRouter = None  # type: ignore[assignment]
    CameraSource = None  # type: ignore[assignment]

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
    """Encapsulates SLAM submission and audio routing logic."""

    def __init__(self, audio_router: Optional[NavigationAudioRouter]) -> None:
        self.audio_router = audio_router

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
