"""
SLAM camera audio routing and cross-camera deduplication.

This module provides audio routing for peripheral SLAM cameras (SLAM1, SLAM2) with
intelligent deduplication to prevent announcing the same object multiple times when
it appears in multiple cameras.

Features:
- Critical-only filtering (only announce close/very_close objects from SLAM)
- Cross-camera deduplication using GlobalObjectTracker track IDs
- RGB-SLAM deduplication (avoid re-announcing objects seen in RGB frontal camera)
- Priority-based message generation (CRITICAL for vehicles, HIGH for people)
- Distance-aware warning messages for critical situations
- Configurable duplicate grace period

Architecture:
    SLAM Frame → Worker → Events → SlamAudioRouter → NavigationAudioRouter
                                       ↓
                                  GlobalObjectTracker (track ID enrichment)
                                       ↓
                                  RgbAudioRouter (notify about RGB announcements)

Deduplication Strategy:
1. Track-based: Use GlobalObjectTracker track_id to detect same object across cameras
2. Class-based fallback: Use class name + timestamp when track_id unavailable
3. RGB coordination: RgbAudioRouter notifies SLAM router to avoid duplicates

Usage:
    slam_router = SlamAudioRouter(audio_router, global_tracker)
    slam_router.submit_and_route(state, CameraSource.SLAM1, frame)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from core.audio.message_formatter import MessageFormatter
    from core.audio.navigation_audio_router import EventPriority, NavigationAudioRouter
    from core.vision.slam_detection_worker import CameraSource, SlamDetectionEvent
    from utils.config import Config
    from utils.config_sections import SlamAudioConfig, load_slam_audio_config
except Exception:  # pragma: no cover - fallback for import issues in docs
    MessageFormatter = None  # type: ignore[assignment]
    NavigationAudioRouter = None  # type: ignore[assignment]
    CameraSource = None  # type: ignore[assignment]
    Config = None  # type: ignore[assignment]
    SlamAudioConfig = None  # type: ignore[assignment]

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

    def __init__(
        self,
        audio_router: Optional[NavigationAudioRouter],
        global_tracker=None,
        message_formatter: Optional[MessageFormatter] = None,
        config: Optional[SlamAudioConfig] = None,
    ) -> None:
        self.audio_router = audio_router
        self.global_tracker = global_tracker  # Reference to GlobalObjectTracker for cross-camera tracking
        self.message_formatter = message_formatter or MessageFormatter()
        # Load typed configuration
        self.config = config or load_slam_audio_config()
        # Track RGB frontal announcements to avoid SLAM duplicates
        self._rgb_class_history: Dict[str, float] = {}
        # Use typed config instead of scattered getattr calls
        self.duplicate_grace = self.config.duplicate_grace
        self.critical_only = self.config.critical_only
        self.critical_distances_walking = self.config.critical_distances_walking
        self.critical_distances_stationary = self.config.critical_distances_stationary
    
    def register_rgb_announcement(self, class_name: str) -> None:
        """Register an RGB frontal announcement to avoid SLAM duplicates."""
        self._rgb_class_history[class_name] = time.time()

    def _is_duplicate_with_rgb(self, class_name: str) -> bool:
        """Check if this class was recently announced from RGB frontal."""
        if class_name not in self._rgb_class_history:
            return False
        elapsed = time.time() - self._rgb_class_history[class_name]
        return elapsed < self.duplicate_grace

    def _enrich_with_track_ids(self, events: List["SlamDetectionEvent"], source: CameraSource) -> None:
        """Enrich SLAM events with global track IDs for cross-camera tracking."""
        if not self.global_tracker:
            return

        # Convert camera source to string format
        camera_str = source.value  # "slam1" or "slam2"

        # Convert events to detection format expected by tracker
        detections = []
        for event in events:
            detections.append({
                "class": event.object_name,
                "bbox": event.bbox,
                "zone": event.zone,
                "confidence": event.confidence,
            })

        # Get track IDs from global tracker (no cooldown check, just matching)
        # We use a very short cooldown (0.01s) since we handle cooldowns separately in audio router
        cooldown_per_class = {event.object_name: 0.01 for event in events}
        tracking_results = self.global_tracker.update_and_check(
            detections, cooldown_per_class, camera_source=camera_str
        )

        # Enrich events with track_ids
        for i, (detection, track_id, should_announce) in enumerate(tracking_results):
            if i < len(events):
                events[i].track_id = track_id

    def _is_duplicate_with_track_id(self, event: "SlamDetectionEvent") -> bool:
        """
        Check if this event's track was recently announced (cross-camera dedup).

        Uses track_id to detect same object across cameras (e.g., person seen in SLAM1
        then appearing in RGB should not be re-announced).
        """
        if not self.global_tracker or event.track_id is None:
            # Fallback to class-based dedup if no tracker or no track_id
            return self._is_duplicate_with_rgb(event.object_name)

        # Check if this track was recently announced
        track = self.global_tracker.tracks.get(event.track_id)
        if not track:
            return False

        # If announced recently, it's a duplicate
        time_since_announce = time.time() - track.last_announced
        return time_since_announce < self.duplicate_grace

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

        # Enrich events with global track_ids for cross-camera tracking
        if self.global_tracker:
            self._enrich_with_track_ids(events, source)

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

                # Filter: skip if recently announced by track_id (cross-camera dedup)
                if self._is_duplicate_with_track_id(event):
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

    def _build_slam_message(self, event: "SlamDetectionEvent") -> str:
        """Build detailed SLAM message with zone and distance information."""
        from utils.config import Config

        zone_text = self.message_formatter.format_zone(event.zone)
        name = self.message_formatter.format_object_name(event.object_name)
        distance = (event.distance or "").lower()

        # Warning messages for critical situations
        if distance in {"close", "very_close"} and event.object_name in {"car", "truck", "bus"}:
            return f"Warning, {name.lower()} approaching on the {zone_text}"
        if event.object_name == "person" and distance in {"close", "very_close"}:
            return f"{name} close on the {zone_text}"

        # Standard distance announcements
        if distance and distance != "unknown":
            distance_text = Config.AUDIO_DISTANCE_LABELS.get(distance, distance.replace("_", " "))
            return f"{name} {distance_text} on the {zone_text}"

        return f"{name} on the {zone_text}"


__all__ = [
    "SlamAudioRouter",
    "SlamRoutingState",
]
