"""Centralized audio routing for multi-camera navigation events."""

from __future__ import annotations

import time
import threading
import queue
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from core.audio.audio_system import AudioSystem
from core.vision.slam_detection_worker import CameraSource, SlamDetectionEvent


class EventPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class NavigationEvent:
    timestamp: float
    source: CameraSource
    priority: EventPriority
    message: str
    raw_event: Optional[SlamDetectionEvent] = None


class NavigationAudioRouter:
    """Priority queue for navigation events across RGB and SLAM feeds."""

    def __init__(self, audio_system: AudioSystem) -> None:
        self.audio = audio_system
        self.event_queue: "queue.PriorityQueue[tuple[int, int, NavigationEvent | None]]" = queue.PriorityQueue(maxsize=16)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._counter = 0

        self._last_global_announcement = 0.0
        self._last_source_announcement: dict[CameraSource, float] = {}

        self.source_cooldown = {
            CameraSource.SLAM1: 3.0,
            CameraSource.SLAM2: 3.0,
        }
        # RGB cooldown se gestiona en Coordinator mediante AudioSystem
        self.global_cooldown = 0.8

        self.events_processed = 0
        self.events_spoken = 0
        self.events_dropped = 0

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._counter = 0
        self._thread = threading.Thread(target=self._run, name="NavigationAudioRouter", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            self.event_queue.put_nowait((EventPriority.LOW.value, self._counter, None))
        except queue.Full:
            pass
        self._counter += 1
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    # ------------------------------------------------------------------
    # queue interface
    # ------------------------------------------------------------------

    def enqueue(self, event: NavigationEvent) -> None:
        try:
            self.event_queue.put_nowait((event.priority.value, self._counter, event))
            self._counter += 1
        except queue.Full:
            self.events_dropped += 1

    def enqueue_from_slam(self, slam_event: SlamDetectionEvent, message: str, priority: EventPriority) -> None:
        nav_event = NavigationEvent(
            timestamp=slam_event.timestamp,
            source=slam_event.source,
            priority=priority,
            message=message,
            raw_event=slam_event,
        )
        self.enqueue(nav_event)

    # ------------------------------------------------------------------
    # worker
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while self._running:
            try:
                priority, _order, event = self.event_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if event is None:
                break

            self.events_processed += 1
            if self._should_announce(event):
                self.audio.speak_async(event.message)
                self.events_spoken += 1
                self._last_global_announcement = time.time()
                self._last_source_announcement[event.source] = self._last_global_announcement
            else:
                self.events_dropped += 1

    def _should_announce(self, event: NavigationEvent) -> bool:
        now = time.time()

        if now - self._last_global_announcement < self.global_cooldown:
            return False

        if event.priority == EventPriority.CRITICAL:
            return True

        last_source = self._last_source_announcement.get(event.source, 0.0)
        source_cooldown = self.source_cooldown.get(event.source, 2.0)
        if now - last_source < source_cooldown:
            return False

        if event.source != CameraSource.SLAM1 and event.source != CameraSource.SLAM2:
            return True

        # For peripheral events, ensure RGB has been quiet for a while
        primary_last = self._last_source_announcement.get(CameraSource.SLAM1, 0.0)
        primary_last = max(primary_last, self._last_source_announcement.get(CameraSource.SLAM2, 0.0))
        if now - primary_last < 4.0:
            return False

        return True


__all__ = ["NavigationAudioRouter", "EventPriority", "NavigationEvent"]
