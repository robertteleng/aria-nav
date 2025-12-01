"""
Centralized audio routing for multi-camera navigation events.

This module provides a priority-based audio event queue that coordinates TTS announcements
from multiple camera sources (RGB frontal, SLAM1/SLAM2 peripheral) with intelligent
cooldown management and anti-stutter logic.

Features:
- Priority-based event queue (CRITICAL > HIGH > MEDIUM > LOW)
- Per-source cooldown management (RGB: 1.2s, SLAM: 3.0s default)
- Global cooldown to prevent audio spam (0.8s default)
- Anti-stutter duplicate message detection
- CRITICAL event interruption with grace period
- Telemetry integration for audio event tracking
- Thread-safe metrics collection

Priority Levels:
- CRITICAL: Can interrupt ongoing speech after grace period (0.25s)
- HIGH/MEDIUM/LOW: Respect global/source cooldowns, cannot interrupt

Architecture:
    RGB/SLAM Event → NavigationEvent → PriorityQueue → Worker Thread → AudioSystem
                                            ↓
                                      Cooldown Logic
                                            ↓
                                      TTS speak_async()

Usage:
    router = NavigationAudioRouter(audio_system, telemetry)
    router.start()
    router.enqueue_from_rgb("Person", EventPriority.HIGH)
    router.enqueue_from_slam(slam_event, "Car approaching", EventPriority.CRITICAL)
"""

from __future__ import annotations

import time
import threading
import queue
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from utils.config import Config
from utils.config_sections import AudioRouterConfig, load_audio_router_config
from core.telemetry.loggers.navigation_logger import get_navigation_logger
from core.audio.audio_system import AudioSystem
from core.vision.slam_detection_worker import CameraSource, SlamDetectionEvent
from core.telemetry.loggers.telemetry_logger import TelemetryLogger


class EventPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


# Camera source constants (imported from Config for consistency)
from utils.config import Config

RGB_SOURCE = Config.CAMERA_SOURCE_RGB
SLAM1_SOURCE = Config.CAMERA_SOURCE_SLAM1
SLAM2_SOURCE = Config.CAMERA_SOURCE_SLAM2


@dataclass
class NavigationEvent:
    timestamp: float
    source: str
    priority: EventPriority
    message: str
    metadata: Optional[Dict[str, Any]] = None
    raw_event: Optional[SlamDetectionEvent] = None


class NavigationAudioRouter:
    """Priority queue for navigation events across RGB and SLAM feeds."""
    def __init__(
        self,
        audio_system: AudioSystem,
        telemetry: Optional[TelemetryLogger] = None
    ) -> None:
        self.audio = audio_system
        self.telemetry = telemetry
        
        self.event_queue: "queue.PriorityQueue[tuple[int, int, NavigationEvent | None]]" = queue.PriorityQueue(maxsize=16)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._counter = 0

        self._last_global_announcement = 0.0
        self._last_source_announcement: Dict[str, float] = {}
        self._last_message: Optional[str] = None  # Track last message for anti-stutter
        self._last_message_time: float = 0.0
        self._speaking_started: float = 0.0  # When current message started

        self.default_source_cooldown = 2.0
        self.source_cooldown: Dict[str, float] = {
            SLAM1_SOURCE: 3.0,
            SLAM2_SOURCE: 3.0,
            RGB_SOURCE: 1.2,
        }

        # Load audio router config (typed section)
        self._router_config = load_audio_router_config()
        self.global_cooldown = self._router_config.global_cooldown
        self.interrupt_grace = self._router_config.interrupt_grace  # Anti-stutter grace

        self.events_enqueued = 0
        self.events_processed = 0
        self.events_spoken = 0
        self.events_skipped = 0
        self.events_dropped = 0

        self._metrics_lock = threading.Lock()
        self._session_start_ts = time.time()

        self.metrics: Dict[str, Any] = {
            "events_enqueued": 0,
            "events_processed": 0,
            "events_spoken": 0,
            "events_skipped": 0,
            "events_dropped": 0,
            "queue_maxsize": self.event_queue.maxsize,
            "queue_size": 0,
            "last_decision_ts": 0.0,
            "session_start_ts": self._session_start_ts,
            "per_source": {
                RGB_SOURCE: self._make_source_stats(),
                SLAM1_SOURCE: self._make_source_stats(),
                SLAM2_SOURCE: self._make_source_stats(),
            },
        }

    def _make_source_stats(self) -> Dict[str, Any]:
        return {
            "enqueued": 0,
            "spoken": 0,
            "skipped": 0,
            "dropped": 0,
            "last_event_ts": 0.0,
            "last_spoken_ts": 0.0,
        }

    def _ensure_source_stats(self, source: str) -> Dict[str, Any]:
        per_source = self.metrics.setdefault("per_source", {})
        if source not in per_source:
            per_source[source] = self._make_source_stats()
        return per_source[source]

    def _reset_metrics(self) -> None:
        self._session_start_ts = time.time()
        self.events_enqueued = 0
        self.events_processed = 0
        self.events_spoken = 0
        self.events_skipped = 0
        self.events_dropped = 0
        self._last_global_announcement = 0.0
        self._last_source_announcement.clear()
        with self._metrics_lock:
            self.metrics.update(
                {
                    "events_enqueued": 0,
                    "events_processed": 0,
                    "events_spoken": 0,
                    "events_skipped": 0,
                    "events_dropped": 0,
                    "queue_size": self.event_queue.qsize(),
                    "last_decision_ts": 0.0,
                    "session_start_ts": self._session_start_ts,
                }
            )
            per_source = self.metrics.setdefault("per_source", {})
            for key in list(per_source.keys()):
                per_source[key] = self._make_source_stats()
            for key in (RGB_SOURCE, SLAM1_SOURCE, SLAM2_SOURCE):
                per_source.setdefault(key, self._make_source_stats())

    def _update_metrics(self, event: NavigationEvent, action: str, reason: Optional[str] = None) -> None:
        """Update metrics for audio event tracking with centralized telemetry."""
        now = time.time()
        with self._metrics_lock:
            stats = self._ensure_source_stats(event.source)
            if action == "enqueued":
                self.events_enqueued += 1
                self.metrics["events_enqueued"] = self.events_enqueued
                stats["enqueued"] += 1
                stats["last_event_ts"] = now
            elif action == "processed":
                self.events_processed += 1
                self.metrics["events_processed"] = self.metrics["last_decision_ts"] = now
            elif action == "spoken":
                self.events_spoken += 1
                self.metrics["events_spoken"] = self.events_spoken
                stats["spoken"] += 1
                stats["last_spoken_ts"] = now
            elif action == "skipped":
                self.events_skipped += 1
                self.metrics["events_skipped"] = self.events_skipped
                stats["skipped"] += 1
            elif action == "dropped":
                self.events_dropped += 1
                self.metrics["events_dropped"] = self.events_dropped
                stats["dropped"] += 1

            self.metrics["queue_size"] = max(self.event_queue.qsize(), 0)

        # Log to centralized telemetry
        if self.telemetry:
            self.telemetry.log_audio_event(
                action=action,
                source=event.source,
                priority=int(event.priority.value if isinstance(event.priority, EventPriority) else event.priority),
                message=event.message,
                reason=reason
            )

    def _on_event_processed(self, event: NavigationEvent) -> None:
        self._update_metrics(event, "processed")

    def get_metrics(self) -> Dict[str, Any]:
        with self._metrics_lock:
            import json
            return json.loads(json.dumps(self.metrics))

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._counter = 0
        self._reset_metrics()
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
            priority_value = int(event.priority.value if isinstance(event.priority, EventPriority) else event.priority)
            self.event_queue.put_nowait((priority_value, self._counter, event))
            self._counter += 1
        except queue.Full:
            self._update_metrics(event, "dropped", reason="queue_full")
            return
        self._update_metrics(event, "enqueued")

    def enqueue_from_slam(self, slam_event: SlamDetectionEvent, message: str, priority: EventPriority) -> None:
        source_value = slam_event.source.value if getattr(slam_event, "source", None) is not None else SLAM1_SOURCE
        nav_event = NavigationEvent(
            timestamp=slam_event.timestamp,
            source=source_value,
            priority=priority,
            message=message,
            metadata=None,
            raw_event=slam_event,
        )
        self.enqueue(nav_event)

    def enqueue_from_rgb(
        self,
        message: str,
        priority: EventPriority,
        *,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        nav_event = NavigationEvent(
            timestamp=timestamp or time.time(),
            source=RGB_SOURCE,
            priority=priority,
            message=message,
            metadata=dict(metadata) if metadata else None,
        )
        self.enqueue(nav_event)

    # ------------------------------------------------------------------
    # worker
    # ------------------------------------------------------------------

    def _run(self) -> None:
        logger = get_navigation_logger().routing
        while self._running:
            try:
                _priority, _order, event = self.event_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if event is None:
                break

            self._on_event_processed(event)

            should_announce, trigger = self._should_announce(event)
            logger.debug(f"Event '{event.message}': should_announce={should_announce}, trigger={trigger}")
            
            if should_announce:
                self._apply_audio_settings(event)
                logger.info(f"Speaking: '{event.message}' (priority={event.priority.name}, source={event.source})")
                speak_ok = self.audio.speak_async(event.message)
                if speak_ok:
                    now = time.time()
                    self._last_global_announcement = now
                    self._last_source_announcement[event.source] = now
                    
                    # Track speaking state for anti-stutter logic
                    self._speaking_started = now
                    self._last_message = event.message
                    self._last_message_time = now
                    self._update_metrics(event, "spoken")
                    logger.debug(f"✓ Spoken successfully")
                else:
                    self._update_metrics(event, "skipped", reason="tts_rejected")
                    logger.warning(f"✗ TTS rejected message")
            else:
                self._update_metrics(event, "skipped", reason=trigger)
                logger.debug(f"✗ Skipped: {trigger}")

    def _should_announce(self, event: NavigationEvent) -> Tuple[bool, str]:
        """Determine if event should be announced with anti-entrecorte logic.
        
        New logic:
        - CRITICAL can interrupt anything after interrupt_grace period
        - MEDIUM/LOW cannot interrupt CRITICAL
        - Same message within grace period is skipped to avoid stuttering
        """
        now = time.time()

        metadata = event.metadata or {}
        cooldown_hint = metadata.get("cooldown")
        if cooldown_hint is not None:
            try:
                cooldown_value = max(0.0, float(cooldown_hint))
            except (TypeError, ValueError):
                cooldown_value = None
            if cooldown_value is not None:
                self.source_cooldown[event.source] = cooldown_value

        # Anti-stutter: skip if same message was just announced
        if self._last_message == event.message and now - self._last_message_time < 2.0:
            return False, "duplicate_message"

        # CRITICAL priority can interrupt after grace period
        if event.priority == EventPriority.CRITICAL:
            # Check if currently speaking
            if self.audio.is_speaking:
                time_speaking = now - self._speaking_started
                if time_speaking < self.interrupt_grace:
                    # Too soon to interrupt, even for critical
                    return False, "interrupt_grace_active"
                # Allow interrupt after grace period
            
            # Check global cooldown (reduced for critical)
            if now - self._last_global_announcement < (self.global_cooldown * 0.5):
                return False, "global_cooldown"
            
            return True, "priority_critical"

        # Non-critical events respect global cooldown fully
        if now - self._last_global_announcement < self.global_cooldown:
            return False, "global_cooldown"

        # Cannot interrupt if TTS is speaking (for non-critical)
        if self.audio.is_speaking:
            return False, "audio_busy"

        # Check source-specific cooldown
        last_source = self._last_source_announcement.get(event.source, 0.0)
        source_cooldown = self.source_cooldown.get(event.source, self.default_source_cooldown)
        if now - last_source < source_cooldown:
            return False, "source_cooldown"

        # SLAM-specific logic
        if not event.source.startswith("slam"):
            return True, "non_slam"

        # For peripheral events, check spacing between both SLAM channels
        slam_grace = self._router_config.slam_duplicate_grace
        primary_last = max(
            self._last_source_announcement.get(SLAM1_SOURCE, 0.0),
            self._last_source_announcement.get(SLAM2_SOURCE, 0.0),
        )
        if now - primary_last < slam_grace:
            return False, "slam_spacing"

        return True, "slam_allowed"

    def set_source_cooldown(self, source: str, cooldown: float) -> None:
        try:
            value = float(cooldown)
            if value < 0.0:
                value = 0.0
            self.source_cooldown[source] = value
        except (TypeError, ValueError):
            pass

    def _apply_audio_settings(self, event: NavigationEvent) -> None:
        metadata = event.metadata or {}

        cooldown_hint = metadata.get("cooldown")
        if cooldown_hint is not None:
            try:
                cooldown_value = max(0.0, float(cooldown_hint))
            except (TypeError, ValueError):
                cooldown_value = None
            if cooldown_value is not None:
                self.set_source_cooldown(event.source, cooldown_value)
                if hasattr(self.audio, "set_repeat_cooldown"):
                    self.audio.set_repeat_cooldown(cooldown_value)
                if hasattr(self.audio, "set_announcement_cooldown"):
                    self.audio.set_announcement_cooldown(max(0.0, cooldown_value * 0.5))


__all__ = ["NavigationAudioRouter", "EventPriority", "NavigationEvent"]