"""Asynchronous YOLO processing for SLAM peripheral cameras."""

from __future__ import annotations

import threading
import time
import queue
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .yolo_processor import YoloProcessor


class CameraSource(Enum):
    """Enumerate supported SLAM camera sources."""

    SLAM1 = "slam1"
    SLAM2 = "slam2"


@dataclass
class SlamDetectionEvent:
    """Lightweight event emitted by the SLAM worker."""

    timestamp: float
    source: CameraSource
    object_name: str
    confidence: float
    zone: str
    distance: str
    bbox: Tuple[int, int, int, int]
    frame_index: int
    processing_ms: float
    metadata: dict = field(default_factory=dict)
    track_id: Optional[int] = None  # Global track ID (set by GlobalObjectTracker)


class SlamDetectionWorker:
    """Process SLAM frames with a YOLO profile tuned for peripheral vision."""

    def __init__(
        self,
        source: CameraSource,
        *,
        processor: Optional[YoloProcessor] = None,
        target_fps: int = 8,
        queue_size: int = 2,
    ) -> None:
        self.source = source
        self.target_fps = max(1, target_fps)
        self.processor = processor or YoloProcessor.from_profile("slam")

        self.input_queue: "queue.Queue[Optional[dict]]" = queue.Queue(maxsize=queue_size)
        self.worker_thread: Optional[threading.Thread] = None
        self._running = False

        self._frame_counter = 0
        self._latest_events: List[SlamDetectionEvent] = []
        self._processing_window: List[float] = []
        self._lock = threading.Lock()

        print(
            f"[SLAM] Worker {self.source.value} ready | imgsz={self.processor.img_size} "
            f"| skip={self.processor.frame_skip} | target_fps={self.target_fps}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            name=f"SlamWorker-{self.source.value}",
            daemon=True,
        )
        self.worker_thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        try:
            self.input_queue.put_nowait(None)
        except queue.Full:
            pass
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)

    def submit(self, frame: np.ndarray, frame_index: int) -> None:
        """Push a frame to the worker without blocking."""
        item = {"frame": frame, "timestamp": time.time(), "frame_index": frame_index}
        try:
            self.input_queue.put_nowait(item)
        except queue.Full:
            # Drop oldest frame when worker lags behind
            try:
                _ = self.input_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.input_queue.put_nowait(item)
            except queue.Full:
                pass

    def latest_events(self) -> List[SlamDetectionEvent]:
        with self._lock:
            return list(self._latest_events)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        min_frame_time = 1.0 / self.target_fps
        frame_skip = max(1, self.processor.frame_skip)

        while self._running:
            try:
                item = self.input_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                break

            self._frame_counter += 1
            if self._frame_counter % frame_skip:
                continue

            frame = item["frame"]
            timestamp = item["timestamp"]
            frame_index = item["frame_index"]

            if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            start = time.perf_counter()
            detections = self.processor.process_frame(frame)
            processing_ms = (time.perf_counter() - start) * 1000.0

            # Mark detections as SLAM for correct filtering
            for det in detections:
                det["camera_source"] = "slam"

            events = self._convert_to_events(
                detections,
                timestamp,
                frame_index,
                processing_ms,
                frame_width=frame.shape[1] if frame is not None else None,
            )
            with self._lock:
                self._latest_events = events
                self._processing_window.append(processing_ms)
                if len(self._processing_window) > 30:
                    self._processing_window.pop(0)

            elapsed = time.perf_counter() - start
            if elapsed < min_frame_time:
                time.sleep(min_frame_time - elapsed)

    def _convert_to_events(
        self,
        detections: List[dict],
        timestamp: float,
        frame_index: int,
        processing_ms: float,
        frame_width: Optional[int],
    ) -> List[SlamDetectionEvent]:
        events: List[SlamDetectionEvent] = []
        for det in detections:
            bbox = det.get("bbox", (0, 0, 0, 0))
            zone = self._peripheral_zone(bbox, det.get("zone", "unknown"), frame_width)
            events.append(
                SlamDetectionEvent(
                    timestamp=timestamp,
                    source=self.source,
                    object_name=det.get("name", "object"),
                    confidence=float(det.get("confidence", 0.0)),
                    zone=zone,
                    distance=str(det.get("distance", "unknown")),
                    bbox=tuple(int(x) for x in bbox),
                    frame_index=frame_index,
                    processing_ms=processing_ms,
                    metadata={
                        "relevance_score": det.get("relevance_score"),
                        "raw_zone": det.get("zone"),
                    },
                )
            )
        return events

    def _peripheral_zone(
        self,
        bbox: Tuple[int, int, int, int],
        raw_zone: str,
        frame_width: Optional[int],
    ) -> str:
        if raw_zone in {"left", "right"}:
            return raw_zone

        x1, _, x2, _ = bbox
        if frame_width and frame_width > 0:
            center = (x1 + x2) / 2
            if self.source == CameraSource.SLAM1:
                return "far_left" if center < frame_width * 0.4 else "left"
            if self.source == CameraSource.SLAM2:
                return "right" if center < frame_width * 0.6 else "far_right"
        return raw_zone or "peripheral"


__all__ = [
    "CameraSource",
    "SlamDetectionEvent",
    "SlamDetectionWorker",
]
