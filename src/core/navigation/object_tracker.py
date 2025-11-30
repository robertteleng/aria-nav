"""🆕 Simple object tracker for per-instance cooldowns (NOA-inspired)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class TrackedObject:
    """Represents a tracked object with persistence."""
    track_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]  # (x, y, w, h)
    last_seen: float
    last_announced: float


class ObjectTracker:
    """
    Simple object tracker using IoU matching for per-instance cooldowns.

    Inspired by NOA's approach: assign unique IDs to objects based on
    bounding box overlap, enabling per-instance cooldowns instead of
    per-class cooldowns.

    Example:
        - person_0 at left: announced at t=0
        - person_1 at right: can be announced at t=0.1 (different instance)
        - person_0 moves slightly: cannot be re-announced until cooldown expires
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        max_age: float = 3.0,
    ):
        """
        Initialize object tracker.

        Args:
            iou_threshold: Minimum IoU to consider objects as same instance (default 0.5)
            max_age: Maximum time (seconds) to keep track without seeing object (default 3.0s)
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age

        self.tracked_objects: Dict[int, TrackedObject] = {}  # track_id -> TrackedObject
        self.next_id = 0

    def update_and_check(
        self,
        detections: List[Dict],
        cooldown_per_class: Dict[str, float],
    ) -> List[Tuple[Dict, int, bool]]:
        """
        Update tracker with new detections and check if each should be announced.

        Args:
            detections: List of detection dicts with 'class', 'bbox', etc.
            cooldown_per_class: Dict mapping class_name -> cooldown_seconds

        Returns:
            List of tuples: (detection, track_id, should_announce)
        """
        now = time.time()

        # Clean up old tracks
        self._cleanup_old_tracks(now)

        # Match detections to existing tracks
        results = []
        matched_track_ids = set()

        for detection in detections:
            class_name = detection.get("class", "").lower()
            bbox = detection.get("bbox")
            if not bbox or not class_name:
                continue

            # Try to match with existing track
            track_id, matched = self._match_or_create(class_name, bbox, now)
            matched_track_ids.add(track_id)

            # Check if should announce based on per-instance cooldown
            tracked = self.tracked_objects[track_id]
            cooldown = cooldown_per_class.get(class_name, 2.0)
            time_since_announce = now - tracked.last_announced
            should_announce = time_since_announce >= cooldown

            results.append((detection, track_id, should_announce))

        # Update last_seen for all matched tracks
        for track_id in matched_track_ids:
            if track_id in self.tracked_objects:
                self.tracked_objects[track_id].last_seen = now

        return results

    def mark_announced(self, track_id: int) -> None:
        """Mark a track as announced (updates last_announced timestamp)."""
        if track_id in self.tracked_objects:
            self.tracked_objects[track_id].last_announced = time.time()

    def _match_or_create(
        self,
        class_name: str,
        bbox: Tuple[float, float, float, float],
        now: float,
    ) -> Tuple[int, bool]:
        """
        Match detection to existing track or create new one.

        Returns:
            (track_id, is_matched): tuple of track ID and whether it was matched (vs created)
        """
        best_iou = 0.0
        best_track_id = None

        # Find best matching track of same class
        for track_id, tracked in self.tracked_objects.items():
            if tracked.class_name != class_name:
                continue

            iou = self._calculate_iou(bbox, tracked.bbox)
            if iou > best_iou and iou >= self.iou_threshold:
                best_iou = iou
                best_track_id = track_id

        if best_track_id is not None:
            # Update existing track
            self.tracked_objects[best_track_id].bbox = bbox
            return best_track_id, True

        # Create new track
        new_track_id = self.next_id
        self.next_id += 1
        self.tracked_objects[new_track_id] = TrackedObject(
            track_id=new_track_id,
            class_name=class_name,
            bbox=bbox,
            last_seen=now,
            last_announced=0.0,  # Allow immediate announcement for new objects
        )
        return new_track_id, False

    def _cleanup_old_tracks(self, now: float) -> None:
        """Remove tracks that haven't been seen for max_age seconds."""
        to_remove = [
            track_id
            for track_id, tracked in self.tracked_objects.items()
            if (now - tracked.last_seen) > self.max_age
        ]
        for track_id in to_remove:
            del self.tracked_objects[track_id]

    @staticmethod
    def _calculate_iou(
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float],
    ) -> float:
        """
        Calculate Intersection over Union (IoU) between two bboxes.

        Args:
            bbox1, bbox2: (x, y, w, h) format

        Returns:
            IoU score (0.0 to 1.0)
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Convert to (x1, y1, x2, y2) format
        box1_x2 = x1 + w1
        box1_y2 = y1 + h1
        box2_x2 = x2 + w2
        box2_y2 = y2 + h2

        # Calculate intersection
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)

        if inter_x2 < inter_x1 or inter_y2 < inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def get_stats(self) -> Dict:
        """Get tracker statistics for debugging."""
        return {
            "active_tracks": len(self.tracked_objects),
            "next_id": self.next_id,
            "tracks_by_class": self._count_by_class(),
        }

    def _count_by_class(self) -> Dict[str, int]:
        """Count active tracks per class."""
        counts = {}
        for tracked in self.tracked_objects.values():
            counts[tracked.class_name] = counts.get(tracked.class_name, 0) + 1
        return counts


__all__ = ["ObjectTracker", "TrackedObject"]
