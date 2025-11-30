"""🌍 Global object tracker for cross-camera tracking (RGB + SLAM1 + SLAM2)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from utils.config import Config


@dataclass
class GlobalTrack:
    """Represents a tracked object across multiple cameras."""
    track_id: int
    class_name: str
    last_camera: str  # "rgb", "slam1", "slam2"
    last_bbox: Tuple[float, float, float, float]  # (x, y, w, h)
    last_zone: str  # "far_left", "left", "center", "right", "far_right"
    last_seen: float
    last_announced: float
    history: List[Dict] = field(default_factory=list)  # Last 5 detections

    def add_detection(self, detection: Dict, camera: str, zone: str, now: float):
        """Add detection to history (keep last 5)."""
        self.history.append({
            "camera": camera,
            "bbox": detection.get("bbox"),
            "zone": zone,
            "timestamp": now,
        })
        if len(self.history) > 5:
            self.history.pop(0)

        # Update current state
        self.last_camera = camera
        self.last_bbox = detection.get("bbox")
        self.last_zone = zone
        self.last_seen = now


class GlobalObjectTracker:
    """
    Unified tracker for RGB + SLAM1 + SLAM2 with shared track IDs.

    Tracking strategy:
    - Intra-camera: IoU-based matching (same as ObjectTracker)
    - Cross-camera: Temporal handoff based on class + zone + time

    Example handoff:
        t=0s: SLAM1 detects "person" in "far_left" → track_id=5
        t=1s: RGB detects "person" in "left" → recognizes as same person → track_id=5
    """

    def __init__(
        self,
        iou_threshold: float = 0.5,
        max_age: float = 3.0,
        handoff_timeout: float = 2.0,
    ):
        """
        Initialize global tracker.

        Args:
            iou_threshold: Minimum IoU for intra-camera matching (default 0.5)
            max_age: Maximum time to keep track without seeing object (default 3.0s)
            handoff_timeout: Maximum time for cross-camera handoff (default 2.0s)
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.handoff_timeout = handoff_timeout

        self.tracks: Dict[int, GlobalTrack] = {}  # track_id -> GlobalTrack
        self.next_id = 0

        # Zone transition rules for cross-camera handoff
        # Maps (src_camera, src_zone) → valid (dst_camera, dst_zone) transitions
        self.valid_transitions = self._build_transition_rules()

    def _build_transition_rules(self) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
        """
        Build zone transition rules for cross-camera handoff.

        Rules based on Aria cameras geometry:
        - SLAM1 (left fisheye): covers far_left + left
        - RGB (frontal): covers left + center + right
        - SLAM2 (right fisheye): covers right + far_right
        """
        return {
            # SLAM1 → RGB transitions
            ("slam1", "far_left"): [("rgb", "left"), ("rgb", "center")],
            ("slam1", "left"): [("rgb", "left"), ("rgb", "center")],

            # RGB → SLAM1 transitions (reverse)
            ("rgb", "left"): [("slam1", "left"), ("slam1", "far_left"), ("rgb", "left")],
            ("rgb", "center"): [("slam1", "left"), ("slam2", "right"), ("rgb", "center")],

            # SLAM2 → RGB transitions
            ("slam2", "right"): [("rgb", "right"), ("rgb", "center")],
            ("slam2", "far_right"): [("rgb", "right"), ("rgb", "center")],

            # RGB → SLAM2 transitions (reverse)
            ("rgb", "right"): [("slam2", "right"), ("slam2", "far_right"), ("rgb", "right")],

            # Same camera transitions (always valid)
            ("rgb", "far_left"): [("rgb", "far_left"), ("rgb", "left")],
            ("rgb", "far_right"): [("rgb", "far_right"), ("rgb", "right")],
            ("slam1", "center"): [("slam1", "center"), ("slam1", "left")],
            ("slam2", "center"): [("slam2", "center"), ("slam2", "right")],
        }

    def update_and_check(
        self,
        detections: List[Dict],
        cooldown_per_class: Dict[str, float],
        camera_source: str = "rgb",
    ) -> List[Tuple[Dict, int, bool]]:
        """
        Update tracker with new detections and check if each should be announced.

        Args:
            detections: List of detection dicts with 'class', 'bbox', 'zone', etc.
            cooldown_per_class: Dict mapping class_name → cooldown_seconds
            camera_source: Camera identifier ("rgb", "slam1", "slam2")

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
            zone = detection.get("zone", "center")

            if not bbox or not class_name:
                continue

            # Try to match with existing track (intra-camera or cross-camera)
            track_id = self._match_or_create(
                class_name=class_name,
                bbox=bbox,
                zone=zone,
                camera_source=camera_source,
                now=now,
            )
            matched_track_ids.add(track_id)

            # Check if should announce based on per-instance cooldown
            track = self.tracks[track_id]
            cooldown = cooldown_per_class.get(class_name, 2.0)
            time_since_announce = now - track.last_announced
            should_announce = time_since_announce >= cooldown

            results.append((detection, track_id, should_announce))

        # Update last_seen for all matched tracks
        for track_id in matched_track_ids:
            if track_id in self.tracks:
                self.tracks[track_id].last_seen = now

        return results

    def mark_announced(self, track_id: int) -> None:
        """Mark a track as announced (updates last_announced timestamp)."""
        if track_id in self.tracks:
            self.tracks[track_id].last_announced = time.time()

    def _match_or_create(
        self,
        class_name: str,
        bbox: Tuple[float, float, float, float],
        zone: str,
        camera_source: str,
        now: float,
    ) -> int:
        """
        Match detection to existing track or create new one.

        Strategy:
        1. Try intra-camera matching (same camera, IoU-based)
        2. Try cross-camera handoff (different camera, temporal + zone)
        3. Create new track if no match

        Returns:
            track_id: ID of matched or newly created track
        """
        # 1. Try intra-camera matching (same camera, IoU-based)
        intra_match = self._find_intra_camera_match(class_name, bbox, camera_source)
        if intra_match is not None:
            # Update existing track
            track = self.tracks[intra_match]
            track.add_detection({"bbox": bbox, "class": class_name}, camera_source, zone, now)
            return intra_match

        # 2. Try cross-camera handoff (different camera, temporal + zone)
        handoff_match = self._find_handoff_candidate(class_name, zone, camera_source, now)
        if handoff_match is not None:
            # Handoff existing track to new camera
            track = self.tracks[handoff_match]
            track.add_detection({"bbox": bbox, "class": class_name}, camera_source, zone, now)
            return handoff_match

        # 3. Create new track
        new_track_id = self.next_id
        self.next_id += 1
        self.tracks[new_track_id] = GlobalTrack(
            track_id=new_track_id,
            class_name=class_name,
            last_camera=camera_source,
            last_bbox=bbox,
            last_zone=zone,
            last_seen=now,
            last_announced=0.0,  # Allow immediate announcement for new objects
            history=[],
        )
        self.tracks[new_track_id].add_detection(
            {"bbox": bbox, "class": class_name}, camera_source, zone, now
        )
        return new_track_id

    def _find_intra_camera_match(
        self,
        class_name: str,
        bbox: Tuple[float, float, float, float],
        camera_source: str,
    ) -> Optional[int]:
        """
        Find matching track within same camera using IoU.

        Returns:
            track_id if match found, None otherwise
        """
        best_iou = 0.0
        best_track_id = None

        for track_id, track in self.tracks.items():
            # Must be same camera and same class
            if track.last_camera != camera_source or track.class_name != class_name:
                continue

            iou = self._calculate_iou(bbox, track.last_bbox)
            if iou > best_iou and iou >= self.iou_threshold:
                best_iou = iou
                best_track_id = track_id

        return best_track_id

    def _find_handoff_candidate(
        self,
        class_name: str,
        zone: str,
        camera_source: str,
        now: float,
    ) -> Optional[int]:
        """
        Find cross-camera handoff candidate using temporal + zone matching.

        Rules:
        - Same class
        - Different camera
        - Last seen within handoff_timeout
        - Zone transition is valid (e.g., SLAM1 far_left → RGB left)

        Returns:
            track_id if handoff candidate found, None otherwise
        """
        candidates = []

        for track_id, track in self.tracks.items():
            # Must be same class
            if track.class_name != class_name:
                continue

            # Must be different camera
            if track.last_camera == camera_source:
                continue

            # Must be recently seen
            time_since_seen = now - track.last_seen
            if time_since_seen > self.handoff_timeout:
                continue

            # Check if zone transition is valid
            if self._is_valid_transition(
                src_camera=track.last_camera,
                src_zone=track.last_zone,
                dst_camera=camera_source,
                dst_zone=zone,
            ):
                candidates.append((track_id, time_since_seen))

        # Return most recent candidate
        if candidates:
            candidates.sort(key=lambda x: x[1])  # Sort by time_since_seen (ascending)
            return candidates[0][0]

        return None

    def _is_valid_transition(
        self,
        src_camera: str,
        src_zone: str,
        dst_camera: str,
        dst_zone: str,
    ) -> bool:
        """
        Check if zone transition between cameras is geometrically valid.

        Example valid transitions:
        - SLAM1 far_left → RGB left (person moving from left peripheral to frontal)
        - RGB right → SLAM2 far_right (person moving from frontal to right peripheral)

        Example invalid transitions:
        - SLAM1 far_left → RGB far_right (person can't teleport across FOV)
        """
        key = (src_camera, src_zone)
        valid_targets = self.valid_transitions.get(key, [])
        return (dst_camera, dst_zone) in valid_targets

    def _cleanup_old_tracks(self, now: float) -> None:
        """Remove tracks that haven't been seen for max_age seconds."""
        to_remove = [
            track_id
            for track_id, track in self.tracks.items()
            if (now - track.last_seen) > self.max_age
        ]
        for track_id in to_remove:
            del self.tracks[track_id]

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
            "active_tracks": len(self.tracks),
            "next_id": self.next_id,
            "tracks_by_class": self._count_by_class(),
            "tracks_by_camera": self._count_by_camera(),
        }

    def _count_by_class(self) -> Dict[str, int]:
        """Count active tracks per class."""
        counts = {}
        for track in self.tracks.values():
            counts[track.class_name] = counts.get(track.class_name, 0) + 1
        return counts

    def _count_by_camera(self) -> Dict[str, int]:
        """Count active tracks per camera."""
        counts = {}
        for track in self.tracks.values():
            counts[track.last_camera] = counts.get(track.last_camera, 0) + 1
        return counts


__all__ = ["GlobalObjectTracker", "GlobalTrack"]
