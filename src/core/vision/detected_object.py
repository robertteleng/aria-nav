"""
Detected object data structure for YOLO detection results.

This module defines the DetectedObject dataclass which stores all information
about a detected object including bounding box, spatial zone, depth estimation,
and relevance scoring for navigation decisions.

Usage:
    obj = DetectedObject(
        name="person",
        confidence=0.95,
        bbox=(100, 150, 200, 300),
        center_x=0.5,
        center_y=0.6,
        area=0.12,
        zone="center",
        distance_bucket="close",
        relevance_score=0.85,
        depth_value=0.7
    )
"""

from typing import Tuple
from dataclasses import dataclass


@dataclass
class DetectedObject:
    """
    A detected object with complete spatial and confidence information.

    Attributes:
        name: Object class name (e.g., "person", "chair", "car")
        confidence: Detection confidence score (0-1)
        bbox: Bounding box (x, y, width, height) in pixels
        center_x: Normalized center X coordinate (0-1)
        center_y: Normalized center Y coordinate (0-1)
        area: Normalized bounding box area (0-1)
        zone: Spatial zone ("left", "center", "right", "far_left", "far_right")
        distance_bucket: Distance category ("very_close", "close", "medium", "far")
        relevance_score: Navigation relevance score (0-1)
        depth_value: Normalized depth (0 = far, 1 = near)
        depth_raw: Optional raw depth value in model-specific scale
    """
    name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center_x: float
    center_y: float
    area: float
    zone: str
    distance_bucket: str
    relevance_score: float
    depth_value: float = 0.5  # Normalized depth (0 = far, 1 = near)
    depth_raw: float | None = None  # Mean raw depth value (model-specific scale)
