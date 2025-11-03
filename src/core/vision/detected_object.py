from typing import Tuple
from dataclasses import dataclass

@dataclass
class DetectedObject:
    """A detected object with spatial information"""
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
