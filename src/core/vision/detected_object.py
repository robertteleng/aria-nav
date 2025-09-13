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
    depth_value: float = 0.5  # AÑADIR ESTA LÍNEA (0=lejos, 1=cerca)