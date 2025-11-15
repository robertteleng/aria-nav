"""Mensajes compartidos en la fase 2 multiprocesamiento."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class FrameMessage:
    frame_id: int
    camera: str  # 'central', 'slam1', 'slam2'
    frame: np.ndarray
    timestamp: float


@dataclass
class ResultMessage:
    frame_id: int
    camera: str
    detections: List[Dict[str, Any]]
    depth_map: Optional[np.ndarray] = None
    depth_raw: Optional[np.ndarray] = None
    latency_ms: float = 0.0
    profiling: Dict[str, float] = field(default_factory=dict)
