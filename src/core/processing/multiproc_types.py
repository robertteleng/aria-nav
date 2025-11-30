"""
Shared message types for Phase 2 multiprocessing architecture.

This module defines dataclass message types used for inter-process communication
in the multiprocessing vision pipeline. Messages are passed through queues between
the main process and worker processes.

Message Types:
- FrameMessage: Input frames sent to workers
- ResultMessage: Detection results returned from workers

Usage:
    # Send frame to worker
    frame_msg = FrameMessage(
        frame_id=42,
        camera='rgb',
        frame=frame_array,
        timestamp=time.time()
    )

    # Receive results from worker
    result_msg = ResultMessage(
        frame_id=42,
        camera='rgb',
        detections=[...],
        depth_map=depth_array,
        latency_ms=15.2
    )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class FrameMessage:
    """
    Input frame message sent to worker process.

    Attributes:
        frame_id: Unique frame identifier
        camera: Camera source ('rgb', 'slam1', 'slam2')
        frame: BGR frame array
        timestamp: Frame capture timestamp
    """
    frame_id: int
    camera: str  # 'rgb', 'slam1', 'slam2'
    frame: np.ndarray
    timestamp: float


@dataclass
class ResultMessage:
    """
    Detection result message returned from worker process.

    Attributes:
        frame_id: Matching frame identifier
        camera: Camera source
        detections: List of detected objects with bbox, class, confidence
        depth_map: Optional estimated depth map
        depth_raw: Optional raw depth values
        latency_ms: Processing latency in milliseconds
        profiling: Optional profiling metrics dict
    """
    frame_id: int
    camera: str
    detections: List[Dict[str, Any]]
    depth_map: Optional[np.ndarray] = None
    depth_raw: Optional[np.ndarray] = None
    latency_ms: float = 0.0
    profiling: Dict[str, float] = field(default_factory=dict)
