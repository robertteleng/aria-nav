"""Modular pipeline for RGB frame processing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from utils.config import Config

try:
    from core.vision.depth_estimator import DepthEstimator
except Exception:
    DepthEstimator = None


@dataclass
class PipelineResult:
    """Container for pipeline outputs."""

    frame: np.ndarray
    detections: Any
    depth_map: Optional[np.ndarray]
    timings: Dict[str, float]


class NavigationPipeline:
    """Encapsulates enhancement, depth estimation and YOLO detection."""

    def __init__(
        self,
        yolo_processor,
        *,
        image_enhancer=None,
        depth_estimator=None,
    ) -> None:
        self.yolo_processor = yolo_processor
        self.image_enhancer = image_enhancer
        self.depth_estimator = depth_estimator or self._build_depth_estimator()

        self.depth_frame_skip = max(1, getattr(Config, "DEPTH_FRAME_SKIP", 1))
        self.latest_depth_map: Optional[np.ndarray] = None
        self.frames_processed = 0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray, *, profile: bool = False) -> PipelineResult:
        """Run the RGB processing pipeline."""

        self.frames_processed += 1

        timings: Dict[str, float] = {}

        processed_frame = frame
        if self.image_enhancer is not None:
            enhance_start = time.perf_counter() if profile else None
            try:
                processed_frame = self.image_enhancer.enhance_frame(frame)
            except Exception as err:
                print(f"[WARN] Image enhancement skipped: {err}")
                processed_frame = frame
            if profile and enhance_start is not None:
                timings["enhance"] = time.perf_counter() - enhance_start

        depth_map = None
        if self.depth_estimator is not None and getattr(self.depth_estimator, "model", None) is not None:
            depth_start = time.perf_counter() if profile else None
            try:
                if self.frames_processed % self.depth_frame_skip == 0:
                    depth_candidate = self.depth_estimator.estimate_depth(processed_frame)
                    if depth_candidate is not None:
                        self.latest_depth_map = depth_candidate
                depth_map = self.latest_depth_map
            except Exception as err:
                print(f"[WARN] Depth estimation skipped: {err}")
            if profile and depth_start is not None:
                timings["depth"] = time.perf_counter() - depth_start

        yolo_start = time.perf_counter() if profile else None
        detections = self.yolo_processor.process_frame(processed_frame, depth_map)
        if profile and yolo_start is not None:
            timings["yolo"] = time.perf_counter() - yolo_start

        return PipelineResult(
            frame=processed_frame,
            detections=detections,
            depth_map=depth_map,
            timings=timings,
        )

    def get_latest_depth_map(self) -> Optional[np.ndarray]:
        return self.latest_depth_map

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_depth_estimator(self):
        if not getattr(Config, "DEPTH_ENABLED", False) or DepthEstimator is None:
            return None
        try:
            estimator = DepthEstimator()
            if getattr(estimator, "model", None) is None:
                print("[WARN] Depth estimator initialized without model (disabled)")
            return estimator
        except Exception as err:
            print(f"[WARN] Depth estimator init failed: {err}")
            return None


__all__ = ["NavigationPipeline", "PipelineResult"]
