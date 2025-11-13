"""Modular pipeline for RGB frame processing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from utils.config import Config
from utils.depth_logger import get_depth_logger

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
    depth_raw: Optional[np.ndarray]
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
        self.latest_depth_raw: Optional[np.ndarray] = None
        self.frames_processed = 0
        
        # Log depth estimator status
        if self.depth_estimator is None:
            print("[WARN] ⚠️ Depth estimator is None - depth estimation disabled")
        elif getattr(self.depth_estimator, "model", None) is None:
            print("[WARN] ⚠️ Depth estimator model failed to load - depth estimation disabled")
        else:
            print(f"[INFO] ✅ Depth estimator initialized: {getattr(self.depth_estimator, 'backend', 'unknown')}")

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
        depth_raw = None
        if self.depth_estimator is not None and getattr(self.depth_estimator, "model", None) is not None:
            depth_start = time.perf_counter() if profile else None
            try:
                if self.frames_processed % self.depth_frame_skip == 0:
                    depth_prediction = None
                    if hasattr(self.depth_estimator, "estimate_depth_with_details"):
                        depth_prediction = self.depth_estimator.estimate_depth_with_details(processed_frame)
                        if depth_prediction is not None:
                            self.latest_depth_map = depth_prediction.map_8bit
                            self.latest_depth_raw = getattr(depth_prediction, "raw", None)
                            if self.frames_processed % 100 == 0:  # Log cada 100 frames
                                print(f"[DEPTH] Frame {self.frames_processed}: Depth computed, shape={depth_prediction.map_8bit.shape}, inference={depth_prediction.inference_ms:.1f}ms")
                    else:
                        depth_candidate = self.depth_estimator.estimate_depth(processed_frame)
                        if depth_candidate is not None:
                            self.latest_depth_map = depth_candidate
                            self.latest_depth_raw = None
                            if self.frames_processed % 100 == 0:
                                print(f"[DEPTH] Frame {self.frames_processed}: Depth computed, shape={depth_candidate.shape}")
                depth_map = self.latest_depth_map
                depth_raw = self.latest_depth_raw
            except Exception as err:
                print(f"[WARN] Depth estimation skipped: {err}")
            if profile and depth_start is not None:
                timings["depth"] = time.perf_counter() - depth_start

        yolo_start = time.perf_counter() if profile else None
        detections = self.yolo_processor.process_frame(
            processed_frame,
            depth_map,
            depth_raw,
        )
        if profile and yolo_start is not None:
            timings["yolo"] = time.perf_counter() - yolo_start

        return PipelineResult(
            frame=processed_frame,
            detections=detections,
            depth_map=depth_map,
            depth_raw=depth_raw,
            timings=timings,
        )

    def get_latest_depth_map(self) -> Optional[np.ndarray]:
        return self.latest_depth_map

    def get_latest_depth_raw(self) -> Optional[np.ndarray]:
        return self.latest_depth_raw

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_depth_estimator(self):
        logger = get_depth_logger()
        logger.section("NavigationPipeline: Building Depth Estimator")
        
        depth_enabled = getattr(Config, "DEPTH_ENABLED", False)
        logger.log(f"DEPTH_ENABLED={depth_enabled}")
        logger.log(f"DepthEstimator class={'Available' if DepthEstimator else 'None'}")
        print(f"[DEBUG] _build_depth_estimator: DEPTH_ENABLED={depth_enabled}, DepthEstimator={'Available' if DepthEstimator else 'None'}")
        
        if not depth_enabled:
            logger.log("Depth estimation disabled in config - returning None")
            print("[INFO] Depth estimation disabled in config")
            return None
        if DepthEstimator is None:
            logger.log("DepthEstimator class not available (import failed) - returning None")
            print("[WARN] DepthEstimator class not available (import failed)")
            return None
            
        try:
            logger.log("Creating DepthEstimator instance...")
            print("[INFO] Creating DepthEstimator instance...")
            estimator = DepthEstimator()
            
            if getattr(estimator, "model", None) is None:
                logger.log("⚠️ Estimator created but model is None")
                print("[WARN] ⚠️ Depth estimator initialized without model (disabled)")
            else:
                logger.log(f"✅ Estimator created successfully with backend: {getattr(estimator, 'backend', 'unknown')}")
            
            return estimator
        except Exception as err:
            logger.log(f"❌ Exception during estimator creation: {err}")
            print(f"[ERROR] ❌ Depth estimator init failed: {err}")
            import traceback
            tb = traceback.format_exc()
            logger.log(f"Traceback:\n{tb}")
            traceback.print_exc()
            return None


__all__ = ["NavigationPipeline", "PipelineResult"]
