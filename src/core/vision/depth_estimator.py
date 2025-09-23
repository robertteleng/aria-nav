"""Depth estimation helpers tuned for MPS on Apple Silicon."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch

from utils.config import Config
from .mps_utils import configure_mps_environment, empty_mps_cache, get_preferred_device


@dataclass
class DepthPrediction:
    map_8bit: np.ndarray
    raw: np.ndarray
    inference_ms: float


class DepthEstimator:
    """MiDaS / Depth-Anything depth estimator with MPS fallbacks."""

    def __init__(self) -> None:
        if not Config.DEPTH_ENABLED:
            self.model = None
            self.backend = None
            self.device = torch.device("cpu")
            self.transform = None
            self.processor = None
            self.last_inference_ms = 0.0
            print("[INFO] Depth estimator disabled via configuration")
            return

        configure_mps_environment(getattr(Config, "YOLO_FORCE_MPS", False))

        self.backend = getattr(Config, "DEPTH_BACKEND", "midas").lower()
        self.device = get_preferred_device(getattr(Config, "MIDAS_DEVICE", "mps"))
        self.transform = None
        self.processor = None
        self.last_inference_ms = 0.0
        self.input_size = getattr(Config, "DEPTH_INPUT_SIZE", 384)

        print(f"[INFO] Loading depth backend '{self.backend}' on {self.device.type}...")

        try:
            if self.backend == "midas":
                self._load_midas()
            elif self.backend == "depth_anything_v2":
                self._load_depth_anything()
            else:
                raise ValueError(f"Unsupported depth backend: {self.backend}")
            print("[INFO] ✓ Depth estimator ready")
        except Exception as err:  # noqa: BLE001
            print(f"[ERROR] MiDaS loading failed: {err}")
            self.model = None

    def _load_midas(self) -> None:
        model_name = getattr(Config, "MIDAS_MODEL", "MiDaS_small")
        self.model = torch.hub.load("intel-isl/MiDaS", model_name)
        self.model.to(self.device)
        self.model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_name in {"DPT_Large", "DPT_Hybrid"}:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def _load_depth_anything(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        variant = getattr(Config, "DEPTH_ANYTHING_VARIANT", "Small")
        name = f"depth-anything/Depth-Anything-V2-{variant}-hf"

        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModelForDepthEstimation.from_pretrained(name)
        self.model.to(self.device)
        self.model.eval()

    def estimate_depth(self, rgb_frame: np.ndarray) -> Optional[np.ndarray]:
        """Estimate an 8-bit depth map from an RGB frame."""
        prediction = self.estimate_depth_with_details(rgb_frame)
        if prediction is None:
            return None
        return prediction.map_8bit

    def estimate_depth_with_details(self, rgb_frame: np.ndarray) -> Optional[DepthPrediction]:
        if self.model is None:
            return None

        start = time.perf_counter()

        try:
            if rgb_frame.ndim == 3:
                rgb_input = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_input = rgb_frame

            if self.backend == "midas":
                depth = self._run_midas(rgb_input)
            else:
                depth = self._run_depth_anything(rgb_input)

            depth_8bit = cv2.normalize(
                depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )

            self.last_inference_ms = (time.perf_counter() - start) * 1000.0

            if self.device.type == "mps":
                empty_mps_cache()

            return DepthPrediction(map_8bit=depth_8bit, raw=depth, inference_ms=self.last_inference_ms)
        except Exception as err:  # noqa: BLE001
            print(f"[ERROR] Depth estimation failed: {err}")
            return None

    def _run_midas(self, rgb_input: np.ndarray) -> np.ndarray:
        assert self.transform is not None

        if self.input_size and max(rgb_input.shape[:2]) > self.input_size:
            scale = self.input_size / max(rgb_input.shape[:2])
            new_size = (int(rgb_input.shape[1] * scale), int(rgb_input.shape[0] * scale))
            rgb_resized = cv2.resize(rgb_input, new_size, interpolation=cv2.INTER_AREA)
        else:
            rgb_resized = rgb_input

        input_tensor = self.transform(rgb_resized).to(self.device)

        with torch.inference_mode():
            prediction = self.model(input_tensor)

            if prediction.device.type == "mps":
                prediction = prediction.to("cpu")

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb_input.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        return prediction.cpu().numpy()

    def _run_depth_anything(self, rgb_input: np.ndarray) -> np.ndarray:
        assert self.processor is not None

        from torch.nn.functional import interpolate

        if self.input_size and max(rgb_input.shape[:2]) > self.input_size:
            scale = self.input_size / max(rgb_input.shape[:2])
            new_size = (int(rgb_input.shape[1] * scale), int(rgb_input.shape[0] * scale))
            rgb_resized = cv2.resize(rgb_input, new_size, interpolation=cv2.INTER_AREA)
        else:
            rgb_resized = rgb_input

        inputs = self.processor(images=rgb_resized, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            depth = self.model(**inputs).predicted_depth

            if depth.device.type == "mps":
                depth = depth.to("cpu")

            depth = interpolate(
                depth.unsqueeze(1),
                size=rgb_input.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        return depth.cpu().numpy()
