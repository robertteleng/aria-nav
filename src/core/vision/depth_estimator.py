"""Depth estimation helpers tuned for MPS on Apple Silicon."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch

from utils.config import Config
from utils.depth_logger import get_depth_logger
from .gpu_utils import configure_mps_environment, empty_mps_cache, get_preferred_device


@dataclass
class DepthPrediction:
    map_8bit: np.ndarray
    raw: np.ndarray
    inference_ms: float


class DepthEstimator:
    """MiDaS / Depth-Anything depth estimator with MPS fallbacks."""

    def __init__(self) -> None:
        logger = get_depth_logger()
        logger.section("DepthEstimator Initialization")
        
        if not Config.DEPTH_ENABLED:
            logger.log("DEPTH_ENABLED = False - Depth estimation disabled")
            self.model = None
            self.backend = None
            self.device = torch.device("cpu")
            self.transform = None
            self.processor = None
            self.last_inference_ms = 0.0
            print("[INFO] ⚠️ Depth estimator disabled via configuration")
            return

        logger.log(f"DEPTH_ENABLED = {Config.DEPTH_ENABLED}")
        
        configure_mps_environment(getattr(Config, "YOLO_FORCE_MPS", False))

        self.backend = getattr(Config, "DEPTH_BACKEND", "midas").lower()
        logger.log(f"Backend selected: {self.backend}")
        
        # Usar el device correcto: DEPTH_ANYTHING_DEVICE para depth_anything, MIDAS_DEVICE para midas
        if self.backend == "depth_anything_v2":
            device_config = getattr(Config, "DEPTH_ANYTHING_DEVICE", "cuda")
        else:
            device_config = getattr(Config, "MIDAS_DEVICE", "cuda")
        
        logger.log(f"Device config from Config: {device_config}")
        self.device = get_preferred_device(device_config)
        logger.log(f"Device resolved: {self.device.type}")
        
        self.transform = None
        self.processor = None
        self.last_inference_ms = 0.0
        self.input_size = getattr(Config, "DEPTH_INPUT_SIZE", 384)
        self._last_map_8bit: Optional[np.ndarray] = None
        
        # FASE 1 / Tarea 3: Habilitar pinned memory y non-blocking transfers
        self.use_pinned_memory = getattr(Config, 'PINNED_MEMORY', False) and torch.cuda.is_available()
        self.non_blocking = getattr(Config, 'NON_BLOCKING_TRANSFER', False)
        if self.use_pinned_memory:
            logger.log("✓ Depth: Pinned memory enabled")
        if self.non_blocking:
            logger.log("✓ Depth: Non-blocking transfers enabled")

        logger.log(f"Input size: {self.input_size}")
        logger.log(f"Frame skip: {getattr(Config, 'DEPTH_FRAME_SKIP', 1)}")
        print(f"[INFO] 🔍 Initializing depth backend '{self.backend}' on {self.device.type}...")
        print(f"[INFO]   - Input size: {self.input_size}")
        print(f"[INFO]   - Frame skip: {getattr(Config, 'DEPTH_FRAME_SKIP', 1)}")

        try:
            logger.log(f"Loading model for backend: {self.backend}")
            if self.backend == "midas":
                self._load_midas()
            elif self.backend == "depth_anything_v2":
                self._load_depth_anything()
            else:
                raise ValueError(f"Unsupported depth backend: {self.backend}")
            
            logger.log(f"✅ Model loaded successfully")
            logger.log(f"Model device: {self.device}")
            logger.log(f"Model type: {type(self.model).__name__}")
            print(f"[INFO] ✅ Depth estimator ready ({self.backend})")
        except Exception as err:  # noqa: BLE001
            logger.log(f"❌ ERROR during model loading: {err}")
            print(f"[ERROR] ❌ Depth loading failed: {err}")
            import traceback
            tb = traceback.format_exc()
            logger.log(f"Traceback:\n{tb}")
            traceback.print_exc()
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

        variant = getattr(Config, "DEPTH_ANYTHING_MODEL", "Small")  # FIX: usar DEPTH_ANYTHING_MODEL
        name = f"depth-anything/Depth-Anything-V2-{variant}-hf"
        
        print(f"[INFO] Loading Depth Anything V2 model: {name}")
        print(f"[INFO] Target device: {self.device}")

        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModelForDepthEstimation.from_pretrained(name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[INFO] ✓ Depth Anything V2 '{variant}' loaded successfully on {self.device}")

    def estimate_depth(self, rgb_frame: np.ndarray) -> Optional[np.ndarray]:
        """Estimate an 8-bit depth map from an RGB frame."""
        prediction = self.estimate_depth_with_details(rgb_frame)
        if prediction is None:
            return None
        return prediction.map_8bit

    def estimate_depth_with_details(self, rgb_frame: np.ndarray) -> Optional[DepthPrediction]:
        logger = get_depth_logger()
        
        if self.model is None:
            logger.log("estimate_depth called but model is None")
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

            depth_min = float(np.nanmin(depth))
            depth_max = float(np.nanmax(depth))
            if depth_max - depth_min < 1e-6:
                print("[WARN] Depth map range too small, reusing previous depth frame")
                depth_8bit = (
                    self._last_map_8bit
                    if self._last_map_8bit is not None
                    else np.zeros_like(rgb_frame[:, :, 0], dtype=np.uint8)
                )
            else:
                depth_8bit = cv2.normalize(
                    depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )
                self._last_map_8bit = depth_8bit

            self.last_inference_ms = (time.perf_counter() - start) * 1000.0

            if self.device.type == "mps":
                empty_mps_cache()

            return DepthPrediction(map_8bit=depth_8bit, raw=depth, inference_ms=self.last_inference_ms)
        except Exception as err:  # noqa: BLE001
            print(f"[ERROR] Depth estimation failed: {err}")
            return None

    # def _run_midas(self, rgb_input: np.ndarray) -> np.ndarray:
    #     assert self.transform is not None

    #     if self.input_size and max(rgb_input.shape[:2]) > self.input_size:
    #         scale = self.input_size / max(rgb_input.shape[:2])
    #         new_size = (int(rgb_input.shape[1] * scale), int(rgb_input.shape[0] * scale))
    #         rgb_resized = cv2.resize(rgb_input, new_size, interpolation=cv2.INTER_AREA)
    #     else:
    #         rgb_resized = rgb_input

    #     input_tensor = self.transform(rgb_resized).to(self.device)

    #     with torch.inference_mode():
    #         prediction = self.model(input_tensor)

    #         if prediction.device.type == "mps":
    #             prediction = prediction.to("cpu")

    #         prediction = torch.nn.functional.interpolate(
    #             prediction.unsqueeze(1),
    #             size=rgb_input.shape[:2],
    #             mode="bicubic",
    #             align_corners=False,
    #         ).squeeze()

    
    def _run_midas(self, rgb_input: np.ndarray) -> np.ndarray:
        assert self.transform is not None

        if self.input_size and max(rgb_input.shape[:2]) > self.input_size:
            scale = self.input_size / max(rgb_input.shape[:2])
            new_size = (int(rgb_input.shape[1] * scale), int(rgb_input.shape[0] * scale))
            rgb_resized = cv2.resize(rgb_input, new_size, interpolation=cv2.INTER_AREA)
        else:
            rgb_resized = rgb_input

        # FASE 1 / Tarea 3: Transferencia optimizada con non-blocking
        input_tensor = self.transform(rgb_resized)
        if self.non_blocking and self.device.type == 'cuda':
            input_tensor = input_tensor.to(self.device, non_blocking=True)
        else:
            input_tensor = input_tensor.to(self.device)

        with torch.inference_mode():
            prediction = self.model(input_tensor)

            # FASE 1 / Tarea 3: Usar bicubic en CUDA (mejor calidad), bilinear en MPS
            if self.device.type == "cuda":
                # CUDA soporta bicubic perfectamente
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=rgb_input.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            elif self.device.type == "mps":
                # MPS solo soporta bilinear de forma confiable
                try:
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=rgb_input.shape[:2],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze()
                except RuntimeError:
                    # Fallback a CPU si MPS falla
                    prediction = prediction.to("cpu")
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=rgb_input.shape[:2],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze()
            else:
                # CPU: usar bilinear (bicubic es muy lento en CPU)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=rgb_input.shape[:2],
                    mode="bilinear",
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

        # FASE 1 / Tarea 3: Transferencia optimizada con non-blocking
        inputs = self.processor(images=rgb_resized, return_tensors="pt")
        if self.non_blocking and self.device.type == 'cuda':
            # Transferir cada tensor del dict con non_blocking
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            depth = self.model(**inputs).predicted_depth

            # FASE 1 / Tarea 3: Usar bicubic en CUDA (mejor calidad), bilinear en MPS
            if self.device.type == "cuda":
                # CUDA soporta bicubic perfectamente
                depth = interpolate(
                    depth.unsqueeze(1),
                    size=rgb_input.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            elif self.device.type == "mps":
                # MPS solo soporta bilinear de forma confiable
                try:
                    depth = interpolate(
                        depth.unsqueeze(1),
                        size=rgb_input.shape[:2],
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze()
                except RuntimeError:
                    # Fallback a CPU si MPS falla
                    depth = depth.to("cpu")
                    depth = interpolate(
                        depth.unsqueeze(1),
                        size=rgb_input.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()
            else:
                # CPU: usar bilinear (bicubic es muy lento en CPU)
                depth = interpolate(
                    depth.unsqueeze(1),
                    size=rgb_input.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze()

        return depth.cpu().numpy()


    # def _run_depth_anything(self, rgb_input: np.ndarray) -> np.ndarray:
    #     assert self.processor is not None

    #     from torch.nn.functional import interpolate

    #     if self.input_size and max(rgb_input.shape[:2]) > self.input_size:
    #         scale = self.input_size / max(rgb_input.shape[:2])
    #         new_size = (int(rgb_input.shape[1] * scale), int(rgb_input.shape[0] * scale))
    #         rgb_resized = cv2.resize(rgb_input, new_size, interpolation=cv2.INTER_AREA)
    #     else:
    #         rgb_resized = rgb_input

    #     inputs = self.processor(images=rgb_resized, return_tensors="pt").to(self.device)

    #     with torch.inference_mode():
    #         depth = self.model(**inputs).predicted_depth

    #         if depth.device.type == "mps":
    #             depth = depth.to("cpu")

    #         depth = interpolate(
    #             depth.unsqueeze(1),
    #             size=rgb_input.shape[:2],
    #             mode="bicubic",
    #             align_corners=False,
    #         ).squeeze()

    #     return depth.cpu().numpy()
