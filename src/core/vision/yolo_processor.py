import numpy as np
import torch
import torchvision
import time
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple
from ultralytics import YOLO

from utils.config import Config
from .detected_object import DetectedObject
from .gpu_utils import (
    configure_mps_environment,
    empty_mps_cache,
    get_preferred_device,
)

import logging
log = logging.getLogger("YoloProcessor")


@dataclass(frozen=True)
class YoloRuntimeConfig:
    """Runtime configuration for a YOLO processor instance."""

    model: str
    device: str
    confidence: float
    image_size: int
    max_detections: int
    iou_threshold: float
    frame_skip: int
    force_mps: bool
    profile_name: str = "custom"

    @classmethod
    def from_defaults(cls) -> "YoloRuntimeConfig":
        return cls(
            model=Config.YOLO_MODEL,
            device=Config.YOLO_DEVICE,
            confidence=Config.YOLO_CONFIDENCE,
            image_size=getattr(Config, "YOLO_IMAGE_SIZE", 416),
            max_detections=getattr(Config, "YOLO_MAX_DETECTIONS", 20),
            iou_threshold=getattr(Config, "YOLO_IOU_THRESHOLD", 0.5),
            frame_skip=max(1, getattr(Config, "YOLO_FRAME_SKIP", 1)),
            force_mps=getattr(Config, "YOLO_FORCE_MPS", False),
            profile_name="rgb-default",
        )

    @classmethod
    def for_profile(cls, profile: str) -> "YoloRuntimeConfig":
        profile = profile.lower()
        if profile in {"rgb", "default"}:
            return cls.from_defaults().with_overrides(profile_name="rgb")
        if profile == "slam":
            base = cls.from_defaults()
            return base.with_overrides(
                profile_name="slam",
                image_size=256,
                confidence=0.60,
                max_detections=8,
                frame_skip=3,
            )
        raise ValueError(f"Unknown YOLO profile '{profile}'")

    def with_overrides(self, **overrides) -> "YoloRuntimeConfig":
        mapped = {}
        for key, value in overrides.items():
            if key in {"imgsz", "image_size"}:
                mapped["image_size"] = int(value)
            elif key in {"max_det", "max_detections"}:
                mapped["max_detections"] = int(value)
            elif key in {"conf", "confidence"}:
                mapped["confidence"] = float(value)
            elif key in {"iou", "iou_threshold"}:
                mapped["iou_threshold"] = float(value)
            elif key in {"frame_skip", "skip"}:
                mapped["frame_skip"] = max(1, int(value))
            elif key == "device":
                mapped["device"] = str(value)
            elif key in {"force_mps", "force"}:
                mapped["force_mps"] = bool(value)
            elif key in {"model", "weights"}:
                mapped["model"] = str(value)
            elif key == "profile_name":
                mapped["profile_name"] = str(value)
            else:
                raise ValueError(f"Unsupported YOLO override '{key}'")

        data = asdict(self)
        data.update(mapped)
        return YoloRuntimeConfig(**data)


def _enable_mps_nms_fallback() -> None:
    """Patch torchvision NMS: execute on CPU when running via MPS."""
    try:
        original_nms = torchvision.ops.nms
    except AttributeError:
        return

    def nms_mps_safe(boxes: torch.Tensor, scores: torch.Tensor, threshold: float) -> torch.Tensor:
        if boxes.device.type == "mps":
            keep = original_nms(boxes.cpu(), scores.cpu(), threshold)
            return keep.to(boxes.device)
        return original_nms(boxes, scores, threshold)

    if getattr(torchvision.ops.nms, "__name__", "") != "nms_mps_safe":
        torchvision.ops.nms = nms_mps_safe  # type: ignore[assignment]


_enable_mps_nms_fallback()


class YoloProcessor:
    """YOLO-based object detector with configurable runtime profiles."""

    def __init__(
        self,
        runtime_config: Optional[YoloRuntimeConfig] = None,
        *,
        profile: Optional[str] = None,
        **overrides,
    ) -> None:
        if runtime_config is not None and (profile or overrides):
            raise ValueError("Provide either runtime_config or profile/overrides, not both")

        if runtime_config is None:
            base = (
                YoloRuntimeConfig.for_profile(profile)
                if profile is not None
                else YoloRuntimeConfig.from_defaults()
            )
            self.runtime_config = base.with_overrides(**overrides) if overrides else base
        else:
            self.runtime_config = runtime_config

        # LOG A: resumen de configuración efectiva
        log.info(
            "Init profile=%s model=%s device_pref=%s imgsz=%s conf=%s iou=%s skip=%s max_det=%s",
            getattr(self.runtime_config, "profile_name", "custom"),
            self.runtime_config.model,
            self.runtime_config.device,
            self.runtime_config.image_size,
            self.runtime_config.confidence,
            self.runtime_config.iou_threshold,
            self.runtime_config.frame_skip,
            self.runtime_config.max_detections,
        )
        print("[INFO] Loading YOLO model...")
        configure_mps_environment(self.runtime_config.force_mps)

        try:
            torch.set_num_threads(2)
        except Exception:
            pass

        self.model = YOLO(self.runtime_config.model)
        self.device = get_preferred_device(self.runtime_config.device)
        self.device_str = self.device.type
        self.model.to(self.device)
        try:
            self.model.fuse()
        except Exception:
            pass
        
        # FASE 1 / Tarea 2: Habilitar pinned memory y non-blocking transfers
        self.use_pinned_memory = getattr(Config, 'PINNED_MEMORY', False) and torch.cuda.is_available()
        self.non_blocking = getattr(Config, 'NON_BLOCKING_TRANSFER', False)
        if self.use_pinned_memory:
            log.info("  ✓ YOLO: Pinned memory enabled")
        if self.non_blocking:
            log.info("  ✓ YOLO: Non-blocking transfers enabled")


        self.frame_skip = max(1, self.runtime_config.frame_skip)
        self.img_size = self.runtime_config.image_size
        self.max_det = self.runtime_config.max_detections
        self.iou_threshold = self.runtime_config.iou_threshold
        self.conf_threshold = self.runtime_config.confidence
        # LOG B: confirma dispositivo final y parámetros de ejecución
        log.info(
            "Model ready on %s | imgsz=%s | conf=%s | iou=%s | skip=%s | max_det=%s",
            self.device_str,
            self.img_size,
            self.conf_threshold,
            self.iou_threshold,
            self.frame_skip,
            self.max_det,
        )

        self.latest_detections: List[dict] = []
        self._cached_results = None
        self._frame_index = 0
        self._profile = getattr(Config, "PROFILE_PIPELINE", False)
        self._inference_acc = 0.0
        self._post_acc = 0.0
        self._profile_frames = 0

        # # Outdoor objects relevant for navigation
        # self.navigation_objects = {
        #     "person": {"priority": 1.0, "name": "person"},
        #     "car": {"priority": 0.9, "name": "car"},
        #     "bicycle": {"priority": 0.8, "name": "bicycle"},
        #     "bus": {"priority": 0.9, "name": "bus"},
        #     "truck": {"priority": 0.8, "name": "truck"},
        #     "motorcycle": {"priority": 0.7, "name": "motorcycle"},
        #     "stop sign": {"priority": 0.9, "name": "stop sign"},
        #     "traffic light": {"priority": 0.6, "name": "traffic light"},
        # }

        # Indoor objects relevant for navigation
        self.navigation_objects = {
            # TIER 1: CRÍTICO - Personas
            "person": {"priority": 1.0, "name": "person"},
            
            # TIER 2: ALTO - Obstáculos grandes/riesgo tropiezo
            "chair": {"priority": 0.9, "name": "chair"},
            "couch": {"priority": 0.8, "name": "couch"},
            "backpack": {"priority": 0.8, "name": "backpack"},
            "bottle": {"priority": 0.7, "name": "bottle"},       
            
            # TIER 3: MEDIO - Referencias espaciales
            "dining table": {"priority": 0.7, "name": "table"},
            "laptop": {"priority": 0.6, "name": "laptop"},
            "handbag": {"priority": 0.6, "name": "handbag"},
            
            # TIER 4: BAJO - Objetos pequeños/contextuales
            "keyboard": {"priority": 0.5, "name": "keyboard"},
            "mouse": {"priority": 0.4, "name": "mouse"},
            "cell phone": {"priority": 0.2, "name": "phone"},
        }

        self.detection_count = 0
        self.min_confidence = max(
            0.0, float(getattr(Config, "NAVIGATION_MIN_CONFIDENCE", 0.4))
        )
        self.min_relevance = max(
            0.0, float(getattr(Config, "NAVIGATION_MIN_RELEVANCE", 0.18))
        )
        self.max_navigation_objects = max(
            1, int(getattr(Config, "NAVIGATION_MAX_OBJECTS", 3))
        )
        self.size_ratio_reference = max(
            1e-6, float(getattr(Config, "NAVIGATION_SIZE_RATIO", 0.08))
        )

            # print(
            #     f"[INFO] ✓ YOLOv11 processor initialized ({self.runtime_config.profile_name}) on {self.device_str} "
            #     f"| imgsz={self.img_size} | max_det={self.max_det} | skip={self.frame_skip}"
            # )

        log.info(
        "✓ YOLOv12 processor initialized (%s) on %s | imgsz=%s | max_det=%s | skip=%s",
        self.runtime_config.profile_name, self.device_str, self.img_size, self.max_det, self.frame_skip
        )

    @classmethod
    def from_profile(cls, profile: str, **overrides) -> "YoloProcessor":
        """Convenience factory that builds a processor for a named profile."""
        return cls(profile=profile, **overrides)

    def process_frame(
        self,
        frame: np.ndarray,
        depth_map: np.ndarray | None = None,
        depth_raw: np.ndarray | None = None,
    ) -> List[dict]:
        """Run YOLO inference with optional frame skipping."""
        self._frame_index += 1

        try:
            run_inference = self._frame_index % self.frame_skip == 0 or self._cached_results is None

            if run_inference:
                start_inf = time.perf_counter() if self._profile else 0.0
                
                # FASE 1 / Tarea 2: Preparar frame con pinned memory si está habilitado
                # Nota: ultralytics hace preprocesamiento interno, pero podemos
                # asegurar que el modelo use transferencias optimizadas
                input_frame = frame
                if self.use_pinned_memory and self.device_str == 'cuda':
                    # Convertir a tensor con pinned memory para transferencia rápida
                    frame_tensor = torch.from_numpy(frame).pin_memory()
                    # Ultralytics acepta tensors directamente
                    input_frame = frame_tensor
                
                results = self.model.predict(
                    source=input_frame,
                    device=self.device_str,
                    imgsz=self.img_size,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    max_det=self.max_det,
                    verbose=False,
                    stream=False,
                )
                if self._profile:
                    self._inference_acc += time.perf_counter() - start_inf
                self._cached_results = results
            else:
                results = self._cached_results

            post_start = time.perf_counter() if self._profile else 0.0

            detected_objects = self._analyze_detections(
                results,
                frame.shape[1],
                frame.shape[0],
                depth_map,
                depth_raw,
            )
            if self._profile:
                self._post_acc += time.perf_counter() - post_start
                self._profile_frames += 1
                if self._profile_frames >= getattr(Config, "PROFILE_WINDOW_FRAMES", 30):
                    self._log_local_profile()

            detections: List[dict] = []
            for obj in detected_objects:
                detections.append(
                    {
                        "bbox": obj.bbox,
                        "name": obj.name,
                        "confidence": obj.confidence,
                        "zone": obj.zone,
                        "distance": obj.distance_bucket,
                        "distance_normalized": obj.depth_value,
                        "distance_raw": obj.depth_raw,
                        "relevance_score": obj.relevance_score,
                    }
                )

            self.detection_count += len(detections)
            self.latest_detections = detections

            if self.device_str == "mps":
                empty_mps_cache()

            return detections
        except Exception as exc:
            print(f"[WARN] YOLO processing failed: {exc}")
            return []
    def _log_local_profile(self) -> None:
        frames = max(1, self._profile_frames)
        inf_ms = (self._inference_acc / frames) * 1000.0
        post_ms = (self._post_acc / frames) * 1000.0
        print(f"[PROFILE][YOLO] inference={inf_ms:.1f}ms | post={post_ms:.1f}ms | imgsz={self.img_size}")
        self._inference_acc = 0.0
        self._post_acc = 0.0
        self._profile_frames = 0

    def _analyze_detections(
        self,
        yolo_results,
        frame_width: int,
        frame_height: int,
        depth_map: np.ndarray | None = None,
        depth_raw: np.ndarray | None = None,
    ) -> List[DetectedObject]:
        """Convert raw YOLO output to structured detection objects."""
        objects: List[DetectedObject] = []
        if not yolo_results:
            return objects

        for detection in yolo_results[0].boxes.data:
            x1, y1, x2, y2, confidence, class_id = detection.cpu().numpy()

            class_name = yolo_results[0].names[int(class_id)]
            if class_name not in self.navigation_objects:
                continue

            if confidence < self.min_confidence:
                continue

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)

            zone = self._classify_zone(center_x, center_y, frame_width, frame_height)

            bbox = (int(x1), int(y1), int(x2), int(y2))

            depth_value, depth_raw_mean = self._extract_depth_features(
                bbox=bbox,
                depth_map=depth_map,
                depth_raw=depth_raw,
            )
            distance_bucket = self._estimate_distance_with_depth(area, frame_width, depth_value)

            base_priority = self.navigation_objects[class_name]["priority"]
            frame_area = max(1.0, float(frame_width) * float(frame_height))
            area_ratio = max(0.0, area / frame_area)
            size_factor = min(area_ratio / self.size_ratio_reference, 1.0)
            relevance_score = (
                base_priority
                * float(confidence)
                * (0.6 + 0.4 * size_factor)
            )

            if relevance_score < self.min_relevance:
                continue

            obj = DetectedObject(
                name=self.navigation_objects[class_name]["name"],
                confidence=float(confidence),
                bbox=bbox,
                center_x=center_x,
                center_y=center_y,
                area=area,
                zone=zone,
                distance_bucket=distance_bucket,
                relevance_score=relevance_score,
                depth_value=depth_value,
                depth_raw=depth_raw_mean,
            )

            objects.append(obj)

        objects.sort(key=lambda item: item.relevance_score, reverse=True)
        return objects[: self.max_navigation_objects]

    def _extract_depth_features(
        self,
        bbox: tuple,
        depth_map: np.ndarray | None,
        depth_raw: np.ndarray | None,
    ) -> Tuple[float, Optional[float]]:
        depth_value = 0.5
        raw_mean: Optional[float] = None

        if depth_raw is not None:
            crop_raw = self._crop_region(depth_raw, bbox)
            if crop_raw.size > 0:
                finite_crop = crop_raw[np.isfinite(crop_raw)]
                if finite_crop.size > 0:
                    raw_mean = float(np.mean(finite_crop))

        if depth_map is not None:
            depth_value = self._calculate_depth_from_map(depth_map, bbox)

        return depth_value, raw_mean

    @staticmethod
    def _crop_region(array: np.ndarray, bbox: tuple) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = max(x1 + 1, x2)
        y2 = max(y1 + 1, y2)
        return array[y1:y2, x1:x2]

    def _classify_zone(
        self, center_x: float, center_y: float, frame_width: int, frame_height: int
    ) -> str:
        from utils.config import Config as LocalConfig

        if LocalConfig.ZONE_SYSTEM == "five_zones":
            center_margin_x = frame_width * LocalConfig.CENTER_ZONE_WIDTH_RATIO
            center_margin_y = frame_height * LocalConfig.CENTER_ZONE_HEIGHT_RATIO

            center_left = frame_width / 2 - center_margin_x / 2
            center_right = frame_width / 2 + center_margin_x / 2
            center_top = frame_height / 2 - center_margin_y / 2
            center_bottom = frame_height / 2 + center_margin_y / 2

            if (
                center_left <= center_x <= center_right
                and center_top <= center_y <= center_bottom
            ):
                return "center"

        mid_x = frame_width / 2
        mid_y = frame_height / 2

        if center_y < mid_y:
            return "top_left" if center_x < mid_x else "top_right"
        return "bottom_left" if center_x < mid_x else "bottom_right"

    def _calculate_depth_from_map(self, depth_map: np.ndarray, bbox: tuple) -> float:
        crop = self._crop_region(depth_map, bbox)
        if crop.size == 0:
            return 0.5
        return float(np.mean(crop) / 255.0)

    def _estimate_distance_with_depth(
        self, area: float, frame_width: int, depth_value: float
    ) -> str:
        if depth_value >= Config.DEPTH_CLOSE_THRESHOLD:
            return "very_close"
        if depth_value >= Config.DEPTH_MEDIUM_THRESHOLD:
            return "medium"

        area_ratio = area / (frame_width * frame_width)
        if area_ratio > Config.DISTANCE_VERY_CLOSE:
            return "very_close"
        if area_ratio > Config.DISTANCE_CLOSE:
            return "close"
        if area_ratio > Config.DISTANCE_MEDIUM:
            return "medium"
        return "far"
