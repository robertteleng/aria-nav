"""
Typed configuration sections for ARIA Navigation System.

This module provides strongly-typed configuration sections to replace
scattered getattr(Config, ...) calls with proper type hints and defaults.

Benefits:
- Type safety: IDE autocomplete and type checking
- Discoverability: All config options visible in one place
- Default values: Centralized and documented
- Better testing: Can mock entire config sections
"""

from dataclasses import dataclass, field
from typing import Dict, Set


@dataclass
class SlamAudioConfig:
    """Configuration for SLAM audio routing and filtering."""

    # Duplicate detection
    duplicate_grace: float = 1.0  # Seconds to wait before re-announcing same object

    # Critical-only filtering
    critical_only: bool = True  # Only announce critical distance objects from SLAM

    # Critical distance definitions by motion state
    critical_distances_walking: Set[str] = field(
        default_factory=lambda: {"very_close", "close"}
    )
    critical_distances_stationary: Set[str] = field(
        default_factory=lambda: {"very_close"}
    )

    # Frame processing
    frame_skip: int = 3  # Process every Nth SLAM frame (performance optimization)


@dataclass
class CriticalDetectionConfig:
    """Configuration for critical object detection and cooldowns."""

    # Allowed critical classes
    critical_allowed_classes: Set[str] = field(
        default_factory=lambda: {"person", "car", "truck", "bus", "bicycle", "motorcycle"}
    )

    # Cooldowns by motion state (seconds)
    critical_cooldown_walking: float = 1.0  # Cooldown when user is walking
    critical_cooldown_stationary: float = 2.0  # Cooldown when user is stationary

    # Yellow zone requirement
    critical_require_yellow_zone: bool = True  # Require center zone for critical alerts

    # Zone tolerance
    center_tolerance: float = 0.3  # ±30% from center is "yellow zone"


@dataclass
class NormalDetectionConfig:
    """Configuration for normal (non-critical) object detection."""

    # Allowed normal classes
    normal_allowed_classes: Set[str] = field(
        default_factory=lambda: {"chair", "couch", "bed", "table", "door", "bottle"}
    )

    # Persistence requirement
    normal_min_frames: int = 3  # Must appear N consecutive frames

    # Yellow zone requirement
    normal_require_yellow_zone: bool = True  # Require center zone for normal objects

    # Cooldown
    normal_cooldown: float = 2.5  # Seconds between normal announcements


@dataclass
class YoloConfig:
    """Configuration for YOLO object detection."""

    # RGB profile (main camera)
    rgb_image_size: int = 640
    rgb_confidence: float = 0.50
    rgb_max_detections: int = 20

    # SLAM profile (peripheral cameras)
    slam_image_size: int = 256
    slam_confidence: float = 0.60
    slam_max_detections: int = 8

    # General settings
    iou_threshold: float = 0.45
    frame_skip: int = 1  # Process every Nth frame


@dataclass
class DepthConfig:
    """Configuration for depth estimation."""

    enabled: bool = True  # Enable/disable depth processing
    frame_skip: int = 0  # Process every Nth frame (0 = all frames)
    model_type: str = "depth_anything_v2"  # Depth model to use


@dataclass
class AudioConfig:
    """Configuration for audio system and routing."""

    # Object labels (technical class -> user-friendly name)
    object_labels: Dict[str, str] = field(default_factory=dict)

    # Zone labels (zone ID -> user-friendly name)
    zone_labels: Dict[str, str] = field(default_factory=dict)

    # Distance labels (distance ID -> user-friendly name)
    distance_labels: Dict[str, str] = field(default_factory=dict)


def load_slam_audio_config() -> SlamAudioConfig:
    """
    Load SLAM audio configuration from Config with fallback defaults.

    Returns:
        SlamAudioConfig with values from Config or defaults
    """
    from utils.config import Config

    return SlamAudioConfig(
        duplicate_grace=getattr(Config, "SLAM_AUDIO_DUPLICATE_GRACE", 1.0),
        critical_only=getattr(Config, "SLAM_CRITICAL_ONLY", True),
        critical_distances_walking=getattr(
            Config, "CRITICAL_DISTANCE_WALKING", {"very_close", "close"}
        ),
        critical_distances_stationary=getattr(
            Config, "CRITICAL_DISTANCE_STATIONARY", {"very_close"}
        ),
        frame_skip=getattr(Config, "SLAM_FRAME_SKIP", 3),
    )


def load_critical_detection_config() -> CriticalDetectionConfig:
    """
    Load critical detection configuration from Config with fallback defaults.

    Returns:
        CriticalDetectionConfig with values from Config or defaults
    """
    from utils.config import Config

    return CriticalDetectionConfig(
        critical_allowed_classes=getattr(
            Config,
            "CRITICAL_ALLOWED_CLASSES",
            {"person", "car", "truck", "bus", "bicycle", "motorcycle"},
        ),
        critical_cooldown_walking=getattr(Config, "CRITICAL_COOLDOWN_WALKING", 1.0),
        critical_cooldown_stationary=getattr(Config, "CRITICAL_COOLDOWN_STATIONARY", 2.0),
        critical_require_yellow_zone=getattr(Config, "CRITICAL_REQUIRE_YELLOW_ZONE", True),
        center_tolerance=getattr(Config, "CENTER_TOLERANCE", 0.3),
    )


def load_normal_detection_config() -> NormalDetectionConfig:
    """
    Load normal detection configuration from Config with fallback defaults.

    Returns:
        NormalDetectionConfig with values from Config or defaults
    """
    from utils.config import Config

    return NormalDetectionConfig(
        normal_allowed_classes=getattr(
            Config,
            "NORMAL_ALLOWED_CLASSES",
            {"chair", "couch", "bed", "table", "door", "bottle"},
        ),
        normal_min_frames=getattr(Config, "NORMAL_MIN_FRAMES", 3),
        normal_require_yellow_zone=getattr(Config, "NORMAL_REQUIRE_YELLOW_ZONE", True),
        normal_cooldown=getattr(Config, "NORMAL_COOLDOWN", 2.5),
    )


def load_yolo_config() -> YoloConfig:
    """
    Load YOLO configuration from Config with fallback defaults.

    Returns:
        YoloConfig with values from Config or defaults
    """
    from utils.config import Config

    return YoloConfig(
        rgb_image_size=getattr(Config, "YOLO_RGB_IMAGE_SIZE", 640),
        rgb_confidence=getattr(Config, "YOLO_RGB_CONFIDENCE", 0.50),
        rgb_max_detections=getattr(Config, "YOLO_RGB_MAX_DETECTIONS", 20),
        slam_image_size=getattr(Config, "YOLO_SLAM_IMAGE_SIZE", 256),
        slam_confidence=getattr(Config, "YOLO_SLAM_CONFIDENCE", 0.60),
        slam_max_detections=getattr(Config, "YOLO_SLAM_MAX_DETECTIONS", 8),
        iou_threshold=getattr(Config, "YOLO_IOU_THRESHOLD", 0.45),
        frame_skip=getattr(Config, "YOLO_FRAME_SKIP", 1),
    )


def load_depth_config() -> DepthConfig:
    """
    Load depth configuration from Config with fallback defaults.

    Returns:
        DepthConfig with values from Config or defaults
    """
    from utils.config import Config

    return DepthConfig(
        enabled=getattr(Config, "DEPTH_ENABLED", True),
        frame_skip=getattr(Config, "DEPTH_FRAME_SKIP", 0),
        model_type=getattr(Config, "DEPTH_MODEL_TYPE", "depth_anything_v2"),
    )


def load_audio_config() -> AudioConfig:
    """
    Load audio configuration from Config with fallback defaults.

    Returns:
        AudioConfig with values from Config or defaults
    """
    from utils.config import Config

    return AudioConfig(
        object_labels=getattr(Config, "AUDIO_OBJECT_LABELS", {}),
        zone_labels=getattr(Config, "AUDIO_ZONE_LABELS", {}),
        distance_labels=getattr(Config, "AUDIO_DISTANCE_LABELS", {}),
    )
