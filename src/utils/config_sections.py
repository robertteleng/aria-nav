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


@dataclass
class PipelineConfig:
    """Configuration for NavigationPipeline processing modes."""

    # Multiprocessing mode
    multiproc_enabled: bool = False  # Enable multiprocessing with workers

    # CUDA streams (parallel Depth + YOLO)
    cuda_streams: bool = False  # Enable CUDA streams for parallelization
    phase6_hybrid_streams: bool = False  # Hybrid: streams in main + workers for SLAM

    # Double buffering (Phase 7)
    double_buffering: bool = False  # Use 2x workers per type
    worker_health_check_interval: float = 5.0  # Seconds between health checks
    graceful_degradation: bool = True  # Continue with remaining workers if one fails

    # Queue sizes
    result_queue_maxsize: int = 10  # Max results in result queue
    central_queue_maxsize: int = 2  # Max frames in RGB worker queue
    slam_queue_maxsize: int = 4  # Max frames in SLAM worker queue

    # Shared memory (Phase 3)
    use_shared_memory: bool = False  # Use shared memory for IPC

    # Input resizing (optimization)
    input_resize_enabled: bool = False  # Resize frames before IPC
    input_resize_width: int = 1024  # Target width for resized frames
    input_resize_height: int = 1024  # Target height for resized frames

    # Stats reporting
    stats_interval: float = 5.0  # Seconds between stats printing


@dataclass
class CoordinatorConfig:
    """Configuration for NavigationCoordinator profiling and monitoring."""

    # Performance profiling
    profile_enabled: bool = False  # Enable pipeline profiling
    profile_window_frames: int = 30  # Number of frames to average for profiling stats


@dataclass
class TrackerConfig:
    """Configuration for object tracking across cameras."""

    # Tracker parameters
    iou_threshold: float = 0.5  # IoU threshold for matching detections
    max_age: float = 3.0  # Maximum age before track expires (seconds)
    handoff_timeout: float = 2.0  # Timeout for RGB-to-SLAM handoff (seconds)

    # 3D geometric validation
    use_3d_validation: bool = False  # Enable 3D geometric validation
    max_3d_distance: float = 0.5  # Maximum 3D distance for same object (meters)


@dataclass
class PeripheralVisionConfig:
    """Configuration for peripheral vision (SLAM cameras) system."""

    # Peripheral vision control
    enabled: bool = False  # Enable peripheral SLAM cameras

    # SLAM camera settings
    slam_target_fps: int = 8  # Target FPS for SLAM camera processing


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


def load_pipeline_config() -> PipelineConfig:
    """
    Load pipeline configuration from Config with fallback defaults.

    Returns:
        PipelineConfig with values from Config or defaults
    """
    from utils.config import Config

    return PipelineConfig(
        multiproc_enabled=getattr(Config, "PHASE2_MULTIPROC_ENABLED", False),
        cuda_streams=getattr(Config, "CUDA_STREAMS", False),
        phase6_hybrid_streams=getattr(Config, "PHASE6_HYBRID_STREAMS", False),
        double_buffering=getattr(Config, "PHASE7_DOUBLE_BUFFERING", False),
        worker_health_check_interval=getattr(Config, "PHASE7_WORKER_HEALTH_CHECK_INTERVAL", 5.0),
        graceful_degradation=getattr(Config, "PHASE7_GRACEFUL_DEGRADATION", True),
        result_queue_maxsize=getattr(Config, "PHASE2_RESULT_QUEUE_MAXSIZE", 10),
        central_queue_maxsize=getattr(Config, "PHASE2_QUEUE_MAXSIZE", 2),
        slam_queue_maxsize=getattr(Config, "PHASE2_SLAM_QUEUE_MAXSIZE", 4),
        use_shared_memory=getattr(Config, "USE_SHARED_MEMORY", False),
        input_resize_enabled=getattr(Config, "INPUT_RESIZE_ENABLED", False),
        input_resize_width=getattr(Config, "INPUT_RESIZE_WIDTH", 1024),
        input_resize_height=getattr(Config, "INPUT_RESIZE_HEIGHT", 1024),
        stats_interval=getattr(Config, "PHASE2_STATS_INTERVAL", 5.0),
    )


def load_coordinator_config() -> CoordinatorConfig:
    """
    Load coordinator configuration from Config with fallback defaults.

    Returns:
        CoordinatorConfig with values from Config or defaults
    """
    from utils.config import Config

    return CoordinatorConfig(
        profile_enabled=getattr(Config, "PROFILE_PIPELINE", False),
        profile_window_frames=getattr(Config, "PROFILE_WINDOW_FRAMES", 30),
    )


def load_tracker_config() -> TrackerConfig:
    """
    Load tracker configuration from Config with fallback defaults.

    Returns:
        TrackerConfig with values from Config or defaults
    """
    from utils.config import Config

    return TrackerConfig(
        iou_threshold=getattr(Config, "TRACKER_IOU_THRESHOLD", 0.5),
        max_age=getattr(Config, "TRACKER_MAX_AGE", 3.0),
        handoff_timeout=getattr(Config, "TRACKER_HANDOFF_TIMEOUT", 2.0),
        use_3d_validation=getattr(Config, "TRACKER_USE_3D_VALIDATION", False),
        max_3d_distance=getattr(Config, "TRACKER_MAX_3D_DISTANCE", 0.5),
    )


def load_peripheral_vision_config() -> PeripheralVisionConfig:
    """
    Load peripheral vision configuration from Config with fallback defaults.

    Returns:
        PeripheralVisionConfig with values from Config or defaults
    """
    from utils.config import Config

    return PeripheralVisionConfig(
        enabled=getattr(Config, "PERIPHERAL_VISION_ENABLED", False),
        slam_target_fps=getattr(Config, "SLAM_TARGET_FPS", 8),
    )
