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
from typing import Dict, Optional, Set


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

    # Legacy single-profile settings (used as fallback)
    image_size: int = 416
    max_detections: int = 20

    # Performance optimizations
    use_tensorrt: bool = False
    pinned_memory: bool = False
    non_blocking_transfer: bool = False
    force_mps: bool = False  # Force MPS on Apple Silicon


@dataclass
class DepthConfig:
    """Configuration for depth estimation."""

    # Enable/disable depth processing
    enabled: bool = True

    # Backend selection: "midas" or "depth_anything"
    backend: str = "midas"

    # Device configuration
    midas_device: str = "cuda"
    depth_anything_device: str = "cuda"

    # Model variants
    midas_model: str = "MiDaS_small"
    depth_anything_model: str = "Small"  # Small, Base, Large

    # Processing options
    input_size: int = 384
    frame_skip: int = 1

    # Performance optimizations
    use_tensorrt: bool = False
    pinned_memory: bool = False
    non_blocking_transfer: bool = False
    force_mps: bool = False  # Force MPS on Apple Silicon


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
class NavigationFilterConfig:
    """Configuration for navigation-relevant object filtering in YOLO processor."""

    # Minimum confidence threshold for navigation relevance
    min_confidence: float = 0.4

    # Minimum relevance score to include object
    min_relevance: float = 0.18

    # Maximum objects to return for navigation
    max_objects: int = 3

    # Reference size ratio for relevance calculation
    size_ratio: float = 0.08


@dataclass
class ProfilingConfig:
    """Configuration for pipeline performance profiling."""

    # Enable/disable profiling
    enabled: bool = False

    # Number of frames to aggregate before printing stats
    window_frames: int = 30


@dataclass
class PeripheralVisionConfig:
    """Configuration for peripheral SLAM camera processing."""

    # Enable peripheral vision (SLAM cameras)
    enabled: bool = False

    # Target FPS for SLAM processing
    target_fps: int = 8

    # Frame skip for SLAM workers
    frame_skip: int = 3


@dataclass
class AudioBeepConfig:
    """Configuration for audio beep generation."""

    # Enable spatial beeps
    spatial_beeps_enabled: bool = True

    # Critical beep settings
    critical_frequency: int = 1000  # Hz
    critical_duration: float = 0.3  # seconds

    # Normal beep settings
    normal_frequency: int = 500  # Hz
    normal_duration: float = 0.1  # seconds
    normal_gap: float = 0.05  # seconds between beeps
    normal_count: int = 2  # number of beeps

    # Volume
    volume: float = 0.7  # 0.0 to 1.0


@dataclass
class AudioRouterConfig:
    """Configuration for audio routing and cooldowns."""

    # Global cooldown between any audio events
    global_cooldown: float = 0.8  # seconds

    # Grace period to prevent audio interruption
    interrupt_grace: float = 0.25  # seconds

    # SLAM duplicate detection grace period
    slam_duplicate_grace: float = 1.0  # seconds


@dataclass
class TrackerConfig:
    """Configuration for object tracking across frames."""

    # IOU threshold for matching detections
    iou_threshold: float = 0.5

    # Maximum age (seconds) before track is dropped
    max_age: float = 3.0

    # Timeout for cross-camera handoff
    handoff_timeout: float = 2.0

    # 3D geometric validation
    use_3d_validation: bool = False
    max_3d_distance: float = 0.5  # meters


@dataclass
class StreamingConfig:
    """Configuration for Aria glasses streaming connection."""

    # Interface type: "usb" or "wifi"
    interface: str = "usb"

    # WiFi device IP (required for wifi interface)
    wifi_device_ip: Optional[str] = None

    # Streaming profiles
    profile_wifi: str = "profile18"  # Default profile for WiFi
    profile_usb: str = "profile28"  # Default profile for USB


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
class ObserverConfig:
    """Configuration for Aria frame observer."""

    # Color space of RGB camera frames: "BGR" or "RGB"
    rgb_color_space: str = "BGR"


def load_observer_config() -> ObserverConfig:
    """
    Load observer configuration from Config with fallback defaults.

    Returns:
        ObserverConfig with values from Config or defaults
    """
    from utils.config import Config

    return ObserverConfig(
        rgb_color_space=getattr(Config, "RGB_CAMERA_COLOR_SPACE", "BGR"),
    )


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
        image_size=getattr(Config, "YOLO_IMAGE_SIZE", 416),
        max_detections=getattr(Config, "YOLO_MAX_DETECTIONS", 20),
        use_tensorrt=getattr(Config, "USE_TENSORRT", False),
        pinned_memory=getattr(Config, "PINNED_MEMORY", False),
        non_blocking_transfer=getattr(Config, "NON_BLOCKING_TRANSFER", False),
        force_mps=getattr(Config, "YOLO_FORCE_MPS", False),
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
        backend=getattr(Config, "DEPTH_BACKEND", "midas"),
        midas_device=getattr(Config, "MIDAS_DEVICE", "cuda"),
        depth_anything_device=getattr(Config, "DEPTH_ANYTHING_DEVICE", "cuda"),
        midas_model=getattr(Config, "MIDAS_MODEL", "MiDaS_small"),
        depth_anything_model=getattr(Config, "DEPTH_ANYTHING_MODEL", "Small"),
        input_size=getattr(Config, "DEPTH_INPUT_SIZE", 384),
        frame_skip=getattr(Config, "DEPTH_FRAME_SKIP", 1),
        use_tensorrt=getattr(Config, "USE_TENSORRT", False),
        pinned_memory=getattr(Config, "PINNED_MEMORY", False),
        non_blocking_transfer=getattr(Config, "NON_BLOCKING_TRANSFER", False),
        force_mps=getattr(Config, "YOLO_FORCE_MPS", False),
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


def load_navigation_filter_config() -> NavigationFilterConfig:
    """
    Load navigation filter configuration from Config with fallback defaults.

    Returns:
        NavigationFilterConfig with values from Config or defaults
    """
    from utils.config import Config

    return NavigationFilterConfig(
        min_confidence=getattr(Config, "NAVIGATION_MIN_CONFIDENCE", 0.4),
        min_relevance=getattr(Config, "NAVIGATION_MIN_RELEVANCE", 0.18),
        max_objects=getattr(Config, "NAVIGATION_MAX_OBJECTS", 3),
        size_ratio=getattr(Config, "NAVIGATION_SIZE_RATIO", 0.08),
    )


def load_profiling_config() -> ProfilingConfig:
    """
    Load profiling configuration from Config with fallback defaults.

    Returns:
        ProfilingConfig with values from Config or defaults
    """
    from utils.config import Config

    return ProfilingConfig(
        enabled=getattr(Config, "PROFILE_PIPELINE", False),
        window_frames=getattr(Config, "PROFILE_WINDOW_FRAMES", 30),
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
        target_fps=getattr(Config, "SLAM_TARGET_FPS", 8),
        frame_skip=getattr(Config, "SLAM_FRAME_SKIP", 3),
    )


def load_audio_beep_config() -> AudioBeepConfig:
    """
    Load audio beep configuration from Config with fallback defaults.

    Returns:
        AudioBeepConfig with values from Config or defaults
    """
    from utils.config import Config

    return AudioBeepConfig(
        spatial_beeps_enabled=getattr(Config, "AUDIO_SPATIAL_BEEPS_ENABLED", True),
        critical_frequency=getattr(Config, "BEEP_CRITICAL_FREQUENCY", 1000),
        critical_duration=getattr(Config, "BEEP_CRITICAL_DURATION", 0.3),
        normal_frequency=getattr(Config, "BEEP_NORMAL_FREQUENCY", 500),
        normal_duration=getattr(Config, "BEEP_NORMAL_DURATION", 0.1),
        normal_gap=getattr(Config, "BEEP_NORMAL_GAP", 0.05),
        normal_count=getattr(Config, "BEEP_NORMAL_COUNT", 2),
        volume=getattr(Config, "BEEP_VOLUME", 0.7),
    )


def load_audio_router_config() -> AudioRouterConfig:
    """
    Load audio router configuration from Config with fallback defaults.

    Returns:
        AudioRouterConfig with values from Config or defaults
    """
    from utils.config import Config

    return AudioRouterConfig(
        global_cooldown=getattr(Config, "AUDIO_GLOBAL_COOLDOWN", 0.8),
        interrupt_grace=getattr(Config, "AUDIO_INTERRUPT_GRACE", 0.25),
        slam_duplicate_grace=getattr(Config, "SLAM_AUDIO_DUPLICATE_GRACE", 1.0),
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


def load_streaming_config() -> StreamingConfig:
    """
    Load streaming configuration from Config with fallback defaults.

    Returns:
        StreamingConfig with values from Config or defaults
    """
    from utils.config import Config

    return StreamingConfig(
        interface=getattr(Config, "STREAMING_INTERFACE", "usb"),
        wifi_device_ip=getattr(Config, "STREAMING_WIFI_DEVICE_IP", None),
        profile_wifi=getattr(Config, "STREAMING_PROFILE_WIFI", "profile18"),
        profile_usb=getattr(Config, "STREAMING_PROFILE_USB", "profile28"),
    )
