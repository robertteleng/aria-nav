"""
Centralized configuration for the Aria Navigation System.

This module provides all configuration constants and runtime settings for:
- Device detection (CUDA, MPS, CPU)
- YOLO object detection (RGB and SLAM cameras)
- Depth estimation (DepthAnything V2)
- Navigation decision engine
- Audio system (TTS and spatial beeps)
- Cross-camera tracking
- Performance optimization (CUDA, TensorRT, multiprocessing)

The Config class contains all constants as class attributes, making them
accessible throughout the application without instantiation.

Usage:
    from utils.config import Config

    image_size = Config.YOLO_RGB_IMAGE_SIZE
    if Config.TRACKER_USE_3D_VALIDATION:
        # Enable 3D geometric validation
"""

import os
import torch
import platform
import logging

log = logging.getLogger(__name__)

def detect_device() -> str:
    """
    Detect the best available device for PyTorch operations.

    Priority order: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU

    Enables CUDA optimizations if GPU is available:
    - cuDNN benchmark mode (auto-tune algorithms)
    - TensorFloat-32 (TF32) for faster operations
    - High precision matrix multiplication

    Returns:
        str: Device identifier ("cuda", "mps", or "cpu")
    """
    if torch.cuda.is_available():
        device = "cuda"
        is_wsl = "microsoft" in platform.uname().release.lower()
        print(f"GPU: {torch.cuda.get_device_name(0)}{' (WSL2)' if is_wsl else ''}")

        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True  # Auto-tune algorithms
        torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat-32
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        print("  CUDA optimizations enabled (cuDNN benchmark, TF32, high precision)")

        return device
    elif torch.backends.mps.is_available():
        print("Apple MPS")
        return "mps"
    else:
        print("CPU (no GPU acceleration)")
        return "cpu"

DEVICE = detect_device()

class Config:
    """System configuration constants for Aria Navigation System."""

    # ==========================================================================
    # PERFORMANCE: Shared Memory & CUDA Optimizations
    # ==========================================================================

    # Shared memory disabled due to race conditions causing 19s spikes and 36% FPS drop
    USE_SHARED_MEMORY = False

    # CUDA optimizations (Phase 1)
    CUDA_OPTIMIZATIONS = True
    PINNED_MEMORY = True
    NON_BLOCKING_TRANSFER = True
    CUDA_STREAMS = True  # Required in Phase 1
    PHASE6_HYBRID_STREAMS = True  # Phase 6: Enable streams in main process with multiproc

    # ==========================================================================
    # YOLO DETECTION: Image Sizes & Confidence Thresholds
    # ==========================================================================

    # RGB Camera (central - high resolution)
    YOLO_RGB_IMAGE_SIZE = 640           # RGB uses 640x640 (yolo12n.engine)
    YOLO_RGB_CONFIDENCE = 0.50
    YOLO_RGB_MAX_DETECTIONS = 20

    # SLAM Cameras (peripheral - optimized for speed)
    YOLO_SLAM_IMAGE_SIZE = 256          # SLAM uses 256x256 (yolo12n_slam256.engine)
    YOLO_SLAM_CONFIDENCE = 0.60
    YOLO_SLAM_MAX_DETECTIONS = 8

    # Depth Processing
    DEPTH_INPUT_SIZE = 384              # DepthAnything: 384x384

    # ==========================================================================
    # FRAME DIMENSIONS: Native resolutions from cameras
    # ==========================================================================

    # Aria glasses native resolutions
    ARIA_RGB_WIDTH = 1408               # RGB camera native width
    ARIA_RGB_HEIGHT = 1408              # RGB camera native height
    ARIA_SLAM_WIDTH = 640               # SLAM cameras native width
    ARIA_SLAM_HEIGHT = 480              # SLAM cameras native height

    # Common test/fallback dimensions
    TEST_FRAME_WIDTH = 640              # Test frame width
    TEST_FRAME_HEIGHT = 480             # Test frame height

    # ==========================================================================
    # CAMERA SOURCES: Identifiers for RGB and SLAM cameras
    # ==========================================================================

    CAMERA_SOURCE_RGB = "rgb"           # RGB frontal camera identifier
    CAMERA_SOURCE_SLAM1 = "slam1"       # SLAM left camera identifier
    CAMERA_SOURCE_SLAM2 = "slam2"       # SLAM right camera identifier

    # Legacy compatibility (defaults to RGB settings)
    YOLO_IMAGE_SIZE = YOLO_RGB_IMAGE_SIZE
    YOLO_CONFIDENCE = YOLO_RGB_CONFIDENCE
    YOLO_MAX_DETECTIONS = YOLO_RGB_MAX_DETECTIONS

    def __init__(self):
        """
        Initialize Config with TensorRT optimizations.

        Sets up CUDA optimizations if available and configures
        frame skipping parameters for optimal performance.
        """

        print(f"[CONFIG] TensorRT Mode - Image sizes: "
                 f"RGB: {self.YOLO_RGB_IMAGE_SIZE}x{self.YOLO_RGB_IMAGE_SIZE}, "
                 f"SLAM: {self.YOLO_SLAM_IMAGE_SIZE}x{self.YOLO_SLAM_IMAGE_SIZE}, "
                 f"Depth: {self.DEPTH_INPUT_SIZE}x{self.DEPTH_INPUT_SIZE}")

        # Enable CUDA optimizations if available
        if self.CUDA_OPTIMIZATIONS and torch.cuda.is_available():
            self._enable_cuda_optimizations()

        # Device configuration
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Frame skipping (0 = process all frames)
        self.YOLO_SKIP_FRAMES = 0       # Process all frames
        self.DEPTH_SKIP_FRAMES = 0      # Process all frames for TensorRT testing
        self.DEPTH_FRAME_SKIP = 0       # Alias for navigation_pipeline.py

        log.info("Config Phase 1 loaded")

    def _enable_cuda_optimizations(self):
        """Enable CUDA optimizations for faster GPU inference."""

        # cuDNN benchmark (auto-tune algorithms)
        torch.backends.cudnn.benchmark = True
        log.info("  cuDNN benchmark enabled")

        # TensorFloat-32 (RTX 2060 compatible)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        log.info("  TF32 enabled")

        # High precision matrix multiplication
        torch.set_float32_matmul_precision('high')
        log.info("  Float32 precision: high")

        # Clear initial CUDA cache
        torch.cuda.empty_cache()
        log.info("  CUDA cache cleared")

    # ==========================================================================
    # VIDEO PROCESSING
    # ==========================================================================

    TARGET_FPS = 30                         # Target: 30 FPS on NVIDIA GPU
    YOLO_MODEL = "checkpoints/yolo12n.pt"   # Model path (auto-detects .engine)
    YOLO_CONFIDENCE = 0.50
    YOLO_DEVICE = DEVICE                    # AUTO: cuda/mps/cpu
    YOLO_MAX_DETECTIONS = 20                # More simultaneous detections on GPU
    YOLO_IOU_THRESHOLD = 0.45
    YOLO_FRAME_SKIP = 1                     # Process almost all frames
    YOLO_FORCE_MPS = False                  # Disabled

    # Peripheral vision (SLAM cameras)
    PERIPHERAL_VISION_ENABLED = True        # SLAM adds IPC overhead, minimal detections
    SLAM_TARGET_FPS = 15                    # Target: 15 FPS for SLAM
    SLAM_FRAME_SKIP = 4                     # Process every 4th frame

    # ==========================================================================
    # AUDIO SYSTEM
    # ==========================================================================

    TTS_RATE = 190
    AUDIO_COOLDOWN = 3.0
    AUDIO_QUEUE_SIZE = 3

    # ==========================================================================
    # SPATIAL PROCESSING
    # ==========================================================================

    # Zone boundaries (normalized 0-1)
    ZONE_LEFT_BOUNDARY = 0.33
    ZONE_RIGHT_BOUNDARY = 0.67

    # Distance thresholds (area ratios)
    DISTANCE_VERY_CLOSE = 0.10
    DISTANCE_CLOSE = 0.04
    DISTANCE_MEDIUM = 0.015

    # ==========================================================================
    # NAVIGATION: Detection Filtering
    # ==========================================================================

    NAVIGATION_MIN_CONFIDENCE = 0.4
    NAVIGATION_MIN_RELEVANCE = 0.18
    NAVIGATION_MAX_OBJECTS = 3
    NAVIGATION_SIZE_RATIO = 0.08

    # ==========================================================================
    # NAVIGATION: Critical Priority (Immediate Risks)
    # ==========================================================================

    CRITICAL_ALLOWED_CLASSES = {"person", "car", "truck", "bus", "bicycle", "motorcycle"}
    CRITICAL_DISTANCE_WALKING = {"very_close", "close"}
    CRITICAL_DISTANCE_STATIONARY = {"very_close"}
    CRITICAL_CENTER_TOLERANCE = 0.45        # Increased from 0.30 to match NORMAL (±45% yellow zone)
    CRITICAL_REQUIRE_YELLOW_ZONE = False    # If True, only announce objects in center zone
    CRITICAL_BBOX_COVERAGE_THRESHOLD = 0.35 # 35% bbox coverage for close objects
    CRITICAL_REPEAT_GRACE = 1.5             # Seconds before repeating same critical
    CRITICAL_COOLDOWN_WALKING = 1.0
    CRITICAL_COOLDOWN_STATIONARY = 2.0

    # ==========================================================================
    # NAVIGATION: Normal Priority (Obstacles, Furniture)
    # ==========================================================================

    NORMAL_ALLOWED_CLASSES = {"chair", "table", "bottle", "door", "laptop", "couch", "bed"}
    NORMAL_DISTANCE = {"close", "medium"}
    NORMAL_CENTER_TOLERANCE = 0.45          # Increased from 0.30 to 0.45 (±45% yellow zone)
    NORMAL_REQUIRE_YELLOW_ZONE = True       # Obstacles must be in yellow zone
    NORMAL_PERSISTENCE_FRAMES = 2           # Must be detected for 2+ consecutive frames
    NORMAL_COOLDOWN = 2.5                   # Longer cooldown for normal objects

    # ==========================================================================
    # CROSS-CAMERA TRACKING: Global Object Tracker
    # ==========================================================================

    TRACKER_IOU_THRESHOLD = 0.5             # Minimum IoU for intra-camera matching
    TRACKER_MAX_AGE = 3.0                   # Maximum time (seconds) to keep track without seeing object
    TRACKER_HANDOFF_TIMEOUT = 2.0           # Maximum time for cross-camera handoff (SLAM to RGB)

    # 3D Geometric Validation (Phase 3 - optional, disabled by default)
    TRACKER_USE_3D_VALIDATION = False       # Enable 3D geometry for cross-camera matching
    TRACKER_MAX_3D_DISTANCE = 0.5           # Maximum 3D distance (meters) for valid handoff

    # ==========================================================================
    # AUDIO ROUTING
    # ==========================================================================

    AUDIO_GLOBAL_COOLDOWN = 0.3             # Minimum time between any two announcements (reduced for faster response)
    AUDIO_INTERRUPT_GRACE = 0.25            # Grace period before interrupting (prevents audio cutoff)
    AUDIO_QUEUE_SIZE = 12                   # Increased from 3

    # Audio TTS labels (centralized to avoid duplication across routers)
    AUDIO_ZONE_LABELS = {
        "far_left": "far left side",
        "left": "left side",
        "center": "straight ahead",
        "right": "right side",
        "far_right": "far right side",
    }

    AUDIO_OBJECT_LABELS = {
        "person": "Person",
        "car": "Car",
        "truck": "Truck",
        "bus": "Bus",
        "bicycle": "Bicycle",
        "motorcycle": "Motorcycle",
        "motorbike": "Motorbike",
        "chair": "Chair",
        "table": "Table",
        "bottle": "Bottle",
        "door": "Door",
        "laptop": "Laptop",
        "couch": "Couch",
        "bed": "Bed",
        "stairs": "Stairs",
        "stop sign": "Stop sign",
        "traffic light": "Traffic light",
    }

    AUDIO_DISTANCE_LABELS = {
        "very_close": "very close",
        "close": "close",
        "medium": "at medium distance",
        "far": "far",
    }

    # ==========================================================================
    # SPATIAL AUDIO BEEPS
    # ==========================================================================

    AUDIO_SPATIAL_BEEPS_ENABLED = True      # Enable spatial beeps before TTS
    BEEP_CRITICAL_FREQUENCY = 1000          # Hz - High pitch for critical alerts
    BEEP_CRITICAL_DURATION = 0.3            # seconds - Long beep for critical
    BEEP_NORMAL_FREQUENCY = 500             # Hz - Low pitch for normal obstacles
    BEEP_NORMAL_DURATION = 0.1              # seconds - Short beeps for normal
    BEEP_NORMAL_COUNT = 2                   # Two short beeps for normal objects
    BEEP_NORMAL_GAP = 0.05                  # Gap between normal beeps
    BEEP_VOLUME = 0.7                       # Volume (0.0 to 1.0)

    # ==========================================================================
    # SLAM AUDIO FILTERING
    # ==========================================================================

    SLAM_AUDIO_DUPLICATE_GRACE = 1.0        # Don't repeat if RGB mentioned same class recently
    SLAM_CRITICAL_ONLY = True               # Only emit SLAM events for critical distances

    # ==========================================================================
    # PERFORMANCE: Input Resolution
    # ==========================================================================

    INPUT_RESIZE_ENABLED = True             # Resize frames before processing (reduces IPC overhead)
    INPUT_RESIZE_WIDTH = 896                # Aggressive resize (was 1024, originally 1408)
    INPUT_RESIZE_HEIGHT = 896               # ~60% fewer pixels = faster inference + less IPC

    # ==========================================================================
    # ARIA STREAMING
    # ==========================================================================

    STREAMING_PROFILE = "profile28"         # 30 FPS (both profiles support 30 FPS)
    STREAMING_INTERFACE = "usb"             # "usb" or "wifi"
    STREAMING_PROFILE_USB = "profile28"
    STREAMING_PROFILE_WIFI = "profile15"
    STREAMING_WIFI_DEVICE_IP = "192.168.0.204"

    # ==========================================================================
    # PERFORMANCE: Detection History
    # ==========================================================================

    DETECTION_HISTORY_SIZE = 30             # Larger history on NVIDIA GPU
    CONSISTENCY_THRESHOLD = 2
    DEBUG_FRAME_INTERVAL = 100

    # ==========================================================================
    # CAMERA INPUT
    # ==========================================================================

    # Aria RGB camera delivers RGB ordering (not BGR)
    RGB_CAMERA_COLOR_SPACE = "RGB"

    # ==========================================================================
    # DEPTH ESTIMATION
    # ==========================================================================

    # Distance estimation strategy
    DISTANCE_METHOD = "depth_only"          # Options: "depth_only", "area_only", "hybrid"
    DEPTH_ENABLED = True                    # Re-enabled: Testing depth impact with TensorRT YOLO

    # TensorRT backend
    USE_TENSORRT = True                     # Use .engine files instead of .pt

    # Backend selection
    DEPTH_BACKEND = "depth_anything_v2"     # Options: "midas" or "depth_anything_v2"

    # MiDaS Configuration
    MIDAS_MODEL = "MiDaS_small"             # Small model for stability
    MIDAS_DEVICE = DEVICE                   # AUTO: cuda/mps/cpu

    # Depth Anything V2 Configuration
    DEPTH_ANYTHING_MODEL = "Small"          # Small model for stability
    DEPTH_ANYTHING_DEVICE = DEVICE          # AUTO: cuda/mps/cpu

    # Distance thresholds (normalized depth values 0-1)
    DEPTH_CLOSE_THRESHOLD = 0.7             # Objects closer than this = "close"
    DEPTH_MEDIUM_THRESHOLD = 0.4            # Objects closer than this = "medium"

    # ==========================================================================
    # SPATIAL ZONES
    # ==========================================================================

    ZONE_SYSTEM = "five_zones"              # Options: "four_quadrants" or "five_zones"
    CENTER_ZONE_WIDTH_RATIO = 0.4           # 40% of total width
    CENTER_ZONE_HEIGHT_RATIO = 0.4          # 40% of total height
    CENTER_ZONE_PRIORITY_BOOST = 1.5        # Priority multiplier for center zone

    # ==========================================================================
    # IMAGE ENHANCEMENT (Low-Light)
    # ==========================================================================

    LOW_LIGHT_ENHANCEMENT = True            # Enable/disable enhancement system
    GAMMA_CORRECTION = 1.1                  # 1.0 = no correction, >1.0 = brighter (recommended: 1.1-1.8)
    AUTO_ENHANCEMENT = True                 # Auto-detect low light conditions
    LOW_LIGHT_THRESHOLD = 120.0             # Brightness threshold (0-255), lower = more sensitive
    ENHANCEMENT_DEBUG = True                # Show debug messages
    CLAHE_CLIP_LIMIT = 3.0                  # Contrast enhancement strength
    CLAHE_TILE_SIZE = (8, 8)                # CLAHE tile grid size

    # ==========================================================================
    # PROFILING
    # ==========================================================================

    PROFILE_PIPELINE = True
    PROFILE_WINDOW_FRAMES = 30

    # ==========================================================================
    # MULTIPROCESSING (Phase 2)
    # ==========================================================================

    # Re-enabled to test parallel execution with TensorRT
    PHASE2_MULTIPROC_ENABLED = True         # Best performance: 16.6 FPS (vs 10.9 without)
    PHASE2_QUEUE_MAXSIZE = 1                # Optimized: Smaller queue = less latency
    PHASE2_SLAM_QUEUE_MAXSIZE = 1           # Optimized: Minimize IPC overhead
    PHASE2_RESULT_QUEUE_MAXSIZE = 2         # Optimized: Just enough for overlap
    PHASE2_STATS_INTERVAL = 5.0
    PHASE2_BACKPRESSURE_TIMEOUT = 0.1       # Faster timeout for queue.put()

    # Phase 7: Double Buffering (disabled - doesn't solve IPC overhead)
    PHASE7_DOUBLE_BUFFERING = False
    PHASE7_WORKER_HEALTH_CHECK_INTERVAL = 5.0   # Check worker health every 5s
    PHASE7_GRACEFUL_DEGRADATION = True          # Fallback to single worker on crash
