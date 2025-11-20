"""Centralized configuration for the navigation system"""

import os
import torch
import platform
import logging

log = logging.getLogger(__name__)

def detect_device():
    """Detecta el mejor device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = "cuda"
        is_wsl = "microsoft" in platform.uname().release.lower()
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}{' (WSL2)' if is_wsl else ''}")
        
        # FASE 1: CUDA Optimizations
        torch.backends.cudnn.benchmark = True  # Auto-tune de algoritmos
        torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat-32
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        print("  ✓ CUDA optimizations enabled (cuDNN benchmark, TF32, high precision)")
        
        return device
    elif torch.backends.mps.is_available():
        print("🍎 Apple MPS")
        return "mps"
    else:
        print("⚠️ CPU")
        return "cpu"

DEVICE = detect_device()

class Config:
    """System configuration constants"""
    
    # ========== FASE 1: CUDA Optimizations (Class Attributes) ==========
    CUDA_OPTIMIZATIONS = True
    PINNED_MEMORY = True
    NON_BLOCKING_TRANSFER = True
    CUDA_STREAMS = True  # OBLIGATORIO en FASE 1
    PHASE6_HYBRID_STREAMS = True  # PHASE 6: Enable streams in main process with multiproc
    
    # ========== FASE 4: Resoluciones TensorRT ==========
    # RGB Camera (central - alta resolución)
    YOLO_RGB_IMAGE_SIZE = 640           # RGB usa 640x640 (yolo12n.engine)
    YOLO_RGB_CONFIDENCE = 0.50
    YOLO_RGB_MAX_DETECTIONS = 20
    
    # SLAM Cameras (periféricas - optimizadas)
    YOLO_SLAM_IMAGE_SIZE = 256          # SLAM usa 256x256 (yolo12n_slam256.engine)
    YOLO_SLAM_CONFIDENCE = 0.60
    YOLO_SLAM_MAX_DETECTIONS = 8
    
    # Depth Processing
    DEPTH_INPUT_SIZE = 384              # DepthAnything: 384x384
    
    # Legacy compatibility (apunta a RGB por defecto)
    YOLO_IMAGE_SIZE = YOLO_RGB_IMAGE_SIZE
    YOLO_CONFIDENCE = YOLO_RGB_CONFIDENCE
    YOLO_MAX_DETECTIONS = YOLO_RGB_MAX_DETECTIONS
    
    def __init__(self):
        """Initialize config with FASE 4 TensorRT optimizations"""
        
        print(f"[CONFIG] FASE 4 TensorRT Mode - Image sizes: "
                 f"RGB: {self.YOLO_RGB_IMAGE_SIZE}x{self.YOLO_RGB_IMAGE_SIZE}, "
                 f"SLAM: {self.YOLO_SLAM_IMAGE_SIZE}x{self.YOLO_SLAM_IMAGE_SIZE}, "
                 f"Depth: {self.DEPTH_INPUT_SIZE}x{self.DEPTH_INPUT_SIZE}")
        
        # Habilitar optimizaciones CUDA
        if self.CUDA_OPTIMIZATIONS and torch.cuda.is_available():
            self._enable_cuda_optimizations()
        
        # Device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # ========== Frame Skipping (FASE 1 optimizado) ==========
        self.YOLO_SKIP_FRAMES = 0       # Procesar todos (0 = no skip)
        self.DEPTH_SKIP_FRAMES = 0      # Procesar todos para testing TensorRT
        self.DEPTH_FRAME_SKIP = 0       # Alias para navegation_pipeline.py (0 = procesar todos)
        
        log.info("✓ Config FASE 1 cargada")
    
    def _enable_cuda_optimizations(self):
        """FASE 1 Quick Wins - Optimizaciones CUDA"""
        
        # cuDNN benchmark (auto-tune de algoritmos)
        torch.backends.cudnn.benchmark = True
        log.info("  ✓ cuDNN benchmark enabled")
        
        # TensorFloat-32 (RTX 2060 compatible)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        log.info("  ✓ TF32 enabled")
        
        # High precision matmul
        torch.set_float32_matmul_precision('high')
        log.info("  ✓ Float32 precision: high")
        
        # Limpiar cache inicial
        torch.cuda.empty_cache()
        log.info("  ✓ CUDA cache cleared")
    
    # Video processing (resto de constantes como class variables)
    TARGET_FPS = 60                         # GPU NVIDIA: 60 FPS
    YOLO_MODEL = "checkpoints/yolo12n.pt"   # Path to model (will auto-detect .engine)
    YOLO_CONFIDENCE = 0.50
    YOLO_DEVICE = DEVICE                    # AUTO: cuda/mps/cpu
    YOLO_MAX_DETECTIONS = 20                # GPU NVIDIA: Más detecciones simultáneas
    YOLO_IOU_THRESHOLD = 0.45
    YOLO_FRAME_SKIP = 1                     # Procesar casi todos los frames
    YOLO_FORCE_MPS = False                  # Desactivado

    # Peripheral vision (SLAM)
    PERIPHERAL_VISION_ENABLED = True
    SLAM_TARGET_FPS = 15                    # Modelo small: 15 FPS para SLAM
    SLAM_FRAME_SKIP = 4                     # Procesar cada 4 frames              
    
    # Audio system
    TTS_RATE = 190
    AUDIO_COOLDOWN = 3.0
    AUDIO_QUEUE_SIZE = 3
    
    # Spatial processing
    ZONE_LEFT_BOUNDARY = 0.33
    ZONE_RIGHT_BOUNDARY = 0.67
    
    # Distance thresholds (area ratios)
    DISTANCE_VERY_CLOSE = 0.10
    DISTANCE_CLOSE = 0.04
    DISTANCE_MEDIUM = 0.015
    
    # Navigation detection filtering
    NAVIGATION_MIN_CONFIDENCE = 0.4
    NAVIGATION_MIN_RELEVANCE = 0.18
    NAVIGATION_MAX_OBJECTS = 3
    NAVIGATION_SIZE_RATIO = 0.08
    
    # Critical priority detection (immediate risks)
    CRITICAL_ALLOWED_CLASSES = {"person", "car", "truck", "bus", "bicycle", "motorcycle"}
    CRITICAL_DISTANCE_WALKING = {"very_close", "close"}
    CRITICAL_DISTANCE_STATIONARY = {"very_close"}
    CRITICAL_CENTER_TOLERANCE = 0.30  # ±30% from center (zona amarilla)
    CRITICAL_REQUIRE_YELLOW_ZONE = False  # If True, only announce objects in center zone
    CRITICAL_BBOX_COVERAGE_THRESHOLD = 0.35  # 35% bbox coverage for close objects
    CRITICAL_REPEAT_GRACE = 1.5  # Seconds before repeating same critical
    CRITICAL_COOLDOWN_WALKING = 1.0
    CRITICAL_COOLDOWN_STATIONARY = 2.0
    
    # Normal priority detection (obstacles, furniture)
    NORMAL_ALLOWED_CLASSES = {"chair", "table", "bottle", "door", "laptop", "couch", "bed"}
    NORMAL_DISTANCE = {"close", "medium"}
    NORMAL_CENTER_TOLERANCE = 0.30  # Same yellow zone ±30%
    NORMAL_REQUIRE_YELLOW_ZONE = True  # Obstacles must be in yellow zone
    NORMAL_PERSISTENCE_FRAMES = 2  # Must be detected for 2+ consecutive frames
    NORMAL_COOLDOWN = 2.5  # Longer cooldown for normal objects
    
    # Audio routing
    AUDIO_GLOBAL_COOLDOWN = 0.8  # Minimum between any two announcements
    AUDIO_INTERRUPT_GRACE = 0.25  # Grace period before interrupting (anti-entrecorte)
    AUDIO_QUEUE_SIZE = 12  # Increased from 3
    
    # Spatial audio beeps
    AUDIO_SPATIAL_BEEPS_ENABLED = True  # Enable spatial beeps before TTS
    BEEP_CRITICAL_FREQUENCY = 1000  # Hz - High pitch for critical alerts
    BEEP_CRITICAL_DURATION = 0.3  # seconds - Long beep for critical
    BEEP_NORMAL_FREQUENCY = 500  # Hz - Low pitch for normal obstacles
    BEEP_NORMAL_DURATION = 0.1  # seconds - Short beeps for normal
    BEEP_NORMAL_COUNT = 2  # Two short beeps for normal objects
    BEEP_NORMAL_GAP = 0.05  # Gap between normal beeps
    BEEP_VOLUME = 0.7  # 0.0 to 1.0
    
    # SLAM audio filtering
    SLAM_AUDIO_DUPLICATE_GRACE = 1.0  # Don't repeat if RGB mentioned same class recently
    SLAM_CRITICAL_ONLY = True  # Only emit SLAM events for critical distances
    
    # Aria streaming
    STREAMING_PROFILE = "profile28"  # Back to 28 for 10min test
    STREAMING_INTERFACE = "usb"  # "usb" or "wifi"
    STREAMING_PROFILE_USB = "profile28"
    STREAMING_PROFILE_WIFI = "profile15"
    STREAMING_WIFI_DEVICE_IP = "192.168.0.204"
    
    # Performance
    DETECTION_HISTORY_SIZE = 30             # GPU NVIDIA: Mayor historial
    CONSISTENCY_THRESHOLD = 2
    DEBUG_FRAME_INTERVAL = 100

    # Camera input (Aria RGB camera delivers RGB ordering)
    RGB_CAMERA_COLOR_SPACE = "RGB"  # "RGB" or "BGR"

      # ===================================================================
    # DEPTH ESTIMATION CONFIGURATION
    # ===================================================================
    
    # Distance estimation strategy
    DISTANCE_METHOD = "depth_only"  # "depth_only", "area_only", "hybrid"
    # Re-enabled: Testing depth impact with TensorRT YOLO
    DEPTH_ENABLED = True  # os.environ.get("ARIA_SKIP_DEPTH", "0") != "1"
    # DEPTH_FRAME_SKIP movido a __init__ (FASE 1)
    # DEPTH_INPUT_SIZE movido a __init__ (FASE 1)
    
    # ========== FASE 4: TensorRT ==========
    USE_TENSORRT = True  # Use .engine files instead of .pt
    
    # Backend selection
    DEPTH_BACKEND = "depth_anything_v2"  # "midas" or "depth_anything_v2"
    
    # MiDaS Configuration
    # Models: MiDaS_small (fastest), MiDaS, DPT_Large (most accurate)
    MIDAS_MODEL = "MiDaS_small"             # Modelo small para estabilidad
    MIDAS_DEVICE = DEVICE           # AUTO: cuda/mps/cpu
    
    # Depth Anything V2 Configuration
    # Models: Small (fastest), Base, Large (most accurate)
    DEPTH_ANYTHING_MODEL = "Small"          # Modelo small para estabilidad
    DEPTH_ANYTHING_DEVICE = DEVICE  # AUTO: cuda/mps/cpu
    
    # Distance thresholds (normalized depth values 0-1)
    DEPTH_CLOSE_THRESHOLD = 0.7     # Objects closer than this = "close"
    DEPTH_MEDIUM_THRESHOLD = 0.4    # Objects closer than this = "medium"
    
    # ===================================================================

    # Spatial zone configuration (improved with center zone)
    ZONE_SYSTEM = "five_zones"  # "four_quadrants" or "five_zones"
    CENTER_ZONE_WIDTH_RATIO = 0.4  # 40% del ancho total
    CENTER_ZONE_HEIGHT_RATIO = 0.25  # 25% del alto total
    CENTER_ZONE_PRIORITY_BOOST = 1.5  # Multiplicador de prioridad para zona central

    # IMAGE ENHANCEMENT SETTINGS
    # Low-light enhancement activation
    LOW_LIGHT_ENHANCEMENT = True    # Enable/disable enhancement system

    # Gamma correction settings  
    GAMMA_CORRECTION = 1.1          # 1.0 = no correction, >1.0 = brighter
                                    # Recommended: 1.1-1.8

    # Auto detection settings
    AUTO_ENHANCEMENT = True         # Auto-detect low light conditions
    LOW_LIGHT_THRESHOLD = 120.0     # Brightness threshold (0-255)
                                    # Lower = more sensitive to darkness

    # Performance settings
    ENHANCEMENT_DEBUG = True        # Show debug messages
    CLAHE_CLIP_LIMIT = 3.0         # Contrast enhancement strength
    CLAHE_TILE_SIZE = (8, 8)       # CLAHE tile grid size

    # Profiling
    PROFILE_PIPELINE = True
    PROFILE_WINDOW_FRAMES = 30

    # Phase 2: Multiprocessing controls
    # Use: python run.py benchmark
    # FASE 4: Re-enabled to test parallel execution with TensorRT
    PHASE2_MULTIPROC_ENABLED = True  # Test: Parallel RGB/Depth + SLAM workers
    PHASE2_QUEUE_MAXSIZE = 4  # Increased: Reduce queue blocking
    PHASE2_SLAM_QUEUE_MAXSIZE = 6  # Increased: More buffer for SLAM cameras
    PHASE2_RESULT_QUEUE_MAXSIZE = 10  # Increased: More buffer for results
    PHASE2_STATS_INTERVAL = 5.0
    PHASE2_BACKPRESSURE_TIMEOUT = 0.1  # Reduced: Faster timeout for queue.put()