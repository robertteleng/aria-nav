"""Centralized configuration for the navigation system"""

class Config:
    """System configuration constants"""
    
    # Video processing
    TARGET_FPS = 30
    YOLO_MODEL = "yolo12n.pt"              
    YOLO_CONFIDENCE = 0.50
    YOLO_DEVICE = "mps"                    
    YOLO_IMAGE_SIZE = 256                  
    YOLO_MAX_DETECTIONS = 8                
    YOLO_IOU_THRESHOLD = 0.45
    YOLO_FRAME_SKIP = 3                    
    YOLO_FORCE_MPS = True

    # Peripheral vision (SLAM)
    PERIPHERAL_VISION_ENABLED = True
    SLAM_TARGET_FPS = 8
    SLAM_FRAME_SKIP = 12              
    
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
    STREAMING_PROFILE = "profile28"
    STREAMING_INTERFACE = "usb"  # "usb" or "wifi"
    STREAMING_PROFILE_USB = "profile28"
    STREAMING_PROFILE_WIFI = "profile18"
    STREAMING_WIFI_DEVICE_IP = "192.168.0.201"
    
    # Performance
    DETECTION_HISTORY_SIZE = 8
    CONSISTENCY_THRESHOLD = 2
    DEBUG_FRAME_INTERVAL = 100

    # Camera input (Aria RGB camera delivers RGB ordering)
    RGB_CAMERA_COLOR_SPACE = "RGB"  # "RGB" or "BGR"

    # Model depth estimation
    # Distance estimation strategy
    DISTANCE_METHOD = "depth_only"  # "depth_only", "area_only", "hybrid"
    DEPTH_ENABLED = True        # Activar profundidad con MPS
    DEPTH_FRAME_SKIP = 12        # Profundidad cada 12 frames para equilibrar rendimiento
    DEPTH_INPUT_SIZE = 80     # Tamaño de entrada para el modelo de profundidad

    #MiDas
    MIDAS_MODEL = "MiDaS_small"  # Opciones: MiDaS_small, MiDaS, DPT_Large
    MIDAS_DEVICE = "mps"         # Evitar problemas MPS

    # Depth Anything v2
    DEPTH_BACKEND = "midas"     # "midas" o "depth_anything_v2"
    DEPTH_ANYTHING_VARIANT = "Small"
    
    # Distance calculation with depth
    DEPTH_CLOSE_THRESHOLD = 0.7   # Profundidad normalizada para "cerca"
    DEPTH_MEDIUM_THRESHOLD = 0.4  # Profundidad normalizada para "medio"

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
