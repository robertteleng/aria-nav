"""Centralized configuration for the navigation system"""

class Config:
    """System configuration constants"""
    
    # Video processing
    TARGET_FPS = 30
    YOLO_MODEL = "yolo12n.pt"
    YOLO_CONFIDENCE = 0.45
    YOLO_DEVICE = "cpu"  # Avoid MPS bug
    
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
    
    # Aria streaming
    STREAMING_PROFILE = "profile28"
    STREAMING_INTERFACE = "usb"
    
    # Performance
    DETECTION_HISTORY_SIZE = 8
    CONSISTENCY_THRESHOLD = 2
    DEBUG_FRAME_INTERVAL = 100

    # MiDaS depth estimation
    # Distance estimation strategy
    DISTANCE_METHOD = "depth_only"  # "depth_only", "area_only", "hybrid"
    MIDAS_MODEL = "MiDaS_small"  # Opciones: MiDaS_small, MiDaS, DPT_Large
    MIDAS_DEVICE = "cpu"         # Evitar problemas MPS
    DEPTH_ENABLED = False        # DESACTIVADO por rendimiento (CPU-only)
    
    # Distance calculation with depth
    DEPTH_CLOSE_THRESHOLD = 0.7   # Profundidad normalizada para "cerca"
    DEPTH_MEDIUM_THRESHOLD = 0.4  # Profundidad normalizada para "medio"

    # Spatial zone configuration (improved with center zone)
    ZONE_SYSTEM = "five_zones"  # "four_quadrants" or "five_zones"
    CENTER_ZONE_WIDTH_RATIO = 0.4  # 40% del ancho total
    CENTER_ZONE_HEIGHT_RATIO = 0.25  # 25% del alto total
    CENTER_ZONE_PRIORITY_BOOST = 1.5  # Multiplicador de prioridad para zona central

    # IMAGE ENHANCEMENT SETTINGS - Day 6

    # Low-light enhancement activation
    LOW_LIGHT_ENHANCEMENT = True    # Enable/disable enhancement system

    # Gamma correction settings  
    GAMMA_CORRECTION = 1.2          # 1.0 = no correction, >1.0 = brighter
                                # Recommended: 1.1-1.8

    # Auto detection settings
    AUTO_ENHANCEMENT = True         # Auto-detect low light conditions
    LOW_LIGHT_THRESHOLD = 60.0     # Brightness threshold (0-255)
                                # Lower = more sensitive to darkness

    # Performance settings
    ENHANCEMENT_DEBUG = True        # Show debug messages
    CLAHE_CLIP_LIMIT = 3.0         # Contrast enhancement strength
    CLAHE_TILE_SIZE = (8, 8)       # CLAHE tile grid size
