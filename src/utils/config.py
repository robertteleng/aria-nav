"""Centralized configuration for the navigation system"""

class Config:
    """System configuration constants"""
    
    # Video processing
    TARGET_FPS = 30
    YOLO_MODEL = "yolo11n.pt"
    YOLO_CONFIDENCE = 0.6
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
    DEPTH_ENABLED = True         # Flag para activar/desactivar
    
    # Distance calculation with depth
    DEPTH_CLOSE_THRESHOLD = 0.7   # Profundidad normalizada para "cerca"
    DEPTH_MEDIUM_THRESHOLD = 0.4  # Profundidad normalizada para "medio"