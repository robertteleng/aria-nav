import rerun as rr
import numpy as np
import time
from typing import Dict, List, Any, Optional

class RerunDashboard:
    """Professional dashboard para TFM presentation"""
    
    def __init__(self, app_id: str = "aria_navigation_tfm"):
        """Initialize Rerun dashboard"""
        rr.init(app_id, spawn=True)
        
        # Performance tracking
        self.start_time = time.time()
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.current_fps = 0.0
        
        print("[DASHBOARD] Rerun dashboard initialized")
        print("[DASHBOARD] Opening browser...")
        
    def log_rgb_frame(self, frame: np.ndarray, timestamp: Optional[float] = None):
        """Log RGB frame con timestamp"""
        if timestamp is None:
            timestamp = time.time() - self.start_time
            
        rr.set_time_seconds("timeline", timestamp)
        rr.log("camera/rgb", rr.Image(frame))
        
    def log_depth_map(self, depth_map: np.ndarray, timestamp: Optional[float] = None):
        """Log depth map visualization"""
        if depth_map is None:
            return
            
        if timestamp is None:
            timestamp = time.time() - self.start_time
            
        rr.set_time_seconds("timeline", timestamp)
        
        # Normalize depth para visualización
        depth_normalized = (depth_map * 255).astype(np.uint8)
        rr.log("camera/depth", rr.DepthImage(depth_normalized))
        
    def log_detections(self, detections: List[Dict], frame_shape: tuple, timestamp: Optional[float] = None):
        """Log YOLO detections como 2D boxes"""
        if not detections:
            return
            
        if timestamp is None:
            timestamp = time.time() - self.start_time
            
        rr.set_time_seconds("timeline", timestamp)
        
        # Convertir detections a formato Rerun
        boxes = []
        labels = []
        colors = []
        
        for detection in detections:
            bbox = detection['bbox']  # [x1, y1, x2, y2]
            
            # Box en formato Rerun [min_x, min_y, width, height]
            box = [
                float(bbox[0]), float(bbox[1]),  # min_x, min_y
                float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])  # width, height
            ]
            boxes.append(box)
            
            # Label con info completa
            name = detection.get('name', 'unknown')
            zone = detection.get('zone', 'unknown')
            distance = detection.get('distance', 'unknown')
            confidence = detection.get('confidence', 0.0)
            
            label = f"{name} ({zone}) {distance} - {confidence:.2f}"
            labels.append(label)
            
            # Color por zona
            zone_colors = {
                'center': [255, 0, 0],      # Rojo para centro
                'upper_left': [255, 255, 0], # Amarillo
                'upper_right': [255, 165, 0], # Naranja
                'lower_left': [0, 0, 255],   # Azul
                'lower_right': [255, 0, 255] # Magenta
            }
            color = zone_colors.get(zone, [128, 128, 128])
            colors.append(color)
        
        if boxes:
            rr.log("detections/boxes", rr.Boxes2D(
                array=boxes,
                labels=labels,
                colors=colors
            ))
    
    def log_performance_metrics(self, timestamp: Optional[float] = None):
        """Log performance metrics"""
        if timestamp is None:
            timestamp = time.time() - self.start_time
            
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_update >= 1.0:
            elapsed = current_time - self.last_fps_update
            self.current_fps = self.frame_count / elapsed if elapsed > 0 else 0
            self.frame_count = 0
            self.last_fps_update = current_time
        
        rr.set_time_seconds("timeline", timestamp)
        # API correcta para scalar:
        rr.log("metrics/fps", rr.Scalar(float(self.current_fps)))

    def log_motion_state(self, motion_state: str, magnitude: float, timestamp: Optional[float] = None):
        """Log motion detection state"""
        if timestamp is None:
            timestamp = time.time() - self.start_time
            
        rr.set_time_seconds("timeline", timestamp)
        # API correcta para texto y scalar:
        rr.log("motion/state", rr.TextLog(str(motion_state)))
        rr.log("motion/magnitude", rr.Scalar(float(magnitude)))

    def log_detections(self, detections: List[Dict], frame_shape: tuple, timestamp: Optional[float] = None):
        """Log YOLO detections como 2D boxes"""
        if not detections:
            return
            
        if timestamp is None:
            timestamp = time.time() - self.start_time
            
        rr.set_time_seconds("timeline", timestamp)
        
        boxes = []
        labels = []
        
        for detection in detections:
            bbox = detection['bbox']
            box = [float(bbox[0]), float(bbox[1]), float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])]
            boxes.append(box)
            
            name = detection.get('name', 'unknown')
            zone = detection.get('zone', 'unknown') 
            distance = detection.get('distance', 'unknown')
            confidence = detection.get('confidence', 0.0)
            
            label = f"{name} ({zone}) {distance} - {confidence:.2f}"
            labels.append(label)
        
        if boxes:
            # API correcta con array_format:
            rr.log("detections/boxes", rr.Boxes2D(
                array=boxes,
                array_format=rr.Box2DFormat.XYWH,
                labels=labels
            ))

    def log_audio_command(self, command: str, timestamp: Optional[float] = None):
        """Log audio commands sent to TTS"""
        if timestamp is None:
            timestamp = time.time() - self.start_time
            
        rr.set_time_seconds("timeline", timestamp)
        rr.log("audio/commands", rr.TextLog(str(command)))