import rerun as rr
import numpy as np
import time
import cv2
from typing import Dict, List, Optional


class RerunDashboard:
    """Dashboard simple para TFM - Solo lo esencial"""
    
    def __init__(self, app_id: str = "aria_navigation_tfm"):
        rr.init(app_id, spawn=True)
        
        self.start_time = time.time()
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.current_fps = 0.0
        self.audio_commands_sent = 0
        self.total_detections = 0
        
        print("[DASHBOARD] Rerun dashboard initialized")
        
    def log_rgb_frame(self, frame: np.ndarray):
        """Log RGB frame principal"""
        timestamp = time.time() - self.start_time
        rr.set_time_seconds("timeline", timestamp)
        
        # Convertir BGR a RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame
            
        rr.log("camera/rgb", rr.Image(frame_rgb))
        self.frame_count += 1
        
    def log_detections(self, detections: List[Dict], frame_shape: tuple):
        """Log detecciones YOLO"""
        if not detections:
            return
            
        timestamp = time.time() - self.start_time
        rr.set_time_seconds("timeline", timestamp)
        
        boxes = []
        labels = []
        colors = []
        
        for detection in detections:
            bbox = detection['bbox']
            
            # Box formato [x, y, width, height]
            box = [
                float(bbox[0]), float(bbox[1]),
                float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])
            ]
            boxes.append(box)
            
            name = detection.get('name', 'unknown')
            zone = detection.get('zone', 'unknown')
            confidence = detection.get('confidence', 0.0)
            
            label = f"{name.upper()}\n({zone})\nConf: {confidence:.2f}"
            labels.append(label)
            
            # Color por prioridad
            if name in ['person', 'car', 'truck']:
                color = [255, 0, 0]  # Rojo
            elif name in ['bicycle', 'motorcycle']:
                color = [255, 255, 0]  # Amarillo
            else:
                color = [0, 255, 0]  # Verde
            colors.append(color)
        
        rr.log("detections/objects", rr.Boxes2D(
            array=boxes,
            array_format=rr.Box2DFormat.XYWH,
            labels=labels,
            colors=colors
        ))
        
        self.total_detections += len(detections)
        
    def log_performance_metrics(self):
        """Log métricas básicas"""
        timestamp = time.time() - self.start_time
        current_time = time.time()
        
        # Calcular FPS
        if current_time - self.last_fps_update >= 1.0:
            elapsed = current_time - self.last_fps_update
            self.current_fps = self.frame_count / elapsed if elapsed > 0 else 0
            self.frame_count = 0
            self.last_fps_update = current_time
        
        rr.set_time_seconds("timeline", timestamp)
        
        # Log métricas como texto en lugar de Scalar
        rr.log("metrics/info", rr.TextLog(f"FPS: {self.current_fps:.1f} | Uptime: {timestamp/60.0:.1f}min | Detections: {self.total_detections} | Audio: {self.audio_commands_sent}"))
            
    def log_audio_command(self, command: str, priority: int = 5):
        """Log comando de audio"""
        timestamp = time.time() - self.start_time
        rr.set_time_seconds("timeline", timestamp)
        
        self.audio_commands_sent += 1
        
        # Log como texto
        full_command = f"[P{priority}] {command}"
        rr.log("audio/commands", rr.TextLog(full_command))
        
    def log_depth_map(self, depth_data: np.ndarray):
        """Log mapa de profundidad"""
        if depth_data is None:
            return
            
        timestamp = time.time() - self.start_time
        rr.set_time_seconds("timeline", timestamp)
        
        if depth_data.dtype != np.float32:
            depth_data = depth_data.astype(np.float32)
            
        depth_clipped = np.clip(depth_data, 0.1, 10.0)
        rr.log("camera/depth", rr.DepthImage(depth_clipped, meter=1.0))
        
    def log_motion_state(self, motion_state: str, imu_magnitude: float):
        """Log estado de movimiento"""
        timestamp = time.time() - self.start_time
        rr.set_time_seconds("timeline", timestamp)
        
        state_text = f"MOTION: {motion_state.upper()}"
        rr.log("motion/state", rr.TextLog(state_text))
        
        try:
            rr.log("motion/magnitude", rr.Scalar(imu_magnitude))
        except:
            pass

    def log_slam_frames(self, slam1_frame=None, slam2_frame=None):
        """Log SLAM camera frames"""
        timestamp = time.time() - self.start_time
        rr.set_time_seconds("timeline", timestamp)
        
        if slam1_frame is not None:
            # Convertir a RGB si es grayscale
            if len(slam1_frame.shape) == 2:
                slam1_frame = cv2.cvtColor(slam1_frame, cv2.COLOR_GRAY2RGB)
            rr.log("camera/slam1", rr.Image(slam1_frame))
        
        if slam2_frame is not None:
            if len(slam2_frame.shape) == 2:
                slam2_frame = cv2.cvtColor(slam2_frame, cv2.COLOR_GRAY2RGB)
            rr.log("camera/slam2", rr.Image(slam2_frame))
                
    def shutdown(self):
        """Shutdown con resumen básico"""
        duration = (time.time() - self.start_time) / 60.0
        print(f"[DASHBOARD] Session: {duration:.1f}min, Commands: {self.audio_commands_sent}, Detections: {self.total_detections}")