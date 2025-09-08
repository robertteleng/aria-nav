import cv2
import numpy as np
import time
from typing import Dict, List, Optional


class OpenCVDashboard:
    """Dashboard simple con múltiples ventanas OpenCV"""
    
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.current_fps = 0.0
        self.audio_commands_sent = 0
        self.total_detections = 0
        
        # Configurar ventanas
        self._setup_windows()
        
        print("[DASHBOARD] OpenCV Dashboard inicializado")
        print("Ventanas: RGB+YOLO | Depth | SLAM1 | SLAM2 | Metrics | Logs")
    
    def _setup_windows(self):
        """Configurar ventanas con posiciones específicas"""
        # Ventana principal RGB
        cv2.namedWindow("RGB + YOLO", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RGB + YOLO", 640, 480)
        cv2.moveWindow("RGB + YOLO", 50, 50)
        
        # Depth map
        cv2.namedWindow("Depth Map", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Depth Map", 480, 360)
        cv2.moveWindow("Depth Map", 720, 50)
        
        # SLAM cámaras
        cv2.namedWindow("SLAM1", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("SLAM1", 400, 300)
        cv2.moveWindow("SLAM1", 1230, 50)
        
        cv2.namedWindow("SLAM2", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("SLAM2", 400, 300)
        cv2.moveWindow("SLAM2", 1230, 380)
        
        # Métricas como imagen
        cv2.namedWindow("Performance", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Performance", 400, 200)
        cv2.moveWindow("Performance", 50, 560)
        
        # Logs como imagen
        cv2.namedWindow("System Logs", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("System Logs", 600, 200)
        cv2.moveWindow("System Logs", 470, 560)
    
    def log_rgb_frame(self, frame: np.ndarray):
        """Mostrar RGB + YOLO en ventana principal"""
        if frame is not None:
            cv2.imshow("RGB + YOLO", frame)
            self.frame_count += 1
    
    def log_depth_map(self, depth_data: np.ndarray):
        """Mostrar depth map coloreado"""
        if depth_data is None:
            # Crear imagen vacía
            empty = np.zeros((360, 480, 3), dtype=np.uint8)
            cv2.putText(empty, "No Depth Data", (150, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Depth Map", empty)
            return
        
        # Normalizar y aplicar colormap
        depth_normalized = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        
        # Añadir info de distancia
        min_dist = np.min(depth_data)
        max_dist = np.max(depth_data)
        cv2.putText(depth_colored, f"Min: {min_dist:.1f}m", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(depth_colored, f"Max: {max_dist:.1f}m", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Depth Map", depth_colored)
    
    def log_slam1_frame(self, slam1_frame: np.ndarray):
        """Mostrar SLAM1"""
        if slam1_frame is not None:
            # Añadir título
            frame_copy = slam1_frame.copy()
            cv2.putText(frame_copy, "SLAM1 (Left)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("SLAM1", frame_copy)
    
    def log_slam2_frame(self, slam2_frame: np.ndarray):
        """Mostrar SLAM2"""
        if slam2_frame is not None:
            # Añadir título
            frame_copy = slam2_frame.copy()
            cv2.putText(frame_copy, "SLAM2 (Right)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("SLAM2", frame_copy)
    
    def log_performance_metrics(self):
        """Crear imagen con métricas de performance"""
        # Calcular FPS
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            elapsed = current_time - self.last_fps_update
            self.current_fps = self.frame_count / elapsed if elapsed > 0 else 0
            self.frame_count = 0
            self.last_fps_update = current_time
        
        # Crear imagen de métricas
        metrics_img = np.zeros((200, 400, 3), dtype=np.uint8)
        
        # Título
        cv2.putText(metrics_img, "PERFORMANCE METRICS", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Métricas
        uptime = current_time - self.start_time
        metrics = [
            f"FPS: {self.current_fps:.1f}",
            f"Detections: {self.total_detections}",
            f"Audio Commands: {self.audio_commands_sent}",
            f"Uptime: {uptime/60:.1f} min"
        ]
        
        for i, metric in enumerate(metrics):
            y_pos = 70 + i * 30
            cv2.putText(metrics_img, metric, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Performance", metrics_img)
    
    def log_system_message(self, message: str, level: str = "INFO"):
        """Añadir mensaje a buffer de logs (simplificado)"""
        if not hasattr(self, 'log_buffer'):
            self.log_buffer = []
        
        timestamp = time.strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] [{level}] {message}"
        
        # Mantener solo últimos 6 mensajes
        self.log_buffer.append(formatted_msg)
        if len(self.log_buffer) > 6:
            self.log_buffer.pop(0)
        
        # Crear imagen con logs
        logs_img = np.zeros((200, 600, 3), dtype=np.uint8)
        
        # Título
        cv2.putText(logs_img, "SYSTEM LOGS", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Mostrar logs
        for i, log_msg in enumerate(self.log_buffer):
            y_pos = 50 + i * 25
            color = (0, 255, 0)  # Verde por defecto
            if "ERROR" in log_msg:
                color = (0, 0, 255)  # Rojo
            elif "DETECT" in log_msg:
                color = (0, 255, 255)  # Amarillo
            elif "AUDIO" in log_msg:
                color = (255, 0, 255)  # Magenta
            
            # Truncar mensaje si es muy largo
            display_msg = log_msg[:70] + "..." if len(log_msg) > 70 else log_msg
            cv2.putText(logs_img, display_msg, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        cv2.imshow("System Logs", logs_img)
    
    def log_detections(self, detections: List[Dict], frame_shape: tuple):
        """Procesar detecciones para métricas"""
        if detections:
            self.total_detections += len(detections)
            
            # Log detecciones importantes
            for det in detections[:2]:  # Top 2
                name = det.get('name', 'unknown')
                zone = det.get('zone', 'unknown')
                priority = det.get('priority', 0)
                self.log_system_message(f"DETECT: {name} en {zone} (P{priority})", "DETECT")
    
    def log_audio_command(self, command: str, priority: int = 5):
        """Log comando de audio"""
        self.audio_commands_sent += 1
        self.log_system_message(f"AUDIO [P{priority}]: {command}", "AUDIO")
    
    def log_motion_state(self, motion_state: str, imu_magnitude: float):
        """Log estado de movimiento"""
        self.log_system_message(f"MOTION: {motion_state.upper()} (mag: {imu_magnitude:.2f})", "IMU")
    
    def update_all(self):
        """Actualizar todas las ventanas - llamar periódicamente"""
        # Actualizar métricas
        self.log_performance_metrics()
        
        # Procesar eventos OpenCV
        key = cv2.waitKey(1) & 0xFF
        return key
    
    def shutdown(self):
        """Cerrar todas las ventanas"""
        duration = (time.time() - self.start_time) / 60.0
        self.log_system_message(f"Sistema cerrándose - Sesión: {duration:.1f}min", "SYSTEM")
        
        print(f"[DASHBOARD] OpenCV Session: {duration:.1f}min, Commands: {self.audio_commands_sent}, Detections: {self.total_detections}")
        
        cv2.destroyAllWindows()


# INTEGRACIÓN EN TU OBSERVER EXISTENTE
"""
Para usar este dashboard en tu Observer, solo cambiar:

# En __init__:
from utils.opencv_dashboard import OpenCVDashboard
self.dashboard = OpenCVDashboard() if enable_dashboard else None

# En _processing_loop(), cambiar las llamadas de dashboard por:
if self.dashboard:
    if 'rgb' in processed_frames:
        rgb_with_overlay = self.frame_renderer.draw_navigation_overlay(...)
        self.dashboard.log_rgb_frame(rgb_with_overlay)
        self.dashboard.log_detections(rgb_detections, processed_frames['rgb'].shape)
    
    if 'slam1' in processed_frames:
        self.dashboard.log_slam1_frame(processed_frames['slam1'])
    
    if 'slam2' in processed_frames:
        self.dashboard.log_slam2_frame(processed_frames['slam2'])
    
    if depth_map is not None:
        self.dashboard.log_depth_map(depth_map)
    
    # Actualizar cada frame
    key = self.dashboard.update_all()
    if key == ord('q'):
        break
"""