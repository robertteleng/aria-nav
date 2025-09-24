import cv2
import numpy as np
import time
from typing import Dict, List, Optional

from utils.config import Config


class OpenCVDashboard:
    """
    Dashboard OpenCV en UNA sola ventana con cuadrícula 2x3.

    Disposición (2 filas x 3 columnas):
      [RGB+YOLO] [Depth] [SLAM1|SLAM2]
      [Logs]     [Metrics] [Ayuda]
    """

    def __init__(self, cell_size=(512, 384), window_name="ARIA Dashboard (OpenCV)"):
        self.start_time = time.time()
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.current_fps = 0.0
        self._frame_skip_multiplier = max(1, getattr(Config, "YOLO_FRAME_SKIP", 1))
        self.audio_commands_sent = 0
        self.total_detections = 0

        # Configuración de la cuadrícula
        self.cols = 3
        self.rows = 2
        self.cell_w, self.cell_h = int(cell_size[0]), int(cell_size[1])
        self.canvas_w = self.cols * self.cell_w
        self.canvas_h = self.rows * self.cell_h
        self.window_name = window_name

        # Buffers de cada panel
        self.rgb_img = None
        self.depth_img = None
        self.slam1_img = None
        self.slam2_img = None
        self.metrics_img = np.zeros((self.cell_h, self.cell_w, 3), dtype=np.uint8)
        self.logs_img = np.zeros((self.cell_h, self.cell_w, 3), dtype=np.uint8)
        self.slam_event_counts = {'slam1': 0, 'slam2': 0}

        # Ventana única
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.canvas_w, self.canvas_h)

        print("[DASHBOARD] OpenCV Dashboard en 1 ventana (2x3) listo")
    
    def log_rgb_frame(self, frame: np.ndarray):
        """Mostrar RGB + YOLO en ventana principal"""
        if frame is not None:
            self.rgb_img = self._resize_keep(frame, self.cell_w, self.cell_h)
            self.frame_count += 1
    
    def log_depth_map(self, depth_data: np.ndarray):
        """Mostrar depth map coloreado"""
        if depth_data is None:
            self.depth_img = self._depth_placeholder("No Depth Data")
            return

        try:
            depth_array = np.asarray(depth_data)

            if depth_array.ndim == 3 and depth_array.shape[2] > 1:
                depth_array = cv2.cvtColor(depth_array, cv2.COLOR_BGR2GRAY)

            depth_array = np.nan_to_num(depth_array, nan=0.0, posinf=0.0, neginf=0.0)

            if depth_array.dtype not in (np.float32, np.float64):
                depth_metrics = depth_array.astype(np.float32)
            else:
                depth_metrics = depth_array

            depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
            depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)

            min_dist = float(np.nanmin(depth_metrics))
            max_dist = float(np.nanmax(depth_metrics))

            cv2.putText(depth_colored, f"Min: {min_dist:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(depth_colored, f"Max: {max_dist:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            self.depth_img = self._resize_keep(depth_colored, self.cell_w, self.cell_h)
        except cv2.error as err:
            self.depth_img = self._depth_placeholder("Depth error")
            self.log_system_message(f"Depth map render failed: {err}", "ERROR")

    
    def log_slam1_frame(self, slam1_frame: np.ndarray, events: Optional[List[Dict]] = None):
        """Mostrar SLAM1"""
        if slam1_frame is not None:
            frame_copy = slam1_frame.copy()
            if frame_copy.ndim == 2 or (frame_copy.ndim == 3 and frame_copy.shape[2] == 1):
                frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_GRAY2BGR)
            if events:
                self._draw_slam_events(frame_copy, events, (0, 165, 255))
                self.slam_event_counts['slam1'] = len(events)
            else:
                self.slam_event_counts['slam1'] = 0
            cv2.putText(frame_copy, "SLAM1 (Left)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            self.slam1_img = self._resize_keep(frame_copy, self.cell_w//2, self.cell_h)
        else:
            self.slam_event_counts['slam1'] = 0

    def log_slam2_frame(self, slam2_frame: np.ndarray, events: Optional[List[Dict]] = None):
        """Mostrar SLAM2"""
        if slam2_frame is not None:
            frame_copy = slam2_frame.copy()
            if frame_copy.ndim == 2 or (frame_copy.ndim == 3 and frame_copy.shape[2] == 1):
                frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_GRAY2BGR)
            if events:
                self._draw_slam_events(frame_copy, events, (255, 128, 0))
                self.slam_event_counts['slam2'] = len(events)
            else:
                self.slam_event_counts['slam2'] = 0
            cv2.putText(frame_copy, "SLAM2 (Right)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            self.slam2_img = self._resize_keep(frame_copy, self.cell_w//2, self.cell_h)
        else:
            self.slam_event_counts['slam2'] = 0
    
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
        metrics_img = np.zeros((self.cell_h, self.cell_w, 3), dtype=np.uint8)
        
        # Título
        cv2.putText(metrics_img, "PERFORMANCE METRICS", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Métricas
        uptime = current_time - self.start_time
        fps_display = min(self.current_fps * self._frame_skip_multiplier, float(Config.TARGET_FPS))

        metrics = [
            f"FPS: {fps_display:.1f}",
            f"Detections: {self.total_detections}",
            f"Audio Commands: {self.audio_commands_sent}",
            f"SLAM1 det: {self.slam_event_counts['slam1']}",
            f"SLAM2 det: {self.slam_event_counts['slam2']}",
            f"Uptime: {uptime/60:.1f} min"
        ]
        
        for i, metric in enumerate(metrics):
            y_pos = 70 + i * 40
            cv2.putText(metrics_img, metric, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        self.metrics_img = metrics_img

    def _draw_slam_events(self, frame: np.ndarray, events: List[Dict], color: tuple) -> None:
        if frame is None or events is None:
            return
        height, width = frame.shape[:2]
        for event in events:
            bbox = event.get('bbox')
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width - 1, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = event.get('name', 'obj')
            distance = event.get('distance')
            if distance and distance not in {'', 'unknown'}:
                label = f"{label} {distance}"
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    
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
        logs_img = np.zeros((self.cell_h, self.cell_w, 3), dtype=np.uint8)
        
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
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        self.logs_img = logs_img
    
    def log_detections(self, detections: List[Dict], frame_shape: tuple = None):
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
        """Update optimizado - solo recomponer cuando sea necesario"""
        if not hasattr(self, 'update_counter'):
            self.update_counter = 0
        
        self.update_counter += 1
        
        # Solo recomponer canvas cada 3 frames (reduce carga)
        if self.update_counter % 3 == 0:
            self._compose_canvas()
        # Actualizar métricas cada 30 frames (~1 segundo)
        if self.update_counter % 30 == 0:
            self.log_performance_metrics()

        return cv2.waitKey(1) & 0xFF

    def _compose_canvas(self):
        """Componer la cuadrícula y refrescar la ventana única - optimizada"""
        # Componer canvas
        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

        # Celda (0,0) RGB
        if self.rgb_img is not None:
            canvas[0:self.cell_h, 0:self.cell_w] = self._pad_to_cell(self.rgb_img)
        else:
            self._put_label(canvas, (0, 0), "RGB + YOLO")

        # Celda (0,1) Depth
        if self.depth_img is not None:
            canvas[0:self.cell_h, self.cell_w:2*self.cell_w] = self._pad_to_cell(self.depth_img)
        else:
            self._put_label(canvas, (0, 1), "Depth Map")

        # Celda (0,2) SLAM compositado (dos mitades)
        slam_cell = np.zeros((self.cell_h, self.cell_w, 3), dtype=np.uint8)
        if self.slam1_img is not None:
            slam_cell[:, 0:self.cell_w//2] = self._pad_to_size(self.slam1_img, self.cell_h, self.cell_w//2)
        if self.slam2_img is not None:
            slam_cell[:, self.cell_w//2:self.cell_w] = self._pad_to_size(self.slam2_img, self.cell_h, self.cell_w//2)
        if self.slam1_img is None and self.slam2_img is None:
            cv2.putText(slam_cell, "SLAM1 | SLAM2", (20, self.cell_h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        canvas[0:self.cell_h, 2*self.cell_w:3*self.cell_w] = slam_cell

        # Fila 2: Logs y Métricas
        canvas[self.cell_h:2*self.cell_h, 0:self.cell_w] = self.logs_img
        canvas[self.cell_h:2*self.cell_h, self.cell_w:2*self.cell_w] = self.metrics_img

        # Celda (1,2) Ayuda
        help_cell = np.zeros((self.cell_h, self.cell_w, 3), dtype=np.uint8)
        help_lines = [
            "ARIA OpenCV Dashboard",
            "q: salir  |  Ventana unica",
            f"FPS: {self.current_fps:.1f}",
        ]
        for i, line in enumerate(help_lines):
            cv2.putText(help_cell, line, (10, 40 + i*40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        canvas[self.cell_h:2*self.cell_h, 2*self.cell_w:3*self.cell_w] = help_cell

        # Líneas de la cuadrícula
        for c in range(1, self.cols):
            x = c * self.cell_w
            cv2.line(canvas, (x, 0), (x, self.canvas_h), (50, 50, 50), 1)
        for r in range(1, self.rows):
            y = r * self.cell_h
            cv2.line(canvas, (0, y), (self.canvas_w, y), (50, 50, 50), 1)

        cv2.imshow(self.window_name, canvas)
        return canvas
    def shutdown(self):
        """Cerrar todas las ventanas"""
        duration = (time.time() - self.start_time) / 60.0
        self.log_system_message(f"Sistema cerrándose - Sesión: {duration:.1f}min", "SYSTEM")
        
        print(f"[DASHBOARD] OpenCV Session: {duration:.1f}min, Commands: {self.audio_commands_sent}, Detections: {self.total_detections}")
        cv2.destroyAllWindows()

    # --------- helpers internos ---------
    def _resize_keep(self, img, target_w, target_h):
        h, w = img.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _pad_to_cell(self, img):
        return self._pad_to_size(img, self.cell_h, self.cell_w)

    @staticmethod
    def _pad_to_size(img, target_h, target_w):
        h, w = img.shape[:2]
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y = (target_h - h) // 2
        x = (target_w - w) // 2
        canvas[y:y+h, x:x+w] = img[: target_h - y, : target_w - x]
        return canvas

    def _put_label(self, canvas, cell_pos, text):
        r, c = cell_pos
        x0, y0 = c * self.cell_w, r * self.cell_h
        cv2.putText(canvas, text, (x0 + 20, y0 + self.cell_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 180, 180), 2)

    def _depth_placeholder(self, message: str) -> np.ndarray:
        tile = np.zeros((self.cell_h, self.cell_w, 3), dtype=np.uint8)
        cv2.putText(tile, message, (30, self.cell_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return tile
