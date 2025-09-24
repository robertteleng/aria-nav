import rerun as rr
import numpy as np
import time
import cv2
from typing import Dict, List, Optional


class RerunDashboard:
    """Dashboard 2x3 fijo para TFM Aria (compatible con rerun-sdk 0.24.x con fallback).

    Layout objetivo (2x3):
        RGB+YOLO | Depth Map | SLAM1+SLAM2
        Logs     | Metrics   | Acceleration Graph

    Notas de compatibilidad:
    - Intenta configurar el layout fijo con la API moderna de 0.24.x: rr.blueprint + contenedores.
    - Si algunos contenedores no existen en tu build, hace fallback a auto-layout, pero
      mantiene los mismos 'orígenes' (paths) para que el viewer muestre todos los paneles.
    - SLAM1+SLAM2 se renderiza como una única imagen compuesta (side-by-side) para
      asegurar que se vean a la vez en el mismo Spatial2DView.
    - El panel Acceleration usa rr.Scalars para traza temporal.
    """

    def __init__(self, app_id: str = "aria_navigation_tfm", slam_mode: str = "side_by_side"):
        # Inicia la app y lanza el viewer embebido (spawn=True) — 0.24.x
        rr.init(app_id, spawn=True)

        self.start_time = time.time()
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.current_fps = 0.0
        self.audio_commands_sent = 0
        self.total_detections = 0
        self.frame_index = 0
        # Modo de visualización SLAM: "side_by_side" (por defecto) o "overlay"
        self.slam_mode = slam_mode if slam_mode in ("side_by_side", "overlay") else "side_by_side"
        self._last_slam1: Optional[np.ndarray] = None
        self._last_slam2: Optional[np.ndarray] = None
        self._last_slam1_events: List[Dict] = []
        self._last_slam2_events: List[Dict] = []

        # --- Intento de layout 2x3 fijo con API moderna ---
        try:
            # Intento 1: contenedores Vertical/Horizontal para 2x3 fijo
            blueprint = rr.blueprint(
                rr.Vertical(
                    rr.Horizontal(
                        rr.Spatial2DView(origin=self.RGB_PATH, name="RGB + YOLO"),
                        rr.Spatial2DView(origin=self.DEPTH_PATH, name="Depth Map"),
                        rr.Spatial2DView(origin=self.SLAM_PATH, name="SLAM"),
                        name="Top Row",
                    ),
                    rr.Horizontal(
                        rr.TextLogView(origin=self.LOGS_PATH, name="Logs"),
                        rr.TimeSeriesView(origin=self.METRICS_PATH, name="Metrics"),
                        rr.TimeSeriesView(origin=self.MOTION_PATH, name="Motion"),
                        name="Bottom Row",
                    ),
                    name="TFM Navigation Dashboard (2x3)",
                )
            )
            rr.send_blueprint(blueprint)
        except AttributeError as e:
            # Intento 2: blueprint plano (sin contenedores). El viewer suele generar grid automáticamente.
            try:
                flat = rr.blueprint(
                    rr.Spatial2DView(origin=self.RGB_PATH, name="RGB + YOLO"),
                    rr.Spatial2DView(origin=self.DEPTH_PATH, name="Depth Map"),
                    rr.Spatial2DView(origin=self.SLAM_PATH, name="SLAM"),
                    rr.TextLogView(origin=self.LOGS_PATH, name="Logs"),
                    rr.TimeSeriesView(origin=self.METRICS_PATH, name="Metrics"),
                    rr.TimeSeriesView(origin=self.MOTION_PATH, name="Motion"),
                )
                rr.send_blueprint(flat)
                print("[DEBUG] Applied flat blueprint (no containers).")
            except Exception as e2:
                print(f"[DEBUG] Flat blueprint also failed ({e2}). Using auto layout.")
        except Exception as e:
            print(f"[DEBUG] Could not set blueprint layout ({e}). Using auto layout.")

        print("[DASHBOARD] Layout 2x3 listo:")
        print(f"  {self.RGB_PATH}: RGB + YOLO detecciones")
        print(f"  {self.DEPTH_PATH}: Depth Map")
        print(f"  {self.SLAM_PATH}: SLAM1+SLAM2 (composición)")
        print(f"  {self.LOGS_PATH}: Logs del sistema")
        print(f"  {self.METRICS_PATH}: Métricas (Scalars)")
        print(f"  {self.MOTION_PATH}: Aceleración / Movimiento")

    # ---------------------- UTILIDADES ----------------------
    def _ts(self) -> float:
        ts = time.time() - self.start_time
        rr.set_time_seconds("timeline", ts)
        return ts

    def _advance_frame_time(self):
        # Avanza un índice de frame entero para que todos los logs del mismo ciclo compartan timestamp
        self.frame_index += 1
        rr.set_time_sequence("frame", self.frame_index)
        # Sincroniza también el tiempo en segundos para el snapshot del frame
        self._ts()

    @staticmethod
    def _ensure_rgb(img: np.ndarray) -> np.ndarray:
        if img is None:
            return img
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.ndim == 3 and img.shape[2] == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @staticmethod
    def _stack_horizontal(img_left: Optional[np.ndarray], img_right: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if img_left is None and img_right is None:
            return None
        if img_left is None:
            return img_right
        if img_right is None:
            return img_left
        # Igualar alturas para apilado limpio
        h1, w1 = img_left.shape[:2]
        h2, w2 = img_right.shape[:2]
        target_h = min(h1, h2)

        def resize_h(img, h):
            ih, iw = img.shape[:2]
            if ih == h:
                return img
            new_w = int(iw * (h / ih))
            return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)

        left_r = resize_h(img_left, target_h)
        right_r = resize_h(img_right, target_h)
        return np.hstack([left_r, right_r])

    @staticmethod
    def _draw_slam_events(frame: np.ndarray, events: List[Dict], color: tuple) -> np.ndarray:
        if frame is None or not events:
            return frame
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

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
        return frame

    def _append_slam_message(self, message: str) -> None:
        if not message:
            return
        self.log_system_message(f"SLAM: {message}", level="SLAM")

    # ---------------------- LOGGING PANELES ----------------------
    def log_rgb_frame(self, frame: np.ndarray):
        self._advance_frame_time()
        rr.log(f"{self.RGB_PATH}/image", rr.Image(self._ensure_rgb(frame)))
        self.frame_count += 1

    def log_detections(self, *args, **kwargs):
        """Registra cajas 2D sobre la imagen RGB.

        Compatibilidad de firma:
        - log_detections(detections)
        - log_detections(frame, detections)
        - log_detections(detections=detections)

        Acepta listas de dicts, arrays Nx4/Nx5, o listas de filas.
        Ignora el frame si viene en la primera posición.
        """
        detections = None

        # kwargs explícitos
        if 'detections' in kwargs:
            detections = kwargs.get('detections')
        else:
            # args posicionales: [detections] o [frame, detections]
            if len(args) == 1:
                detections = args[0]
            elif len(args) >= 2:
                detections = args[1]

        # Normaliza y valida
        try:
            boxes, labels, colors, n = self._normalize_dets(detections)
        except Exception as e:
            self.log_system_message(f"DETECTIONS PARSE ERROR: {e}", level="ERROR")
            return
        if n == 0:
            return

        rr.log(
            f"{self.RGB_PATH}/dets",
            rr.Boxes2D(array=boxes, array_format=rr.Box2DFormat.XYWH, labels=labels, colors=colors),
        )
        self.total_detections += n
        for label in labels[:2]:
            self.log_system_message(f"DETECT: {label.splitlines()[0]}", level="DETECT")

    def _normalize_dets(self, detections) -> tuple:
        """Convierte detecciones en (boxes_xywh, labels, colors, count).

        Admite:
        - None, 0, int → 0 dets
        - lista de dicts con claves: 'bbox'(x1,y1,x2,y2) o 'xyxy' o 'xywh'
        - lista/np.ndarray de filas [x1,y1,x2,y2,(conf),(cls_id)]
        - dict con clave 'detections' (envoltorio)
        """
        if detections is None:
            return [], [], [], 0
        # ints o np.int → sin dets (tratamos cualquier int como contador, no como lista)
        if isinstance(detections, (int, np.integer)):
            return [], [], [], 0
        # Envoltorio {'detections': [...]}
        if isinstance(detections, dict) and 'detections' in detections:
            detections = detections['detections']

        # ndarray → filas
        if isinstance(detections, np.ndarray):
            det_list = detections.tolist()
        else:
            det_list = detections

        if not isinstance(det_list, (list, tuple)):
            return [], [], [], 0

        boxes, labels, colors = [], [], []

        def spanish_name(name: str) -> str:
            mapping = {
                'person': 'Persona',
                'car': 'Coche',
                'truck': 'Camión',
                'bus': 'Autobús',
                'bicycle': 'Bicicleta',
                'motorcycle': 'Moto',
                'stop sign': 'Señal de stop',
                'traffic light': 'Semáforo',
            }
            key = name.lower()
            return mapping.get(key, name.capitalize())

        def to_xywh_from_xyxy(row):
            x1, y1, x2, y2 = map(float, row[:4])
            return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]

        for det in det_list:
            if det is None:
                continue
            # dict con campos conocidos
            if isinstance(det, dict):
                if 'bbox' in det and isinstance(det['bbox'], (list, tuple, np.ndarray)) and len(det['bbox']) >= 4:
                    xywh = to_xywh_from_xyxy(det['bbox'])
                elif 'xyxy' in det and isinstance(det['xyxy'], (list, tuple, np.ndarray)) and len(det['xyxy']) >= 4:
                    xywh = to_xywh_from_xyxy(det['xyxy'])
                elif 'xywh' in det and isinstance(det['xywh'], (list, tuple, np.ndarray)) and len(det['xywh']) >= 4:
                    x, y, w, h = det['xywh'][:4]
                    xywh = [float(x), float(y), float(w), float(h)]
                else:
                    continue
                raw_name = str(det.get('name', det.get('class', 'object')))
                name = spanish_name(raw_name)
                zone = str(det.get('zone', ''))
                conf = float(det.get('confidence', det.get('conf', 0.0)))
                pri = int(det.get('priority', 0))
            # fila tipo [x1,y1,x2,y2,(conf),(cls)]
            elif isinstance(det, (list, tuple, np.ndarray)) and len(det) >= 4:
                xywh = to_xywh_from_xyxy(det)
                conf = float(det[4]) if len(det) >= 5 else 0.0
                cls_name = str(det[5]) if len(det) >= 6 else 'object'
                name, zone, pri = spanish_name(cls_name), '', 0
            else:
                continue

            label_zone = zone.replace('_', ' ') if zone else ''
            if label_zone:
                label = f"{name}\n{label_zone} | P{pri}\nConf: {conf:.2f}"
            else:
                label = f"{name}\nP{pri}\nConf: {conf:.2f}"
            if pri >= 8:
                color = [255, 0, 0]
            elif pri >= 5:
                color = [255, 165, 0]
            else:
                color = [0, 255, 0]
            boxes.append(xywh)
            labels.append(label)
            colors.append(color)

        return boxes, labels, colors, len(boxes)

    def log_depth_map(self, depth_data: Optional[np.ndarray]):
        if depth_data is None:
            return
        if depth_data.ndim == 2:
            depth_norm = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
            depth_colored = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)
        else:
            depth_colored = depth_data
        rr.log(f"{self.DEPTH_PATH}/image", rr.Image(self._ensure_rgb(depth_colored)))

    def log_slam1_frame(self, slam1_frame: Optional[np.ndarray], events: Optional[List[Dict]] = None):
        if slam1_frame is None:
            return
        self._last_slam1 = slam1_frame
        self._last_slam1_events = events or []
        self._compose_and_log_slam()

    def log_slam2_frame(self, slam2_frame: Optional[np.ndarray], events: Optional[List[Dict]] = None):
        if slam2_frame is None:
            return
        self._last_slam2 = slam2_frame
        self._last_slam2_events = events or []
        self._compose_and_log_slam()

    # Compat: permitir log combinado explícito
    def log_slam_frames(self, slam1_frame: Optional[np.ndarray] = None, slam2_frame: Optional[np.ndarray] = None):
        if slam1_frame is not None:
            self._last_slam1 = slam1_frame
            self._last_slam1_events = []
        if slam2_frame is not None:
            self._last_slam2 = slam2_frame
            self._last_slam2_events = []
        self._compose_and_log_slam()

    def _compose_and_log_slam(self):
        # Render SLAM1+SLAM2 según modo seleccionado
        img1 = getattr(self, "_last_slam1", None)
        img2 = getattr(self, "_last_slam2", None)
        if img1 is None and img2 is None:
            return
        events1 = getattr(self, "_last_slam1_events", [])
        events2 = getattr(self, "_last_slam2_events", [])
        if img1 is not None and events1:
            img1 = self._draw_slam_events(img1.copy(), events1, (0, 165, 255))
        if img2 is not None and events2:
            img2 = self._draw_slam_events(img2.copy(), events2, (255, 128, 0))
        if self.slam_mode == "overlay":
            if img1 is not None:
                rr.log(f"{self.SLAM_PATH}/slam1", rr.Image(self._ensure_rgb(img1)))
            if img2 is not None:
                rr.log(f"{self.SLAM_PATH}/slam2", rr.Image(self._ensure_rgb(img2)))
        else:
            stacked = self._stack_horizontal(self._ensure_rgb(img1), self._ensure_rgb(img2))
            rr.log(f"{self.SLAM_PATH}/image", rr.Image(stacked))

    def set_slam_mode(self, mode: str):
        """Cambia el modo de SLAM en caliente: 'side_by_side' o 'overlay'."""
        if mode in ("side_by_side", "overlay"):
            self.slam_mode = mode

    def log_system_message(self, message: str, level: str = "INFO"):
        rr.log(f"{self.LOGS_PATH}", rr.TextLog(f"[{level}] {message}"))

    # Métricas en Panel 5 (TimeSeriesView)
    def log_performance_metrics(self):
        # No movemos el tiempo aquí; compartirá snapshot con el último frame
        ts = time.time() - self.start_time
        now = time.time()
        if now - self.last_fps_update >= 1.0:
            elapsed = now - self.last_fps_update
            self.current_fps = self.frame_count / elapsed if elapsed > 0 else 0.0
            self.frame_count = 0
            self.last_fps_update = now
        # Series temporales
        rr.log(f"{self.METRICS_PATH}/fps", rr.Scalars(self.current_fps))
        rr.log(f"{self.METRICS_PATH}/detections", rr.Scalars(float(self.total_detections)))
        rr.log(f"{self.METRICS_PATH}/audio_commands", rr.Scalars(float(self.audio_commands_sent)))
        rr.log(f"{self.METRICS_PATH}/uptime", rr.Scalars(ts))

    # Panel 6: Aceleración (magnitud IMU)
    def log_acceleration_magnitude(self, accel_mag: float):
        rr.log(f"{self.MOTION_PATH}/mag", rr.Scalars(float(accel_mag)))

    # Utilidades adicionales
    def log_audio_command(self, command: str, priority: int = 5):
        self.audio_commands_sent += 1
        self.log_system_message(f"AUDIO [P{priority}]: {command}", "AUDIO")

    def log_motion_state(self, motion_state: str, imu_magnitude: float):
        self.log_system_message(f"MOTION: {motion_state.upper()} (mag: {imu_magnitude:.2f})", "IMU")
        self.log_acceleration_magnitude(imu_magnitude)

    def shutdown(self):
        duration = (time.time() - self.start_time) / 60.0
        self.log_system_message(f"Sistema cerrándose - Sesión: {duration:.1f}min", "SYSTEM")
        print(
            f"[DASHBOARD] Session: {duration:.1f}min, Commands: {self.audio_commands_sent}, Detections: {self.total_detections}"
        )
