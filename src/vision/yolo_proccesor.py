import numpy as np
from ultralytics import YOLO
from typing import List, Tuple
from vision.detected_object import DetectedObject
from vision.depth_estimator import DepthEstimator

from utils.config import Config



class YoloProcessor:
    """YOLO-based object detection optimized for navigation"""
    
    def __init__(self):
        print("[INFO] Loading YOLO model...")
        self.model = YOLO('yolo12n.pt')
        self.device = 'cpu'  # Avoid MPS NMS bug
        self.model.to(self.device)
        
        # Navigation-relevant objects with priorities
        self.navigation_objects = {
            'person': {'priority': 1.0, 'name': 'person'},
            'car': {'priority': 0.9, 'name': 'car'},
            'bicycle': {'priority': 0.8, 'name': 'bicycle'}, 
            'bus': {'priority': 0.9, 'name': 'bus'},
            'truck': {'priority': 0.8, 'name': 'truck'},
            'motorcycle': {'priority': 0.7, 'name': 'motorcycle'},
            'stop sign': {'priority': 0.9, 'name': 'stop sign'},
            'traffic light': {'priority': 0.6, 'name': 'traffic light'}
        }
        
        self.detection_count = 0
        # self.depth_estimator = DepthEstimator()

        print("[INFO] ✓ YOLOv11 processor initialized")
    
    def process_frame(self, frame: np.array, depth_map: np.array = None) -> List[dict]:
        """
        Process frame through YOLO and return structured detections
        
        Returns:
            List of detection dictionaries for audio and visualization
        """
        try:
            # Run YOLO inference
            results = self.model(frame, device=self.device, verbose=False)
            
            # Convert to structured objects
            # detected_objects = self._analyze_detections(results, frame.shape[1])

            # depth_map = self.depth_estimator.estimate_depth(frame)
            detected_objects = self._analyze_detections(results, frame.shape[1], depth_map)

            
            # Convert to simple dictionaries for other modules
            detections = []
            for obj in detected_objects:
                detections.append({
                    'bbox': obj.bbox,
                    'name': obj.name,
                    'confidence': obj.confidence,
                    'zone': obj.zone,
                    'distance': obj.distance_bucket,
                    'relevance_score': obj.relevance_score
                })
            
            self.detection_count += len(detections)
            return detections
            
        except Exception as e:
            print(f"[WARN] YOLO processing failed: {e}")
            return []
    
    def _analyze_detections(self, yolo_results, frame_width: int, depth_map: np.array = None) -> List[DetectedObject]:
        """Convert raw YOLO output to structured detection objects"""
        objects = []
        
        # Zone boundaries
        left_boundary = frame_width * 0.33
        right_boundary = frame_width * 0.67
        
        for detection in yolo_results[0].boxes.data:
            x1, y1, x2, y2, confidence, class_id = detection.cpu().numpy()
            
            # Filter 1: Known navigation objects only
            class_name = yolo_results[0].names[int(class_id)]
            if class_name not in self.navigation_objects:
                continue
            
            # Filter 2: Confidence threshold
            if confidence < 0.6:
                continue
                
            # Spatial analysis
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)
            
            # Zone classification
            zone = self._classify_zone(center_x, center_y, frame_width)

            # Distance estimation:
            bbox = (int(x1), int(y1), int(x2), int(y2))
            depth_value = 0.5  # Default
            
            if depth_map is not None:
                # Solo usar el depth_map que ya tenemos, NO crear nuevo estimator
                depth_value = self._calculate_depth_from_map(depth_map, bbox)
                print(f"[DEBUG] Object: {class_name}, Depth: {depth_value:.2f}")  # AÑADIR ESTO

            # Distance estimation
            distance_bucket = self._estimate_distance_with_depth(area, frame_width, depth_value)

            # Relevance scoring
            base_priority = self.navigation_objects[class_name]['priority']
            size_factor = min(area / (frame_width * 300), 1.0)
            relevance_score = base_priority * float(confidence) * (0.3 + 0.7 * size_factor)
            
            # Filter 3: Minimum relevance threshold
            if relevance_score < 0.6:
                continue

            # # ANTES de crear DetectedObject, añadir:
            # depth_value = 0.5  # Default
            # if depth_map is not None and self.depth_estimator.model is not None:
            #     depth_value = self.depth_estimator.get_object_depth(depth_map, (int(x1), int(y1), int(x2), int(y2)))
            #     # Actualizar distance_bucket con profundidad
            #     distance_bucket = self._estimate_distance_with_depth(area, frame_width, depth_value)

            obj = DetectedObject(
                name=self.navigation_objects[class_name]['name'],
                confidence=float(confidence),
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                center_x=center_x,
                center_y=center_y,
                area=area,
                zone=zone,
                distance_bucket=distance_bucket,
                relevance_score=relevance_score,
                depth_value=depth_value
            )
            
            objects.append(obj)
        
        # Return top detection only to reduce audio spam
        objects.sort(key=lambda x: x.relevance_score, reverse=True)
        return objects[:1]
    
    def _classify_zone(self, center_x: float, center_y: float, frame_width: int) -> str:
        """Classify object position into spatial zones (4 quadrants + optional center)"""
        from utils.config import Config
        
        frame_height = int(frame_width * 0.75)  # Assume 4:3 aspect ratio
        
        # Si usamos el sistema de 5 zonas
        if Config.ZONE_SYSTEM == "five_zones":
            # Definir zona central
            center_margin_x = frame_width * Config.CENTER_ZONE_WIDTH_RATIO
            center_margin_y = frame_height * Config.CENTER_ZONE_HEIGHT_RATIO
            
            center_left = frame_width/2 - center_margin_x/2
            center_right = frame_width/2 + center_margin_x/2
            center_top = frame_height/2 - center_margin_y/2
            center_bottom = frame_height/2 + center_margin_y/2
            
            # Verificar si está en zona central
            if (center_left <= center_x <= center_right and 
                center_top <= center_y <= center_bottom):
                return "center"
        
        # Si no está en centro, usar cuadrantes tradicionales
        mid_x = frame_width / 2
        mid_y = frame_height / 2
        
        if center_y < mid_y:
            return "top_left" if center_x < mid_x else "top_right"
        else:
            return "bottom_left" if center_x < mid_x else "bottom_right"
    
    def _estimate_distance(self, area: float, frame_width: int) -> str:
        """Estimate relative distance based on bounding box area"""
        frame_height = int(frame_width * 0.75)
        frame_area = frame_width * frame_height
        
        area_ratio = area / frame_area
        
        if area_ratio >= 0.10:
            return "very close"
        elif area_ratio >= 0.04:
            return "close"
        elif area_ratio >= 0.015:
            return "medium"
        else:
            return "far"
        
    def _estimate_distance_with_depth(self, area: float, frame_width: int, depth_value: float) -> str:
        """Enhanced distance estimation with selectable strategy"""
        
        print(f"[DEBUG] Config method: {Config.DISTANCE_METHOD}, Depth: {depth_value:.2f}")

        # AQUÍ ESTÁ LA LÓGICA DE ELECCIÓN QUE FALTABA
        if Config.DISTANCE_METHOD == "depth_only":
            # Solo usar MiDaS
            if depth_value > Config.DEPTH_CLOSE_THRESHOLD:
                result = "very close"
            elif depth_value > Config.DEPTH_MEDIUM_THRESHOLD:
                result = "close"
            else:
                result = "far"
            
            print(f"[DEBUG] Depth method result: {result}")
            return result
        
        elif Config.DISTANCE_METHOD == "area_only":
            # Solo usar área (tu método original)
            return self._estimate_distance(area, frame_width)
        
        elif Config.DISTANCE_METHOD == "hybrid":
            # Combinar ambos métodos
            depth_category = self._depth_to_category(depth_value)
            area_category = self._estimate_distance(area, frame_width)
            
            # MiDaS tiene prioridad para objetos muy cercanos
            if depth_category == "very close":
                return "very close"
            elif depth_category == "close" or area_category == "very close":
                return "close"
            elif area_category == "close":
                return "close"
            else:
                return "far"
        
        else:
            # Fallback por defecto
            return self._estimate_distance(area, frame_width)

    def _depth_to_category(self, depth_value: float) -> str:
        """Helper: convert depth value to distance category"""

        if depth_value > Config.DEPTH_CLOSE_THRESHOLD:
            return "very close"
        elif depth_value > Config.DEPTH_MEDIUM_THRESHOLD:
            return "close"
        else:
            return "far"
        

    # Añadir este método al final de la clase YoloProcessor:
    def _calculate_depth_from_map(self, depth_map: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """Calculate average depth for an object bbox directly from depth map"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            h, w = depth_map.shape
            
            # Clamp coordinates
            x1, x2 = max(0, x1), min(w-1, x2)
            y1, y2 = max(0, y1), min(h-1, y2)
            
            if x2 <= x1 or y2 <= y1:
                return 0.5
            
            # Extract ROI and calculate mean
            roi = depth_map[y1:y2, x1:x2]
            mean_depth = np.mean(roi) / 255.0
            
            return 1.0 - mean_depth  # Invert: 1=close, 0=far
            
        except Exception:
            return 0.5