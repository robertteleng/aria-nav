import numpy as np
from ultralytics import YOLO
from typing import List
from src.vision.detected_object import DetectedObject

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
        print("[INFO] ✓ YOLOv11 processor initialized")
    
    def process_frame(self, frame: np.array) -> List[dict]:
        """
        Process frame through YOLO and return structured detections
        
        Returns:
            List of detection dictionaries for audio and visualization
        """
        try:
            # Run YOLO inference
            results = self.model(frame, device=self.device, verbose=False)
            
            # Convert to structured objects
            detected_objects = self._analyze_detections(results, frame.shape[1])
            
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
    
    def _analyze_detections(self, yolo_results, frame_width: int) -> List[DetectedObject]:
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
            
            # Distance estimation
            distance_bucket = self._estimate_distance(area, frame_width)
            
            # Relevance scoring
            base_priority = self.navigation_objects[class_name]['priority']
            size_factor = min(area / (frame_width * 300), 1.0)
            relevance_score = base_priority * float(confidence) * (0.3 + 0.7 * size_factor)
            
            # Filter 3: Minimum relevance threshold
            if relevance_score < 0.6:
                continue
            
            obj = DetectedObject(
                name=self.navigation_objects[class_name]['name'],
                confidence=float(confidence),
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                center_x=center_x,
                center_y=center_y,
                area=area,
                zone=zone,
                distance_bucket=distance_bucket,
                relevance_score=relevance_score
            )
            
            objects.append(obj)
        
        # Return top detection only to reduce audio spam
        objects.sort(key=lambda x: x.relevance_score, reverse=True)
        return objects[:1]
    
    def _classify_zone(self, center_x: float, center_y: float, frame_width: int) -> str:
        """Classify object position into spatial zones"""
        frame_height = int(frame_width * 0.75)  # Assume 4:3 aspect ratio
        
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
