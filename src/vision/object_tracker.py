from collections import deque
from typing import List
import time

class ObjectTracker:
    """Track objects over time for consistency and movement detection"""
    
    def __init__(self, history_size=8):
        self.detection_history = deque(maxlen=history_size)
        self.object_paths = {}
        self.last_update = time.time()
    
    def update(self, detections: List[dict]) -> List[dict]:
        """
        Update tracking with new detections, return stable objects
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            List of confirmed/stable detections
        """
        current_time = time.time()
        self.detection_history.append((current_time, detections))
        
        # Implement consistency checking
        stable_detections = self._filter_consistent_detections()
        
        self.last_update = current_time
        return stable_detections
    
    def _filter_consistent_detections(self) -> List[dict]:
        """Filter detections that appear consistently over multiple frames"""
        if len(self.detection_history) < 3:
            return []
        
        # Get recent detections
        recent_frames = list(self.detection_history)[-5:]
        if not recent_frames:
            return []
        
        # Find objects that appear in multiple recent frames
        consistent_objects = []
        latest_detections = recent_frames[-1][1]
        
        for detection in latest_detections:
            consistency_count = 0
            object_key = f"{detection['name']}_{detection['zone']}"
            
            # Check appearance in recent frames
            for timestamp, frame_detections in recent_frames:
                for frame_det in frame_detections:
                    frame_key = f"{frame_det['name']}_{frame_det['zone']}"
                    if frame_key == object_key:
                        consistency_count += 1
                        break
            
            # Require appearance in at least 2 of last 3-5 frames
            required_consistency = min(2, len(recent_frames) - 1)
            if consistency_count >= required_consistency:
                consistent_objects.append(detection)
        
        return consistent_objects