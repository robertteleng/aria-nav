"""
Visualization utilities for navigation overlay rendering
"""

import cv2
import numpy as np
from typing import List
from utils.config import Config


class FrameRenderer:
    """Handle all frame annotation and overlay rendering"""
    
    def __init__(self):
        self.quadrant_colors = {
            'center': (255, 0, 0),             # Azul
            'top_left': (0, 255, 255),         # Yellow
            'top_right': (0, 165, 255),        # Orange
            'bottom_left': (0, 0, 255),        # Red  
            'bottom_right': (255, 0, 255)      # Magenta
        }
    
        self.zone_titles = {
            'center': 'Center',                
            'top_left': 'Upper Left',
            'top_right': 'Upper Right', 
            'bottom_left': 'Lower Left',
            'bottom_right': 'Lower Right'
        }
    
    def draw_navigation_overlay(self, frame: np.array, detections: List[dict], 
                              audio_system, depth_map: np.array = None) -> np.array:
        """Draw complete navigation overlay on frame"""
        height, width = frame.shape[:2]
        annotated_frame = frame.copy()
        
        # Draw zone grid
        self._draw_zone_grid(annotated_frame, width, height)
        
        # Draw detections
        self._draw_detections(annotated_frame, detections)
        
        # Draw system status
        self._draw_system_status(annotated_frame, audio_system, width, height)

        # Draw depth overlay if available
        if depth_map is not None:
            self._draw_depth_overlay(annotated_frame, depth_map, width, height)
        
        return annotated_frame
        
    def _draw_zone_grid(self, frame: np.array, width: int, height: int):
        """Draw zone grid overlay"""
        mid_x, mid_y = width // 2, height // 2
        
        # SIEMPRE dibujar líneas principales
        cv2.line(frame, (mid_x, 0), (mid_x, height - 1), (255, 255, 255), 1)
        cv2.line(frame, (0, mid_y), (width - 1, mid_y), (255, 255, 255), 1)

        if Config.ZONE_SYSTEM == "five_zones":
            # Zona central
            center_margin_x = width * Config.CENTER_ZONE_WIDTH_RATIO
            center_margin_y = height * Config.CENTER_ZONE_HEIGHT_RATIO
            
            center_left = int(width/2 - center_margin_x/2)
            center_right = int(width/2 + center_margin_x/2)
            center_top = int(height/2 - center_margin_y/2)
            center_bottom = int(height/2 + center_margin_y/2)
            
            # Rectángulo zona central
            cv2.rectangle(frame, (center_left, center_top), (center_right, center_bottom), 
                         (255, 0, 0), 2)
            
            # Labels de zonas
            cv2.putText(frame, "Upper Left", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Upper Right", (mid_x + 10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Lower Left", (10, mid_y + 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Lower Right", (mid_x + 10, mid_y + 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "CENTER", (center_left + 10, center_top + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            # Labels 4 cuadrantes
            cv2.putText(frame, "Top Left", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Top Right", (mid_x + 10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Bottom Left", (10, mid_y + 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Bottom Right", (mid_x + 10, mid_y + 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_detections(self, frame: np.array, detections: List[dict]):
        """Draw detection boxes and labels"""
        for det in detections:
            try:
                x1, y1, x2, y2 = det['bbox']
                name = det.get('name', 'object')
                zone = det.get('zone', 'center')
                distance = det.get('distance', '')
                color = self.quadrant_colors.get(zone, (255, 255, 255))

                # Bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                # Label
                zone_text = self.zone_titles.get(zone, zone.replace('_', ' ').title())
                label = f"{name} ({zone_text})"
                if distance:
                    label += f" - {distance}"
                cv2.putText(frame, label, (int(x1), max(20, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            except Exception:
                continue

    def _draw_system_status(self, frame: np.array, audio_system, width: int, height: int):
        """Draw audio system status at bottom of frame"""
        try:
            queue_len = len(audio_system.audio_queue) if hasattr(audio_system, 'audio_queue') else 0
            status_text = "SPEAKING" if audio_system.is_speaking else f"Queue: {queue_len}"
            cv2.putText(frame, f"Audio: {status_text}", (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception:
            cv2.putText(frame, "Audio: Unknown", (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
    def _draw_depth_overlay(self, frame: np.array, depth_map: np.array, width: int, height: int):
        """Draw depth information overlay in top-right corner"""
        try:
            depth_small = cv2.resize(depth_map, (120, 90))
            depth_colored = cv2.applyColorMap(depth_small, cv2.COLORMAP_JET)
            
            start_x = width - 130
            start_y = 10
            end_x = width - 10
            end_y = 100
            
            if start_x > 0 and start_y >= 0 and end_x <= width and end_y <= height:
                frame[start_y:end_y, start_x:end_x] = depth_colored
                cv2.putText(frame, "Depth", (start_x, start_y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        except Exception:
            pass