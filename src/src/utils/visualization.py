#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization utilities for navigation overlay rendering
"""

import cv2
import numpy as np
from typing import List


class FrameRenderer:
    """Handle all frame annotation and overlay rendering"""
    
    def __init__(self):
        self.quadrant_colors = {
            'top_left': (0, 255, 255),     # Yellow
            'top_right': (0, 165, 255),    # Orange
            'bottom_left': (0, 0, 255),    # Red  
            'bottom_right': (255, 0, 255)  # Magenta
        }
        
        self.zone_titles = {
            'top_left': 'Upper Left',
            'top_right': 'Upper Right', 
            'bottom_left': 'Lower Left',
            'bottom_right': 'Lower Right',
            'center': 'Center'
        }
    
    def draw_navigation_overlay(self, frame: np.array, detections: List[dict], 
                              audio_system) -> np.array:
        """Draw complete navigation overlay on frame"""
        height, width = frame.shape[:2]
        annotated_frame = frame.copy()
        
        # Draw zone grid
        self._draw_zone_grid(annotated_frame, width, height)
        
        # Draw detections
        self._draw_detections(annotated_frame, detections)
        
        # Draw system status
        self._draw_system_status(annotated_frame, audio_system, width, height)
        
        return annotated_frame
    
    def _draw_zone_grid(self, frame: np.array, width: int, height: int):
        """Draw 4-quadrant grid overlay"""
        mid_x, mid_y = width // 2, height // 2
        # Cross lines
        cv2.line(frame, (mid_x, 0), (mid_x, height - 1), (255, 255, 255), 1)
        cv2.line(frame, (0, mid_y), (width - 1, mid_y), (255, 255, 255), 1)

        # Labels (Title Case)
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
                # Skip malformed detection entries
                continue

    def _draw_system_status(self, frame: np.array, audio_system, width: int, height: int):
        """Draw audio system status at the bottom-left corner"""
        try:
            queue_len = len(audio_system.audio_queue) if hasattr(audio_system, 'audio_queue') else 0
            status_text = "🔊 SPEAKING" if audio_system.is_speaking else f"Queue: {queue_len}"
            cv2.putText(frame, status_text, (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception:
            pass
        # Draw grid lines
        cv2.line(frame, (mid_x, 0), (mid_x, height - 1), (255, 255, 255), 1)
        cv2.line(frame, (0, mid_y), (width - 1, mid_y), (255, 255, 255), 1)
        
        # Zone labels
        cv2.putText(frame, "Upper Left", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Upper Right", (mid_x + 10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Lower Left", (10, mid_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Lower Right", (mid_x + 10, mid_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def _draw_detections(self, frame: np.array, detections: List[dict]):
        """Draw bounding boxes and labels for detected objects"""
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            zone = detection.get('zone', 'center')
            color = self.quadrant_colors.get(zone, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            zone_title = self.zone_titles.get(zone, zone.replace('_', ' ').title())
            distance = detection.get('distance', 'unknown')
            label = f"{detection['name']} ({zone_title}) - {distance}"
            
            # Calculate label position
            label_y = max(20, y1 - 8)
            cv2.putText(frame, label, (x1, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    def _draw_system_status(self, frame: np.array, audio_system, width: int, height: int):
        """Draw audio system status at bottom of frame"""
        try:
            queue_len = len(audio_system.audio_queue) if hasattr(audio_system, 'audio_queue') else 0
            status_text = "SPEAKING" if audio_system.is_speaking else f"Queue: {queue_len}"
            
            cv2.putText(frame, f"Audio: {status_text}", (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception as e:
            # Fallback if audio_system has issues
            cv2.putText(frame, "Audio: Unknown", (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
