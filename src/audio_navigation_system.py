#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio navigation system for blind users with Meta Aria glasses
Day 2: Directional audio commands (refined)

Date: 2025-08-31
Version: 2.1 - Reduced audio spam and simplified prompts
Changes: longer cooldown, higher thresholds, simpler messages
"""

import signal
import cv2
import numpy as np
import torch
import aria.sdk as aria
from ultralytics import YOLO
import platform
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)


@dataclass
class DetectedObject:
    """A detected object with spatial info"""
    name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center_x: float
    center_y: float
    area: float
    direction: str
    relevance_score: float


class AudioNavigationSystem:
    """Directional audio command system optimized to avoid spam"""
    
    def __init__(self):
        # Configure TTS (macOS `say`)
        self._setup_tts()
        
        # Stricter audio control
        self.audio_queue = deque(maxlen=3)  # reduced from 5 to 3
        self.last_announcement_time = time.time()
        self.announcement_cooldown = 3.0  # lowered from 5s to 3s
        
        # Only most relevant navigation objects (reduced set)
        self.navigation_objects = {
            'person': {'priority': 1.0, 'name': 'person'},
            'car': {'priority': 0.9, 'name': 'car'},
            'bicycle': {'priority': 0.8, 'name': 'bicycle'}, 
            'bus': {'priority': 0.9, 'name': 'bus'},
            'truck': {'priority': 0.8, 'name': 'truck'},
            # removed less critical classes to reduce spam
        }
        
        # Larger history for stability
        self.detection_history = deque(maxlen=8)  # increased from 5 to 8
        
        # TTS state control
        self.tts_speaking = False
        self._say_warned = False
        self._last_debug_ts = 0.0
        self._debug_interval = 2.0  # seconds

        # Frame dimensions (updated by the observer)
        self.frame_width = None
        self.frame_height = None
        
        print("[INFO] ✓ Audio navigation system initialized (anti-spam)")

        # Persistent `say` process optional (disabled: unreliable with stdin for `say`)
        self.use_persistent_say = False
        self.say_proc = None
        self._say_lock = threading.Lock()
        if self.say_available and self.use_persistent_say:
            self._start_say_proc()
        print(f"[INFO] say_available={self.say_available}, voice={self.selected_voice}, rate={self.tts_rate}, persistent={self.use_persistent_say}")

    @property
    def is_speaking(self) -> bool:
        """Observer compatibility: current speaking state"""
        return self.tts_speaking

    def update_frame_dimensions(self, width: int, height: int) -> None:
        """Store last frame dimensions for zone analysis"""
        self.frame_width = int(width)
        self.frame_height = int(height)
    
    def _setup_tts(self):
        """Configure TTS using macOS `say` command"""
        self.tts_rate = 190  # faster WPM for `say` (-r)
        self.say_available = platform.system() == 'Darwin' and shutil.which('say') is not None
        # Preferred English voices available in macOS
        self.voice_preferences = [
            'Samantha', # en-US
            'Alex',     # en-US
            'Victoria', # en-US
            'Daniel'    # en-GB
        ]
        self.selected_voice = None

        if self.say_available:
            # Try to select the first available preferred voice
            for v in self.voice_preferences:
                if self._test_say_voice(v):
                    self.selected_voice = v
                    print(f"[INFO] Selected macOS voice: {v}")
                    break
            if not self.selected_voice:
                print("[INFO] Using default macOS `say` voice")
        else:
            print("[WARN] macOS `say` not available. Audio disabled.")

    def _start_say_proc(self) -> None:
        """Start a persistent `say` process with stdin for low-latency speech"""
        try:
            cmd = ["say", "-r", str(self.tts_rate)]
            if self.selected_voice:
                cmd.extend(["-v", self.selected_voice])
            # Start process with a pipe; universal_newlines ensures text mode
            self.say_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            print("[INFO] Persistent `say` process started")
        except Exception as e:
            self.say_proc = None
            print(f"[WARN] Failed to start persistent `say`: {e}")

    def _stop_say_proc(self) -> None:
        """Stop the persistent `say` process if running"""
        try:
            if self.say_proc and self.say_proc.poll() is None:
                # Send EOF to let `say` finish gracefully
                try:
                    self.say_proc.stdin.close()
                except Exception:
                    pass
                self.say_proc.terminate()
        except Exception:
            pass

    def _test_say_voice(self, voice_name: str) -> bool:
        """Quick check whether a voice exists in `say`"""
        try:
            # Run a minimal `say` and ignore output/errors
            completed = subprocess.run(
                ["say", "-v", voice_name, "-r", str(self.tts_rate), ""],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2
            )
            return completed.returncode == 0
        except Exception:
            return False
    
    def speak_async(self, message: str):
        """Speak a message using macOS `say` with cooldown control"""
        def _speak():
            try:
                if not self.tts_speaking and self.say_available:
                    self.tts_speaking = True
                    print(f"[AUDIO] 🔊 {message}")
                    # Add to visual queue while speaking
                    self.audio_queue.append(message)

                    # Estimate duration based on words and WPM
                    words = max(1, len(message.split()))
                    est_duration = (words / max(100, self.tts_rate)) * 60.0 + 0.15

                    # One-shot `say` (non-blocking): most reliable behavior
                    run_cmd = ["say", "-r", str(self.tts_rate)]
                    if self.selected_voice:
                        run_cmd.extend(["-v", self.selected_voice])
                    run_cmd.append(message)
                    subprocess.Popen(run_cmd)

                    # Keep speaking flag for estimated duration then clear
                    time.sleep(est_duration)
            except Exception as e:
                print(f"[WARN] TTS error (say): {e}")
            finally:
                # Remove from queue and release speaking state
                if len(self.audio_queue) > 0 and self.audio_queue[-1] == message:
                    try:
                        self.audio_queue.pop()
                    except Exception:
                        pass
                self.tts_speaking = False

        # Stricter cooldown control
        current_time = time.time()
        if (current_time - self.last_announcement_time >= self.announcement_cooldown 
            and not self.tts_speaking and self.say_available):
            self.last_announcement_time = current_time
            threading.Thread(target=_speak, daemon=True).start()
        else:
            # Rate-limited debug messages with clear reason
            now = time.time()
            if not self.say_available:
                if not self._say_warned:
                    print("[WARN] `say` not available on this system (audio disabled)")
                    self._say_warned = True
            elif now - self._last_debug_ts >= self._debug_interval:
                remaining = max(0.0, self.announcement_cooldown - (current_time - self.last_announcement_time))
                print(f"[DEBUG] Audio blocked by cooldown ({remaining:.1f}s)")
                self._last_debug_ts = now

    def speak_force(self, message: str):
        """Speak immediately, bypassing cooldown and speaking flag (for tests)"""
        if not self.say_available:
            print("[WARN] `say` not available to speak_force")
            return
        print(f"[AUDIO][FORCE] 🔊 {message}")
        # One-shot `say` to guarantee output
        try:
            run_cmd = ["say", "-r", str(self.tts_rate)]
            if self.selected_voice:
                run_cmd.extend(["-v", self.selected_voice])
            run_cmd.append(message)
            subprocess.Popen(run_cmd)
        except Exception as e:
            print(f"[WARN] speak_force failed: {e}")

    def close(self) -> None:
        """Public cleanup for TTS resources"""
        try:
            self._stop_say_proc()
        except Exception:
            pass
    
    def analyze_detections(self, yolo_results, frame_width: int) -> List[DetectedObject]:
        """More selective analysis of detections"""
        objects = []
        
        # Direction boundaries
        left_boundary = frame_width * 0.33
        right_boundary = frame_width * 0.67
        
        for detection in yolo_results[0].boxes.data:
            x1, y1, x2, y2, confidence, class_id = detection.cpu().numpy()
            
            # Filter 1: only known classes
            class_name = yolo_results[0].names[int(class_id)]
            if class_name not in self.navigation_objects:
                continue
            
            # Filter 2: confidence threshold (lower for earlier detection)
            if confidence < 0.6:  # lowered from 0.7 to 0.6
                continue
                
            # Compute properties
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)
            
            # Determine direction
            if center_x < left_boundary:
                direction = "left"
            elif center_x > right_boundary:
                direction = "right"
            else:
                direction = "center"
            
            # Stricter relevance score
            base_priority = self.navigation_objects[class_name]['priority']
            size_factor = min(area / (frame_width * 300), 1.0)  # require bigger objects
            relevance_score = base_priority * float(confidence) * (0.3 + 0.7 * size_factor)
            
            # Filter 3: higher minimal relevance
            if relevance_score < 0.6:  # increased from 0.4 to 0.6
                continue
            
            obj = DetectedObject(
                name=self.navigation_objects[class_name]['name'],
                confidence=float(confidence),
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                center_x=center_x,
                center_y=center_y,
                area=area,
                direction=direction,
                relevance_score=relevance_score
            )
            
            objects.append(obj)
        
        # Keep top-1 most relevant object
        objects.sort(key=lambda x: x.relevance_score, reverse=True)
        return objects[:1]  # at most one per frame

    def _zone_from_center(self, cx: float, cy: float) -> str:
        """Determine quadrant name in English using current dimensions"""
        if not self.frame_width or not self.frame_height:
            # Fallback: unknown dims
            return "center"
        midx, midy = self.frame_width / 2.0, self.frame_height / 2.0
        if cy < midy and cx < midx:
            return "top_left"
        if cy < midy and cx >= midx:
            return "top_right"
        if cy >= midy and cx < midx:
            return "bottom_left"
        return "bottom_right"

    def _distance_bucket(self, area: float) -> str:
        """Simple distance heuristic based on bbox area"""
        if not self.frame_width or not self.frame_height:
            return ""
        frame_area = float(self.frame_width * self.frame_height)
        ratio = area / max(frame_area, 1.0)
        if ratio >= 0.10:
            return "very close"
        if ratio >= 0.04:
            return "close"
        if ratio >= 0.015:
            return "medium"
        return "far"

    def process_detections(self, yolo_results):
        """
        Integrate analysis and audio control, return overlay-friendly dicts:
        [{'bbox':(x1,y1,x2,y2), 'spanish_name':..., 'zone':..., 'distance':...}, ...]
        """
        try:
            fw = self.frame_width if self.frame_width else 0
            objects = self.analyze_detections(yolo_results, fw)

            # Announce using history and cooldown
            if self.update_detections(objects):
                cmd = self.generate_audio_command(objects)
                if cmd:
                    self.speak_async(cmd)

            # Format output for drawing overlay
            detections = []
            for obj in objects:
                zone = self._zone_from_center(obj.center_x, obj.center_y)
                distance = self._distance_bucket(obj.area)
                detections.append({
                    'bbox': obj.bbox,
                    'name': obj.name,
                    'zone': zone,
                    'distance': distance
                })
            return detections
        except Exception as e:
            print(f"[WARN] process_detections failed: {e}")
            return []
    
    def _speakable_zone(self, zone: str) -> str:
        """Convert internal zone key to a natural English phrase"""
        mapping = {
            'top_left': 'top left',
            'top_right': 'top right',
            'bottom_left': 'bottom left',
            'bottom_right': 'bottom right',
            'center': 'center',
        }
        return mapping.get(zone, zone.replace('_', ' '))

    def generate_audio_command(self, objects: List[DetectedObject]) -> Optional[str]:
        """Generate a simple English command like 'person upper right'"""
        if not objects:
            return None
        
        # Single object -> simple phrase in English
        obj = objects[0]
        zone = self._zone_from_center(obj.center_x, obj.center_y)
        return f"{obj.name} {self._speakable_zone(zone)}"  # e.g., "person upper right"
    
    def update_detections(self, objects: List[DetectedObject]) -> bool:
        """Controlled announcements with short warm-up and consistency requirement"""
        self.detection_history.append(objects)
        
        # Use a sliding window of up to 5 frames; allow early decisions with >=3 frames
        window = min(5, len(self.detection_history))
        if window < 3:
            return False
        
        if not objects:
            return False
            
        obj = objects[0]
        key = f"{obj.name}_{obj.direction}"
        
        # Count appearances in the last `window` frames
        recent_frames = list(self.detection_history)[-window:]
        recent_count = 0
        for frame_objects in recent_frames:
            for frame_obj in frame_objects:
                if f"{frame_obj.name}_{frame_obj.direction}" == key:
                    recent_count += 1
                    break
        
        # Required consistency: 2/3, 3/4, or 3/5
        required = 2 if window == 3 else 3
        return recent_count >= required
