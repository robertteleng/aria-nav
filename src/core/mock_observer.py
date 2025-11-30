#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mock observer for Aria SDK testing without physical hardware.

This module provides a drop-in replacement for the real Observer that enables
development and testing without Aria glasses by providing:
1. Synthetic frame generation with simulated objects
2. Video file replay in loop
3. Static images with small variations

The MockObserver is 100% API-compatible with the real Observer for seamless
switching between mock and real hardware modes.

Operating modes:
- 'synthetic': Generates synthetic frames with simulated objects (people, chairs, tables)
- 'video': Replays a recorded video file in loop
- 'static': Returns a static image with small random variations (noise, shift)

Usage:
    # Synthetic mode (default)
    observer = MockObserver(mode='synthetic', fps=60)

    # Video mode
    observer = MockObserver(mode='video', video_path='data/session.mp4')

    # Static mode
    observer = MockObserver(mode='static', image_path='data/frame.jpg')
"""

import numpy as np
import cv2
import threading
import time
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
from pathlib import Path
import logging

log = logging.getLogger("MockObserver")


class MockObserver:
    """
    Mock Observer for Aria development without physical hardware.

    Operating modes:
    - 'synthetic': Generates synthetic frames with simulated objects
    - 'video': Replays a video in loop
    - 'static': Static image with small random variations

    See module docstring for usage examples.
    """

    def __init__(
        self,
        mode: str = 'synthetic',
        fps: int = 60,
        resolution: Tuple[int, int] = (1408, 1408),
        video_path: Optional[str] = None,
        image_path: Optional[str] = None,
        rgb_calib: Optional[Any] = None,
        buffer_size: int = 30,
    ) -> None:
        """
        Initialize the MockObserver.

        Args:
            mode: 'synthetic', 'video', or 'static'
            fps: Frames per second to simulate
            resolution: (width, height) of generated frames
            video_path: Path to video file for 'video' mode
            image_path: Path to image file for 'static' mode
            rgb_calib: RGB calibration (optional, for API compatibility)
            buffer_size: Size of frame buffer
        """
        self.mode = mode
        self.fps = fps
        self.resolution = resolution
        self.video_path = video_path
        self.image_path = image_path
        self.rgb_calib = rgb_calib
        self.buffer_size = buffer_size

        # Circular buffer for frames (like real Observer)
        self.frame_buffer = deque(maxlen=buffer_size)
        self.frame_lock = threading.Lock()

        # State
        self.running = False
        self.frame_count = 0
        self.start_time = None
        self._generator_thread = None

        # Initialize based on mode
        self._init_mode()

        print(f"[MockObserver] Initialized in '{mode}' mode @ {fps} FPS, resolution {resolution}")

    def _init_mode(self) -> None:
        """Initialize resources based on selected mode."""
        if self.mode == 'video':
            if not self.video_path or not Path(self.video_path).exists():
                raise ValueError(f"Video file not found: {self.video_path}")
            self.video_capture = cv2.VideoCapture(self.video_path)
            if not self.video_capture.isOpened():
                raise ValueError(f"Cannot open video: {self.video_path}")
            print(f"[MockObserver] Loaded video: {self.video_path}")
            
        elif self.mode == 'static':
            if not self.image_path or not Path(self.image_path).exists():
                raise ValueError(f"Image file not found: {self.image_path}")
            self.static_image = cv2.imread(self.image_path)
            if self.static_image is None:
                raise ValueError(f"Cannot read image: {self.image_path}")
            # Resize to target resolution
            self.static_image = cv2.resize(self.static_image, self.resolution)
            print(f"[MockObserver] Loaded image: {self.image_path}")
            
        elif self.mode == 'synthetic':
            # Nothing to initialize for synthetic mode
            print(f"[MockObserver] Synthetic frame generation ready")
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def start(self) -> None:
        """Start frame generation (compatible with real Observer API)."""
        if self.running:
            print("[MockObserver] Already running")
            return
        
        self.running = True
        self.start_time = time.time()
        self.frame_count = 0

        # Start frame generation thread
        self._generator_thread = threading.Thread(target=self._generate_frames, daemon=True)
        self._generator_thread.start()

        print(f"[MockObserver] Started frame generation")

    def stop(self) -> None:
        """Stop frame generation."""
        self.running = False
        if self._generator_thread:
            self._generator_thread.join(timeout=2.0)
        
        if self.mode == 'video' and hasattr(self, 'video_capture'):
            self.video_capture.release()
        
        print(f"[MockObserver] Stopped (generated {self.frame_count} frames)")

    def _generate_frames(self) -> None:
        """Thread loop that generates frames based on mode."""
        frame_interval = 1.0 / self.fps

        while self.running:
            loop_start = time.time()

            # Generate frame based on mode
            if self.mode == 'synthetic':
                frame = self._generate_synthetic_frame()
            elif self.mode == 'video':
                frame = self._get_video_frame()
            elif self.mode == 'static':
                frame = self._get_static_frame()
            else:
                frame = None

            if frame is not None:
                # Add to buffer (thread-safe)
                with self.frame_lock:
                    self.frame_buffer.append({
                        'frame': frame,
                        'timestamp': time.time(),
                        'frame_id': self.frame_count
                    })

                self.frame_count += 1

            # Sleep to maintain target FPS
            elapsed = time.time() - loop_start
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _generate_synthetic_frame(self) -> np.ndarray:
        """
        Generate a synthetic frame with simulated objects.
        Simulates an indoor scene with people, chairs, tables, etc.
        """
        # Base: gray background with noise
        frame = np.random.randint(100, 150, (*self.resolution[::-1], 3), dtype=np.uint8)

        # Add some simulated objects
        num_objects = np.random.randint(2, 6)

        for _ in range(num_objects):
            # Random object (simulating people, chairs, etc)
            obj_type = np.random.choice(['person', 'chair', 'table', 'bottle'])

            # Random position and size
            x = np.random.randint(100, self.resolution[0] - 200)
            y = np.random.randint(100, self.resolution[1] - 200)
            w = np.random.randint(80, 300)
            h = np.random.randint(100, 400)

            # Color based on type
            if obj_type == 'person':
                color = (180, 150, 120)  # Skin tone
            elif obj_type == 'chair':
                color = (139, 69, 19)    # Brown
            elif obj_type == 'table':
                color = (160, 82, 45)    # Light brown
            else:
                color = (0, 150, 200)    # Blue (bottle)

            # Draw rectangle with simple gradient
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)

            # Darker border
            darker = tuple(int(c * 0.7) for c in color)
            cv2.rectangle(frame, (x, y), (x + w, y + h), darker, 3)

        # Add timestamp as text (useful for debugging)
        timestamp = f"Frame: {self.frame_count} | {time.time():.2f}s"
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (255, 255, 255), 2)

        # Add mode indicator
        cv2.putText(frame, "MOCK MODE: SYNTHETIC", (10, self.resolution[1] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame

    def _get_video_frame(self) -> Optional[np.ndarray]:
        """Read next frame from video (infinite loop)."""
        ret, frame = self.video_capture.read()

        if not ret:
            # Restart video at end
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.video_capture.read()
            print("[MockObserver] Video loop restarted")

        if ret and frame is not None:
            # Resize to expected size
            frame = cv2.resize(frame, self.resolution)

            # Add indicator
            cv2.putText(frame, f"MOCK MODE: VIDEO | Frame {self.frame_count}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return frame

        return None

    def _get_static_frame(self) -> np.ndarray:
        """
        Return static image with small variations.
        Useful for testing stability and consistency.
        """
        if self.static_image is None:
            # Fallback to black frame if no image
            frame = np.zeros((*self.resolution[::-1], 3), dtype=np.uint8)
        else:
            frame = self.static_image.copy()

        # Add small random noise (simulates natural vibration)
        noise = np.random.randint(-5, 5, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Small random shift (simulates minimal movement)
        shift_x = int(np.random.randint(-2, 3))
        shift_y = int(np.random.randint(-2, 3))
        M = np.array([[1.0, 0.0, float(shift_x)], [0.0, 1.0, float(shift_y)]], dtype=np.float32)
        frame = cv2.warpAffine(frame, M, self.resolution, flags=cv2.INTER_LINEAR)

        # Indicator
        cv2.putText(frame, f"MOCK MODE: STATIC | Frame {self.frame_count}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return frame
    
    def get_latest_frame(self, camera: str = 'rgb') -> Optional[np.ndarray]:
        """
        Get the most recent frame from buffer.
        Compatible with real Observer API.

        Args:
            camera: 'rgb', 'slam1', or 'slam2' (only 'rgb' implemented in mock)
        """
        # Mock only supports RGB, ignores slam1/slam2
        if camera not in ['rgb', 'slam1', 'slam2']:
            return None

        with self.frame_lock:
            if not self.frame_buffer:
                return None
            return self.frame_buffer[-1]['frame'].copy()

    def get_frame_data(self) -> Optional[Dict[str, Any]]:
        """
        Get complete data for most recent frame.
        Compatible with real Observer API.
        """
        with self.frame_lock:
            if not self.frame_buffer:
                return None
            return self.frame_buffer[-1].copy()

    def get_buffer_size(self) -> int:
        """Return number of frames in buffer."""
        with self.frame_lock:
            return len(self.frame_buffer)

    def get_motion_state(self) -> Dict[str, Any]:
        """
        Return simulated motion state.
        Compatible with real Observer API.
        """
        # Mock simulates stationary state
        return {
            'state': 'stationary',
            'magnitude': 9.8,  # Standard gravity
            'timestamp': time.time(),
            'history_length': 0
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return operation statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        actual_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        return {
            'mode': self.mode,
            'frames_generated': self.frame_count,
            'elapsed_time': elapsed,
            'target_fps': self.fps,
            'actual_fps': actual_fps,
            'buffer_size': self.get_buffer_size(),
            'running': self.running
        }
    
    def __enter__(self):
        """Context manager support."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.stop()
        return False
