import numpy as np
import time
from collections import deque
from typing import Optional
from projectaria_tools.core.sensor_data import MotionData

from motion.orientation_state import OrientationState

class MotionProcessor:
    """Process IMU and magnetometer data for spatial orientation."""
    
    def __init__(self, history_size=20):
        # Current orientation state
        self.orientation = OrientationState(
            heading=0.0, pitch=0.0, yaw=0.0, roll=0.0,
            is_moving=False, movement_speed=0.0,
            last_updated=time.time()
        )
        
        # Data buffers for smoothing
        self.accel_history = deque(maxlen=history_size)
        self.gyro_history = deque(maxlen=history_size)
        self.magneto_history = deque(maxlen=history_size)
        
        # Calibration and filtering
        self.magneto_calibration = None
        self.gravity_filter = np.array([0.0, 0.0, -9.81])  # Initial gravity estimate
        self.alpha = 0.8  # Low-pass filter coefficient
        
        # Movement detection
        self.movement_threshold = 1.5  # m/s² for walking detection
        self.stationary_time = 0.0
        
        print("[INFO] IMU processor initialized (ready for Day 4)")
    
    def update_motion(self, samples, imu_idx: int) -> None:
        """
        Process IMU samples (accelerometer + gyroscope)
        
        Args:
            samples: List of MotionData from Aria SDK
            imu_idx: IMU index (0 or 1 for dual IMUs)
        """
        # TODO: Day 4 implementation
        for sample in samples:
            timestamp = sample.capture_timestamp_ns * 1e-9
            
            # Store raw data
            self.accel_history.append((timestamp, sample.accel_msec2))
            self.gyro_history.append((timestamp, sample.gyro_radsec))
            
            # Process accelerometer for movement detection
            self._detect_movement(sample.accel_msec2, timestamp)
            
            # Process gyroscope for head orientation  
            self._update_head_orientation(sample.gyro_radsec, timestamp)
    
    def update_heading(self, sample: MotionData) -> None:
        """
        Process magnetometer sample for compass heading
        
        Args:
            sample: MotionData with magnetometer reading
        """
        # TODO: Day 4 implementation
        timestamp = sample.capture_timestamp_ns * 1e-9
        
        # Store magnetometer data
        self.magneto_history.append((timestamp, sample.mag_tesla))

        # Calculate compass heading
        heading = self._calculate_compass_heading(sample.mag_tesla)
        if heading is not None:
            self.orientation.heading = heading
            self.orientation.last_updated = timestamp
    
    def _detect_movement(self, accel_data, timestamp: float) -> None:
        """Detect if user is walking based on accelerometer patterns"""
        # TODO: Day 4 implementation
        # Basic movement detection using acceleration magnitude
        accel_magnitude = np.linalg.norm(accel_data)
        
        # Simple threshold-based detection (placeholder)
        if accel_magnitude > self.movement_threshold:
            self.orientation.is_moving = True
            self.stationary_time = 0.0
        else:
            self.stationary_time += 0.033  # Assume ~30fps IMU
            if self.stationary_time > 1.0:  # Stationary for 1+ seconds
                self.orientation.is_moving = False
    
    def _update_head_orientation(self, gyro_data, timestamp: float) -> None:
        """Update head orientation from gyroscope"""
        # TODO: Day 4 implementation
        # Integrate gyroscope for pitch/yaw/roll
        dt = 0.033  # Approximate time step
        
        # Simple integration (placeholder - needs proper quaternion math)
        self.orientation.pitch += gyro_data[0] * dt
        self.orientation.yaw += gyro_data[1] * dt  
        self.orientation.roll += gyro_data[2] * dt
    
    def _calculate_compass_heading(self, mag_data) -> Optional[float]:
        """
        Calculate compass heading from magnetometer reading
        
        Returns:
            Heading in degrees (0° = North, 90° = East)
        """
        # TODO: Day 4 implementation
        try:
            # Basic 2D compass calculation (ignores tilt compensation)
            mag_x, mag_y, mag_z = mag_data
            
            # Calculate heading from X,Y components
            heading_rad = np.arctan2(mag_y, mag_x)
            heading_deg = np.degrees(heading_rad)
            
            # Normalize to 0-360°
            if heading_deg < 0:
                heading_deg += 360
                
            return heading_deg
            
        except Exception as e:
            print(f"[WARN] Compass calculation failed: {e}")
            return None
    
    def get_relative_direction(self, object_center_x: float, frame_width: int) -> str:
        """
        Convert frame coordinates to user-relative directions
        
        Args:
            object_center_x: X coordinate of object in frame
            frame_width: Width of frame for normalization
            
        Returns:
            Relative direction string
        """
        # TODO: Day 4 implementation with actual heading
        # For now, return simple frame-relative directions
        
        left_boundary = frame_width * 0.33
        right_boundary = frame_width * 0.67
        
        if object_center_x < left_boundary:
            return "to your left"
        elif object_center_x > right_boundary:
            return "to your right"  
        else:
            return "directly ahead"
    
    def get_movement_context(self) -> str:
        """Get current movement context for audio commands"""
        if self.orientation.is_moving:
            return "while walking"
        else:
            return "while stationary"
    
    def get_compass_direction(self, relative_direction: str) -> str:
        """
        Convert relative direction to absolute compass direction
        
        Args:
            relative_direction: "left", "right", "ahead"
            
        Returns:
            Compass direction like "north", "southeast"
        """
        # TODO: Day 4 implementation
        # This will use self.orientation.heading to convert
        # relative directions to absolute compass bearings
        
        # Placeholder: return relative for now
        return relative_direction
    
    def calibrate_magnetometer(self, duration_seconds=10):
        """
        Perform magnetometer calibration by collecting data
        while user rotates in place
        """
        # TODO: Day 4 implementation
        print(f"[INFO] Magnetometer calibration needed ({duration_seconds}s)")
        print("[INFO] Slowly turn in a full circle to calibrate compass")
        # Collect min/max values for each axis during rotation
        # Calculate offset and scale factors
        pass
    
    def reset_orientation(self):
        """Reset orientation to initial state"""
        self.orientation.heading = 0.0
        self.orientation.pitch = 0.0
        self.orientation.yaw = 0.0
        self.orientation.roll = 0.0
        self.orientation.is_moving = False
        self.orientation.last_updated = time.time()
        
        # Clear history buffers
        self.accel_history.clear()
        self.gyro_history.clear() 
        self.magneto_history.clear()
        
        print("[INFO] Orientation reset to defaults")


