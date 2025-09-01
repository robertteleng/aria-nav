import numpy as np
from typing import Tuple

class OrientationCalculator:
    """
    Advanced orientation calculations using sensor fusion.
    Implements complementary filter for IMU + magnetometer.
    """
    
    def __init__(self, alpha=0.98):
        """
        Args:
            alpha: Complementary filter coefficient (gyro vs accel/magneto)
        """
        self.alpha = alpha
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        self.last_timestamp = None
        
    def update(self, accel_data, gyro_data, mag_data, timestamp: float) -> Tuple[float, float, float]:
        """
        Sensor fusion update using complementary filter
        
        Args:
            accel_data: [ax, ay, az] in m/s²
            gyro_data: [gx, gy, gz] in rad/s
            mag_data: [mx, my, mz] in Tesla
            timestamp: Time in seconds
            
        Returns:
            (heading, pitch, roll) in degrees
        """
        # TODO: Day 4 implementation
        # This will implement proper quaternion-based sensor fusion
        
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            return 0.0, 0.0, 0.0
            
        dt = timestamp - self.last_timestamp
        self.last_timestamp = timestamp
        
        # Placeholder: simple calculations
        # Real implementation needs quaternion integration
        heading = self._simple_compass_heading(mag_data)
        pitch, roll = self._simple_tilt_angles(accel_data)
        
        return heading, pitch, roll
    
    def _simple_compass_heading(self, mag_data) -> float:
        """Simple 2D compass heading calculation"""
        mx, my, mz = mag_data
        heading_rad = np.arctan2(my, mx)
        heading_deg = np.degrees(heading_rad)
        
        # Normalize to 0-360°
        if heading_deg < 0:
            heading_deg += 360
            
        return heading_deg
    
    def _simple_tilt_angles(self, accel_data) -> Tuple[float, float]:
        """Simple tilt calculation from accelerometer"""
        ax, ay, az = accel_data
        
        # Calculate pitch and roll from gravity vector
        pitch = np.degrees(np.arctan2(-ax, np.sqrt(ay**2 + az**2)))
        roll = np.degrees(np.arctan2(ay, az))
        
        return pitch, roll
    
    def get_relative_bearing(self, object_angle: float, user_heading: float) -> float:
        """
        Calculate relative bearing of object from user perspective
        
        Args:
            object_angle: Angle of object in frame (-90° to +90°)
            user_heading: User's compass heading (0-360°)
            
        Returns:
            Relative bearing in degrees
        """
        # TODO: Day 4 implementation
        absolute_bearing = (user_heading + object_angle) % 360
        return absolute_bearing
    
    def bearing_to_direction(self, bearing: float) -> str:
        """Convert bearing to readable direction"""
        directions = [
            "north", "northeast", "east", "southeast",
            "south", "southwest", "west", "northwest"
        ]
        
        # Each direction covers 45° (360° / 8)
        index = round(bearing / 45) % 8
        return directions[index]