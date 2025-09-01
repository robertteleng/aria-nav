from dataclasses import dataclass

@dataclass
class OrientationState:
    """Current user orientation and movement state"""
    heading: float          # Compass heading (0° = North)
    pitch: float           # Head tilt up/down
    yaw: float             # Head turn left/right  
    roll: float            # Head tilt sideways
    is_moving: bool        # Whether user is walking
    movement_speed: float  # Relative speed estimate
    last_updated: float    # Timestamp