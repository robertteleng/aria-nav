"""Camera geometry for 3D projections and cross-camera matching validation."""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, Dict
import logging

log = logging.getLogger(__name__)


class CameraGeometry:
    """
    3D geometry utilities for cross-camera tracking using Aria SDK calibrations.

    Uses camera intrinsics (focal length, principal point) and extrinsics
    (rotation, translation) to:
    - Project 2D bboxes + depth to 3D world points
    - Transform 3D points between camera coordinate systems
    - Validate cross-camera handoffs with geometric consistency

    Example usage:
        geometry = CameraGeometry(rgb_calib, slam1_calib, slam2_calib)

        # Project SLAM1 detection to 3D
        point_3d = geometry.bbox_to_3d_point(bbox, depth, "slam1")

        # Check if it's geometrically consistent with RGB detection
        distance = geometry.compute_3d_distance(point_slam1, point_rgb)
        is_valid = distance < 0.5  # meters
    """

    def __init__(
        self,
        rgb_calib: Optional[object] = None,
        slam1_calib: Optional[object] = None,
        slam2_calib: Optional[object] = None,
    ):
        """
        Initialize camera geometry with Aria SDK calibrations.

        Args:
            rgb_calib: RGB camera calibration (from sensors_calib.get_camera_calib("camera-rgb"))
            slam1_calib: SLAM1 left camera calibration
            slam2_calib: SLAM2 right camera calibration
        """
        self.rgb_calib = rgb_calib
        self.slam1_calib = slam1_calib
        self.slam2_calib = slam2_calib

        # Cache calibration data
        self.calibrations = {
            "rgb": rgb_calib,
            "slam1": slam1_calib,
            "slam2": slam2_calib,
        }

        # Extract intrinsics and extrinsics
        self._extract_camera_params()

        log.info(f"[CameraGeometry] Initialized with calibrations: "
                 f"RGB={rgb_calib is not None}, "
                 f"SLAM1={slam1_calib is not None}, "
                 f"SLAM2={slam2_calib is not None}")

    def _extract_camera_params(self) -> None:
        """Extract intrinsics and extrinsics from calibrations."""
        self.intrinsics: Dict[str, Optional[Dict]] = {}
        self.extrinsics: Dict[str, Optional[Dict]] = {}

        for camera_name, calib in self.calibrations.items():
            if calib is None:
                self.intrinsics[camera_name] = None
                self.extrinsics[camera_name] = None
                continue

            try:
                # Extract intrinsics (focal length, principal point, distortion)
                # Aria SDK provides: projection_params, distortion_params
                projection = calib.projection_params()

                # For pinhole/fisheye models, first 2 params are focal lengths (fx, fy)
                # Next 2 are principal point (cx, cy)
                focal_x = projection[0] if len(projection) > 0 else 1.0
                focal_y = projection[1] if len(projection) > 1 else focal_x
                center_x = projection[2] if len(projection) > 2 else 0.0
                center_y = projection[3] if len(projection) > 3 else 0.0

                self.intrinsics[camera_name] = {
                    "focal_x": focal_x,
                    "focal_y": focal_y,
                    "center_x": center_x,
                    "center_y": center_y,
                    "projection_params": projection,
                }

                # Extract extrinsics (T_Device_Camera transform)
                # Aria SDK provides transformation from device to camera
                T_device_camera = calib.get_transform_device_camera()

                # Convert to numpy arrays for easier manipulation
                rotation = np.array(T_device_camera.rotation().to_matrix())
                translation = np.array(T_device_camera.translation())

                self.extrinsics[camera_name] = {
                    "rotation": rotation,  # 3x3 rotation matrix
                    "translation": translation,  # 3x1 translation vector
                    "transform": T_device_camera,
                }

                log.debug(f"[CameraGeometry] {camera_name} intrinsics: "
                         f"fx={focal_x:.2f}, fy={focal_y:.2f}, cx={center_x:.2f}, cy={center_y:.2f}")
                log.debug(f"[CameraGeometry] {camera_name} translation: {translation}")

            except Exception as e:
                log.warning(f"[CameraGeometry] Failed to extract params for {camera_name}: {e}")
                self.intrinsics[camera_name] = None
                self.extrinsics[camera_name] = None

    def bbox_to_3d_point(
        self,
        bbox: Tuple[float, float, float, float],
        depth: float,
        camera_source: str,
    ) -> Optional[np.ndarray]:
        """
        Project 2D bbox center + depth to 3D point in camera coordinates.

        Args:
            bbox: (x, y, w, h) bounding box in pixels
            depth: Depth in meters (from depth estimation)
            camera_source: "rgb", "slam1", or "slam2"

        Returns:
            3D point [X, Y, Z] in camera coordinate system, or None if failed
        """
        intrinsics = self.intrinsics.get(camera_source)
        if intrinsics is None:
            log.debug(f"[CameraGeometry] No intrinsics for {camera_source}")
            return None

        if depth is None or depth <= 0:
            log.debug(f"[CameraGeometry] Invalid depth: {depth}")
            return None

        x, y, w, h = bbox
        # Bbox center in pixels
        u = x + w / 2
        v = y + h / 2

        # Unproject to 3D using pinhole camera model:
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        # Z = depth
        fx = intrinsics["focal_x"]
        fy = intrinsics["focal_y"]
        cx = intrinsics["center_x"]
        cy = intrinsics["center_y"]

        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth

        point_3d = np.array([X, Y, Z])

        log.debug(f"[CameraGeometry] {camera_source} bbox center ({u:.1f}, {v:.1f}) "
                 f"+ depth {depth:.2f}m → 3D point {point_3d}")

        return point_3d

    def transform_point_to_device(
        self,
        point_camera: np.ndarray,
        camera_source: str,
    ) -> Optional[np.ndarray]:
        """
        Transform 3D point from camera coordinates to device coordinates.

        Args:
            point_camera: 3D point [X, Y, Z] in camera coordinate system
            camera_source: "rgb", "slam1", or "slam2"

        Returns:
            3D point in device coordinate system, or None if failed
        """
        extrinsics = self.extrinsics.get(camera_source)
        if extrinsics is None:
            log.debug(f"[CameraGeometry] No extrinsics for {camera_source}")
            return None

        # T_Device_Camera: transforms from camera to device
        # point_device = R * point_camera + t
        R = extrinsics["rotation"]
        t = extrinsics["translation"]

        point_device = R @ point_camera + t

        log.debug(f"[CameraGeometry] Transform {camera_source} camera→device: "
                 f"{point_camera} → {point_device}")

        return point_device

    def transform_point_between_cameras(
        self,
        point_camera1: np.ndarray,
        src_camera: str,
        dst_camera: str,
    ) -> Optional[np.ndarray]:
        """
        Transform 3D point from one camera coordinate system to another.

        Args:
            point_camera1: 3D point in src_camera coordinates
            src_camera: Source camera ("rgb", "slam1", "slam2")
            dst_camera: Destination camera ("rgb", "slam1", "slam2")

        Returns:
            3D point in dst_camera coordinates, or None if failed
        """
        # Transform: camera1 → device → camera2

        # Step 1: camera1 → device
        point_device = self.transform_point_to_device(point_camera1, src_camera)
        if point_device is None:
            return None

        # Step 2: device → camera2 (inverse transform)
        extrinsics2 = self.extrinsics.get(dst_camera)
        if extrinsics2 is None:
            return None

        # T_Camera2_Device = (T_Device_Camera2)^-1
        # point_camera2 = R^T * (point_device - t)
        R2 = extrinsics2["rotation"]
        t2 = extrinsics2["translation"]

        point_camera2 = R2.T @ (point_device - t2)

        log.debug(f"[CameraGeometry] Transform {src_camera}→{dst_camera}: "
                 f"{point_camera1} → {point_camera2}")

        return point_camera2

    def compute_3d_distance(
        self,
        point1: np.ndarray,
        point2: np.ndarray,
    ) -> float:
        """
        Compute Euclidean distance between two 3D points.

        Args:
            point1: 3D point [X, Y, Z]
            point2: 3D point [X, Y, Z]

        Returns:
            Distance in meters
        """
        return np.linalg.norm(point1 - point2)

    def validate_handoff_geometry(
        self,
        bbox1: Tuple[float, float, float, float],
        depth1: float,
        camera1: str,
        bbox2: Tuple[float, float, float, float],
        depth2: float,
        camera2: str,
        max_distance: float = 0.5,
    ) -> bool:
        """
        Validate cross-camera handoff using 3D geometric consistency.

        Args:
            bbox1: Bbox in camera1
            depth1: Depth in camera1 (meters)
            camera1: Source camera ("slam1", "slam2")
            bbox2: Bbox in camera2
            depth2: Depth in camera2 (meters)
            camera2: Destination camera ("rgb")
            max_distance: Maximum allowed 3D distance (meters) for valid match

        Returns:
            True if handoff is geometrically valid, False otherwise
        """
        # Project both bboxes to 3D in their respective camera coordinates
        point1_cam1 = self.bbox_to_3d_point(bbox1, depth1, camera1)
        point2_cam2 = self.bbox_to_3d_point(bbox2, depth2, camera2)

        if point1_cam1 is None or point2_cam2 is None:
            log.debug(f"[CameraGeometry] Cannot validate handoff: 3D projection failed")
            return False

        # Transform point1 to camera2 coordinates
        point1_in_cam2 = self.transform_point_between_cameras(
            point1_cam1, src_camera=camera1, dst_camera=camera2
        )

        if point1_in_cam2 is None:
            log.debug(f"[CameraGeometry] Cannot validate handoff: transform failed")
            return False

        # Compute 3D distance in camera2 coordinate system
        distance = self.compute_3d_distance(point1_in_cam2, point2_cam2)

        is_valid = distance < max_distance

        log.debug(f"[CameraGeometry] Handoff {camera1}→{camera2}: "
                 f"3D distance={distance:.3f}m, max={max_distance:.3f}m, valid={is_valid}")

        return is_valid

    def is_available(self) -> bool:
        """Check if geometry is available (at least 2 cameras calibrated)."""
        calibrated = sum(1 for c in self.calibrations.values() if c is not None)
        return calibrated >= 2


__all__ = ["CameraGeometry"]
