#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Navigation Coordinator - TFM Navigation System

Orchestrates data flow between vision processing, navigation decisions, and audio output.
This enhanced coordinator supports motion detection and hybrid architecture with both
RGB and peripheral SLAM cameras for 360° awareness.

Features:
- Multi-camera processing (RGB + 2 SLAM peripheral cameras)
- Motion-aware navigation decisions (stationary vs. walking)
- Cross-camera object tracking with 3D geometric validation
- Audio routing with priority-based queueing
- Real-time performance profiling
- Dashboard integration for telemetry

Pipeline Flow:
    Image → Enhancement → YOLO Detection → Depth Estimation →
    Navigation Decision → Audio Routing → Frame Rendering

Version: 1.1 - Enhanced with Motion Support + Hybrid Architecture Ready
Date: September 2025
"""

import time
from typing import Optional, Dict, List, Any

from utils.config import Config
from utils.config_sections import ProfilingConfig, load_profiling_config, TrackerConfig, load_tracker_config

from core.audio.message_formatter import MessageFormatter
from core.navigation.navigation_decision_engine import (
    NavigationDecisionEngine,
)
from core.navigation.rgb_audio_router import RgbAudioRouter
from core.navigation.navigation_pipeline import NavigationPipeline
from core.navigation.slam_audio_router import SlamAudioRouter, SlamRoutingState

try:
    from core.vision.slam_detection_worker import (
        CameraSource,
        SlamDetectionWorker,
    )
    from core.audio.navigation_audio_router import (
        NavigationAudioRouter,
        EventPriority,
    )
except Exception:
    from enum import Enum

    CameraSource = Any  # type: ignore[assignment]
    SlamDetectionWorker = Any  # type: ignore[assignment]
    NavigationAudioRouter = Any  # type: ignore[assignment]

    class _FallbackPriority(Enum):
        CRITICAL = 1
        HIGH = 2
        MEDIUM = 3
        LOW = 4

    EventPriority = _FallbackPriority  # type: ignore[assignment]


class Coordinator:
    """
    Orchestrates data flow between vision, navigation, and audio modules.

    This coordinator receives pre-configured dependencies via dependency injection
    and focuses solely on coordinating the processing pipeline. It does not create
    or initialize subsystems.

    Processing Pipeline:
        Image → Enhancement → YOLO → Navigation → Audio → Rendering

    Attributes:
        pipeline: NavigationPipeline for vision processing
        decision_engine: NavigationDecisionEngine for navigation logic
        audio_system: Audio output system for TTS and spatial beeps
        frame_renderer: Optional frame visualization
        dashboard: Optional telemetry dashboard
        audio_router: Optional priority-based audio queue router
        slam_router: SLAM camera event router
        rgb_router: RGB camera event router
        camera_geometry: Optional 3D geometric validation for cross-camera tracking
    """
    
    def __init__(
        self,
        yolo_processor,
        audio_system,
        frame_renderer=None,
        image_enhancer=None,
        dashboard=None,
        audio_router: Optional[Any] = None,
        navigation_pipeline: Optional[NavigationPipeline] = None,
        decision_engine: Optional[NavigationDecisionEngine] = None,
        telemetry=None,
    ):
        """
        Initialize coordinator with injected dependencies.

        Args:
            yolo_processor: Pre-configured YoloProcessor instance
            audio_system: Pre-configured AudioSystem instance
            frame_renderer: Optional FrameRenderer instance for visualization
            image_enhancer: Optional ImageEnhancer instance for low-light enhancement
            dashboard: Optional Dashboard instance for telemetry
            audio_router: Optional NavigationAudioRouter for priority queuing
            navigation_pipeline: Optional NavigationPipeline (created if not provided)
            decision_engine: Optional NavigationDecisionEngine (created if not provided)
            telemetry: Optional telemetry system for logging
        """
        # Injected dependencies
        self.audio_system = audio_system
        self.frame_renderer = frame_renderer
        self.dashboard = dashboard
        self.audio_router: Optional[Any] = audio_router
        self.telemetry = telemetry

        # Initialize pipeline and decision engine
        # NOTE: Fallback construction violates Dependency Inversion Principle
        # TODO: Require all dependencies in constructor, remove fallback construction
        # Builder should ensure all dependencies are created before Coordinator
        self.pipeline = navigation_pipeline or NavigationPipeline(
            yolo_processor=yolo_processor,
            image_enhancer=image_enhancer,
        )
        self.decision_engine = decision_engine or NavigationDecisionEngine()

        # Backward compatibility aliases
        self.yolo_processor = self.pipeline.yolo_processor
        self.image_enhancer = self.pipeline.image_enhancer
        self.depth_estimator = self.pipeline.depth_estimator

        # Internal state
        self.frames_processed = 0
        self.last_announcement_time = 0.0
        self.current_detections: List[Dict[str, Any]] = []

        # Load profiling configuration (typed section)
        profiling_config = load_profiling_config()
        self.profile_enabled = profiling_config.enabled
        self.profile_window = max(1, profiling_config.window_frames)
        self._profile_acc = {
            'enhance': 0.0,
            'depth': 0.0,
            'yolo': 0.0,
            'nav_audio': 0.0,
            'render': 0.0,
            'total': 0.0,
        }
        self._profile_frames = 0

        # Peripheral SLAM support
        self.peripheral_enabled = False
        self.slam_state = SlamRoutingState(
            workers={},
            frame_counters={},
            last_indices={},
            latest_events={},
        )
        # Create shared MessageFormatter for both RGB and SLAM routers
        message_formatter = MessageFormatter()

        # Symmetric layer to SlamAudioRouter: formats RGB events before queuing.
        # Pass global_tracker for cross-camera tracking
        self.slam_router = SlamAudioRouter(
            self.audio_router,
            global_tracker=self.decision_engine.global_tracker,
            message_formatter=message_formatter
        )
        self.rgb_router = RgbAudioRouter(
            audio_system,
            self.audio_router,
            self.slam_router,
            message_formatter=message_formatter
        )
        if self.audio_router and not getattr(self.audio_router, "_running", False):
            self.audio_router.start()

        # Camera geometry for 3D tracking (optional, set later with calibrations)
        self.camera_geometry = None

        print(f"[INFO] Coordinator initialized")
        print(f"  - YOLO: {type(self.yolo_processor).__name__}")
        print(f"  - Audio: {type(self.audio_system).__name__}")
        print(f"  - Frame Renderer: {type(self.frame_renderer).__name__ if self.frame_renderer else 'None'}")
        print(f"  - Image Enhancer: {type(self.image_enhancer).__name__ if self.image_enhancer else 'None'}")
        print(f"  - Dashboard: {type(self.dashboard).__name__ if self.dashboard else 'None'}")

    def set_camera_calibrations(
        self,
        rgb_calib: Optional[object] = None,
        slam1_calib: Optional[object] = None,
        slam2_calib: Optional[object] = None,
    ) -> None:
        """
        Set camera calibrations for 3D geometric tracking (Phase 3).

        Enables cross-camera object tracking with 3D geometric validation when
        all three camera calibrations are available. This allows the system to
        verify that objects detected in multiple cameras are the same physical
        object based on their 3D positions.

        Args:
            rgb_calib: RGB camera calibration from Aria SDK
            slam1_calib: SLAM1 camera calibration from Aria SDK
            slam2_calib: SLAM2 camera calibration from Aria SDK
        """
        try:
            from core.vision.camera_geometry import CameraGeometry

            self.camera_geometry = CameraGeometry(rgb_calib, slam1_calib, slam2_calib)

            # Enable 3D validation if configured (typed section)
            tracker_config = load_tracker_config()

            if tracker_config.use_3d_validation and self.camera_geometry.is_available():
                # Update global tracker with camera geometry
                self.decision_engine.global_tracker.camera_geometry = self.camera_geometry
                self.decision_engine.global_tracker.use_3d_validation = True
                self.decision_engine.global_tracker.max_3d_distance = tracker_config.max_3d_distance

                print(f"[Coordinator] 3D geometric validation ENABLED "
                      f"(max_distance={tracker_config.max_3d_distance}m)")
            else:
                print("[Coordinator] 3D validation available but disabled in Config")

        except Exception as e:
            print(f"[Coordinator] Could not initialize CameraGeometry: {e}")
            self.camera_geometry = None

    def process_frame(self, frame: "np.ndarray", motion_state: str = "stationary", frames_dict: Optional[Dict[str, "np.ndarray"]] = None) -> "np.ndarray":
        """
        Process complete frame through the vision-navigation-audio pipeline.

        Args:
            frame: Input BGR frame from RGB camera
            motion_state: User motion state ("stationary", "walking")
            frames_dict: Optional dict with frames from 3 cameras for multiprocess mode

        Returns:
            np.ndarray: Processed frame with navigation overlay annotations
        """
        self.frames_processed += 1

        total_start = time.perf_counter() if self.profile_enabled else 0.0

        pipeline_result = self.pipeline.process(frame, profile=self.profile_enabled, frames_dict=frames_dict)
        processed_frame = pipeline_result.frame
        detections = pipeline_result.detections
        depth_map = pipeline_result.depth_map

        if self.profile_enabled:
            timings = pipeline_result.timings
            self._profile_acc['enhance'] += timings.get('enhance', 0.0)
            self._profile_acc['depth'] += timings.get('depth', 0.0)
            self._profile_acc['yolo'] += timings.get('yolo', 0.0)

        # 3. Navigation Analysis
        nav_start = time.perf_counter() if self.profile_enabled else 0.0
        navigation_objects = self.decision_engine.analyze(detections, depth_map)

        # 4. Audio Commands (with motion-aware cooldown)
        decision_candidate = self.decision_engine.evaluate(navigation_objects, motion_state)
        if decision_candidate is not None:
            self.rgb_router.route(decision_candidate)
        if self.profile_enabled:
            self._profile_acc['nav_audio'] += time.perf_counter() - nav_start

        self.last_announcement_time = self.decision_engine.last_announcement_time

        # 5. Frame Rendering (if available)
        render_start = time.perf_counter() if self.profile_enabled else 0.0
        annotated_frame = processed_frame
        if self.frame_renderer is not None:
            try:
                # Draw only navigation_objects (already filtered by decision engine)
                # This avoids phantom boxes from irrelevant detections
                annotated_frame = self.frame_renderer.draw_navigation_overlay(
                    processed_frame, navigation_objects, self.audio_system, depth_map
                )
            except Exception as err:
                print(f"[WARN] Frame rendering skipped: {err}")
                annotated_frame = processed_frame
        if self.profile_enabled:
            self._profile_acc['render'] += time.perf_counter() - render_start

        # 6. Dashboard Update (if available)
        if self.dashboard:
            try:
                self.dashboard.update_detections(detections)
                self.dashboard.update_navigation_status(navigation_objects)
            except Exception as err:
                print(f"[WARN] Dashboard update skipped: {err}")

        # Save current state: use navigation_objects (filtered) instead of detections (raw)
        # This avoids showing irrelevant objects that didn't pass navigation filter
        self.current_detections = navigation_objects

        if self.profile_enabled:
            self._profile_acc['total'] += time.perf_counter() - total_start
            self._profile_frames += 1
            if self._profile_frames >= self.profile_window:
                self._log_profile_metrics()

        return annotated_frame

    # ------------------------------------------------------------------
    # Peripheral vision (SLAM) integration
    # ------------------------------------------------------------------

    def attach_peripheral_system(
        self,
        slam_workers: Dict[Any, Any],
        audio_router: Optional[Any] = None,
    ) -> None:
        if CameraSource is None:
            print("[WARN] Peripheral system not available (missing dependencies)")
            return

        self.peripheral_enabled = True
        self.slam_state.workers = slam_workers
        if audio_router is not None:
            self.audio_router = audio_router
            self.slam_router = SlamAudioRouter(audio_router)
            self.rgb_router.set_audio_router(self.audio_router)
            self.rgb_router.set_slam_router(self.slam_router)

        for source, worker in slam_workers.items():
            worker.start()
            self.slam_state.frame_counters[source] = 0
            self.slam_state.last_indices[source] = -1
            self.slam_state.latest_events[source] = []

        if self.audio_router and not getattr(self.audio_router, "_running", False):
            self.audio_router.start()

        print("[PERIPHERAL] SLAM workers attached")

    def handle_slam_frames(
        self,
        slam1_frame: Optional["np.ndarray"] = None,
        slam2_frame: Optional["np.ndarray"] = None,
    ) -> None:
        if not self.peripheral_enabled or CameraSource is None:
            return

        slam1_source = getattr(CameraSource, 'SLAM1', None)
        slam2_source = getattr(CameraSource, 'SLAM2', None)

        if slam1_frame is not None and slam1_source in self.slam_state.workers:
            self.slam_router.submit_and_route(self.slam_state, slam1_source, slam1_frame)

        if slam2_frame is not None and slam2_source in self.slam_state.workers:
            self.slam_router.submit_and_route(self.slam_state, slam2_source, slam2_frame)

    def get_slam_events(self) -> Dict[str, List[dict]]:
        if not self.peripheral_enabled:
            return {}

        event_dict: Dict[str, List[dict]] = {}
        for source, events in self.slam_state.latest_events.items():
            simplified = []
            for event in events:
                simplified.append(
                    {
                        'bbox': event.bbox,
                        'name': event.object_name,
                        'confidence': event.confidence,
                        'zone': event.zone,
                        'distance': event.distance,
                        'message': self.slam_router.describe_event(event),
                        'camera_source': 'slam',  # Mark as SLAM for filtering
                    }
                )
            event_dict[source.value] = simplified
        return event_dict

    def _log_profile_metrics(self) -> None:
        frame_count = max(1, self._profile_frames)
        averaged = {
            key: (value / frame_count) * 1000.0 for key, value in self._profile_acc.items()
        }
        msg = "[PROFILE] enhance={enhance:.1f}ms | depth={depth:.1f}ms | yolo={yolo:.1f}ms | nav+audio={nav_audio:.1f}ms | render={render:.1f}ms | total={total:.1f}ms".format(**averaged)
        print(msg)
        
        # Log to telemetry if available
        if hasattr(self, 'telemetry') and self.telemetry:
            try:
                self.telemetry.log_system_event({
                    'event_type': 'profile_metrics',
                    'enhance_ms': averaged['enhance'],
                    'depth_ms': averaged['depth'],
                    'yolo_ms': averaged['yolo'],
                    'nav_audio_ms': averaged['nav_audio'],
                    'render_ms': averaged['render'],
                    'total_ms': averaged['total'],
                    'frames_averaged': frame_count
                })
            except Exception:
                pass
        
        for key in self._profile_acc:
            self._profile_acc[key] = 0.0
        self._profile_frames = 0

    def get_latest_depth_map(self) -> Optional["np.ndarray"]:
        """Get the latest estimated depth map."""
        return self.pipeline.get_latest_depth_map()

    def get_status(self):
        """
        Get current coordinator status with metrics and information.

        Returns:
            dict: Status dictionary containing frames processed, detection count,
                  audio queue size, peripheral events, and subsystem availability
        """
        audio_queue_size = 0
        beep_stats = {}
        try:
            if hasattr(self.audio_system, 'get_queue_size'):
                audio_queue_size = self.audio_system.get_queue_size()
            elif hasattr(self.audio_system, 'audio_queue'):
                audio_queue_size = len(self.audio_system.audio_queue)
            
            # Get beep statistics if available
            if hasattr(self.audio_system, 'get_beep_stats'):
                beep_stats = self.audio_system.get_beep_stats()
        except:
            pass
        
        status = {
            'frames_processed': self.frames_processed,
            'current_detections_count': len(self.current_detections),
            'audio_queue_size': audio_queue_size,
            'has_dashboard': self.dashboard is not None,
            'slam_events': (
                sum(len(events) for events in self.slam_state.latest_events.values())
                if self.peripheral_enabled
                else 0
            ),
            'has_frame_renderer': self.frame_renderer is not None,
            'has_image_enhancer': self.image_enhancer is not None,
            'last_announcement': self.last_announcement_time
        }
        
        # Add beep statistics
        status.update(beep_stats)
        
        return status
    
    def get_current_detections(self):
        """
        Get current detections (for compatibility with Observer).

        Returns:
            list: Copy of current detections list
        """
        return self.current_detections.copy()

    def scan_scene(self):
        """
        SCAN MODE: Audible summary of current scene (NOA-inspired).

        Announces 3-5 main objects grouped by zone.
        Example: "Scanning. Ahead: person, chair. Left: table."
        """
        if hasattr(self.audio_system, 'scan_scene'):
            self.audio_system.scan_scene(self.current_detections)
            print("[COORDINATOR] Scene scan triggered")
        else:
            print("[COORDINATOR] scan_scene() not available in audio_system")

    def test_audio(self):
        """
        Test the audio system with a simple message.
        """
        try:
            if hasattr(self.audio_system, 'speak_force'):
                self.audio_system.speak_force("Navigation system test")
            else:
                self.audio_system.speak_async("Navigation system test")
            print("[COORDINATOR] Audio test emitted")
        except Exception as e:
            print(f"[COORDINATOR] Audio test failed: {e}")
    
    def print_stats(self):
        """
        Print coordinator statistics and current state.
        """
        status = self.get_status()

        print(f"\n[COORDINATOR STATS]")
        print(f"  Frames processed: {status['frames_processed']}")
        print(f"  Current detections: {status['current_detections_count']}")
        print(f"  Audio queue size: {status['audio_queue_size']}")
        print(f"  Dashboard: {'Yes' if status['has_dashboard'] else 'No'}")
        print(f"  Frame Renderer: {'Yes' if status['has_frame_renderer'] else 'No'}")
        print(f"  Image Enhancer: {'Yes' if status['has_image_enhancer'] else 'No'}")

        # Show latest detections
        if self.current_detections:
            print(f"  Latest detections:")
            for det in self.current_detections[:3]:  # Top 3
                name = det.get('name', 'unknown')
                zone = det.get('zone', 'unknown')
                conf = det.get('confidence', 0)
                print(f"    - {name} in {zone} (conf: {conf:.2f})")
    
    def cleanup(self):
        """
        Clean up resources and shutdown all subsystems.
        """
        print("[INFO] Cleaning up Coordinator...")

        # Shutdown pipeline multiproc workers
        if hasattr(self.pipeline, 'shutdown'):
            try:
                self.pipeline.shutdown()
            except Exception as err:
                print(f"  [WARN] Pipeline shutdown error: {err}")

        if self.peripheral_enabled:
            try:
                if self.audio_router:
                    self.audio_router.stop()
            except Exception as err:
                print(f"  [WARN] Peripheral audio router cleanup error: {err}")

            for source, worker in self.slam_state.workers.items():
                try:
                    worker.stop()
                except Exception as err:
                    print(f"  [WARN] SLAM worker {source.value} cleanup error: {err}")
            self.slam_state = SlamRoutingState(
                workers={},
                frame_counters={},
                last_indices={},
                latest_events={},
            )

        try:
            if self.audio_system:
                if hasattr(self.audio_system, 'cleanup'):
                    self.audio_system.cleanup()
                elif hasattr(self.audio_system, 'close'):
                    self.audio_system.close()
                print("  [INFO] Audio system cleanup complete")
        except Exception as e:
            print(f"  [WARN] Audio cleanup error: {e}")

        try:
            if self.dashboard:
                if hasattr(self.dashboard, 'cleanup'):
                    self.dashboard.cleanup()
                elif hasattr(self.dashboard, 'shutdown'):
                    self.dashboard.shutdown()
                print("  [INFO] Dashboard cleanup complete")
        except Exception as e:
            print(f"  [WARN] Dashboard cleanup error: {e}")

        # Frame renderer and image enhancer typically don't need cleanup

        # Reset state
        self.current_detections = []

        print("[INFO] Coordinator cleanup complete")
