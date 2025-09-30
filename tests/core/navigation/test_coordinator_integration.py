"""Integration tests for Coordinator wiring with audio routers."""

from __future__ import annotations

import time
from types import SimpleNamespace

from core.audio.navigation_audio_router import EventPriority
from core.navigation.coordinator import Coordinator
from core.navigation.navigation_decision_engine import DecisionCandidate

try:
    from core.vision.slam_detection_worker import CameraSource
except Exception:  # pragma: no cover - fallback when SLAM enums unavailable
    CameraSource = SimpleNamespace(SLAM1=SimpleNamespace(value="slam1"))  # type: ignore


class DummyAudioSystem:
    def __init__(self) -> None:
        self.repeat = None
        self.announcement = None
        self.messages = []

    def set_repeat_cooldown(self, value: float) -> None:
        self.repeat = value

    def set_announcement_cooldown(self, value: float) -> None:
        self.announcement = value

    def queue_message(self, message: str) -> None:
        self.messages.append(message)

    def speak_async(self, message: str) -> bool:
        self.messages.append(message)
        return True


class FakePipeline:
    def __init__(self, detections):
        self._detections = detections
        self.yolo_processor = object()
        self.image_enhancer = object()
        self.depth_estimator = None
        self.processed_frames = []

    def process(self, frame, profile: bool = False):
        self.processed_frames.append(frame)
        return SimpleNamespace(
            frame="processed-frame",
            detections=self._detections,
            depth_map=None,
            timings={},
        )

    def get_latest_depth_map(self):  # pragma: no cover - coordinator may call indirectly
        return None


class FakeDecisionEngine:
    def __init__(self, candidate: DecisionCandidate):
        self._candidate = candidate
        self.last_announcement_time = 0.0
        self.analyzed_inputs = []

    def analyze(self, detections):
        self.analyzed_inputs.append(detections)
        return detections

    def evaluate(self, navigation_objects, motion_state="stationary"):
        self.last_announcement_time = time.time()
        return self._candidate


class FakeNavigationAudioRouter:
    _running = True

    def __init__(self) -> None:
        self.rgb_calls = []
        self.slam_calls = []
        self.cooldowns = []
        self.started = False

    def set_source_cooldown(self, source: str, value: float) -> None:
        self.cooldowns.append((source, value))

    def enqueue_from_rgb(self, message: str, priority, metadata):
        self.rgb_calls.append((message, priority, metadata))

    def enqueue_from_slam(self, event, message: str, priority):
        self.slam_calls.append((event, message, priority))

    def start(self) -> None:
        self.started = True


class DummySlamWorker:
    def __init__(self):
        self.submissions = []
        self._events = []
        self.started = False

    def start(self):
        self.started = True

    def submit(self, frame, frame_index):
        self.submissions.append((frame, frame_index))

    def latest_events(self):
        return list(self._events)


def test_coordinator_processes_rgb_and_slam_paths():
    detections = [{"class": "person", "zone": "center", "distance": "cerca", "priority": 10}]
    candidate = DecisionCandidate(
        nav_object=detections[0],
        metadata={"cooldown": 1.0, "class": "person", "distance": "cerca"},
        priority=EventPriority.CRITICAL,
    )

    pipeline = FakePipeline(detections)
    decision_engine = FakeDecisionEngine(candidate)
    audio_system = DummyAudioSystem()
    nav_router = FakeNavigationAudioRouter()

    coordinator = Coordinator(
        yolo_processor=object(),
        audio_system=audio_system,
        frame_renderer=None,
        image_enhancer=None,
        dashboard=None,
        audio_router=nav_router,
        navigation_pipeline=pipeline,
        decision_engine=decision_engine,
    )

    coordinator.process_frame(frame="raw-frame")

    assert nav_router.rgb_calls
    rgb_message, rgb_priority, rgb_metadata = nav_router.rgb_calls[0]
    assert "person" in rgb_message.lower()
    assert rgb_priority is EventPriority.CRITICAL
    assert rgb_metadata["cooldown"] == 1.0

    worker = DummySlamWorker()
    slam_event = SimpleNamespace(
        frame_index=1,
        source=CameraSource.SLAM1,
        object_name="car",
        distance="close",
        zone="left",
        timestamp=123.0,
        confidence=0.8,
        bbox=(0, 0, 10, 10),
    )
    worker._events = [slam_event]

    coordinator.attach_peripheral_system({CameraSource.SLAM1: worker}, audio_router=nav_router)
    coordinator.handle_slam_frames(slam1_frame="frame-data")

    assert worker.submissions
    assert nav_router.slam_calls
    _, slam_message, slam_priority = nav_router.slam_calls[0]
    assert "car" in slam_message.lower()
    assert slam_priority.name == "CRITICAL"
