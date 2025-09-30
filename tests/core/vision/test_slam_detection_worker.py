"""Behaviour tests for SlamDetectionWorker with stubbed YOLO processor."""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from core.vision import slam_detection_worker as slam_module
from core.vision.slam_detection_worker import CameraSource, SlamDetectionWorker


class StubProcessor:
    def __init__(self):
        self.frame_skip = 1
        self.img_size = 320

    def process_frame(self, frame):
        return [
            {
                "name": "person",
                "confidence": 0.8,
                "zone": "left",
                "distance": "close",
                "bbox": (0, 0, frame.shape[1] // 2, frame.shape[0] // 2),
                "relevance_score": 0.9,
            }
        ]


@pytest.fixture()
def stubbed_worker(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(slam_module, "YoloProcessor", SimpleNamespace(from_profile=lambda *_: StubProcessor()))
    worker = SlamDetectionWorker(CameraSource.SLAM1, processor=StubProcessor(), target_fps=30, queue_size=1)
    worker.start()
    yield worker
    worker.stop()


def create_frame(width=100, height=80):
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_submit_creates_events(stubbed_worker: SlamDetectionWorker):
    stubbed_worker.submit(create_frame(), frame_index=1)
    time.sleep(0.1)
    events = stubbed_worker.latest_events()
    assert events
    event = events[0]
    assert event.object_name == "person"
    assert event.zone == "peripheral_left"
    assert event.frame_index == 1


def test_queue_overflow_replaces_old_frame(stubbed_worker: SlamDetectionWorker):
    stubbed_worker.submit(create_frame(), frame_index=1)
    stubbed_worker.submit(create_frame(), frame_index=2)
    stubbed_worker.submit(create_frame(), frame_index=3)
    time.sleep(0.1)
    events = stubbed_worker.latest_events()
    assert events and events[0].frame_index in {2, 3}


def test_stop_is_idempotent(stubbed_worker: SlamDetectionWorker):
    stubbed_worker.stop()
    stubbed_worker.stop()
