"""Tests for SlamAudioRouter routing behaviour."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.navigation.slam_audio_router import SlamAudioRouter, SlamRoutingState
from core.vision.slam_detection_worker import CameraSource


class FakeWorker:
    def __init__(self, events):
        self._events = events
        self.submissions = []

    def set_events(self, events):
        self._events = events

    def submit(self, frame, frame_index):
        self.submissions.append((frame, frame_index))

    def latest_events(self):
        return self._events


class DummyRouter:
    def __init__(self):
        self.enqueued = []

    def enqueue_from_slam(self, event, message, priority):
        self.enqueued.append((event, message, priority))


@pytest.fixture()
def slam_state():
    source = CameraSource.SLAM1
    worker = FakeWorker([])
    return (
        source,
        worker,
        SlamRoutingState(
            workers={source: worker},
            frame_counters={source: 0},
            last_indices={source: -1},
            latest_events={source: []},
        ),
    )


def create_event(frame_index: int, source: CameraSource, **kwargs):
    defaults = {
        "object_name": "car",
        "distance": "close",
        "zone": "left",
        "timestamp": 123.0,
        "bbox": (0, 0, 10, 10),
        "confidence": 0.9,
    }
    defaults.update(kwargs)
    return SimpleNamespace(frame_index=frame_index, source=source, **defaults)


def test_submit_and_route_enqueues_new_events(slam_state):
    source, worker, state = slam_state
    router = DummyRouter()
    slam_router = SlamAudioRouter(router)

    event = create_event(1, source)
    worker.set_events([event])

    slam_router.submit_and_route(state, source, frame="frame")

    assert worker.submissions == [("frame", 1)]
    assert state.last_indices[source] == 1
    assert len(router.enqueued) == 1
    _, message, priority = router.enqueued[0]
    assert message.startswith("Warning")
    assert priority.name == "CRITICAL"


def test_submit_and_route_skips_duplicate_events(slam_state):
    source, worker, state = slam_state
    router = DummyRouter()
    slam_router = SlamAudioRouter(router)

    event = create_event(5, source)
    worker.set_events([event])

    slam_router.submit_and_route(state, source, frame="frame-a")
    assert len(router.enqueued) == 1

    # Same frame index -> should not enqueue again.
    slam_router.submit_and_route(state, source, frame="frame-b")
    assert len(router.enqueued) == 1


def test_submit_and_route_clears_latest_events_when_empty(slam_state):
    source, worker, state = slam_state
    router = DummyRouter()
    slam_router = SlamAudioRouter(router)

    # First populate
    worker.set_events([create_event(2, source)])
    slam_router.submit_and_route(state, source, frame="frame")
    assert state.latest_events[source]

    # Now worker reports no detections; list should clear and no enqueue.
    worker.set_events([])
    slam_router.submit_and_route(state, source, frame="frame2")

    assert state.latest_events[source] == []
    assert len(router.enqueued) == 1  # unchanged from first pass


def test_person_events_get_high_priority(slam_state):
    source, worker, state = slam_state
    router = DummyRouter()
    slam_router = SlamAudioRouter(router)

    event = create_event(3, source, object_name="person", distance="close")
    worker.set_events([event])

    slam_router.submit_and_route(state, source, frame="frame")

    assert router.enqueued[-1][2].name == "HIGH"


def test_submit_without_shared_router_is_noop(slam_state):
    source, worker, state = slam_state
    slam_router = SlamAudioRouter(audio_router=None)

    event = create_event(4, source)
    worker.set_events([event])

    slam_router.submit_and_route(state, source, frame="frame")

    # last_indices still actualized even sin router para permitir dedupe
    assert state.last_indices[source] == 4
