import multiprocessing as mp
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

os.environ.setdefault("ARIA_SKIP_DEPTH", "1")

from core.processing.central_worker import central_gpu_worker
from core.processing.multiproc_types import ResultMessage
from core.processing.slam_worker import slam_gpu_worker


FRAME_SHAPE = (224, 224, 3)


def _build_frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, FRAME_SHAPE, dtype=np.uint8)


def _warm_queue(queue: mp.Queue, data: dict) -> None:
    queue.put(data, timeout=5)


def _collect_result(queue: mp.Queue) -> ResultMessage:
    return queue.get(timeout=30)


def test_central_worker_once():
    mp.set_start_method('spawn', force=True)
    central_queue = mp.Queue(maxsize=2)
    result_queue = mp.Queue(maxsize=2)
    stop_event = mp.Event()

    proc = mp.Process(
        target=central_gpu_worker,
        args=(central_queue, result_queue, stop_event),
        name="CentralWorkerTest",
    )
    proc.start()

    _warm_queue(
        central_queue,
        {
            'frame_id': 1,
            'camera': 'central',
            'frame': _build_frame(seed=1),
            'timestamp': time.time(),
        },
    )

    result: ResultMessage = _collect_result(result_queue)
    print("CentralWorker result:", result.camera, "detections", len(result.detections))

    stop_event.set()
    proc.join(timeout=5)
    if proc.is_alive():
        proc.terminate()
        proc.join()


def test_slam_worker_once():
    mp.set_start_method('spawn', force=True)
    slam_queue = mp.Queue(maxsize=2)
    result_queue = mp.Queue(maxsize=2)
    stop_event = mp.Event()

    proc = mp.Process(
        target=slam_gpu_worker,
        args=(slam_queue, result_queue, stop_event),
        name="SlamWorkerTest",
    )
    proc.start()

    _warm_queue(
        slam_queue,
        {
            'frame_id': 2,
            'camera': 'slam1',
            'frame': _build_frame(seed=2),
            'timestamp': time.time(),
        },
    )

    result: ResultMessage = _collect_result(result_queue)
    print("SlamWorker result:", result.camera, "detections", len(result.detections))

    stop_event.set()
    proc.join(timeout=5)
    if proc.is_alive():
        proc.terminate()
        proc.join()


def main():
    test_central_worker_once()
    test_slam_worker_once()


if __name__ == "__main__":
    main()
