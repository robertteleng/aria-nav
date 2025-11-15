import time
import threading
import psutil
import os

try:
    import pynvml
    pynvml.nvmlInit()
    _HAS_PYNVML = True
except Exception:
    _HAS_PYNVML = False


def _sample_gpu():
    if not _HAS_PYNVML:
        return {"gpu_present": False}
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        return {
            "gpu_present": True,
            "gpu_mem_used_mb": int(mem.used // (1024 * 1024)),
            "gpu_mem_total_mb": int(mem.total // (1024 * 1024)),
            "gpu_util_pct": int(util.gpu),
        }
    except Exception:
        return {"gpu_present": False}


def sample_resources():
    vm = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=None)
    gpu = _sample_gpu()
    return {
        "timestamp": time.time(),
        "cpu_pct": float(cpu),
        "ram_used_mb": int(vm.used // (1024 * 1024)),
        "ram_total_mb": int(vm.total // (1024 * 1024)),
        **gpu,
        "pid": os.getpid(),
    }


class ResourceMonitor(threading.Thread):
    def __init__(self, interval=1.0, callback=None):
        super().__init__(daemon=True)
        self.interval = interval
        self.callback = callback
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            data = sample_resources()
            if self.callback:
                try:
                    self.callback(data)
                except Exception:
                    pass
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()
