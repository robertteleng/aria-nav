# Professional Profiling Tools for aria-nav Phase 2+

## 1. py-spy (RECOMENDADO para production)
# Sampling profiler, overhead mínimo (~3-5%)
pip install py-spy

# Uso:
py-spy record -o profile.svg --native -- python src/main.py debug
py-spy top --native -- python src/main.py debug

# Genera flamegraph SVG - muy visual


## 2. cProfile + snakeviz (built-in)
pip install snakeviz

# Uso:
python tools/profile_runner.py 60  # 60 segundos
snakeviz logs/profiling/latest.prof  # Abre browser


## 3. torch.profiler (para CUDA)
# Ya en torch, para medir GPU kernels

# En NavigationPipeline:
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
) as prof:
    result = pipeline.process(frame)

prof.export_chrome_trace("trace.json")
# Ver en chrome://tracing


## 4. Custom PerformanceProfiler (creado)
# Ligero, sin deps extra, métricas custom
from utils.profiler import get_profiler

profiler = get_profiler()
with profiler.measure("component_name"):
    do_work()
profiler.print_report()


## 5. nvidia-smi + gpustat (monitoring)
pip install gpustat

# Loop monitoring:
watch -n 1 gpustat

# O en script:
nvidia-smi dmon -i 0 -s pucvmet -c 60 > gpu_stats.txt


## 6. memory_profiler (para memory leaks)
pip install memory-profiler

# Decorador en funciones sospechosas:
@profile
def _process_multiproc(self, frames_dict, profile):
    ...

python -m memory_profiler src/main.py debug


## 7. psutil (system monitoring)
pip install psutil nvidia-ml-py

# CPU/RAM/Network/Disk monitoring
# Ver src/utils/system_monitor.py para wrapper completo
from utils.system_monitor import get_monitor

monitor = get_monitor()
metrics = monitor.get_metrics()
monitor.print_metrics(metrics)

# Alertas automáticas
if metrics.gpu_memory_percent > 90:
    print("⚠️ GPU VRAM crítico!")


## RECOMENDACIÓN FASE 2:
1. **SystemMonitor** (custom con psutil) - CPU/RAM/GPU real-time
2. **PerformanceProfiler** (custom) - Tiempos por componente
3. **py-spy** para flamegraphs (visual, bajo overhead)
4. **torch.profiler** si sospechas GPU bottleneck

## Uso integrado:
```python
from utils.system_monitor import get_monitor
from utils.profiler import get_profiler

sys_monitor = get_monitor()
profiler = get_profiler()

# Cada frame
with profiler.measure("frame_processing"):
    result = pipeline.process(frame)

# Cada 5s - stats
if time.time() - last_check > 5.0:
    metrics = sys_monitor.get_metrics()
    if metrics.gpu_memory_percent > 90:
        log.warning("⚠️ GPU VRAM > 90%!")
```


## INSTALACIÓN RÁPIDA:
import psutil

# CPU por core
cpu_percent = psutil.cpu_percent(interval=1, percpu=True)

# RAM
mem = psutil.virtual_memory()
print(f"RAM: {mem.percent}% used, {mem.available / 1024**3:.2f}GB available")

# GPU (via pynvml/nvidia-ml-py)
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(f"GPU VRAM: {mem_info.used / 1024**3:.2f}GB / {mem_info.total / 1024**3:.2f}GB")

# Network I/O
net = psutil.net_io_counters()
print(f"Network sent: {net.bytes_sent / 1024**2:.2f}MB, recv: {net.bytes_recv / 1024**2:.2f}MB")


## INSTALACIÓN RÁPIDA:
pip install py-spy snakeviz gpustat memory-profiler psutil pynvml
