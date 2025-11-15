#!/usr/bin/env python3
"""
System resource monitor using psutil + pynvml
Lightweight, real-time CPU/RAM/GPU tracking
"""

import time
import psutil
from typing import Dict, Optional
from dataclasses import dataclass, asdict

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("[WARN] pynvml not available. Install: pip install nvidia-ml-py")

@dataclass
class SystemMetrics:
    """System resource snapshot"""
    timestamp: float
    
    # CPU
    cpu_percent: float              # Total CPU usage %
    cpu_per_core: list              # Per-core usage %
    cpu_freq_mhz: float            # Current CPU frequency
    
    # RAM
    ram_total_gb: float
    ram_used_gb: float
    ram_available_gb: float
    ram_percent: float
    
    # GPU (if available)
    gpu_utilization: Optional[float] = None      # GPU usage %
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_temperature_c: Optional[int] = None
    gpu_power_watts: Optional[float] = None
    
    # Network
    network_sent_mb: Optional[float] = None
    network_recv_mb: Optional[float] = None
    
    # Process specific (if PID provided)
    process_cpu_percent: Optional[float] = None
    process_ram_mb: Optional[float] = None
    process_num_threads: Optional[int] = None

class SystemMonitor:
    """
    Real-time system resource monitor
    
    Usage:
        monitor = SystemMonitor()
        
        while running:
            metrics = monitor.get_metrics()
            if metrics.ram_percent > 90:
                print("⚠️ Low memory!")
    """
    
    def __init__(self, enable_gpu: bool = True, process_pid: Optional[int] = None):
        self.enable_gpu = enable_gpu and PYNVML_AVAILABLE
        self.process_pid = process_pid
        self.process = None
        
        if self.process_pid:
            try:
                self.process = psutil.Process(self.process_pid)
            except psutil.NoSuchProcess:
                print(f"[WARN] Process {process_pid} not found")
        
        # Initialize GPU
        self.gpu_handle = None
        if self.enable_gpu:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception as e:
                print(f"[WARN] Failed to initialize GPU monitoring: {e}")
                self.enable_gpu = False
        
        # Network baseline
        self.net_baseline = psutil.net_io_counters()
        self.last_check_time = time.time()
    
    def get_metrics(self) -> SystemMetrics:
        """Get current system metrics snapshot"""
        timestamp = time.time()
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_freq = psutil.cpu_freq()
        cpu_freq_mhz = cpu_freq.current if cpu_freq else 0.0
        
        # RAM
        mem = psutil.virtual_memory()
        ram_total_gb = mem.total / (1024**3)
        ram_used_gb = mem.used / (1024**3)
        ram_available_gb = mem.available / (1024**3)
        ram_percent = mem.percent
        
        # GPU
        gpu_util = None
        gpu_mem_used = None
        gpu_mem_total = None
        gpu_mem_percent = None
        gpu_temp = None
        gpu_power = None
        
        if self.enable_gpu and self.gpu_handle:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_util = float(util.gpu)
                
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_mem_used = mem_info.used / (1024**3)
                gpu_mem_total = mem_info.total / (1024**3)
                gpu_mem_percent = (mem_info.used / mem_info.total) * 100
                
                gpu_temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
                
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                    gpu_power = power / 1000.0  # mW to W
                except:
                    pass
            except Exception as e:
                print(f"[WARN] GPU metrics error: {e}")
        
        # Network
        net = psutil.net_io_counters()
        net_sent_mb = (net.bytes_sent - self.net_baseline.bytes_sent) / (1024**2)
        net_recv_mb = (net.bytes_recv - self.net_baseline.bytes_recv) / (1024**2)
        
        # Process specific
        proc_cpu = None
        proc_ram = None
        proc_threads = None
        
        if self.process:
            try:
                proc_cpu = self.process.cpu_percent(interval=0.1)
                proc_mem = self.process.memory_info()
                proc_ram = proc_mem.rss / (1024**2)  # MB
                proc_threads = self.process.num_threads()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return SystemMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            cpu_per_core=cpu_per_core,
            cpu_freq_mhz=cpu_freq_mhz,
            ram_total_gb=ram_total_gb,
            ram_used_gb=ram_used_gb,
            ram_available_gb=ram_available_gb,
            ram_percent=ram_percent,
            gpu_utilization=gpu_util,
            gpu_memory_used_gb=gpu_mem_used,
            gpu_memory_total_gb=gpu_mem_total,
            gpu_memory_percent=gpu_mem_percent,
            gpu_temperature_c=gpu_temp,
            gpu_power_watts=gpu_power,
            network_sent_mb=net_sent_mb,
            network_recv_mb=net_recv_mb,
            process_cpu_percent=proc_cpu,
            process_ram_mb=proc_ram,
            process_num_threads=proc_threads,
        )
    
    def print_metrics(self, metrics: SystemMetrics = None):
        """Print formatted metrics"""
        if metrics is None:
            metrics = self.get_metrics()
        
        print("\n" + "="*60)
        print("📊 SYSTEM METRICS")
        print("="*60)
        
        print(f"\n🖥️  CPU:")
        print(f"  Usage: {metrics.cpu_percent:.1f}%")
        print(f"  Freq:  {metrics.cpu_freq_mhz:.0f} MHz")
        
        print(f"\n💾 RAM:")
        print(f"  Used:      {metrics.ram_used_gb:.2f} / {metrics.ram_total_gb:.2f} GB ({metrics.ram_percent:.1f}%)")
        print(f"  Available: {metrics.ram_available_gb:.2f} GB")
        
        if metrics.gpu_utilization is not None:
            print(f"\n🎮 GPU:")
            print(f"  Usage:  {metrics.gpu_utilization:.1f}%")
            print(f"  VRAM:   {metrics.gpu_memory_used_gb:.2f} / {metrics.gpu_memory_total_gb:.2f} GB ({metrics.gpu_memory_percent:.1f}%)")
            if metrics.gpu_temperature_c:
                print(f"  Temp:   {metrics.gpu_temperature_c}°C")
            if metrics.gpu_power_watts:
                print(f"  Power:  {metrics.gpu_power_watts:.1f}W")
        
        if metrics.network_sent_mb is not None:
            print(f"\n🌐 Network (since start):")
            print(f"  Sent: {metrics.network_sent_mb:.2f} MB")
            print(f"  Recv: {metrics.network_recv_mb:.2f} MB")
        
        if metrics.process_cpu_percent is not None:
            print(f"\n🔧 Process:")
            print(f"  CPU:     {metrics.process_cpu_percent:.1f}%")
            print(f"  RAM:     {metrics.process_ram_mb:.1f} MB")
            print(f"  Threads: {metrics.process_num_threads}")
        
        print("="*60)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.enable_gpu:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

# Global monitor instance
_global_monitor = None

def get_monitor() -> SystemMonitor:
    """Get or create global monitor"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = SystemMonitor()
    return _global_monitor

if __name__ == "__main__":
    # Demo
    import os
    monitor = SystemMonitor(process_pid=os.getpid())
    
    print("Monitoring for 10 seconds...")
    for i in range(10):
        metrics = monitor.get_metrics()
        monitor.print_metrics(metrics)
        time.sleep(1)
    
    monitor.cleanup()
