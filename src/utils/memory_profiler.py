"""Memory profiling utilities using tracemalloc"""

import tracemalloc
import time
from typing import Optional, List, Dict, Any
from pathlib import Path
import json


class MemoryProfiler:
    """
    Lightweight memory profiler using tracemalloc.
    
    Captures periodic snapshots and logs top allocations.
    """
    
    def __init__(self, enabled: bool = True, snapshot_interval: float = 30.0):
        """
        Args:
            enabled: Whether to enable profiling
            snapshot_interval: Seconds between snapshots
        """
        self.enabled = enabled
        self.snapshot_interval = snapshot_interval
        self.snapshots: List[tracemalloc.Snapshot] = []
        self.last_snapshot_time = 0.0
        self.start_time = 0.0
        
        if self.enabled:
            tracemalloc.start(25)  # Track 25 frames deep
            self.start_time = time.time()
            print(f"[MEMORY PROFILER] Started (snapshot interval: {snapshot_interval}s)")
    
    def maybe_take_snapshot(self) -> bool:
        """
        Take snapshot if interval elapsed.
        
        Returns:
            True if snapshot was taken
        """
        if not self.enabled:
            return False
        
        now = time.time()
        if now - self.last_snapshot_time >= self.snapshot_interval:
            snapshot = tracemalloc.take_snapshot()
            self.snapshots.append(snapshot)
            self.last_snapshot_time = now
            
            elapsed = now - self.start_time
            print(f"[MEMORY PROFILER] Snapshot {len(self.snapshots)} at {elapsed:.1f}s")
            
            return True
        
        return False
    
    def get_top_stats(self, limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        """
        Get top memory allocations from latest snapshot.
        
        Returns:
            List of dicts with filename, lineno, size_mb
        """
        if not self.enabled or not self.snapshots:
            return None
        
        snapshot = self.snapshots[-1]
        top_stats = snapshot.statistics('lineno')
        
        results = []
        for stat in top_stats[:limit]:
            results.append({
                "filename": stat.traceback.format()[0] if stat.traceback else "unknown",
                "size_mb": stat.size / 1024 / 1024,
                "count": stat.count
            })
        
        return results
    
    def get_diff_stats(self, limit: int = 10) -> Optional[List[Dict[str, Any]]]:
        """
        Compare last two snapshots to see growth.
        
        Returns:
            List of dicts with filename, size_diff_mb, count_diff
        """
        if not self.enabled or len(self.snapshots) < 2:
            return None
        
        snapshot1 = self.snapshots[-2]
        snapshot2 = self.snapshots[-1]
        
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        results = []
        for stat in top_stats[:limit]:
            results.append({
                "filename": stat.traceback.format()[0] if stat.traceback else "unknown",
                "size_diff_mb": stat.size_diff / 1024 / 1024,
                "count_diff": stat.count_diff
            })
        
        return results
    
    def log_to_file(self, output_dir: Path) -> None:
        """
        Save profiling results to JSON file.
        
        Args:
            output_dir: Directory to save profile.json
        """
        if not self.enabled or not self.snapshots:
            return
        
        output_file = output_dir / "memory_profile.json"
        
        data = {
            "start_time": self.start_time,
            "snapshot_count": len(self.snapshots),
            "top_allocations": self.get_top_stats(20),
            "growth": self.get_diff_stats(20) if len(self.snapshots) >= 2 else None
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[MEMORY PROFILER] Saved results to {output_file}")
    
    def print_summary(self) -> None:
        """Print summary of memory usage to console"""
        if not self.enabled or not self.snapshots:
            return
        
        print("\n" + "="*70)
        print("MEMORY PROFILING SUMMARY")
        print("="*70)
        
        # Current memory usage
        current = tracemalloc.get_traced_memory()
        print(f"\nCurrent traced memory:")
        print(f"  Current: {current[0] / 1024 / 1024:.1f} MB")
        print(f"  Peak: {current[1] / 1024 / 1024:.1f} MB")
        
        # Top allocations
        print(f"\nTop 10 memory allocations:")
        top_stats = self.get_top_stats(10)
        if top_stats:
            for i, stat in enumerate(top_stats, 1):
                print(f"  {i}. {stat['size_mb']:.2f} MB - {stat['filename']}")
        
        # Growth between snapshots
        if len(self.snapshots) >= 2:
            print(f"\nMemory growth (last 2 snapshots):")
            diff_stats = self.get_diff_stats(10)
            if diff_stats:
                for i, stat in enumerate(diff_stats, 1):
                    if stat['size_diff_mb'] > 0:
                        print(f"  {i}. +{stat['size_diff_mb']:.2f} MB - {stat['filename']}")
        
        print("="*70 + "\n")
    
    def stop(self) -> None:
        """Stop profiling and cleanup"""
        if self.enabled:
            tracemalloc.stop()
            print("[MEMORY PROFILER] Stopped")
