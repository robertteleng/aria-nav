"""Dedicated logger for depth estimation debugging"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional


class DepthLogger:
    """Writes depth-related logs to a dedicated file in the session directory"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, session_dir: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, session_dir: Optional[str] = None):
        if self._initialized:
            return
            
        if session_dir:
            # Use provided session directory
            self.log_dir = Path(session_dir)
        else:
            # Fallback: create in project root logs/
            project_root = Path(__file__).resolve().parents[4]
            log_dir = project_root / "logs"
            log_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.log_dir = log_dir / f"session_{timestamp}_depth_only"
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create depth log files directly in session directory
        self.log_file = self.log_dir / "depth_debug.log"
        
        # Setup Python logger for metrics
        self.metrics_logger = logging.getLogger("depth.metrics")
        self.metrics_logger.setLevel(logging.INFO)
        self.metrics_logger.handlers.clear()
        
        handler = logging.FileHandler(self.log_dir / "depth_metrics.log")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        self.metrics_logger.addHandler(handler)
        
        DepthLogger._initialized = True
        
        self.log(f"=" * 80)
        self.log(f"DEPTH ESTIMATION DEBUG LOG")
        self.log(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f"Log directory: {self.log_dir}")
        self.log(f"=" * 80)
        self.log("")
    
    def log(self, message: str):
        """Write message to log file and print to console"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_line = f"[{timestamp}] {message}"
        
        # Write to file
        with open(self.log_file, "a") as f:
            f.write(log_line + "\n")
        
        # Also print to console with distinctive marker
        print(f"[DEPTH] {message}")
    
    def log_metric(self, metric_data: dict):
        """Log a metric entry as JSON"""
        metric_data['timestamp'] = datetime.now().timestamp()
        self.metrics_logger.info(json.dumps(metric_data))
    
    def section(self, title: str):
        """Create a section header"""
        self.log("")
        self.log("-" * 60)
        self.log(f"  {title}")
        self.log("-" * 60)


# Singleton instance
_depth_logger = None

def get_depth_logger(session_dir: Optional[str] = None) -> DepthLogger:
    """Get or create the global depth logger instance"""
    global _depth_logger
    if _depth_logger is None:
        _depth_logger = DepthLogger(session_dir=session_dir)
    return _depth_logger

def init_depth_logger(session_dir: str):
    """Initialize depth logger with a specific session directory"""
    global _depth_logger
    _depth_logger = DepthLogger(session_dir=session_dir)
    return _depth_logger
