"""Logger dedicado para debug de navegación y audio."""

import logging
import os
from pathlib import Path
from datetime import datetime

class NavigationLogger:
    """Singleton logger for navigation and audio debugging."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, session_dir: Path = None):
        if self._initialized:
            return
        
        # Use provided session directory or create new one
        if session_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            session_root = Path("logs") / f"session_{timestamp}"
        else:
            session_root = Path(session_dir)
        
        # Create telemetry subdirectory
        self.log_dir = session_root / "telemetry"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup loggers
        self._setup_logger("decision", "decision_engine.log")
        self._setup_logger("audio", "audio_system.log")
        self._setup_logger("renderer", "frame_renderer.log")
        self._setup_logger("routing", "audio_routing.log")
        
        self._initialized = True
    
    def _setup_logger(self, name: str, filename: str):
        """Setup individual logger with file handler."""
        logger = logging.getLogger(f"nav.{name}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler
        fh = logging.FileHandler(self.log_dir / filename, mode='w')
        fh.setLevel(logging.DEBUG)
        
        # Console handler (optional, for critical messages)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        
        # Format
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        setattr(self, name, logger)
    
    def close(self):
        """Close all handlers."""
        for name in ['decision', 'audio', 'renderer', 'routing']:
            logger = getattr(self, name, None)
            if logger:
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)


# Global instance
_nav_logger = None

def get_navigation_logger(session_dir: Path = None):
    """Get or create navigation logger instance."""
    global _nav_logger
    if _nav_logger is None:
        _nav_logger = NavigationLogger(session_dir=session_dir)
    return _nav_logger
