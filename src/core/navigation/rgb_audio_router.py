"""RGB-specific routing layer that formats decisions before audio output."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

from core.audio.audio_system import AudioSystem
from core.audio.navigation_audio_router import NavigationAudioRouter
from core.navigation.navigation_decision_engine import DecisionCandidate
from core.telemetry.loggers.navigation_logger import get_navigation_logger

if TYPE_CHECKING:
    from core.navigation.slam_audio_router import SlamAudioRouter


class RgbAudioRouter:
    """Translate decision candidates into spoken messages and route them."""

    def __init__(
        self,
        audio_system: AudioSystem,
        audio_router: Optional[NavigationAudioRouter] = None,
        slam_router: Optional["SlamAudioRouter"] = None,
    ) -> None:
        self.audio_system = audio_system
        self.audio_router = audio_router
        self.slam_router = slam_router

    def set_audio_router(self, audio_router: Optional[NavigationAudioRouter]) -> None:
        """Allow the coordinator to swap the shared audio router at runtime."""
        self.audio_router = audio_router
    
    def set_slam_router(self, slam_router: Optional["SlamAudioRouter"]) -> None:
        """Set reference to SLAM router for duplicate tracking."""
        self.slam_router = slam_router

    def route(self, candidate: DecisionCandidate) -> None:
        """Build the RGB message and send it through the appropriate channel."""
        metadata = dict(candidate.metadata or {})
        zone = str(candidate.nav_object.get("zone", "center")).strip()
        class_name = str((candidate.nav_object.get("class") or "")).strip()
        distance = str(candidate.nav_object.get("distance", "")).strip() or None

        # Determine if critical based on priority
        from core.audio.navigation_audio_router import EventPriority
        is_critical = (candidate.priority == EventPriority.CRITICAL or
                      (isinstance(candidate.priority, EventPriority) and
                       candidate.priority.value == 1))

        # Play spatial beep FIRST (instant directional feedback) with distance-based volume
        self.audio_system.play_spatial_beep(zone, is_critical=is_critical, distance=distance)
        
        # Build simplified TTS message (just object name)
        message = self._build_simple_message(candidate.nav_object)
        cooldown = self._parse_cooldown(metadata)
        
        # DEBUG: Log audio routing
        logger = get_navigation_logger().routing
        logger.info(f"RGB: {class_name} -> '{message}' (zone={zone}, critical={is_critical}, cooldown={cooldown}s)")

        router = self.audio_router
        if router is not None and hasattr(router, "enqueue_from_rgb"):
            if getattr(router, "_running", True):
                if hasattr(router, "set_source_cooldown"):
                    try:
                        router.set_source_cooldown("rgb", cooldown)
                    except Exception as err:
                        print(f"[WARN] Audio router cooldown failed: {err}")
                        router = None
                if router is not None:
                    try:
                        router.enqueue_from_rgb(
                            message=message,
                            priority=candidate.priority,
                            metadata=metadata,
                        )
                        logger.info(f"✓ Enqueued to NavigationAudioRouter")
                        # Notify SLAM router to avoid duplicates
                        if self.slam_router and class_name:
                            self.slam_router.register_rgb_announcement(class_name)
                        return
                    except Exception as err:
                        print(f"[WARN] Audio router enqueue failed, falling back to TTS: {err}")
                        router = None

        # Fallback to the legacy audio system when the router is unavailable.
        logger.info(f"Using fallback AudioSystem.queue_message()")
        self.audio_system.set_repeat_cooldown(cooldown)
        # Set very low announcement cooldown to allow beep + TTS in quick succession
        if hasattr(self.audio_system, "set_announcement_cooldown"):
            self.audio_system.set_announcement_cooldown(0.0)
        self.audio_system.queue_message(message)
        
        # Notify SLAM router even in fallback path
        if self.slam_router and class_name:
            self.slam_router.register_rgb_announcement(class_name)
    
    @staticmethod
    def _build_simple_message(nav_object: Dict[str, object]) -> str:
        """Build simple TTS message with just the object name."""
        from utils.config import Config

        class_name = str((nav_object.get("class") or "")).strip()
        return Config.AUDIO_OBJECT_LABELS.get(class_name, class_name.capitalize() if class_name else "Object")

    @staticmethod
    def _parse_cooldown(metadata: Dict[str, object]) -> float:
        raw_value = metadata.get("cooldown", 0.0)
        try:
            return float(raw_value or 0.0)
        except (TypeError, ValueError):
            return 0.0


__all__ = ["RgbAudioRouter"]
