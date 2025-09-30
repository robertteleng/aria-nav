"""RGB-specific routing layer that formats decisions before audio output."""

from __future__ import annotations

from typing import Dict, Optional

from core.audio.audio_system import AudioSystem
from core.audio.navigation_audio_router import NavigationAudioRouter
from core.navigation.navigation_decision_engine import DecisionCandidate


class RgbAudioRouter:
    """Translate decision candidates into spoken messages and route them."""

    def __init__(
        self,
        audio_system: AudioSystem,
        audio_router: Optional[NavigationAudioRouter] = None,
    ) -> None:
        self.audio_system = audio_system
        self.audio_router = audio_router

    def set_audio_router(self, audio_router: Optional[NavigationAudioRouter]) -> None:
        """Allow the coordinator to swap the shared audio router at runtime."""
        self.audio_router = audio_router

    def route(self, candidate: DecisionCandidate) -> None:
        """Build the RGB message and send it through the appropriate channel."""
        metadata = dict(candidate.metadata or {})
        message = self._build_rgb_message(candidate.nav_object)
        cooldown = self._parse_cooldown(metadata)

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
                        return
                    except Exception as err:
                        print(f"[WARN] Audio router enqueue failed, falling back to TTS: {err}")
                        router = None

        # Fallback to the legacy audio system when the router is unavailable.
        self.audio_system.set_repeat_cooldown(cooldown)
        if hasattr(self.audio_system, "set_announcement_cooldown"):
            self.audio_system.set_announcement_cooldown(max(0.0, cooldown * 0.5))
        self.audio_system.queue_message(message)

    @staticmethod
    def _parse_cooldown(metadata: Dict[str, object]) -> float:
        raw_value = metadata.get("cooldown", 0.0)
        try:
            return float(raw_value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _build_rgb_message(nav_object: Dict[str, object]) -> str:
        """Mirror the previous phrasing logic in a reusable static helper."""
        zone = str(nav_object.get("zone", "")).strip() or "center"
        distance = str(nav_object.get("distance", "")).strip().lower()
        class_name = (nav_object.get("class") or "").strip()

        speech_labels = {
            "person": "person",
            "car": "car",
            "truck": "truck",
            "bus": "bus",
            "bicycle": "bicycle",
            "motorcycle": "motorcycle",
            "motorbike": "motorbike",
            "stop sign": "stop sign",
            "traffic light": "traffic light",
            "chair": "chair",
            "door": "door",
            "stairs": "stairs",
        }
        name = speech_labels.get(class_name, class_name if class_name else "object")

        zone_english = {
            "left": "left side",
            "center": "straight ahead",
            "right": "right side",
        }
        zone_text = zone_english.get(zone, zone or "straight ahead")

        distance_english = {
            "cerca": "very close",
            "muy_cerca": "very close",
            "very_close": "very close",
            "close": "very close",
            "medio": "at medium distance",
            "medium": "at medium distance",
            "lejos": "far",
            "far": "far",
        }
        distance_text = distance_english.get(distance, distance or "at medium distance")

        priority = float(nav_object.get("priority", 0.0) or 0.0)
        if distance_text in {"very close", "close"} and priority >= 9:
            return f"Warning, {name} {distance_text} on the {zone_text}"
        return f"{name.capitalize()} on the {zone_text}, {distance_text}"


__all__ = ["RgbAudioRouter"]
