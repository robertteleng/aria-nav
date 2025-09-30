"""Unit tests for the AudioSystem class."""

from __future__ import annotations

import pytest

import core.audio.audio_system as audio_module
from core.audio.audio_system import AudioSystem


class SyncThread:
    def __init__(self, target, daemon=False):
        self.target = target
        self.daemon = daemon

    def start(self):
        self.target()


class FakeClock:
    def __init__(self, start: float = 100.0) -> None:
        self.value = start

    def time(self) -> float:
        return self.value

    def advance(self, delta: float) -> None:
        self.value += delta


@pytest.fixture()
def audio_env(monkeypatch: pytest.MonkeyPatch):
    clock = FakeClock()
    popen_calls: list[list[str]] = []

    monkeypatch.setattr(audio_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(audio_module.shutil, "which", lambda _: "/usr/bin/say")
    monkeypatch.setattr(audio_module.subprocess, "Popen", lambda cmd: popen_calls.append(cmd))
    monkeypatch.setattr(audio_module.threading, "Thread", SyncThread)
    monkeypatch.setattr(audio_module.time, "time", clock.time)
    monkeypatch.setattr(audio_module.time, "sleep", lambda _: None)

    system = AudioSystem()
    return system, clock, popen_calls


def test_speak_async_respects_repeat_cooldown(audio_env) -> None:
    system, clock, popen_calls = audio_env

    assert system.speak_async("hello") is True
    assert popen_calls and popen_calls[0][-1] == "hello"

    # Immediate repetition blocked by repeat cooldown.
    assert system.speak_async("hello") is False

    # Advance beyond repeat cooldown and try again.
    clock.advance(system.repeat_cooldown + 0.1)
    assert system.speak_async("hello") is True
    assert len(popen_calls) == 2


def test_speak_async_force_bypasses_cooldown(audio_env) -> None:
    system, clock, _ = audio_env

    assert system.speak_async("warning") is True
    assert system.speak_async("warning") is False
    assert system.speak_async("warning", force=True) is True


def test_setters_and_queue(audio_env) -> None:
    system, _, _ = audio_env

    system.set_repeat_cooldown("invalid")
    assert system.repeat_cooldown == 2.0  # unchanged

    system.set_repeat_cooldown(0.05)
    assert system.repeat_cooldown == pytest.approx(0.1)

    system.set_announcement_cooldown(-1)
    assert system.announcement_cooldown == 0.0

    system.update_frame_dimensions(640, 480)
    assert system.frame_width == 640
    assert system.frame_height == 480

    assert system.queue_message("") is False
