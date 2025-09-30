"""Tests for DeviceManager using stubbed Aria SDK components."""

from __future__ import annotations

import types

import pytest

from core.hardware import device_manager as dm_module
from core.hardware.device_manager import DeviceManager


class StubStreamingClient:
    def __init__(self) -> None:
        self.observer = None
        self.subscribed = False

    def set_streaming_client_observer(self, observer):
        self.observer = observer

    def subscribe(self):
        self.subscribed = True

    def unsubscribe(self):
        self.subscribed = False


class StubStreamingManager:
    def __init__(self) -> None:
        self.streaming_config = None
        self.started = False
        self.stopped = False
        self.streaming_client = StubStreamingClient()

    def start_streaming(self):
        self.started = True

    def stop_streaming(self):
        self.stopped = True

    def sensors_calibration(self):
        return "{}"


class StubDevice:
    def __init__(self):
        self.streaming_manager = StubStreamingManager()


class StubDeviceClient:
    def __init__(self):
        self.config = None
        self.disconnect_calls = []

    def set_client_config(self, config):
        self.config = config

    def connect(self):
        return StubDevice()

    def disconnect(self, device):
        self.disconnect_calls.append(device)


class StubDeviceClientConfig:
    def __init__(self):
        self.ip_v4_address = None


class StubSecurityOptions:
    def __init__(self):
        self.use_ephemeral_certs = False


class StubStreamingConfig:
    def __init__(self):
        self.profile_name = None
        self.streaming_interface = None
        self.security_options = StubSecurityOptions()


class StubStreamingInterface:
    WifiStation = "wifi"
    Usb = "usb"


@pytest.fixture()
def stubbed_env(monkeypatch: pytest.MonkeyPatch):
    aria_stub = types.SimpleNamespace(
        DeviceClient=StubDeviceClient,
        DeviceClientConfig=StubDeviceClientConfig,
        StreamingConfig=StubStreamingConfig,
        StreamingInterface=StubStreamingInterface,
    )
    monkeypatch.setattr(dm_module, "aria", aria_stub)

    def fake_calibration(json_str):
        class _Calib:
            def get_camera_calib(self, name):
                return {"camera": name}
        return _Calib()

    monkeypatch.setattr(dm_module, "device_calibration_from_json_string", fake_calibration)
    return aria_stub


def test_connect_wifi_configures_device(monkeypatch: pytest.MonkeyPatch, stubbed_env):
    monkeypatch.setattr(dm_module.Config, "STREAMING_INTERFACE", "wifi", raising=False)
    monkeypatch.setattr(dm_module.Config, "STREAMING_WIFI_DEVICE_IP", "10.0.0.5", raising=False)

    manager = DeviceManager()
    manager.connect()

    assert isinstance(manager.device_client, StubDeviceClient)
    assert manager.device_client.config.ip_v4_address == "10.0.0.5"
    assert isinstance(manager.device, StubDevice)


def test_start_streaming_sets_profile_and_returns_calibration(monkeypatch: pytest.MonkeyPatch, stubbed_env):
    monkeypatch.setattr(dm_module.Config, "STREAMING_INTERFACE", "usb", raising=False)
    monkeypatch.setattr(dm_module.Config, "STREAMING_PROFILE_USB", "profile28", raising=False)

    manager = DeviceManager()
    manager.connect()
    calib = manager.start_streaming()

    streaming_config = manager.streaming_manager.streaming_config
    assert streaming_config.profile_name == "profile28"
    assert streaming_config.security_options.use_ephemeral_certs is True
    assert manager.streaming_manager.started is True
    assert calib == {"camera": "camera-rgb"}


def test_cleanup_calls_disconnect(monkeypatch: pytest.MonkeyPatch, stubbed_env):
    monkeypatch.setattr(dm_module.Config, "STREAMING_INTERFACE", "usb", raising=False)
    manager = DeviceManager()
    manager.connect()
    manager.start_streaming()
    manager.subscribe()
    manager.cleanup()

    assert manager.streaming_manager.stopped is True
    assert not manager.streaming_manager.streaming_client.subscribed
    assert manager.device_client.disconnect_calls
