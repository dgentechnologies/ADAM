"""Exercise Windows worker dispatch with driver responses, without touching a display."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from hardware import WindowsHardwareWorker


@pytest.mark.parametrize("values, expected", [([None, 45], 45), ([0], 0), (80, 80)])
def test_brightness_reads_skip_unsupported_displays(values, expected):
    driver = SimpleNamespace(get_brightness=Mock(return_value=values))
    worker = object.__new__(WindowsHardwareWorker)
    with patch.dict(sys.modules, {"screen_brightness_control": driver}):
        assert worker._execute("get_bri", ()) == expected


def test_brightness_write_returns_confirmed_value_including_zero():
    driver = SimpleNamespace(set_brightness=Mock(return_value=[None, 0]),
                             get_brightness=Mock(side_effect=AssertionError("Unrelated read")))
    worker = object.__new__(WindowsHardwareWorker)
    with patch.dict(sys.modules, {"screen_brightness_control": driver}):
        assert worker._execute("set_bri", (0,)) == 0
    driver.set_brightness.assert_called_once_with(0, no_return=False)


@pytest.mark.parametrize("values", [[], [None, None], None])
def test_brightness_write_without_acknowledgement_fails(values):
    driver = SimpleNamespace(set_brightness=Mock(return_value=values))
    worker = object.__new__(WindowsHardwareWorker)
    with patch.dict(sys.modules, {"screen_brightness_control": driver}):
        with pytest.raises(RuntimeError, match="No display acknowledged"):
            worker._execute("set_bri", (50,))


def test_brightness_driver_failure_propagates():
    driver = SimpleNamespace(set_brightness=Mock(side_effect=RuntimeError("DDC/CI disabled")))
    worker = object.__new__(WindowsHardwareWorker)
    with patch.dict(sys.modules, {"screen_brightness_control": driver}):
        with pytest.raises(RuntimeError, match="DDC/CI disabled"):
            worker._execute("set_bri", (50,))
