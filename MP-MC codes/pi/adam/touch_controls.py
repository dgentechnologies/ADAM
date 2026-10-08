"""Persistent, nonblocking touch-to-laptop controls.

Consumes the existing ESP32 T/G packets without changing firmware. Raw presses
are classified once. Single taps keep fixed robot reactions; custom gestures
are holds on every pad and double/triple taps on Touch 3 only.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import tempfile
import time
import uuid

import laptop_actions

SENSORS = ("touch1", "touch2", "touch3", "touch4")
EVENTS = ("double", "triple", "hold")
CUSTOM_EVENTS = {sensor: EVENTS if sensor == "touch3" else ("hold",) for sensor in SENSORS}
SENSOR_INFO = {
    "touch1": {"label": "Left cheek", "gpio": 12},
    "touch2": {"label": "Right cheek", "gpio": 14},
    "touch3": {"label": "Stop / petting A", "gpio": 15},
    "touch4": {"label": "Petting B", "gpio": 2},
}
_EXCLUDED = {"dispatch_coding_task", "read_clipboard", "write_clipboard", "clipboard_paste"}


def validate_assignments(value):
    if not isinstance(value, dict) or set(value) - set(SENSORS):
        raise ValueError("Choose a valid touch sensor")
    result = {}
    for sensor, events in value.items():
        if not isinstance(events, dict) or set(events) - set(CUSTOM_EVENTS[sensor]):
            raise ValueError("Single taps are fixed. Only Touch 3 supports custom double/triple taps.")
        result[sensor] = {}
        for event, binding in events.items():
            if not isinstance(binding, dict) or set(binding) - {"action", "value"}:
                raise ValueError("Invalid touch assignment")
            name, argument = binding.get("action"), binding.get("value")
            if not isinstance(name, str):
                raise ValueError("Choose a supported touch action")
            if name == "none":
                argument = None
            else:
                if name not in laptop_actions.CANONICAL_ACTIONS or name in _EXCLUDED:
                    raise ValueError("This action cannot be assigned to a touch sensor")
                valid, argument, _ = laptop_actions.coerce(name, argument)
                if not valid:
                    raise ValueError("The touch action needs a valid value")
            result[sensor][event] = {"action": name, "value": argument}
    return result


class AssignmentStore:
    def __init__(self, path):
        self.path = Path(path)
        self.assignments = {}
        self.revision = 0
        self._lock = asyncio.Lock()

    async def load(self):
        def read():
            if not self.path.exists():
                return {}
            if self.path.stat().st_size > 16384:
                raise ValueError("Touch settings exceed the supported size")
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("version") not in (1, 2):
                raise ValueError("Unsupported touch settings version")
            assignments = data.get("assignments")
            if data.get("version") == 1 and isinstance(assignments, dict):
                assignments = {sensor: {event: binding for event, binding in events.items()
                                       if event in CUSTOM_EVENTS[sensor]}
                               for sensor, events in assignments.items()
                               if sensor in CUSTOM_EVENTS and isinstance(events, dict)}
            return validate_assignments(assignments)
        self.assignments = await asyncio.to_thread(read)

    def snapshot(self):
        return deepcopy(self.assignments)

    async def save(self, assignments):
        validated = validate_assignments(assignments)
        async with self._lock:
            def write():
                self.path.parent.mkdir(parents=True, exist_ok=True)
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                                     prefix=".touch-", delete=False) as handle:
                        temporary = Path(handle.name)
                        json.dump({"version": 2, "assignments": validated}, handle, separators=(",", ":"))
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.chmod(temporary, 0o600)
                    os.replace(temporary, self.path)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            await asyncio.to_thread(write)
            self.assignments = validated
            self.revision += 1
        return self.snapshot()


@dataclass
class _Pad:
    pressed: bool = False
    initialized: bool = False
    started: float = 0
    released: float = -100
    pending: float | None = None
    taps: int = 0
    reaction: int | None = None
    held: bool = False
    blocked: bool = False
    cycle: int = 0


class TouchClassifier:
    DOUBLE_S = 0.30
    HOLD_S = 0.65

    def __init__(self, assignments, emit, legacy, protected=lambda sensor: False):
        self.assignments, self.emit, self.legacy, self.protected = assignments, emit, legacy, protected
        self.pads = {sensor: _Pad() for sensor in SENSORS}
        self.legacy_seen = {}

    def feed(self, states, now):
        if not isinstance(states, (list, tuple)) or len(states) != 4 or any(x not in (0, 1) for x in states):
            return
        self.tick(now)
        for sensor, pressed in zip(SENSORS, states):
            pad = self.pads[sensor]
            if not pad.initialized:
                pad.initialized = True
                if pressed:
                    pad.blocked = True  # Never act on a pad already held at startup.
                    pad.pressed = True
                    continue
            if bool(pressed) == pad.pressed:
                continue
            pad.pressed = bool(pressed)
            if pressed:
                pad.cycle += 1
                pad.started, pad.held = now, False
                if pad.pending is None or now - pad.pending > self.DOUBLE_S:
                    pad.taps = 0
                pad.reaction = None
                pad.blocked = bool(self.protected(sensor))
                if pad.blocked:
                    pad.pending, pad.taps = None, 0
            else:
                pad.released = now
                if pad.blocked or pad.held:
                    pad.pending, pad.taps = None, 0
                elif sensor == "touch3":
                    pad.taps += 1
                    if pad.taps == 3:
                        pad.pending, pad.taps = None, 0
                        self._event(sensor, "triple")
                    else:
                        pad.pending = now
                else:
                    self._fixed_tap(sensor)

    def tick(self, now):
        for sensor, pad in self.pads.items():
            if (pad.pressed or pad.pending is not None) and self.protected(sensor):
                pad.blocked, pad.pending, pad.taps = True, None, 0
            if pad.pressed and not pad.blocked and not pad.held and now - pad.started >= self.HOLD_S:
                pad.held, pad.pending, pad.taps = True, None, 0
                self._event(sensor, "hold")
            if not pad.pressed and pad.pending is not None and now - pad.pending >= self.DOUBLE_S:
                taps = pad.taps
                pad.pending, pad.taps = None, 0
                if taps == 2:
                    self._event(sensor, "double")
                else:
                    self._fixed_tap(sensor)

    def _fixed_tap(self, sensor):
        if not self.assignments().get(sensor):
            return  # Unconfigured firmware gestures were already forwarded.
        code = self.pads[sensor].reaction or {"touch1": 1, "touch2": 1, "touch3": 3, "touch4": 2}[sensor]
        self.legacy(code)

    def _event(self, sensor, event):
        if event not in CUSTOM_EVENTS.get(sensor, ()):
            return
        configured = self.assignments()
        if sensor not in configured or not configured[sensor]:
            return  # Firmware's unchanged gesture supplies the default.
        binding = configured[sensor].get(event)
        if binding is None:
            return
        self.emit(sensor, event, binding)

    def firmware_gesture(self, code, now):
        group = {1: ("touch1", "touch2"), 2: ("touch3", "touch4"), 3: ("touch3",)}.get(code)
        if not group:
            return
        configured = self.assignments()
        active = [s for s in group if self.pads[s].pressed or now - self.pads[s].released < 0.2]
        # T packets precede G packets on the wire. If no corresponding press
        # arrived, discard the stale gesture rather than inventing an action.
        if not active:
            return
        chord = code == 2 and len(active) == 2
        if chord:
            for sensor in active:
                self.pads[sensor].blocked = True
                self.pads[sensor].pending, self.pads[sensor].taps = None, 0
        for sensor in active:
            self.pads[sensor].reaction = code
        available = [s for s in active if not configured.get(s) or self.pads[s].blocked or self.protected(s)]
        if not available:
            return
        signature = tuple((s, self.pads[s].cycle) for s in available)
        if self.legacy_seen.get(code) == signature:
            return  # Firmware sends STOP/slap repeatedly while a pad is held.
        self.legacy_seen[code] = signature
        self.legacy(code)

    def reset_pending(self):
        for pad in self.pads.values():
            pad.pending = None
            pad.taps, pad.reaction = 0, None
            pad.blocked = pad.pressed


class TouchRuntime:
    def __init__(self, link, store, dispatch, broadcast, alarm_ringing=lambda: False):
        self.link, self.store, self.dispatch, self.broadcast = link, store, dispatch, broadcast
        self.alarm_ringing = alarm_ringing
        self.priority = None
        self.priority_token = None
        self.legacy = queue.Queue(maxsize=20)
        self.actions = asyncio.Queue(maxsize=16)
        self.tasks = []
        self.sequence = 0
        self.runtime_id = uuid.uuid4().hex
        self.classifier = TouchClassifier(lambda: self.store.assignments, self._enqueue,
                                          self._legacy, self._protected)

    def _protected(self, sensor):
        try:
            if sensor != "touch4" and self.alarm_ringing():
                return True
            return bool(self.priority and self.priority(sensor))
        except Exception:
            return True  # Keep legacy controls if a safety-state provider fails.

    def _legacy(self, code):
        try:
            self.legacy.put_nowait((time.monotonic(), code))
        except queue.Full:
            self.legacy.get_nowait()
            self.legacy.put_nowait((time.monotonic(), code))

    def next_legacy(self):
        while True:
            stamp, code = self.legacy.get_nowait()
            if time.monotonic() - stamp < 2.0:
                return code

    def _enqueue(self, sensor, event, binding):
        self.sequence += 1
        payload = {"type": "touch", "id": f"{self.runtime_id}-{self.sequence}",
                   "sensor": sensor, "event": event, "action": binding["action"], "handled_by": "pi"}
        try:
            self.actions.put_nowait((self.store.revision, time.monotonic(), payload, deepcopy(binding)))
        except asyncio.QueueFull:
            pass  # Never let a stalled laptop grow memory or stall the audio loop.

    def start(self):
        if self.tasks:
            return
        self.tasks = [asyncio.create_task(self._read(), name="touch-input"),
                      asyncio.create_task(self._dispatch(), name="touch-laptop")]

    async def stop(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks = []
        self.classifier.reset_pending()
        while not self.actions.empty():
            self.actions.get_nowait()
            self.actions.task_done()
        while not self.legacy.empty():
            self.legacy.get_nowait()

    async def _read(self):
        revision = self.store.revision
        while True:
            if revision != self.store.revision:
                self.classifier.reset_pending()
                revision = self.store.revision
            if self.link.connected:
                for _ in range(24):
                    try:
                        self.classifier.feed(self.link.touch_q.get_nowait(), time.monotonic())
                    except queue.Empty:
                        break
                self.classifier.tick(time.monotonic())
                for _ in range(24):
                    try:
                        self.classifier.firmware_gesture(self.link.gesture_q.get_nowait(), time.monotonic())
                    except queue.Empty:
                        break
            else:
                self.classifier.reset_pending()
            await asyncio.sleep(0.02)

    async def _dispatch(self):
        while True:
            revision, created, payload, binding = await self.actions.get()
            try:
                if (revision != self.store.revision or time.monotonic() - created > 2.0
                        or self._protected(payload["sensor"])):
                    continue
                if binding["action"] == "none":
                    payload["status"] = "ignored"
                else:
                    result = await asyncio.to_thread(self.dispatch, binding["action"], binding.get("value"))
                    payload["status"] = "ok" if isinstance(result, dict) and result.get("status") == "ok" else "error"
                await asyncio.wait_for(self.broadcast(payload), timeout=1.0)
            except asyncio.CancelledError:
                raise
            except Exception:
                # The laptop/telemetry channel cannot stop the local touch loop.
                pass
            finally:
                self.actions.task_done()


_runtime = None


async def initialize_touch_controls(link):
    global _runtime
    if _runtime is not None:
        return _runtime
    from config import BASE_DIR
    from laptop_agent_client import laptop_control_sync
    from ws_server import ws_broadcast
    import scheduler
    store = AssignmentStore(Path(BASE_DIR) / "touch_assignments.json")
    try:
        await store.load()
    except Exception:
        print("  Touch settings could not be loaded; original gestures remain active.")
    _runtime = TouchRuntime(link, store, laptop_control_sync, ws_broadcast, scheduler.alarm_ringing)
    _runtime.start()
    return _runtime


async def stop_touch_controls():
    global _runtime
    if _runtime is not None:
        await _runtime.stop()
        _runtime = None


def capabilities():
    return {"touch_assignments": _runtime is not None, "touch_events": _runtime is not None,
            "touch_gesture_policy": 2 if _runtime is not None else 0}


async def save_assignments(assignments):
    if _runtime is None:
        raise RuntimeError("Touch controls have not started")
    return await _runtime.store.save(assignments)


def read_assignments():
    return _runtime.store.snapshot() if _runtime else {}


def next_legacy():
    if _runtime is None:
        raise queue.Empty()
    return _runtime.next_legacy()


def set_priority(callback):
    token = object()
    if _runtime:
        _runtime.priority, _runtime.priority_token = callback, token
    return token


def clear_priority(token):
    if _runtime and _runtime.priority_token is token:
        _runtime.priority = _runtime.priority_token = None
