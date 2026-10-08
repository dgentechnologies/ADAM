"""Verified LAN connection to an ADAM Pi.

The desktop's command server and this client are separate channels. A socket
opening (or a machine answering ping) is never enough to identify a robot.
Only the Pi's versioned HTTP identity response establishes a data connection.
"""

from __future__ import annotations

import ipaddress
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import requests


_HOST_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", re.I)
_MAX_RESPONSE = 1024 * 1024
_HTTP_TIMEOUT = (2.0, 3.0)
_POLL_INTERVAL = 8.0
_SUPPORTED_API = {1}
_EMOTIONS = frozenset({"happy", "sad", "surprised", "angry", "thinking", "excited",
                       "love", "blush", "confused", "smug", "sleep", "rizz", "panic",
                       "shy", "reconnecting"})
_HEAD_GESTURES = frozenset({"none", "nod"})
_TELEMETRY_DEFAULTS = {"emotion": "", "head": "none", "speaking": False,
                       "listening": False, "telemetry_updated_at": 0.0, "last_touch": None}


def _host(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Enter ADAM's hostname or IP address.")
    value = value.strip().lower().rstrip(".")
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    if not value or len(value) > 253 or any(c in value for c in "/\\@?#%"):
        raise ValueError("Use a hostname or IP address without a URL, path or port.")
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        if not all(_HOST_LABEL.fullmatch(label) for label in value.split(".")):
            raise ValueError("Use a valid hostname or IP address without a port.") from None
        return value
    if address.is_unspecified or address.is_multicast:
        raise ValueError("Use ADAM's own hostname or IP address.")
    return address.compressed


def _port(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{label} must be a number from 1 to 65535.")
    if isinstance(value, str) and not value.strip().isdigit():
        raise ValueError(f"{label} must be a number from 1 to 65535.")
    number = int(value)
    if not 1 <= number <= 65535:
        raise ValueError(f"{label} must be a number from 1 to 65535.")
    return number


@dataclass(frozen=True)
class _Target:
    host: str
    sync_port: int
    ws_port: int
    token: str = field(default="", repr=False)

    @property
    def authority(self) -> str:
        return f"[{self.host}]" if ":" in self.host else self.host

    @property
    def identity(self) -> tuple[str, int]:
        return self.host, self.sync_port


@dataclass
class _Run:
    target: _Target
    revision: int
    stop: threading.Event = field(default_factory=threading.Event)
    verified: threading.Event = field(default_factory=threading.Event)
    threads: list[threading.Thread] = field(default_factory=list)
    write_denied: bool = False


def _target(config: dict[str, Any], saved: dict[str, Any]) -> _Target:
    saved_host = saved.get("pi_host") or saved.get("pi_ip") or "adam-pi.local"
    host = _host(config.get("host", config.get("pi_host", config.get("pi_ip", saved_host))))
    port = _port(config.get("sync_port", config.get("pi_sync_port", saved.get("pi_sync_port", 8766))), "Data port")
    ws_port = _port(config.get("ws_port", config.get("pi_ws_port", saved.get("pi_ws_port", 8765))), "Live status port")
    token = ""
    if "sync_token" in config:
        token = config["sync_token"]
    else:
        # A legacy saved token belongs only to its saved endpoint. Never carry
        # it to a newly entered host/port, including a failed connection probe.
        token_host = saved.get("sync_token_host") or saved_host
        token_port = saved.get("sync_token_port", saved.get("pi_sync_port", 8766))
        if (host, port) == (_host(token_host), _port(token_port, "Data port")):
            token = saved.get("sync_token", "")
    if not isinstance(token, str) or len(token) > 4096 or any(ord(c) < 32 or ord(c) > 126 for c in token):
        raise ValueError("Enter a valid connection key.")
    return _Target(host, port, ws_port, token.strip())


def _empty_status(reason: str = "Connect ADAM after completing setup in the mobile app.") -> dict[str, Any]:
    return {
        "connected": False, "data_connected": False, "telemetry_connected": False,
        "host": "", "device_name": "ADAM", "reason": reason, "read_only": True,
        "capabilities": {}, "last_seen": 0.0, "paired": False,
        **_TELEMETRY_DEFAULTS,
    }


def _capabilities(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("capabilities", {})
    if isinstance(value, list):
        return {item: True for item in value if isinstance(item, str) and len(item) <= 80}
    if isinstance(value, dict):
        return {str(key)[:80]: item for key, item in value.items()
                if isinstance(key, str) and isinstance(item, (bool, int, str, dict))}
    return {}


class ConnectionService:
    """A nonblocking monitor with explicit synchronous connection actions.

    ``update_settings`` receives a patch, never a complete settings replacement.
    ``probe``, ``connect`` and ``call`` are bounded network operations intended
    for the HTTP backend's worker threads. ``start``/``stop`` do not wait for
    network I/O, so window and tray lifecycle remain responsive.
    """

    def __init__(self, read_settings: Callable[[], dict[str, Any]],
                 update_settings: Callable[[dict[str, Any]], Any],
                 on_event: Callable[[dict[str, Any]], Any] | None = None):
        self._read_settings = read_settings
        self._update_settings = update_settings
        self._on_event = on_event
        self._lock = threading.RLock()
        self._revision = 0
        self._run: _Run | None = None
        self._state = _empty_status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state))

    def _configuration(self, config: dict[str, Any] | None = None) -> _Target:
        if config is not None and not isinstance(config, dict):
            raise ValueError("Connection settings must be an object.")
        return _target(config or {}, self._read_settings())

    @staticmethod
    def _request(target: _Target, method: str, path: str,
                 body: dict[str, Any] | None = None) -> tuple[dict | None, str | None]:
        if method not in {"GET", "POST", "PUT"} or not re.fullmatch(r"/api/[a-z0-9_/-]+", path):
            return None, "This ADAM request is not supported."
        headers = {"Accept": "application/json"}
        # Identity checks never disclose the private write key.
        if target.token and path != "/api/ping":
            headers["X-ADAM-Token"] = target.token
        try:
            with requests.Session() as session:
                session.trust_env = False  # LAN traffic must not use HTTP_PROXY.
                with session.request(method, f"http://{target.authority}:{target.sync_port}{path}",
                                     json=body if method != "GET" else None,
                                     headers=headers, timeout=_HTTP_TIMEOUT,
                                     allow_redirects=False, stream=True) as response:
                    if 300 <= response.status_code < 400:
                        return None, "ADAM redirected the request. Check the connection address."
                    chunks = bytearray()
                    deadline = time.monotonic() + 8.0
                    for chunk in response.iter_content(16384):
                        chunks.extend(chunk)
                        if len(chunks) > _MAX_RESPONSE or time.monotonic() > deadline:
                            return None, "ADAM returned an oversized or incomplete response."
                    try:
                        payload = json.loads(chunks)
                    except (ValueError, UnicodeError):
                        return None, "The device did not return an ADAM data response."
                    if not isinstance(payload, dict):
                        return None, "The device did not return an ADAM data response."
                    if response.status_code >= 400 or payload.get("ok") is False:
                        if response.status_code in (401, 403):
                            return None, "ADAM did not authorize this change. Check its connection key."
                        reason = payload.get("error") or payload.get("reason")
                        if isinstance(reason, str):
                            if target.token:
                                reason = reason.replace(target.token, "[redacted]")
                            return None, reason[:300]
                        return None, f"ADAM could not complete the request (HTTP {response.status_code})."
                    return payload, None
        except (requests.RequestException, OSError):
            return None, "ADAM did not respond. Check that both devices are on the same network."

    def _identity(self, target: _Target) -> tuple[dict | None, str | None]:
        payload, error = self._request(target, "GET", "/api/ping")
        if error:
            return None, error
        if (payload.get("ok") is not True or payload.get("app") != "adam"
                or payload.get("kind") != "pi"):
            return None, "This address did not identify itself as ADAM."
        if type(payload.get("api")) is not int or payload["api"] not in _SUPPORTED_API:
            return None, "This ADAM data version is not supported by this desktop app."
        return payload, None

    @staticmethod
    def _identified(target: _Target, payload: dict) -> dict[str, Any]:
        state = _empty_status("")
        state.update(connected=True, data_connected=True, host=target.host,
                     device_name=str(payload.get("device_name") or payload.get("name") or "ADAM")[:100],
                     read_only=bool(payload.get("readonly", True)) or not bool(target.token),
                     capabilities=_capabilities(payload), last_seen=time.time())
        return state

    def _refresh_identity(self, run: _Run, payload: dict) -> None:
        # HTTP identity refreshes do not describe the face. Preserve the last
        # verified live event while updating reachability and capabilities.
        telemetry = {key: self._state[key] for key in
                     ("telemetry_connected", *_TELEMETRY_DEFAULTS)}
        self._state.update(self._identified(run.target, payload))
        self._state.update(telemetry, paired=True)
        if run.write_denied:
            self._state.update(read_only=True,
                               reason="ADAM did not authorize changes. Check its connection key.")

    def probe(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Inspect a proposed endpoint without saving it or changing live state."""
        try:
            target = self._configuration(config)
            payload, error = self._identity(target)
        except (ValueError, TypeError):
            return {"ok": False, **_empty_status("Enter a valid ADAM hostname and ports.")}
        if error:
            return {"ok": False, **_empty_status(error), "host": target.host}
        return {"ok": True, **self._identified(target, payload)}

    def connect(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            target = self._configuration(config)
        except (ValueError, TypeError) as error:
            return {"ok": False, **_empty_status(str(error))}
        with self._lock:
            revision = self._revision = self._revision + 1
        payload, error = self._identity(target)
        with self._lock:
            if revision != self._revision:
                return {"ok": False, **_empty_status("The connection attempt was cancelled.")}
            if error:
                return {"ok": False, **_empty_status(error), "host": target.host}
            try:
                self._update_settings({
                    "pi_host": target.host, "pi_ip": target.host,
                    "pi_sync_port": target.sync_port, "pi_ws_port": target.ws_port,
                    "sync_token": target.token, "sync_token_host": target.host,
                    "sync_token_port": target.sync_port, "paired": True,
                })
            except Exception:
                return {"ok": False, **_empty_status("The verified connection could not be saved. Please try again.")}
            if self._run:
                self._run.stop.set()
            self._state = self._identified(target, payload)
            self._state["paired"] = True
            self._launch(target, revision, verified=True)
            return {"ok": True, **self.status()}

    def disconnect(self) -> dict[str, Any]:
        with self._lock:
            self._revision += 1
            if self._run:
                self._run.stop.set()
                self._run = None
            try:
                self._update_settings({"paired": False, "sync_token": "",
                                       "sync_token_host": "", "sync_token_port": 8766})
            except Exception:
                self._state = _empty_status("Disconnected, but the saved connection could not be cleared.")
                return {"ok": False, **self.status()}
            self._state = _empty_status("ADAM is disconnected.")
            return {"ok": True, **self.status()}

    def start(self) -> None:
        with self._lock:
            if self._run and not self._run.stop.is_set():
                return
            try:
                saved = self._read_settings()
                if not saved.get("paired", False):
                    return
                target = _target({}, saved)
            except (ValueError, TypeError):
                self._state = _empty_status("The saved connection needs a valid hostname and ports.")
                return
            self._revision += 1
            self._state = _empty_status("Connecting to ADAM…")
            self._state.update(host=target.host, paired=True)
            self._launch(target, self._revision)

    def _launch(self, target: _Target, revision: int, verified: bool = False) -> None:
        run = self._run = _Run(target, revision)
        if verified:
            run.verified.set()
        for suffix, worker in (("data", self._monitor), ("live", self._telemetry)):
            thread = threading.Thread(target=worker, args=(run,), name=f"ADAM-{suffix}-{revision}", daemon=True)
            run.threads.append(thread)
            thread.start()

    def stop(self) -> None:
        with self._lock:
            self._revision += 1
            if self._run:
                self._run.stop.set()
                self._run = None
            self._state.update(connected=False, data_connected=False, telemetry_connected=False,
                               reason="Connection monitoring is stopped.", **_TELEMETRY_DEFAULTS)

    def _current(self, run: _Run) -> bool:
        return self._run is run and not run.stop.is_set()

    def _monitor(self, run: _Run) -> None:
        while not run.stop.is_set():
            payload, error = self._identity(run.target)
            with self._lock:
                if not self._current(run):
                    return
                if error:
                    run.verified.clear()
                    self._state.update(connected=False, data_connected=False, telemetry_connected=False,
                                       reason=error, **_TELEMETRY_DEFAULTS)
                else:
                    self._refresh_identity(run, payload)
                    run.verified.set()
            run.stop.wait(_POLL_INTERVAL)

    def _event(self, run: _Run, message: Any) -> None:
        if not isinstance(message, str) or len(message) > 65536:
            return
        try:
            payload = json.loads(message)
        except (ValueError, TypeError, RecursionError):
            return
        if not isinstance(payload, dict):
            return
        kind = payload.get("type")
        if not isinstance(kind, str) or kind not in {"emotion", "speaking", "listening", "idle", "heartbeat", "touch"}:
            return
        event = {"type": kind}
        update = {}
        # The Pi emits emotion/head events. Also accept explicit speech/idle
        # events from compatible runtimes without inferring a microphone state
        # from an HTTP connection or a facial expression.
        if kind == "emotion" and "emotion" not in payload:
            return
        if kind != "touch":
            for key, allowed in (("emotion", _EMOTIONS), ("head", _HEAD_GESTURES)):
                if key in payload:
                    value = payload[key]
                    if not isinstance(value, str) or value not in allowed:
                        return
                    update[key] = value
            for key in ("speaking", "listening"):
                if key in payload:
                    if not isinstance(payload[key], bool):
                        return
                    update[key] = payload[key]
            if kind == "emotion":
                update.setdefault("head", "none")
            elif kind in ("speaking", "listening"):
                update.setdefault(kind, True)
                if update[kind]:
                    update["listening" if kind == "speaking" else "speaking"] = False
            elif kind == "idle":
                update.update(speaking=False, listening=False, head="none")
            if update.get("speaking") is True and update.get("listening") is True:
                return
            event.update(update)
        else:
            # This message is emitted AFTER the Pi dispatches the touch action.
            # Keep only a bounded display record; never send it to an action
            # callback, which would execute the same gesture a second time.
            for key, allowed in (("sensor", {"touch1", "touch2", "touch3", "touch4"}),
                                 ("event", {"tap", "double", "hold"}),
                                 ("status", {"ok", "error", "ignored"})):
                value = payload.get(key)
                if not isinstance(value, str) or value not in allowed:
                    return
                event[key] = value
            if payload.get("handled_by") != "pi":
                return
            for key, pattern in (("id", r"[A-Za-z0-9_-]{1,100}"),
                                 ("action", r"[a-z][a-z0-9_]{0,79}")):
                value = payload.get(key)
                if not isinstance(value, str) or not re.fullmatch(pattern, value):
                    return
                event[key] = value
            event["handled_by"] = "pi"
        with self._lock:
            if not self._current(run) or not run.verified.is_set():
                return
            capabilities = self._state["capabilities"]
            if kind == "touch" and capabilities.get("touch_events") is not True:
                return
            now = time.time()
            self._state.update(update, last_seen=now, telemetry_updated_at=now)
            if kind == "touch":
                self._state["last_touch"] = {**event, "received_at": now}
                return
            # Serialize display callbacks with disconnect/replacement. Raw
            # remote fields never enter callbacks or the desktop status.
            if self._on_event:
                try:
                    self._on_event(event)
                except Exception:
                    # A renderer/gesture consumer must not stop reconnection.
                    pass

    def _telemetry(self, run: _Run) -> None:
        try:
            from websockets.sync.client import connect
        except ImportError:
            return
        while not run.stop.is_set():
            if not run.verified.wait(0.25):
                continue
            try:
                with connect(f"ws://{run.target.authority}:{run.target.ws_port}",
                             open_timeout=2, close_timeout=0.25, max_size=65536, proxy=None) as socket:
                    with self._lock:
                        if not self._current(run) or not run.verified.is_set():
                            return
                        self._state["telemetry_connected"] = True
                    while not run.stop.is_set() and run.verified.is_set():
                        try:
                            self._event(run, socket.recv(timeout=0.5))
                        except TimeoutError:
                            continue
            except Exception:
                pass
            finally:
                with self._lock:
                    if self._current(run):
                        self._state.update(telemetry_connected=False, **_TELEMETRY_DEFAULTS)
            run.stop.wait(2.0)

    def call(self, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[dict | None, str | None]:
        """Call only the selected Pi. Writes are never retried or redirected."""
        method = str(method).upper()
        with self._lock:
            run = self._run
            if not run or run.stop.is_set():
                return None, "Connect ADAM before opening or changing its data."
            target = run.target
        # Verify identity before sending a private key. This also allows an
        # explicit Refresh to recover before the background monitor's next tick.
        identity, error = self._identity(target)
        if error:
            with self._lock:
                if self._current(run):
                    run.verified.clear()
                    self._state.update(connected=False, data_connected=False,
                                       telemetry_connected=False, reason=error, **_TELEMETRY_DEFAULTS)
            return None, error
        with self._lock:
            if not self._current(run):
                return None, "The ADAM connection changed. Please try again."
            self._refresh_identity(run, identity)
            run.verified.set()
            if method != "GET" and self._state["read_only"]:
                return None, "ADAM is read-only. Add its connection key to save changes."
        payload, error = self._request(target, method, path, body)
        with self._lock:
            if not self._current(run):
                return None, "The ADAM connection changed before the response arrived."
            if error and "authorize" in error:
                run.write_denied = True
                self._state.update(read_only=True, reason=error)
        return payload, error
