"""
ADAM Windows Companion App — Backend Server & Action Registry
==============================================================
DGEN Technologies Pvt. Ltd.
Maintains the @action decorator registry, Flask HTTP endpoints, mDNS broadcast,
Pi WebSocket listener, Activity Log, and Coding Agent integration.
"""

import os
import sys
import time
import socket
import subprocess
import platform
import threading
import queue
import json
import secrets
import hmac
import hashlib
import html
from urllib.parse import urlsplit
from typing import Callable, Any, Dict, List, Optional
from collections import deque
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory, Response
import requests

from config import (
    APP_DIR, APP_NAME, APP_VERSION, MDNS_SERVICE_TYPE,
    load_settings, save_settings, set_startup_enabled, is_startup_enabled,
    logger, LOG_BUFFER, update_settings, USER_DATA_DIR
)
from coding_agent import CodingAgentManager, TaskState

# laptop_actions.py is THE shared protocol definition (ADAM v41): the identical
# file ships on the Pi, so both sides agree on action names, value types and
# which values must never be logged. Guarded because this app is also run from a
# packaged build where a missing data file must degrade, not crash the tray app —
# without it the registry simply falls back to v40's untyped behaviour.
try:
    import laptop_actions
    LAPTOP_ACTIONS_AVAILABLE = True
except Exception as _e:        # pragma: no cover - packaging safety net
    laptop_actions = None
    LAPTOP_ACTIONS_AVAILABLE = False
    logger.warning(f"laptop_actions.py unavailable ({_e}); "
                   "falling back to untyped action handling")

SYSTEM = platform.system().lower()

# Initialize Flask app
STATIC_FOLDER = str(APP_DIR / "static")
app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path="/static")
app.config["JSON_SORT_KEYS"] = False
app.config["MAX_CONTENT_LENGTH"] = 1_000_000
DESKTOP_SESSION = secrets.token_urlsafe(32)
DESKTOP_INSTANCE = secrets.token_hex(24)


def _loopback_request():
    return request.remote_addr in ("127.0.0.1", "::1") and urlsplit("http://" + request.host).hostname in ("127.0.0.1", "localhost", "::1")


def _local_session():
    supplied = request.headers.get("X-ADAM-Session", "")
    return _loopback_request() and bool(supplied) and hmac.compare_digest(supplied.encode("utf-8"), DESKTOP_SESSION.encode("ascii"))


@app.before_request
def protect_desktop_and_agent():
    origin = request.headers.get("Origin")
    if origin and origin != request.host_url.rstrip("/"):
        return jsonify(status="error", reason="This origin is not allowed"), 403
    if request.method in ("POST", "PUT", "PATCH"):
        if not request.is_json:
            return jsonify(status="error", reason="A JSON request is required"), 415
        if not isinstance(request.get_json(silent=True), dict):
            return jsonify(status="error", reason="A JSON object is required"), 400
    if request.path in ("/ping", "/actions") and request.method == "GET":
        return None
    if request.path == "/" or request.path.startswith("/static/"):
        if not _loopback_request():
            return jsonify(status="error", reason="Open ADAM on this computer"), 403
        return None
    if request.path == "/show_window" and _loopback_request() and not origin:
        return None
    if _local_session():
        if request.path == '/pair/adam':
            return jsonify(status='error', reason='Connect using your account device picker.'), 410
        # Server gate, not just a hidden dashboard. Only authentication and
        # onboarding are reachable until both account and robot are verified.
        bootstrap = request.path.startswith('/account/') or request.path.startswith('/onboarding/')
        if not bootstrap and not onboarding.status()['ready']:
            return jsonify(status='error', reason='Sign in and connect your ADAM to continue.'), 403
        return None
    remote_action = request.path in ("/pair/verify", "/control", "/coding/dispatch", "/coding/input", "/coding/cancel", "/coding_task_status", "/robot/state") or request.path.startswith("/action/")
    if remote_action and _is_request_authenticated(request.get_json(silent=True) or {}):
        if not onboarding.status()["ready"]:
            return jsonify(status="error", reason="Sign in and connect ADAM first."), 403
        return None
    return jsonify(status="error", reason="Authentication required"), 401


@app.after_request
def response_security(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob: https:; font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
    return response


@app.errorhandler(ValueError)
def invalid_input(error):
    return jsonify(status="error", reason=str(error)), 400


@app.errorhandler(RuntimeError)
def unavailable(error):
    return jsonify(status="error", reason=str(error)), 503

# Action registry
ACTIONS: Dict[str, Dict[str, Any]] = {}
ENABLED_ACTIONS: Dict[str, bool] = {}

# Activity Log Ring Buffer (last 100 actions)
ACTIVITY_LOG = deque(maxlen=100)

# Rate limiter deque for /control (sliding window 1s)
_control_call_times = deque(maxlen=50)

def _is_rate_limited() -> bool:
    now = time.time()
    _control_call_times.append(now)
    if len(_control_call_times) >= 25:
        oldest = _control_call_times[0]
        if now - oldest < 1.0:
            return True
    return False

def _note_remote_client_ip():
    # Public discovery requests are not evidence of robot identity or pairing.
    return

# Global runtime state
CURRENT_ROBOT_STATE = {
    "connected": False,
    "pi_ip": "",
    "emotion": "happy",
    "head": "none",
    "speaking": False,
    "listening": False,
    "last_seen": 0,
    "device_name": "ADAM Robot",
    "uptime_s": None,
}

window_show_callback: Optional[Callable[[], None]] = None
coding_manager: Optional[CodingAgentManager] = None
_zeroconf_instance = None
_service_info = None
_ws_client_thread = None
_ws_stop_event = threading.Event()

from account import AccountService
from cloud_sync import CloudSync
from connection import ConnectionService

account = AccountService()
cloud = CloudSync(account)  # read-only legacy recovery source
from canonical_sync import CanonicalSync
from device_catalog import list_devices as _owned_devices
canonical = CanonicalSync(account, _owned_devices)
connection = ConnectionService(load_settings, update_settings)
_sync_thread_lock = threading.Lock()
from onboarding import OnboardingService

def _nearby_units():
    start_adam_discovery()
    cutoff = time.time() - ADAM_STALE_AFTER_S
    return [dict(u) for u in list(_found_adams.values()) if u['seen_at'] > cutoff]

onboarding = OnboardingService(account, connection, _nearby_units)
from pi_bridge import PiBridge
bridge = PiBridge(canonical, onboarding, connection)

@app.route('/sync/robot', methods=['POST'])
def sync_robot():
    canonical.sync()
    result = bridge.run()
    canonical.sync()
    return jsonify(result)

@app.route('/sync/execution', methods=['POST'])
def sync_execution():
    return jsonify(bridge.report_execution())

@app.route('/onboarding/status')
def onboarding_status():
    return jsonify(onboarding.status())

@app.route('/onboarding/refresh', methods=['POST'])
def onboarding_refresh():
    return jsonify(onboarding.refresh())

@app.route('/onboarding/connect', methods=['POST'])
def onboarding_connect():
    data = request.get_json()
    return jsonify(onboarding.connect(data.get('deviceId', ''), data.get('code', '')))

@app.route('/onboarding/cancel', methods=['POST'])
def onboarding_cancel():
    return jsonify(onboarding.cancel())


def _start_cloud_sync():
    if not account.user:
        raise ValueError("Sign in to sync with your mobile app")
    if _sync_thread_lock.acquire(blocking=False):
        def work():
            try:
                canonical.sync()
            except Exception:
                # The sync service keeps its safe error and all unsent edits.
                pass
            finally:
                _sync_thread_lock.release()
        threading.Thread(target=work, name="ADAM account sync", daemon=True).start()


def account_status():
    result = account.status()
    return {**result, "signed_in": result.get("authenticated", False),
            "configured": True, "google_available": result.get("google", {}).get("configured", False),
            "sync": canonical.status(), "preferences": cloud.get_preferences()}


@app.route('/sync/records', methods=['GET', 'POST'])
def canonical_records():
    if request.method == 'POST':
        data = request.get_json()
        return jsonify(canonical.save(data.get('path', ''), data.get('record', {})))
    return jsonify(canonical.records())

@app.route('/sync/run', methods=['POST'])
def canonical_run():
    return jsonify(canonical.sync())

@app.route('/sync/resolve', methods=['POST'])
def canonical_resolve():
    data = request.get_json()
    return jsonify(canonical.resolve(data.get('path', ''), data.get('choice', '')))

@app.route('/sync/migration')
def migration_inventory():
    # Read-only inventory. No guessing original wall time or robot assignment.
    uid = canonical.uid()
    with cloud._lock:
        old = cloud._read(uid)['companion']
    return jsonify(status='review_required', counts={key:len(value) for key,value in old.items() if isinstance(value,dict)},
        reason='Legacy records are preserved. Review device mappings and schedule timezones before importing.')

@app.route("/account/status")
def account_status_endpoint():
    return jsonify(account_status())


@app.route("/account/devices")
def account_devices():
    from device_catalog import list_devices
    return jsonify(devices=list_devices(account))


@app.route("/account/google/start", methods=["POST"])
def google_start():
    account.start_google()
    return jsonify(account_status())


@app.route("/companion/<kind>", methods=["GET", "POST"])
def companion_records(kind):
    if request.method == "POST":
        raise ValueError("Use the canonical account planner. Legacy records are preserved for migration.")
    return jsonify(items=cloud.list_records(kind), sync=cloud.status())


@app.route("/companion/<kind>/delete", methods=["POST"])
def companion_record_delete(kind):
    raise ValueError("Legacy records are preserved for migration. Use the canonical account planner.")
    return jsonify(items=cloud.list_records(kind), sync=cloud.status())


@app.route("/companion/ble-sync", methods=["POST"])
def companion_ble_sync():
    raise ValueError("Desktop physical synchronization uses the authenticated LAN bridge.")


@app.route("/account/google/cancel", methods=["POST"])
def google_cancel():
    account.cancel_google()
    return jsonify(account_status())


@app.route("/account/email", methods=["POST"])
def email_login():
    data = request.get_json()
    if not isinstance(data.get("create", False), bool):
        raise ValueError("Choose sign in or create account")
    account.email_login(data.get("email", ""), data.get("password", ""),
                        create=data.get("create", False), name=data.get("name", ""))
    _start_cloud_sync()
    return jsonify(account_status())


@app.route("/account/reset", methods=["POST"])
def account_reset():
    account.reset_password(request.get_json().get("email", ""))
    return jsonify(status="ok")


@app.route("/account/signout", methods=["POST"])
def account_signout():
    onboarding.invalidate()
    account.signout()
    onboarding.invalidate()
    update_settings({"paused": True})
    return jsonify(account_status())


@app.route("/account/sync", methods=["POST"])
def account_sync():
    _start_cloud_sync()
    return jsonify(account_status())


@app.route("/account/import-guest", methods=["POST"])
def account_import_guest():
    raise ValueError("Legacy import requires reviewing device mappings and schedule timezones first.")
    return jsonify(account_status())


@app.route("/account/preferences", methods=["POST"])
def account_preferences():
    result = cloud.save_preferences(request.get_json())
    return jsonify(status="ok", preferences=result)


@app.route("/memories", methods=["GET", "POST"])
def memories():
    if request.method == "POST":
        raise ValueError("Use the device-scoped canonical memory editor.")
    return jsonify(memories=cloud.list_memories(), sync=cloud.status())


@app.route("/memories/delete", methods=["POST"])
def memory_delete():
    raise ValueError("Legacy memories are preserved for migration.")
    return jsonify(status="ok", memories=cloud.list_memories(), sync=cloud.status())


@app.route("/connection/probe", methods=["POST"])
def connection_probe():
    result = connection.probe(request.get_json())
    return jsonify({**result, "status": "ok" if result.get("ok") else "error"})


@app.route("/connection/connect", methods=["POST"])
def connection_connect():
    result = connection.connect(request.get_json())
    return jsonify({**result, "status": "ok" if result.get("ok") else "error"})


@app.route("/connection/disconnect", methods=["POST"])
def connection_disconnect():
    result = connection.disconnect()
    update_settings({"touch_applied_hash": ""})
    return jsonify(status="ok", robot=result)


@app.route("/connection/credentials", methods=["POST"])
def connection_credentials():
    settings = load_settings()
    return jsonify(token=settings["agent_token"], host=_get_local_ip(), port=settings["agent_port"])


@app.route("/pair/verify")
def pair_verify():
    return jsonify(status="ok", app=APP_NAME, kind="laptop", api=1)


@app.route("/connection/authorize", methods=["POST"])
def authorize_desktop_control():
    state = connection.status()
    if not state.get("capabilities", {}).get("laptop_pairing"):
        raise RuntimeError("Update ADAM's companion service to enable guided laptop pairing, or use manual pairing.")
    settings = load_settings()
    host = _get_local_ip()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
            route.connect((state.get("host") or settings["pi_host"], settings["pi_sync_port"]))
            host = route.getsockname()[0]
    except OSError:
        pass
    payload, error = connection.call("POST", "/api/laptops/pair", {
        "host": host, "port": settings["agent_port"], "token": settings["agent_token"]})
    if (error or not payload or payload.get("ok") is not True
            or type(payload.get("api")) is not int or payload["api"] != 1
            or payload.get("paired") is not True or payload.get("host") != host
            or type(payload.get("port")) is not int or payload["port"] != settings["agent_port"]):
        raise RuntimeError(error or "ADAM did not confirm laptop pairing")
    log_activity("laptop_paired", status="ok", details="ADAM verified this computer's control key")
    return jsonify(status="ok", paired=True, host=host, port=settings["agent_port"])


@app.route("/connection/revoke", methods=["POST"])
def revoke_desktop_control():
    payload, error = connection.call("POST", "/api/laptops/unpair", {})
    if (error or not payload or payload.get("ok") is not True
            or type(payload.get("api")) is not int or payload["api"] != 1
            or payload.get("paired") is not False):
        raise RuntimeError(error or "ADAM did not confirm the disconnection")
    update_settings({"paused": True})
    return jsonify(status="ok", paired=False)


def validate_touch(assignments):
    if not isinstance(assignments, dict) or set(assignments) - {"touch1", "touch2", "touch3", "touch4"}:
        raise ValueError("Choose a valid ADAM touch sensor")
    result = {}
    for sensor in ("touch1", "touch2", "touch3", "touch4"):
        events = assignments.get(sensor, {})
        allowed = ("double", "triple", "hold") if sensor == "touch3" else ("hold",)
        if not isinstance(events, dict) or set(events) - set(allowed):
            raise ValueError("Single taps are fixed ADAM reactions. Only Touch 3 supports custom double and triple taps.")
        result[sensor] = {}
        for event in allowed:
            binding = events.get(event, {"action": "none"})
            if not isinstance(binding, dict):
                raise ValueError("Invalid touch assignment")
            name = binding.get("action", "none")
            value = binding.get("value")
            if not isinstance(name, str):
                raise ValueError("Choose a supported action")
            if name == "none":
                value = None
            if name != "none":
                if name not in ACTIONS:
                    raise ValueError("Unknown touch action")
                valid, value, reason = _coerce_action_value(name, ACTIONS[name], value)
                if not valid:
                    raise ValueError(reason)
                if name in ("dispatch_coding_task", "read_clipboard", "write_clipboard", "clipboard_paste"):
                    raise ValueError("Use Actions for clipboard and coding operations")
            result[sensor][event] = {"action": name, "value": value}
    return result


@app.route("/touch/save", methods=["POST"])
def touch_save():
    assignments = validate_touch(request.get_json().get("assignments", {}))
    update_settings({"touch_assignments": assignments, "touch_applied_hash": ""})
    return jsonify(status="ok", assignments=assignments, applied=False)


@app.route("/touch/test", methods=["POST"])
def touch_test():
    data = request.get_json()
    if not isinstance(data.get("sensor"), str) or not isinstance(data.get("event"), str):
        raise ValueError("Choose a sensor and gesture")
    allowed = ("double", "triple", "hold") if data["sensor"] == "touch3" else ("hold",)
    if data["sensor"] not in ("touch1", "touch2", "touch3", "touch4") or data["event"] not in allowed:
        raise ValueError("This gesture is a fixed ADAM reaction")
    settings = load_settings()
    if settings["paused"]:
        return jsonify(status="error", reason="Resume laptop controls first"), 403
    binding = settings.get("touch_assignments", {}).get(data.get("sensor"), {}).get(data.get("event"))
    if not binding or binding.get("action") == "none":
        raise ValueError("Save an action for this gesture first")
    name = binding["action"]
    if not ENABLED_ACTIONS.get(name, False):
        return jsonify(status="error", reason="Enable this action in Actions first"), 403
    spec = ACTIONS[name]
    result = spec["fn"](binding.get("value")) if spec["needs_value"] else spec["fn"]()
    if isinstance(result, dict) and (result.get("error") or result.get("status") == "error"):
        raise RuntimeError(str(result.get("error") or result.get("reason") or "The action failed"))
    log_activity(name, binding.get("value"), "ok", "Touch control tested locally")
    return jsonify(status="ok", result=result or {})


@app.route("/touch/apply", methods=["POST"])
def touch_apply():
    if not connection.status().get("capabilities", {}).get("touch_assignments"):
        raise RuntimeError("This ADAM software does not support custom touch assignments yet. Your choices are saved on this PC.")
    if connection.status().get("capabilities", {}).get("touch_gesture_policy") != 2:
        raise RuntimeError("Update ADAM's touch service to apply these gesture preferences. Your choices are saved on this PC.")
    assignments = validate_touch(load_settings().get("touch_assignments", {}))
    payload, error = connection.call("POST", "/api/touch/assignments", {"assignments": assignments})
    if error or not payload or payload.get("ok") is not True or payload.get("assignments") != assignments:
        raise RuntimeError(error or "ADAM did not confirm these assignments")
    digest = hashlib.sha256(json.dumps(assignments, sort_keys=True).encode()).hexdigest()
    update_settings({"touch_applied_hash": digest})
    return jsonify(status="ok", applied=True, assignments=assignments)


def log_activity(action: str, value: Any = None, status: str = "ok", details: str = ""):
    """Record an action in the Activity Log.

    The value is redacted for actions whose argument is user content rather than
    a control parameter (clipboard writes, coding instructions). The Activity Log
    is served over /activity_log to any authenticated LAN client and is echoed to
    the rotating log file, so logging a clipboard write verbatim would persist and
    publish whatever the user had copied — including a password-manager buffer.
    Volume 40 still logs as 40; clipboard text logs as "<text 1832 chars>".
    """
    category = "General"
    if action in ACTIONS:
        category = ACTIONS[action].get("category", "General")
    if "coding" in action:
        value = None
        details = "Coding task state updated" if status == "ok" else "Coding task request failed"
    if LAPTOP_ACTIONS_AVAILABLE:
        value = laptop_actions.redact_value(action, value)
        # Only a successful call puts the action's RESULT in `details`; on an
        # error `details` is a reason ("invalid token"), which must stay readable
        # or the log stops being useful for diagnosing a failure.
        if status == "ok":
            details = laptop_actions.redact_details(action, details)
    entry = {
        "timestamp": time.time(),
        "time_str": time.strftime("%H:%M:%S"),
        "action": action,
        "category": category,
        "value": value,
        "status": status,
        "details": details,
    }
    ACTIVITY_LOG.appendleft(entry)
    logger.info(f"Activity: {action} (val={value}) -> {status} {details}")


def action(name: str, description: str, needs_value: bool = False,
           value_hint: str = "", category: str = "General",
           value_type: str = "") -> Callable:
    """Decorator that registers a function as a callable laptop action.

    value_type ("none" | "int" | "str" | "enum") is what the Pi uses to coerce
    the argument. It is normally left blank and taken from laptop_actions.py, so
    the type is declared in exactly one place for both machines; passing it
    explicitly is only for an action that does not exist in the shared manifest.
    """
    def wrapper(fn: Callable) -> Callable:
        vt = value_type
        if not vt and LAPTOP_ACTIONS_AVAILABLE:
            spec = laptop_actions.spec(name)
            vt = (spec["value_type"] if spec
                  else laptop_actions.infer_value_type(needs_value, value_hint, name))
        if not vt:
            vt = "str" if needs_value else "none"
        ACTIONS[name] = {
            "fn": fn,
            "description": description,
            "needs_value": needs_value,
            "value_hint": value_hint,
            "category": category,
            "value_type": vt,
        }
        if name not in ENABLED_ACTIONS:
            saved = load_settings().get("enabled_actions", {})
            ENABLED_ACTIONS[name] = saved.get(name, name not in {"lock_screen", "read_clipboard", "write_clipboard", "clipboard_paste", "dispatch_coding_task"})
        return fn
    return wrapper


def _resolve_action_name(name: str) -> str:
    """Map an alias (clipboard_get) onto a registered action (read_clipboard).

    The Pi normally resolves aliases before sending, but the REST endpoints are
    also called by the dashboard and by hand, so both vocabularies work here too.
    An unknown name is returned untouched so the caller still gets the existing
    "unknown action" error with the list of what is available.
    """
    if not isinstance(name, str) or not name or len(name) > 64:
        raise ValueError("Choose a supported action")
    if name in ACTIONS:
        return name
    if LAPTOP_ACTIONS_AVAILABLE:
        canon = laptop_actions.resolve(name)
        if canon in ACTIONS:
            return canon
    return name


def _coerce_action_value(action_name: str, spec: dict, value: Any):
    """Validate and convert an incoming value. Returns (ok, value, reason).

    Before v41 the handlers took whatever JSON contained: act_volume_set(value)
    would receive the string "50" from any caller that sent text and hand it
    straight to a Windows volume API expecting an int. Now the same table the Pi
    uses decides the type, so "50", 50 and "50%" all arrive as 50, and an
    out-of-range number is clamped rather than passed to the driver.
    """
    if not spec["needs_value"]:
        return True, None, ""
    if not LAPTOP_ACTIONS_AVAILABLE:
        return (True, value, "") if value is not None else (False, None, "value required")
    return laptop_actions.coerce(action_name, value, {action_name: spec})


# ═════════════════════════════════════════════════════════════════════════
# HARDWARE CONTROL BACKENDS (Windows pycaw / screen-brightness-control)
# ═════════════════════════════════════════════════════════════════════════

from hardware import WindowsHardwareWorker
_hw_worker = None

def _hardware_call(task, args=()):
    if _hw_worker is None:
        raise RuntimeError("Windows hardware control is unavailable")
    return _hw_worker.call_sync(task, args)

def get_system_volume():
    return _hardware_call("get_vol")

def set_system_volume(val):
    return _hardware_call("set_vol", (val,))

def set_system_mute(mute):
    return _hardware_call("set_mute", (mute,))

def get_system_brightness():
    return _hardware_call("get_bri")

def set_system_brightness(val):
    return _hardware_call("set_bri", (val,))

VOLUME_STEP = 10
BRIGHTNESS_STEP = 10


# ═════════════════════════════════════════════════════════════════════════
# REGISTERED ACTIONS
# ═════════════════════════════════════════════════════════════════════════

@action("volume_up", "Increase system volume by 10%.", category="Media")
def act_volume_up():
    new_val = min(100, get_system_volume() + VOLUME_STEP)
    return {"volume": set_system_volume(new_val)}


@action("volume_down", "Decrease system volume by 10%.", category="Media")
def act_volume_down():
    new_val = max(0, get_system_volume() - VOLUME_STEP)
    return {"volume": set_system_volume(new_val)}


@action("volume_set", "Set system volume to an exact percentage.",
        needs_value=True, value_hint="0-100", category="Media")
def act_volume_set(value: int):
    try:
        val = max(0, min(100, int(float(value))))
    except (ValueError, TypeError):
        return {"status": "error", "reason": f"Invalid numeric value: {value}"}
    return {"volume": set_system_volume(val)}


@action("volume_mute", "Mute system audio.", category="Media")
def act_volume_mute():
    if SYSTEM == "windows":
        set_system_mute(True)
    return {"muted": True}


@action("volume_unmute", "Unmute system audio.", category="Media")
def act_volume_unmute():
    if SYSTEM == "windows":
        set_system_mute(False)
    return {"muted": False}


@action("brightness_up", "Increase screen brightness by 10%.", category="System")
def act_brightness_up():
    new_val = min(100, get_system_brightness() + BRIGHTNESS_STEP)
    return {"brightness": set_system_brightness(new_val)}


@action("brightness_down", "Decrease screen brightness by 10%.", category="System")
def act_brightness_down():
    new_val = max(0, get_system_brightness() - BRIGHTNESS_STEP)
    return {"brightness": set_system_brightness(new_val)}


@action("brightness_set", "Set screen brightness to an exact percentage.",
        needs_value=True, value_hint="0-100", category="System")
def act_brightness_set(value: int):
    try:
        val = max(0, min(100, int(float(value))))
    except (ValueError, TypeError):
        return {"status": "error", "reason": f"Invalid numeric value: {value}"}
    return {"brightness": set_system_brightness(val)}


def _send_media_key(vk: int):
    if SYSTEM == "windows":
        try:
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            user32.keybd_event.argtypes = [wintypes.BYTE, wintypes.BYTE, wintypes.DWORD, ctypes.c_size_t]
            user32.keybd_event.restype = None
            user32.keybd_event(vk, 0, 0, 0)
            user32.keybd_event(vk, 0, 2, 0)
        except Exception as e:
            logger.warning(f"Failed sending media key {vk}: {e}")


def _send_key_combo(modifier_vk: int, key_vk: int):
    """Hold a modifier, tap a key, release — e.g. Ctrl+V.

    Deliberately built on the SAME ctypes user32.keybd_event call that
    _send_media_key already uses, rather than adding pyautogui/keyboard/SendInput
    as a second automation layer (master prompt §20). The only difference from
    _send_media_key is that the modifier's key-up is deferred until after the
    key tap, which keybd_event cannot express in a single call.
    """
    if SYSTEM != "windows":
        return False
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        user32.keybd_event.argtypes = [wintypes.BYTE, wintypes.BYTE, wintypes.DWORD, ctypes.c_size_t]
        user32.keybd_event.restype = None
        KEYUP = 2
        user32.keybd_event(modifier_vk, 0, 0, 0)       # modifier down
        user32.keybd_event(key_vk, 0, 0, 0)            # key down
        user32.keybd_event(key_vk, 0, KEYUP, 0)        # key up
        user32.keybd_event(modifier_vk, 0, KEYUP, 0)   # modifier up
        return True
    except Exception as e:
        logger.warning(f"Failed sending key combo {modifier_vk}+{key_vk}: {e}")
        return False


@action("lock_screen", "Lock the laptop's screen immediately.", category="System")
def act_lock_screen():
    if SYSTEM == "windows":
        try:
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            user32.LockWorkStation.argtypes = []
            user32.LockWorkStation.restype = wintypes.BOOL
            if not user32.LockWorkStation():
                raise RuntimeError("Windows did not accept the lock request")
        except Exception:
            import subprocess
            raise RuntimeError("Windows could not lock this session") from None
    else:
        raise RuntimeError("Screen locking requires Windows")
    return {"locked": True}


@action("read_clipboard", "Read text content currently copied on the laptop clipboard.", category="Clipboard")
def act_read_clipboard():
    try:
        import pyperclip
        text = pyperclip.paste() or ""
        # Cap from the shared manifest instead of the old hardcoded 500 chars, so
        # the laptop and the Pi truncate at the same point. 500 was too short to
        # discuss even a paragraph the user had copied.
        cap = laptop_actions.CLIPBOARD_MAX_CHARS if LAPTOP_ACTIONS_AVAILABLE else 500
        return {"text": text[:cap], "length": len(text),
                "truncated": len(text) > cap}
    except Exception as e:
        return {"error": f"Failed reading clipboard: {e}"}


@action("write_clipboard", "Copy text to the laptop clipboard.",
        needs_value=True, value_hint="text string", category="Clipboard")
def act_write_clipboard(value: str):
    try:
        import pyperclip
        pyperclip.copy(str(value))
        return {"copied": True, "length": len(str(value))}
    except Exception as e:
        return {"error": f"Failed writing clipboard: {e}"}


@action("clipboard_paste",
        "Press Ctrl+V to paste the clipboard into the focused window. "
        "Only on an explicit request to paste.",
        category="Clipboard")
def act_clipboard_paste():
    """Paste into whatever window currently has focus.

    Added in v41 (§14/§20). Two rules this must not break:
      * It uses the existing keybd_event layer via _send_key_combo — not a new
        automation dependency.
      * It fires only on an explicit instruction. ADAM must never paste merely
        because it generated something, and a touch gesture on the robot must
        never reach this action (§24); both are stated in prompts.txt and the
        generated-content flow ends at write_clipboard, not here.
    """
    VK_CONTROL, VK_V = 0x11, 0x56
    if SYSTEM != "windows":
        return {"pasted": False, "error": f"not supported on {SYSTEM}"}
    sent = _send_key_combo(VK_CONTROL, VK_V)
    return {"pasted": bool(sent)}


@action("media_play_pause", "Toggle play/pause for active laptop media player.", category="Media")
def act_media_play_pause():
    _send_media_key(0xB3)  # VK_MEDIA_PLAY_PAUSE
    return {"media_action": "play_pause"}


@action("media_next", "Skip to next track on active laptop media player.", category="Media")
def act_media_next():
    _send_media_key(0xB0)  # VK_MEDIA_NEXT_TRACK
    return {"media_action": "next"}


@action("media_previous", "Return to previous track on active laptop media player.", category="Media")
def act_media_previous():
    _send_media_key(0xB1)  # VK_MEDIA_PREV_TRACK
    return {"media_action": "previous"}


@action("dispatch_coding_task", "Run a coding task in your selected workspace.",
        needs_value=True, value_hint="instruction string", category="Coding")
def act_dispatch_coding(prompt: str):
    global coding_manager
    if not coding_manager:
        raise RuntimeError("Coding manager not initialized")
    settings = load_settings()
    res = coding_manager.dispatch(str(prompt), tool=settings.get("coding_tool", "codex"), cwd=settings.get("coding_workspace"))
    if res.get("status") == "error":
        raise RuntimeError(res.get("reason", "Could not start the coding task"))
    return res


@action("check_coding_task_status", "Check status of the active or latest coding task.", category="Coding")
def act_check_coding_status():
    global coding_manager
    if not coding_manager:
        return {"status": "error", "reason": "Coding manager not initialized"}
    return coding_manager.get_status()


@action("cancel_coding_task", "Cancel the currently active coding task.", category="Coding")
def act_cancel_coding():
    global coding_manager
    if not coding_manager:
        return {"status": "error", "reason": "Coding manager not initialized"}
    return coding_manager.cancel()


@action("set_robot_emotion",
        "Mirror the robot's current face onto the PC app's 3D model.",
        needs_value=True, category="Robot")
def act_set_emotion(emotion: str):
    """Mirror ADAM's face onto the desktop 3D model.

    The @action decorator above was LOST in the pcAPP -> adam-desktop
    restructure: this function survived, its registration did not, so the agent
    advertised 18 actions instead of 19 and the Pi's live parity check reported
    "missing expected actions: set_robot_emotion". value_type/choices come from
    laptop_actions.py (enum over ROBOT_EMOTIONS), so they are declared once for
    both machines rather than repeated here.

    The value is re-checked against the manifest even though the Pi already
    coerces enums: this endpoint is reachable from the LAN, and an unknown face
    name would otherwise be written straight into the shared state the 3D model
    renders from.
    """
    name = str(emotion).lower().strip()
    if LAPTOP_ACTIONS_AVAILABLE:
        allowed = getattr(laptop_actions, "ROBOT_EMOTIONS", ())
        if allowed and name not in allowed:
            return {"status": "error",
                    "reason": f"unknown emotion {name!r}"}
    CURRENT_ROBOT_STATE["emotion"] = name
    CURRENT_ROBOT_STATE["last_seen"] = time.time()
    return {"emotion": CURRENT_ROBOT_STATE["emotion"]}


# ═════════════════════════════════════════════════════════════════════════
# HTTP API ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    """Serve the dashboard HTML."""
    page = (Path(STATIC_FOLDER) / "index.html").read_text(encoding="utf-8")
    return Response(page.replace("__ADAM_SESSION__", DESKTOP_SESSION), mimetype="text/html")


@app.route("/actions", methods=["GET"])
def list_actions():
    """Self-describing manifest called by the Pi.

    `value_type` was added in v41. The Pi reads it to coerce the argument before
    sending; an older Pi ignores the extra key, so this stays backward
    compatible in both directions.
    """
    _note_remote_client_ip()
    manifest = {}
    for name, spec in ACTIONS.items():
        entry = {
            "description": spec["description"],
            "needs_value": spec["needs_value"],
            "value_hint": spec["value_hint"],
            "value_type": spec.get("value_type", "none"),
            "category": spec.get("category", "General"),
            "enabled": ENABLED_ACTIONS.get(name, True),
        }
        if LAPTOP_ACTIONS_AVAILABLE:
            shared = laptop_actions.spec(name) or {}
            if "choices" in shared:
                entry["choices"] = list(shared["choices"])
            if shared.get("confirm"):
                entry["confirm"] = True
        manifest[name] = entry
    payload = {"platform": SYSTEM, "actions": manifest, "version": APP_VERSION}
    if LAPTOP_ACTIONS_AVAILABLE:
        payload["aliases"] = dict(laptop_actions.ALIASES)
    return jsonify(payload)


def _extract_request_token(data: dict) -> str:
    """Extract authentication token from JSON body, X-ADAM-Token header, or Authorization Bearer header."""
    auth_hdr = request.headers.get("Authorization", "")
    bearer = auth_hdr.split(" ", 1)[1].strip() if auth_hdr.lower().startswith("bearer ") else ""
    return str(data.get("token") or request.headers.get("X-ADAM-Token") or bearer).strip()


def _is_request_authenticated(data):
    if _local_session():
        return True
    configured = str(load_settings().get("agent_token", ""))
    supplied = _extract_request_token(data)
    return bool(configured and supplied and hmac.compare_digest(configured.encode("utf-8"), supplied.encode("utf-8")))


@app.route("/control", methods=["POST"])
def control():
    """Generic action dispatcher."""
    _note_remote_client_ip()
    settings = load_settings()
    data = request.get_json(silent=True) or {}

    if _is_rate_limited():
        log_activity(data.get("action", "unknown"), data.get("value"), "rate_limited", "Rate limit exceeded")
        return jsonify({"status": "error", "reason": "rate limit exceeded"}), 429

    if settings.get("paused"):
        log_activity(data.get("action", "unknown"), data.get("value"), "blocked", "Agent is paused")
        return jsonify({"status": "error", "reason": "laptop control is currently paused"}), 403

    if not _is_request_authenticated(data):
        log_activity(data.get("action", "unknown"), data.get("value"), "unauthorized", "Invalid token")
        return jsonify({"status": "error", "reason": "invalid token"}), 401

    action_name = data.get("action", "")
    value = data.get("value")
    if value is None:
        value = data.get("text") or data.get("prompt")

    action_name = _resolve_action_name(action_name)
    spec = ACTIONS.get(action_name)
    if spec is None:
        log_activity(action_name, value, "error", "Unknown action")
        return jsonify({
            "status": "error",
            "reason": f"unknown action: {action_name}",
            "available": list(ACTIONS.keys())
        }), 400

    if not ENABLED_ACTIONS.get(action_name, True):
        log_activity(action_name, value, "disabled", f"Action {action_name} is disabled")
        return jsonify({"status": "error", "reason": f"action '{action_name}' is disabled in settings"}), 403

    ok, value, reason = _coerce_action_value(action_name, spec, value)
    if not ok:
        log_activity(action_name, value, "error", reason)
        return jsonify({"status": "error", "reason": reason}), 400

    try:
        if spec["needs_value"]:
            result = spec["fn"](value)
        else:
            result = spec["fn"]()
        result = result or {}
        if result.get("error") or result.get("status") == "error":
            raise RuntimeError(str(result.get("error") or result.get("reason") or "The action failed"))
        log_activity(action_name, value, "ok", str(result))
        return jsonify({"status": "ok", "action": action_name, **result})
    except Exception as e:
        log_activity(action_name, value, "error", str(e))
        return jsonify({"status": "error", "reason": f"{type(e).__name__}: {e}"}), 500


@app.route("/action/<action_name>", methods=["POST"])
def direct_action_endpoint(action_name: str):
    """Direct REST action execution endpoint."""
    _note_remote_client_ip()
    settings = load_settings()
    data = request.get_json(silent=True) or {}

    if _is_rate_limited():
        log_activity(action_name, data.get("value"), "rate_limited", "Rate limit exceeded")
        return jsonify({"status": "error", "reason": "rate limit exceeded"}), 429

    if settings.get("paused"):
        log_activity(action_name, data.get("value"), "blocked", "Agent is paused")
        return jsonify({"status": "error", "reason": "laptop control is currently paused"}), 403

    if not _is_request_authenticated(data):
        log_activity(action_name, data.get("value"), "unauthorized", "Invalid token")
        return jsonify({"status": "error", "reason": "invalid token"}), 401

    spec = ACTIONS.get(action_name)
    if spec is None:
        resolved = _resolve_action_name(action_name)
        spec = ACTIONS.get(resolved)
        if spec is not None:
            action_name = resolved
    if spec is None:
        log_activity(action_name, None, "error", "Unknown action")
        return jsonify({
            "status": "error",
            "reason": f"unknown action: {action_name}",
            "available": list(ACTIONS.keys())
        }), 400

    if not ENABLED_ACTIONS.get(action_name, True):
        log_activity(action_name, None, "disabled", f"Action {action_name} is disabled")
        return jsonify({"status": "error", "reason": f"action '{action_name}' is disabled in settings"}), 403

    value = data.get("value")
    if value is None:
        value = data.get("text") if "text" in data else data.get("prompt")

    ok, value, reason = _coerce_action_value(action_name, spec, value)
    if not ok:
        log_activity(action_name, value, "error", reason)
        return jsonify({"status": "error", "reason": reason}), 400

    try:
        if spec["needs_value"]:
            result = spec["fn"](value)
        else:
            result = spec["fn"]()
        result = result or {}
        if result.get("error") or result.get("status") == "error":
            raise RuntimeError(str(result.get("error") or result.get("reason") or "The action failed"))
        log_activity(action_name, value, "ok", str(result))
        return jsonify({"status": "ok", "action": action_name, "result": result, **result})
    except Exception as e:
        log_activity(action_name, value, "error", str(e))
        return jsonify({"status": "error", "reason": f"{type(e).__name__}: {e}"}), 500


@app.route("/ping", methods=["GET"])
def ping():
    _note_remote_client_ip()
    return jsonify({
        "status": "ok",
        "app": APP_NAME,
        "platform": SYSTEM,
        "action_count": len(ACTIONS),
        "version": APP_VERSION,
        **({"instance_id": DESKTOP_INSTANCE, "pid": os.getpid()} if _loopback_request() else {})
    })


@app.route("/status", methods=["GET"])
def full_status():
    settings = public_settings()
    hardware = _hw_worker.snapshot() if _hw_worker else {"volume": None, "brightness": None}
    robot = connection.status()
    return jsonify({"status": "ok", **hardware, "hardware": hardware, "robot": robot,
                    "coding": coding_manager.get_status() if coding_manager else {},
                    "settings": settings, "paused": settings["paused"], "version": APP_VERSION,
                    "system_state": "paused" if settings["paused"] else "ready",
                    "enabled_actions": dict(ENABLED_ACTIONS), "recent_activity": list(ACTIVITY_LOG)[:20]})


try:
    import psutil
except ImportError:
    psutil = None

class _SystemStatsSampler:
    """Share counter history across Flask threads without waiting for a sample."""

    TTL_SECONDS = 2.0
    MAX_NETWORK_MBPS = 1_000_000.0

    def __init__(self, probe, clock=None):
        self._probe = probe
        self._clock = clock or time.monotonic
        self._sample_lock = threading.Lock()
        self._cache = (None, self._empty_snapshot())
        self._previous_cpu = None
        self._previous_net = None

    @staticmethod
    def _empty_snapshot():
        return {
            'cpu_pct': None,
            'ram_pct': None,
            'disk_pct': None,
            'net_connected': False,
            'net_up_mbps': 0.0,
            'net_down_mbps': 0.0,
        }

    @staticmethod
    def _number(value):
        import math
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) else None

    @classmethod
    def _percentage(cls, value):
        number = cls._number(value)
        return None if number is None else round(min(100.0, max(0.0, number)), 1)

    def _cpu_percentage(self, current):
        # Linux guest time is already included in user/nice. I/O wait counts as
        # idle, matching psutil's CPU percentage semantics on that platform.
        counters = {key: self._number(value) for key, value in current._asdict().items()
                    if key not in ('guest', 'guest_nice')}
        previous = self._previous_cpu
        if 'idle' not in counters or any(value is None or value < 0 for value in counters.values()):
            self._previous_cpu = None
            return None
        self._previous_cpu = counters
        if previous is None or previous.keys() != counters.keys():
            return None
        deltas = {key: value - previous[key] for key, value in counters.items()}
        # A reboot, counter reset or changing CPU set needs a fresh baseline.
        if any(value < 0 for value in deltas.values()):
            return None
        total = sum(deltas.values())
        if total <= 0:
            return None
        busy = total - deltas['idle'] - deltas.get('iowait', 0.0)
        return self._percentage(busy / total * 100.0)

    def _network_rates(self, current, sampled_at):
        previous = self._previous_net
        sent = self._number(getattr(current, 'bytes_sent', None))
        received = self._number(getattr(current, 'bytes_recv', None))
        if sent is None or received is None or sent < 0 or received < 0:
            self._previous_net = None
            return 0.0, 0.0
        self._previous_net = (sampled_at, sent, received)
        if previous is None:
            return 0.0, 0.0
        elapsed = sampled_at - previous[0]
        if elapsed <= 0 or sent < previous[1] or received < previous[2]:
            return 0.0, 0.0

        def mbps(delta):
            rate = self._number((delta / elapsed) * 8.0 / 1_000_000.0)
            # A generous ceiling also bounds corrupt/reset counter spikes.
            return 0.0 if rate is None else round(min(self.MAX_NETWORK_MBPS, max(0.0, rate)), 1)

        return mbps(sent - previous[1]), mbps(received - previous[2])

    def _sample(self):
        result = self._empty_snapshot()
        if self._probe is None:
            return result
        cpu = self._probe.cpu_times()
        net = self._probe.net_io_counters()
        sampled_at = self._clock()
        result['cpu_pct'] = self._cpu_percentage(cpu)
        result['net_up_mbps'], result['net_down_mbps'] = self._network_rates(net, sampled_at)
        result['ram_pct'] = self._percentage(self._probe.virtual_memory().percent)
        result['disk_pct'] = self._percentage(self._probe.disk_usage(str(USER_DATA_DIR.anchor)).percent)
        result['net_connected'] = any(
            value.isup for name, value in self._probe.net_if_stats().items()
            if 'loopback' not in name.lower() and name.lower() != 'lo'
        )
        return result

    def snapshot(self):
        sampled_at, cached = self._cache
        if sampled_at is not None and 0 <= self._clock() - sampled_at < self.TTL_SECONDS:
            return dict(cached)
        if not self._sample_lock.acquire(blocking=False):
            return dict(self._cache[1])
        try:
            # Another request may have refreshed the cache before acquisition.
            sampled_at, cached = self._cache
            if sampled_at is not None and 0 <= self._clock() - sampled_at < self.TTL_SECONDS:
                return dict(cached)
            try:
                result = self._sample()
            except Exception as error:
                logger.warning(f"system_stats error: {error}")
                self._previous_cpu = None
                self._previous_net = None
                result = self._empty_snapshot()
            # Replace the complete cache atomically; callers never share a
            # mutable result, and failures are throttled by the same TTL.
            self._cache = (self._clock(), result)
            return dict(result)
        finally:
            self._sample_lock.release()


_system_stats_sampler = _SystemStatsSampler(psutil)


@app.route('/system_stats', methods=['GET'])
def system_stats_endpoint():
    return jsonify(_system_stats_sampler.snapshot())

@app.route('/activity_log', methods=['GET'])
def activity_log_endpoint():
    since = request.args.get('since', type=float, default=0.0)
    limit = request.args.get('limit', type=int, default=100)
    q = request.args.get('q', type=str, default='').lower()
    
    logs = []
    for entry in list(ACTIVITY_LOG):
        if entry["timestamp"] >= since:
            if not q or q in entry["action"].lower() or q in entry["details"].lower():
                logs.append(entry)
                if len(logs) >= limit:
                    break
    
    return jsonify({"log": logs})

@app.route('/activity_log/clear', methods=['POST'])
def activity_log_clear_endpoint():
    ACTIVITY_LOG.clear()
    return jsonify({"status": "ok"})


@app.route("/show_window", methods=["POST"])
def show_window_endpoint():
    """Brings the dashboard window forward (used by single-instance activation)."""
    global window_show_callback
    client_ip = request.remote_addr or ""
    settings = load_settings()
    data = request.get_json(silent=True) or {}
    if data.get("instance_id") and data["instance_id"] != DESKTOP_INSTANCE:
        return jsonify(status="error", reason="This desktop instance has changed"), 409

    # Remote clients must provide valid token to prevent unauthorized window popups
    if client_ip not in ("127.0.0.1", "::1", "localhost"):
        token = settings.get("agent_token", "")
        req_token = _extract_request_token(data)
        if token and req_token != token:
            return jsonify({"status": "error", "reason": "unauthorized"}), 401

    if window_show_callback:
        try:
            threading.Thread(target=window_show_callback, daemon=True).start()
            return jsonify({"status": "ok", "action": "window_restored"})
        except Exception as e:
            return jsonify({"status": "error", "reason": str(e)}), 500
    return jsonify({"status": "error", "reason": "no window callback"}), 400


@app.route("/agent/restart", methods=["POST"])
def agent_restart_endpoint():
    """Restart local mDNS, WS client, and agent services."""
    try:
        restart_backend_services()
        return jsonify({"status": "ok", "message": "Agent services restarted successfully."})
    except Exception as e:
        logger.error(f"Error restarting agent services: {e}")
        return jsonify({"status": "error", "reason": str(e)}), 500


def public_settings():
    settings = load_settings()
    for key in ("agent_token", "sync_token"):
        settings[key + "_configured"] = bool(settings.pop(key, ""))
    settings["startup_on_login"] = is_startup_enabled()
    settings["profile"] = {"name": settings.get("user_name", "")}
    settings["enabled_actions"] = dict(ENABLED_ACTIONS)
    return settings


@app.route("/settings", methods=["GET", "POST"])
def settings_endpoint():
    if request.method == "GET":
        return jsonify(public_settings())
    data = request.get_json(silent=True) or {}
    allowed = {"paused", "startup_on_login", "enabled_actions", "setup_complete", "user_name", "coding_workspace"}
    if set(data) - allowed:
        return jsonify(status="error", reason="Use device connection to change connection settings"), 400
    patch = {}
    for key in ("paused", "startup_on_login", "setup_complete"):
        if key in data:
            if not isinstance(data[key], bool):
                raise ValueError(key + " must be true or false")
            patch[key] = data[key]
    if "user_name" in data:
        if not isinstance(data["user_name"], str) or len(data["user_name"]) > 80:
            raise ValueError("Name must be at most 80 characters")
        patch["user_name"] = data["user_name"].strip()
    if "coding_workspace" in data:
        value = data["coding_workspace"]
        if not isinstance(value, str) or (value and not Path(value).is_dir()):
            raise ValueError("Choose an existing coding workspace folder")
        patch["coding_workspace"] = str(Path(value).resolve()) if value else ""
    if "enabled_actions" in data:
        values = data["enabled_actions"]
        if not isinstance(values, dict) or any(k not in ACTIONS or not isinstance(v, bool) for k,v in values.items()):
            raise ValueError("Invalid action preferences")
        patch["enabled_actions"] = {**ENABLED_ACTIONS, **values}
    if "startup_on_login" in patch and not set_startup_enabled(patch["startup_on_login"]):
        raise RuntimeError("Windows could not update the startup shortcut")
    saved = update_settings(patch)
    ENABLED_ACTIONS.update(saved.get("enabled_actions", {}))
    log_activity("settings_update", status="ok")
    return jsonify(status="ok", settings=public_settings())


@app.route("/logs", methods=["GET"])
def logs_endpoint():
    """Return in-memory log buffer."""
    return jsonify({
        "logs": list(LOG_BUFFER),
        "activity": list(ACTIVITY_LOG)
    })


@app.route("/coding/dispatch", methods=["POST"])
def coding_dispatch_endpoint():
    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "")
    tool = data.get("tool", "codex")
    if not isinstance(prompt, str) or not prompt.strip():
        return jsonify({"status": "error", "reason": "prompt required"}), 400
    settings = load_settings()
    if settings.get("paused") or not ENABLED_ACTIONS.get("dispatch_coding_task", False):
        return jsonify(status="error", reason="Enable coding tasks in Actions and resume laptop controls first"), 403
    if not coding_manager:
        return jsonify({"status": "error", "reason": "Coding manager not initialized"}), 503
    workspace = data.get("cwd") if _local_session() else settings.get("coding_workspace")
    res = coding_manager.dispatch(prompt.strip(), tool=tool, cwd=workspace or settings.get("coding_workspace"))
    log_activity("dispatch_coding_task", status="ok" if res.get("status") == "dispatched" else "error")
    return jsonify(res), (400 if res.get("status") == "error" else 200)


@app.route("/coding/input", methods=["POST"])
def coding_input_endpoint():
    data = request.get_json(silent=True) or {}
    task_id = data.get("task_id", "")
    user_input = data.get("input", "")
    if not isinstance(task_id, str) or not isinstance(user_input, str):
        raise ValueError("Invalid coding task input")
    if not coding_manager:
        raise RuntimeError("Coding tools are not ready")
    res = coding_manager.send_input(task_id, user_input)
    log_activity("coding_input", status=res.get("status", "ok"))
    return jsonify(res)


@app.route("/coding/cancel", methods=["POST"])
def coding_cancel_endpoint():
    data = request.get_json(silent=True) or {}
    task_id = data.get("task_id")
    if task_id is not None and not isinstance(task_id, str):
        raise ValueError("Choose a valid coding task")
    if not coding_manager:
        raise RuntimeError("Coding tools are not ready")
    res = coding_manager.cancel(task_id)
    log_activity("cancel_coding_task", task_id, res.get("status", "ok"))
    return jsonify(res)


@app.route("/coding_task_status", methods=["GET"])
def coding_task_status_endpoint():
    data = request.args.to_dict()
    if not _is_request_authenticated(data):
        return jsonify({"status": "error", "reason": "invalid token"}), 401
    
    coding_status = coding_manager.get_status() if coding_manager else {}
    return jsonify({"status": "ok", **coding_status})


@app.route("/robot/state", methods=["POST"])
def robot_state_endpoint():
    """Update or simulate robot physical state for 3D mirror."""
    data = request.get_json(silent=True) or {}
    for key in ("emotion", "head", "speaking", "listening", "connected", "pi_ip"):
        if key in data:
            CURRENT_ROBOT_STATE[key] = data[key]
    CURRENT_ROBOT_STATE["last_seen"] = time.time()
    return jsonify({"status": "ok", "robot": CURRENT_ROBOT_STATE})


# ═════════════════════════════════════════════════════════════════════════
# PI DATA SYNC PROXY (Clock tab)
# ═════════════════════════════════════════════════════════════════════════
# The Clock tab shows the schedules/todos/memories that live on the Pi. The
# browser COULD call the Pi directly, but it goes through here instead, for
# three reasons:
#
#   1. The Pi binds plain HTTP on the LAN. The dashboard is served over
#      http://127.0.0.1, so a direct cross-origin call would need CORS on the
#      Pi (it has it) but would ALSO expose the Pi's SYNC_TOKEN to the page.
#      Proxying keeps that token server-side, in the same settings.json that
#      already holds pi_ip.
#   2. Discovery is already solved here. CURRENT_ROBOT_STATE["pi_ip"] knows
#      the Pi's address from the WS link; the page does not. Re-deriving it in
#      JavaScript would mean a second, divergent discovery path.
#   3. One place to time out. The Pi may be asleep or off; the browser getting
#      a clean {"ok": false} in ~4s is far better than a hanging fetch that
#      the UI has to guess about.
#
# These routes never raise: a Pi that is off is a normal state the UI renders,
# not an error.

_SYNC_TIMEOUT = (3.05, 6)


def _pi_call(method, path, body=None):
    return connection.call(method, path, body)


@app.route("/pi/snapshot", methods=["GET"])
def pi_snapshot_endpoint():
    """Everything the Clock tab needs, in one round trip."""
    data, err = _pi_call("GET", "/api/snapshot")
    if err:
        return jsonify({"status": "error", "reason": err,
                        "cached": False, "connection": connection.status()}), 200
    return jsonify({"status": "ok", "data": data.get("data", {}),
                    "connection": connection.status(),
                    "fetched_at": time.time()})


def _relay(payload, err, log_name: str, log_value=""):
    """Shape one fine-grained Pi write into the Clock tab's reply envelope.

    Always HTTP 200 with a status field — the tab treats "the Pi said no" and
    "the Pi is off" the same way (show the reason, don't pretend it worked),
    and a 4xx/5xx here would just make the fetch layer throw for no gain.
    """
    if err:
        return jsonify({"status": "error", "reason": err}), 200
    if not payload or not payload.get("ok", False):
        reason = (payload or {}).get("error", "the Pi rejected that change")
        return jsonify({"status": "error", "reason": reason}), 200
    log_activity(log_name, log_value, "ok")
    return jsonify({"status": "ok", "data": payload,
                    "fetched_at": time.time()})


@app.route("/pi/write/<path:target>", methods=["POST"])
def pi_write_endpoint(target: str):
    """Relay a Clock-tab edit to the Pi.

    Operations the Clock tab sends:

        {"toggle": "<todo id>"}        tick a to-do off
        {"delete": "<todo id>"}        remove a to-do
        {"cancel": "<schedule id>"}    remove an alarm/timer/reminder
        {"replace": true, "todos"|"schedules": [...]}   whole-list replace
        anything else                  passed through as a POST (add one)

    The single-item verbs are forwarded to the Pi's own endpoints rather than
    being done here as read-modify-write of the whole list. That is on purpose:
    the Pi's scheduler validates and timestamps its own entries, and doing the
    edit on its side means the entry keeps the fields the display snapshot
    omits (a recurring alarm's stored `at`, its `snoozes` count). Rewriting
    the list from a snapshot could quietly drop a recurring alarm.
    """
    if target not in ("todos", "schedules"):
        return jsonify({"status": "error",
                        "reason": f"unknown target '{target}'"}), 400

    body = request.get_json(silent=True) or {}

    # ── single-item verbs → the Pi's own fine-grained endpoints ─────────
    # POST, not PUT: these are actions, and each returns the refreshed list.
    if "toggle" in body:
        payload, err = _pi_call("POST", "/api/todos/done",
                                {"id": str(body["toggle"])})
        return _relay(payload, err, "pi_todos_done", body["toggle"])

    if "delete" in body and target == "todos":
        payload, err = _pi_call("POST", "/api/todos/delete",
                                {"id": str(body["delete"])})
        return _relay(payload, err, "pi_todos_delete", body["delete"])

    if "cancel" in body and target == "schedules":
        payload, err = _pi_call("POST", "/api/schedules/cancel",
                                {"id": str(body["cancel"])})
        return _relay(payload, err, "pi_schedules_cancel", body["cancel"])

    # ── whole-list replace / plain add ──────────────────────────────────
    replace = bool(body.pop("replace", False))
    verb = "PUT" if replace else "POST"

    payload, err = _pi_call(verb, f"/api/{target}", body)
    if err:
        return jsonify({"status": "error", "reason": err}), 200

    if not payload.get("ok", False):
        return jsonify({"status": "error",
                        "reason": payload.get("error", "the Pi rejected it")}), 200

    log_activity(f"pi_{target}_{'replace' if replace else 'add'}",
                 body.get("text") or body.get("label") or "", "ok")
    return jsonify({"status": "ok", "data": payload,
                    "fetched_at": time.time()})


@app.route("/pi/ping", methods=["GET"])
def pi_ping_endpoint():
    """Is the Pi's sync API up, and is it writable?"""
    payload, err = _pi_call("GET", "/api/ping")
    if err:
        return jsonify({"status": "error", "reachable": False, "reason": err})
    return jsonify({"status": "ok", "reachable": True, "pi": payload})


# =========================================================================
# ADAM DISCOVERY - "Find ADAM", then one click to pair
#
# This is the other half of mDNS. Below, the app ADVERTISES itself as
# `_adam-laptop._tcp` so the Pi can find the laptop it controls. Here we BROWSE
# for `_adam._tcp`, which the Pi publishes (pi/adam/discovery.py).
#
# That direction did not exist before, which is why connecting ADAM meant
# typing an IP into Settings and pasting a token out of the Pi's .env by hand.
# The address then went stale every time DHCP moved the Pi, and the Clock tab
# just reported "not connected".
#
# The ServiceBrowser is kept running for the life of the app rather than
# started per request: mDNS answers arrive asynchronously, so a browser created
# inside a request handler would have to block for seconds and would still miss
# units that answer late. "Find ADAM" reads an already-warm cache instead.
# =========================================================================

_adam_browser = None
_adam_zc = None
_found_adams = {}          # deviceId -> record

ADAM_SERVICE_TYPE = "_adam._tcp.local."
ADAM_STALE_AFTER_S = 120


def _decode_txt(props):
    out = {}
    for k, v in (props or {}).items():
        try:
            key = k.decode() if isinstance(k, bytes) else str(k)
            if v is None:
                continue
            out[key] = v.decode() if isinstance(v, bytes) else str(v)
        except Exception:
            continue
    return out


def start_adam_discovery():
    """Begin browsing for ADAM units. Never raises - discovery is a
    convenience, and the manual IP override in Settings still works."""
    global _adam_browser, _adam_zc
    if _adam_browser is not None:
        return
    try:
        from zeroconf import ServiceBrowser, Zeroconf, ServiceListener
    except ImportError:
        logger.warning("zeroconf not installed - 'Find ADAM' will not work.")
        return

    class _Listener(ServiceListener):
        def _record(self, zc, type_, name):
            try:
                info = zc.get_service_info(type_, name, timeout=3000)
                if not info or not info.addresses:
                    return
                txt = _decode_txt(info.properties)
                ip = socket.inet_ntoa(info.addresses[0])
                dev_id = txt.get("id") or name.split(".")[0]
                _found_adams[name + "@" + ip] = {
                    "id": dev_id,
                    "name": txt.get("name", "ADAM"),
                    "host": ip,
                    "port": int(info.port or 8766),
                    "version": txt.get("ver", ""),
                    "api": txt.get("api", ""),
                    # From the TXT record, so an already-claimed unit can be
                    # shown greyed out rather than letting the user pick it and
                    # then fail with a 409.
                    "paired": txt.get("paired") == "1",
                    "seen_at": time.time(),
                }
                logger.info("[discovery] found %s at %s:%s", dev_id, ip, info.port)
            except Exception as e:
                logger.warning("[discovery] could not read %s: %s", name, e)

        def add_service(self, zc, type_, name):
            self._record(zc, type_, name)

        def update_service(self, zc, type_, name):
            self._record(zc, type_, name)

        def remove_service(self, zc, type_, name):
            for k in list(_found_adams):
                if k.startswith(name + "@") :
                    _found_adams.pop(k, None)

    try:
        _adam_zc = Zeroconf()
        _adam_browser = ServiceBrowser(_adam_zc, ADAM_SERVICE_TYPE, _Listener())
        logger.info("[discovery] browsing for '%s'", ADAM_SERVICE_TYPE)
    except Exception as e:
        logger.warning("[discovery] browse failed: %s", e)
        _adam_browser = None


def stop_adam_discovery():
    global _adam_browser, _adam_zc
    try:
        if _adam_zc is not None:
            _adam_zc.close()
    except Exception:
        pass
    finally:
        _adam_browser, _adam_zc = None, None


@app.route("/discover/adam", methods=["GET"])
def discover_adam_endpoint():
    """What "Find ADAM" renders. Starts the browser on first call, so a user
    who never opens this screen pays nothing for it."""
    start_adam_discovery()
    # Drop units not seen recently: an ADAM that was switched off should leave
    # the list rather than sit there failing to connect.
    cutoff = time.time() - ADAM_STALE_AFTER_S
    units = [dict(u) for u in _found_adams.values() if u["seen_at"] > cutoff]
    units.sort(key=lambda u: u["id"])
    settings = load_settings()
    current = str(settings.get("pi_ip", "")).strip()
    have_token = bool(str(settings.get("sync_token", "")).strip())
    for u in units:
        u["connected"] = bool(current) and u["host"] == current and have_token
    return jsonify({"status": "ok", "units": units, "count": len(units)})


@app.route("/pair/adam", methods=["POST"])
def pair_adam_endpoint():
    return jsonify(status='error', reason='Use the account device picker to authorize ADAM.'), 410


@app.route("/pair/adam/forget", methods=["POST"])
def unpair_adam_endpoint():
    """Release the claim so another device can pair, and forget it locally.

    Order matters: tell the Pi FIRST, while we still hold the token that
    authorises the release. Clearing our settings first would leave the Pi
    permanently claimed by a device that can no longer prove it owns it, and
    the user would have to SSH in to recover.
    """
    payload, err = _pi_call("POST", "/api/pair/release", {})
    # Clear local state through the connection module for the same reason
    # pairing goes through it: it owns paired/sync_token_host/the telemetry run,
    # not just the two settings keys.
    connection.disconnect()
    settings = load_settings()
    settings["pi_device_id"] = ""
    save_settings(settings)
    log_activity("adam_unpaired", "", "ok" if not err else "warn")
    if err:
        return jsonify({"status": "ok",
                        "warning": "Forgotten locally, but ADAM did not confirm: %s" % err})
    return jsonify({"status": "ok"})


# ═════════════════════════════════════════════════════════════════════════
# mDNS BROADCAST
# ═════════════════════════════════════════════════════════════════════════

def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        pass
    try:
        hostname_ip = socket.gethostbyname(socket.gethostname())
        if not hostname_ip.startswith("127."):
            return hostname_ip
    except Exception:
        pass
    return "127.0.0.1"


def start_mdns_broadcast(port: int) -> None:
    global _zeroconf_instance, _service_info
    try:
        from zeroconf import ServiceInfo, Zeroconf
    except ImportError:
        logger.warning("zeroconf not installed — Pi will need LAPTOP_AGENT_IP set manually.")
        return
    try:
        hostname = socket.gethostname()
        local_ip = _get_local_ip()
        _service_info = ServiceInfo(
            MDNS_SERVICE_TYPE,
            f"{hostname}.{MDNS_SERVICE_TYPE}",
            addresses=[socket.inet_aton(local_ip)],
            port=port,
            properties={"platform": SYSTEM, "version": APP_VERSION, "app": APP_NAME},
        )
        _zeroconf_instance = Zeroconf()
        _zeroconf_instance.register_service(_service_info)
        logger.info(f"[mDNS] Registered as '{MDNS_SERVICE_TYPE}' on {local_ip}:{port}")
    except Exception as e:
        logger.warning(f"mDNS broadcast setup warning: {e}")


def stop_mdns_broadcast() -> None:
    global _zeroconf_instance, _service_info
    if _zeroconf_instance and _service_info:
        try:
            _zeroconf_instance.unregister_service(_service_info)
            _zeroconf_instance.close()
            logger.info("mDNS unregistered cleanly.")
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════
# PI WEBSOCKET CLIENT (Phase 4 3D Status Mirror)
# ═════════════════════════════════════════════════════════════════════════

def start_pi_ws_client():
    # Onboarding revalidates ownership and the pinned grant before starting.
    return


def stop_pi_ws_client():
    connection.stop()


# ═════════════════════════════════════════════════════════════════════════
# LIFECYCLE HELPERS
# ═════════════════════════════════════════════════════════════════════════

def restart_backend_services():
    """Restart local mDNS broadcast and Pi WebSocket listener cleanly."""
    settings = load_settings()
    port = settings.get("agent_port", 8642)
    stop_mdns_broadcast()
    stop_pi_ws_client()
    start_mdns_broadcast(port)
    start_pi_ws_client()
    logger.info("Agent backend services restarted.")


def init_backend(on_show_window: Optional[Callable[[], None]] = None):
    """Initialize coding manager, WS client, and callbacks."""
    global window_show_callback, coding_manager, _hw_worker
    window_show_callback = on_show_window

    def on_coding_change(task_dict):
        log_activity(
            f"coding_{task_dict['state']}",
            task_dict.get("id"),
            task_dict["state"],
            "Coding task state changed"
        )

    coding_manager = CodingAgentManager(on_state_change=on_coding_change)
    if SYSTEM == "windows" and not os.environ.get("ADAM_DISABLE_HARDWARE") and _hw_worker is None:
        _hw_worker = WindowsHardwareWorker()
    start_pi_ws_client()


_http_server = None
server_ready = threading.Event()
server_error = ""
server_port = None


def run_flask_server(port):
    global _http_server, server_error, server_port
    listener = None
    server_ready.clear()
    server_error = ""
    try:
        from waitress import create_server
        from desktop_runtime import reserve_listener
        listener = reserve_listener(port)
        server_port = listener.getsockname()[1]
        _http_server = create_server(app, sockets=[listener], threads=8,
                                     max_request_body_size=1_000_000, channel_timeout=30)
        update_settings({"agent_port": server_port})
        if server_port != port:
            logger.info("Preferred port unavailable; using reserved port %s", server_port)
        server_ready.set()
        start_mdns_broadcast(server_port)
        _http_server.run()
    except Exception as error:
        server_error = "The local service could not start. The port may already be in use."
        logger.error("Local service failed: %s", type(error).__name__)
        server_ready.set()
    finally:
        stop_mdns_broadcast()
        if listener:
            listener.close()


def stop_backend():
    connection.stop()
    stop_mdns_broadcast()
    if _hw_worker:
        _hw_worker.stop()
    if coding_manager:
        coding_manager.cancel()
    account.cancel_google()
    if _http_server:
        _http_server.close()
