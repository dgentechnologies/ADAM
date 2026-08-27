"""
ADAM Laptop Agent — MODULAR ACTION REGISTRY EDITION
============================================================================
DGEN Technologies Pvt. Ltd.

Runs on YOUR LAPTOP (not the Pi). Exposes a local HTTP endpoint that
adam_main_pi.py's laptop_control tool calls over the LAN.

──────────────────────────────────────────────────────────────────────────
WHY THIS FILE LOOKS DIFFERENT FROM v1
──────────────────────────────────────────────────────────────────────────
Old version: /control had one big if/elif chain, and adding "mute Spotify"
meant editing this file AND adam_main_pi.py's enum AND its description
string AND its dispatch code. Four edits, two files, easy to typo the enum
and break the tool schema silently.

New version: every action is a small function decorated with @action(...).
The decorator registers it in ACTIONS, a name -> spec dict. Two new
endpoints expose that registry:

    GET /actions        -> full self-describing manifest (used by the Pi
                            to build its Gemini tool schema automatically)
    POST /control        -> generic dispatcher, looks the action up in
                            ACTIONS and calls it — no if/elif needed

TO ADD A NEW CAPABILITY (e.g. "mute_spotify", "lock_screen", "open_app"):
    1. Write one function below, decorated with @action(...).
    2. That's it. Restart the agent. The Pi will pick it up automatically
       next time it calls /actions (on its own startup, or on-demand if
       you wire a refresh — see adam_main_pi.py's `refresh_laptop_actions`).

No enum to edit. No dispatch table to edit. No description string to edit
in two places. One function, one file.

──────────────────────────────────────────────────────────────────────────
SETUP  (same as before)
──────────────────────────────────────────────────────────────────────────
    pip install flask zeroconf
    Windows:  pip install pycaw comtypes screen-brightness-control
    macOS:    brew install brightness
    Linux:    pip install screen-brightness-control   (+ alsa-utils for amixer)

    Copy .env.example -> .env, set AGENT_TOKEN + AGENT_PORT, then:
    python laptop_agent.py

──────────────────────────────────────────────────────────────────────────
SECURITY NOTE — unchanged from v1
──────────────────────────────────────────────────────────────────────────
Binds 0.0.0.0 so the Pi can reach it. The shared token is the real security
boundary — keep it secret, don't port-forward this, LAN only.
"""

import os
import socket
import subprocess
import platform
import inspect
import sys
from pathlib import Path
from typing import Callable, Any

from flask import Flask, request, jsonify
from dotenv import load_dotenv

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

AGENT_TOKEN = os.getenv("AGENT_TOKEN", "").strip()
AGENT_PORT = int(os.getenv("AGENT_PORT", "8642"))

if not AGENT_TOKEN:
    print("⚠️  WARNING: AGENT_TOKEN not set in .env — anyone on your LAN could control this laptop.")

SYSTEM = platform.system().lower()  # 'windows', 'darwin', 'linux'

app = Flask(__name__)


# ═════════════════════════════════════════════════════════════════════════
# ACTION REGISTRY — the whole point of this rewrite
# ═════════════════════════════════════════════════════════════════════════
#
# ACTIONS["volume_set"] = {
#     "fn": <function>,
#     "description": "...",          # fed straight into the Gemini tool schema
#     "needs_value": True/False,     # whether the Pi must send an int 'value'
#     "value_hint": "0-100",         # shown to Gemini in the schema description
# }

ACTIONS: dict[str, dict[str, Any]] = {}


def action(name: str, description: str, needs_value: bool = False,
           value_hint: str = "") -> Callable:
    """Decorator that registers a function as a callable laptop action.

    The decorated function must return a JSON-serialisable dict (or None,
    in which case {"status": "ok"} is assumed). If needs_value=True, the
    function will be called as fn(value) — otherwise fn() with no args.
    """
    def wrapper(fn: Callable) -> Callable:
        ACTIONS[name] = {
            "fn": fn,
            "description": description,
            "needs_value": needs_value,
            "value_hint": value_hint,
        }
        return fn
    return wrapper


# ═════════════════════════════════════════════════════════════════════════
# PLATFORM BACKENDS (unchanged logic, just called from the registry now)
# ═════════════════════════════════════════════════════════════════════════

# ── WINDOWS ──────────────────────────────────────────────────────────────
def _run_with_windows_com(fn: Callable[[], Any]) -> Any:
    """Run a Windows API call from Flask worker threads with COM initialized."""
    import comtypes

    initialized = False
    try:
        comtypes.CoInitialize()
        initialized = True
    except OSError as e:
        if getattr(e, "winerror", None) != -2147417850:
            raise

    try:
        return fn()
    finally:
        if initialized:
            comtypes.CoUninitialize()


def _win_volume_iface():
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))


def _win_get_volume() -> int:
    return _run_with_windows_com(
        lambda: round(_win_volume_iface().GetMasterVolumeLevelScalar() * 100)
    )


def _win_set_volume(pct: int) -> None:
    _run_with_windows_com(
        lambda: _win_volume_iface().SetMasterVolumeLevelScalar(
            max(0, min(100, pct)) / 100.0, None
        )
    )


def _win_set_mute(mute: bool) -> None:
    _run_with_windows_com(lambda: _win_volume_iface().SetMute(1 if mute else 0, None))


def _win_get_brightness() -> int:
    def _get() -> int:
        import screen_brightness_control as sbc
        vals = sbc.get_brightness()
        return vals[0] if isinstance(vals, list) else vals

    return _run_with_windows_com(_get)


def _win_set_brightness(pct: int) -> None:
    def _set() -> None:
        import screen_brightness_control as sbc
        sbc.set_brightness(max(0, min(100, pct)))

    _run_with_windows_com(_set)


# ── macOS ────────────────────────────────────────────────────────────────
def _mac_get_volume() -> int:
    out = subprocess.run(["osascript", "-e", "output volume of (get volume settings)"],
                          check=True, capture_output=True, text=True)
    return int(out.stdout.strip())


def _mac_set_volume(pct: int) -> None:
    subprocess.run(["osascript", "-e", f"set volume output volume {max(0, min(100, pct))}"],
                    check=True)


def _mac_set_mute(mute: bool) -> None:
    subprocess.run(["osascript", "-e", f"set volume output muted {'true' if mute else 'false'}"],
                    check=True)


def _mac_get_brightness() -> int:
    out = subprocess.run(["brightness", "-l"], check=True, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if "brightness" in line.lower():
            try:
                return round(float(line.strip().split()[-1]) * 100)
            except Exception:
                pass
    return -1


def _mac_set_brightness(pct: int) -> None:
    subprocess.run(["brightness", str(max(0, min(100, pct)) / 100.0)], check=True)


# ── Linux ────────────────────────────────────────────────────────────────
def _linux_get_volume() -> int:
    out = subprocess.run(["amixer", "get", "Master"], check=True, capture_output=True, text=True)
    for line in out.stdout.splitlines():
        if "%" in line:
            try:
                return int(line.split("[")[1].split("%")[0])
            except Exception:
                pass
    return -1


def _linux_set_volume(pct: int) -> None:
    subprocess.run(["amixer", "set", "Master", f"{max(0, min(100, pct))}%"], check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _linux_set_mute(mute: bool) -> None:
    subprocess.run(["amixer", "set", "Master", "mute" if mute else "unmute"], check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _linux_get_brightness() -> int:
    import screen_brightness_control as sbc
    vals = sbc.get_brightness()
    return vals[0] if isinstance(vals, list) else vals


def _linux_set_brightness(pct: int) -> None:
    import screen_brightness_control as sbc
    sbc.set_brightness(max(0, min(100, pct)))


# ── Cross-platform dispatch shims (pick the right backend once) ──────────
def _get_volume() -> int:
    return {"windows": _win_get_volume, "darwin": _mac_get_volume}.get(SYSTEM, _linux_get_volume)()


def _set_volume(pct: int) -> None:
    {"windows": _win_set_volume, "darwin": _mac_set_volume}.get(SYSTEM, _linux_set_volume)(pct)


def _set_mute(mute: bool) -> None:
    {"windows": _win_set_mute, "darwin": _mac_set_mute}.get(SYSTEM, _linux_set_mute)(mute)


def _get_brightness() -> int:
    return {"windows": _win_get_brightness, "darwin": _mac_get_brightness}.get(
        SYSTEM, _linux_get_brightness)()


def _set_brightness(pct: int) -> None:
    {"windows": _win_set_brightness, "darwin": _mac_set_brightness}.get(
        SYSTEM, _linux_set_brightness)(pct)


VOLUME_STEP = 10
BRIGHTNESS_STEP = 10


# ═════════════════════════════════════════════════════════════════════════
# REGISTERED ACTIONS — add new capabilities HERE and only here
# ═════════════════════════════════════════════════════════════════════════

@action("volume_up", "Increase system volume by 10%.")
def act_volume_up():
    new_val = min(100, _get_volume() + VOLUME_STEP)
    _set_volume(new_val)
    return {"volume": new_val}


@action("volume_down", "Decrease system volume by 10%.")
def act_volume_down():
    new_val = max(0, _get_volume() - VOLUME_STEP)
    _set_volume(new_val)
    return {"volume": new_val}


@action("volume_set", "Set system volume to an exact percentage.",
        needs_value=True, value_hint="0-100")
def act_volume_set(value: int):
    _set_volume(value)
    return {"volume": value}


@action("volume_mute", "Mute system audio.")
def act_volume_mute():
    _set_mute(True)
    return {}


@action("volume_unmute", "Unmute system audio.")
def act_volume_unmute():
    _set_mute(False)
    return {}


@action("brightness_up", "Increase screen brightness by 10%.")
def act_brightness_up():
    new_val = min(100, _get_brightness() + BRIGHTNESS_STEP)
    _set_brightness(new_val)
    return {"brightness": new_val}


@action("brightness_down", "Decrease screen brightness by 10%.")
def act_brightness_down():
    new_val = max(0, _get_brightness() - BRIGHTNESS_STEP)
    _set_brightness(new_val)
    return {"brightness": new_val}


@action("brightness_set", "Set screen brightness to an exact percentage.",
        needs_value=True, value_hint="0-100")
def act_brightness_set(value: int):
    _set_brightness(value)
    return {"brightness": value}


# ──────────────────────────────────────────────────────────────────────────
# EXAMPLE: adding a brand-new capability is THIS SHORT. Uncomment + adapt.
# No other file needs to change. The Pi discovers this automatically via
# GET /actions on its next refresh.
# ──────────────────────────────────────────────────────────────────────────
#
# @action("lock_screen", "Lock the laptop's screen immediately.")
# def act_lock_screen():
#     if SYSTEM == "windows":
#         subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"], check=True)
#     elif SYSTEM == "darwin":
#         subprocess.run(["pmset", "displaysleepnow"], check=True)
#     else:
#         subprocess.run(["loginctl", "lock-session"], check=True)
#     return {}


# ═════════════════════════════════════════════════════════════════════════
# mDNS BROADCAST (unchanged from v1)
# ═════════════════════════════════════════════════════════════════════════

_zeroconf_instance = None
_service_info = None
MDNS_SERVICE_TYPE = "_adam-laptop._tcp.local."


def _get_local_ip() -> str:
    try:
        hostname_ip = socket.gethostbyname(socket.gethostname())
        if not hostname_ip.startswith("127."):
            return hostname_ip
    except Exception:
        pass
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def start_mdns_broadcast() -> None:
    global _zeroconf_instance, _service_info
    try:
        from zeroconf import ServiceInfo, Zeroconf
    except ImportError:
        print("⚠️  zeroconf not installed — Pi will need LAPTOP_AGENT_IP set manually "
              "(pip install zeroconf to enable auto-discovery)")
        return
    try:
        hostname = socket.gethostname()
        local_ip = _get_local_ip()
        _service_info = ServiceInfo(
            MDNS_SERVICE_TYPE,
            f"{hostname}.{MDNS_SERVICE_TYPE}",
            addresses=[socket.inet_aton(local_ip)],
            port=AGENT_PORT,
            properties={"platform": SYSTEM, "version": "2"},
        )
        _zeroconf_instance = Zeroconf()
        _zeroconf_instance.register_service(_service_info)
        print(f"📡 mDNS broadcasting as '{MDNS_SERVICE_TYPE}' on {local_ip}:{AGENT_PORT}")
    except Exception as e:
        print(f"⚠️  mDNS broadcast failed: {e}")


def stop_mdns_broadcast() -> None:
    global _zeroconf_instance, _service_info
    if _zeroconf_instance and _service_info:
        try:
            _zeroconf_instance.unregister_service(_service_info)
            _zeroconf_instance.close()
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════
# HTTP ENDPOINTS
# ═════════════════════════════════════════════════════════════════════════

@app.route("/actions", methods=["GET"])
def list_actions():
    """Self-describing manifest. The Pi calls this to build its Gemini tool
    schema dynamically — add an @action() here and the Pi's enum grows on
    its own next refresh, no code change needed on the Pi side."""
    manifest = {
        name: {
            "description": spec["description"],
            "needs_value": spec["needs_value"],
            "value_hint": spec["value_hint"],
        }
        for name, spec in ACTIONS.items()
    }
    return jsonify({"platform": SYSTEM, "actions": manifest})


@app.route("/control", methods=["POST"])
def control():
    data = request.get_json(silent=True) or {}

    if not AGENT_TOKEN:
        return jsonify({"status": "error", "reason": "AGENT_TOKEN not configured"}), 503

    if data.get("token") != AGENT_TOKEN:
        return jsonify({"status": "error", "reason": "invalid token"}), 401

    action_name = data.get("action", "")
    value = data.get("value")

    spec = ACTIONS.get(action_name)
    if spec is None:
        return jsonify({"status": "error",
                         "reason": f"unknown action: {action_name}",
                         "available": list(ACTIONS.keys())}), 400

    if spec["needs_value"] and value is None:
        return jsonify({"status": "error", "reason": "value required"}), 400

    try:
        if spec["needs_value"]:
            result = spec["fn"](int(value))
        else:
            result = spec["fn"]()
        result = result or {}
        return jsonify({"status": "ok", "action": action_name, **result})
    except Exception as e:
        return jsonify({"status": "error",
                         "reason": f"{type(e).__name__}: {e}"}), 500


@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "ok", "platform": SYSTEM, "action_count": len(ACTIONS)})


if __name__ == "__main__":
    print(f"🖥️  ADAM Laptop Agent (modular) starting")
    print(f"    Platform: {SYSTEM}")
    print(f"    Port: {AGENT_PORT}")
    print(f"    Token set: {'yes' if AGENT_TOKEN else 'NO — insecure!'}")
    print(f"    Registered actions ({len(ACTIONS)}): {', '.join(ACTIONS.keys())}")
    if SYSTEM not in ("windows", "darwin", "linux"):
        print(f"    ⚠️  Unrecognized platform '{SYSTEM}' — some actions may fail.")

    start_mdns_broadcast()
    try:
        app.run(host="0.0.0.0", port=AGENT_PORT, debug=False)
    finally:
        stop_mdns_broadcast()
