"""
adam_neck_serial.py — ADAM neck servo driver (v2)
==================================================
Wraps serial communication to the Arduino running adam_neck_servo.ino.

New in v2:
  • set_speed(n)  — send SPEED<n> to Arduino (1=slow .. 10=fast)
  • pan() / tilt() accept an optional speed argument
  • All named moves accept an optional speed argument
  • Speed is sent as a separate SPEED command before the move, then
    restored to the global default after — so per-move speed overrides
    are self-contained and don't permanently change the global rate.
"""

import serial
import serial.tools.list_ports
import time
import threading

_ser: serial.Serial | None = None
_lock = threading.Lock()
_ready = False
_default_speed = 5   # matches Arduino default

BAUD = 9600
TIMEOUT = 2.0

# ── Named move speed presets ──────────────────────────────────────
# Maps emotion/context → Arduino speed value (1-10)
_EMOTION_SPEED = {
    "happy":     7,
    "excited":   9,
    "angry":     8,
    "confused":  3,
    "smug":      4,
    "sad":       2,
    "surprised": 9,
    "thinking":  2,
    "love":      4,
    "blush":     3,
}

_MOVE_SPEED = {
    "NOD":          6,
    "NOD_FAST":     10,
    "SHAKE":        5,
    "RESET":        4,
    "LOOK_UP":      5,
    "LOOK_DOWN":    5,
    "LOOK_LEFT":    6,
    "LOOK_RIGHT":   6,
    "TILT_CURIOUS": 3,
}


def _send(cmd: str) -> str | None:
    """Send a newline-terminated command; return the reply line or None."""
    global _ser
    if not _ser or not _ser.is_open:
        return None
    with _lock:
        try:
            _ser.write((cmd.strip() + "\n").encode())
            _ser.flush()
            reply = _ser.readline().decode("utf-8", errors="ignore").strip()
            return reply or None
        except Exception as e:
            print(f"  [neck] serial error: {e}")
            return None


def _auto_detect_port() -> str | None:
    """Return the first USB/ACM port that looks like an Arduino."""
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        hwid = (p.hwid or "").lower()
        if any(k in desc for k in ["arduino", "ch340", "cp210", "ftdi", "uno"]):
            return p.device
        if any(k in hwid for k in ["2341", "1a86", "10c4", "0403"]):
            return p.device
    # Fallback: first ACM/USB port
    for p in serial.tools.list_ports.comports():
        if "ACM" in p.device or "USB" in p.device:
            return p.device
    return None


def init_neck(port: str | None = None, baud: int = BAUD) -> bool:
    """Open serial connection to Arduino.  Auto-detects port if not given."""
    global _ser, _ready
    _port = port or _auto_detect_port()
    if not _port:
        print("  [neck] No Arduino port found — servo disabled")
        return False
    try:
        _ser = serial.Serial(_port, baud, timeout=TIMEOUT)
        time.sleep(2.0)  # Arduino resets on serial open
        _ser.reset_input_buffer()
        # Wait for READY signal
        deadline = time.time() + 4.0
        while time.time() < deadline:
            line = _ser.readline().decode("utf-8", errors="ignore").strip()
            if "READY" in line:
                _ready = True
                print(f"  [neck] Connected on {_port} — servo ready")
                return True
        print(f"  [neck] Timeout waiting for ADAM_SERVO_READY on {_port}")
        return False
    except Exception as e:
        print(f"  [neck] Init failed ({_port}): {e}")
        return False


def is_ready() -> bool:
    return _ready and _ser is not None and _ser.is_open


def close_neck() -> None:
    global _ser, _ready
    if _ser and _ser.is_open:
        try:
            _send("NRESET")
            time.sleep(0.5)
            _ser.close()
        except Exception:
            pass
    _ready = False


# ── Speed control ─────────────────────────────────────────────────

def set_speed(speed: int) -> None:
    """Set global servo speed on Arduino (1=very slow .. 10=very fast)."""
    spd = max(1, min(10, int(speed)))
    _send(f"SPEED{spd}")


# ── Direct angle commands ──────────────────────────────────────────

def pan(angle: int, speed: int | None = None) -> None:
    """Pan to absolute angle.  Optionally set speed for this move only."""
    if speed is not None:
        set_speed(speed)
    _send(f"P{int(angle)}")
    if speed is not None:
        set_speed(_default_speed)  # restore


def tilt(angle: int, speed: int | None = None) -> None:
    """Tilt to absolute angle.  Optionally set speed for this move only."""
    if speed is not None:
        set_speed(speed)
    _send(f"T{int(angle)}")
    if speed is not None:
        set_speed(_default_speed)


def reset_neck() -> None:
    _send("NRESET")


# ── Named moves ───────────────────────────────────────────────────

def named_move(move: str, speed: int | None = None) -> None:
    """
    Execute a named movement preset.
    If speed is None, use the preset's natural speed from _MOVE_SPEED.
    """
    m = move.upper().strip()
    effective_speed = speed if speed is not None else _MOVE_SPEED.get(m, _default_speed)
    set_speed(effective_speed)
    _send(f"N{m}")
    set_speed(_default_speed)  # restore


# ── Emotion → named move ──────────────────────────────────────────

def emotion_move(emotion: str) -> None:
    """
    Trigger the physical movement associated with an emotion.
    Uses emotion-appropriate speed automatically.
    """
    e = emotion.lower().strip()
    speed = _EMOTION_SPEED.get(e, _default_speed)

    move_map = {
        "happy":     "NOD",
        "excited":   "NOD_FAST",
        "surprised": "NOD",
        "love":      "NOD",
        "thinking":  "TILT_CURIOUS",
        "confused":  "TILT_CURIOUS",
    }
    move = move_map.get(e)
    if move:
        set_speed(speed)
        _send(f"N{move}")
        set_speed(_default_speed)