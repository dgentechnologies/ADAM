"""
adam_neck_serial.py — ADAM Servo Neck Controller
==================================================
Drop this file next to adamV24.py.
Import it in adamV24.py and call init_neck() at startup.

Requires:  pip install pyserial

Auto-detects the Arduino COM port on Windows / Linux / macOS.
Falls back to NECK_PORT env variable or manual config below.

Serial protocol (matches adam_neck_servo.ino):
  P<angle>   → Pan  servo  (0–180, centre=90)
  T<angle>   → Tilt servo  (0–180, centre=85)
  N<name>    → Named animation
  S          → Status ping

Named moves: NOD, SHAKE, RESET, LOOK_UP, LOOK_DOWN,
             LOOK_LEFT, LOOK_RIGHT, TILT_CURIOUS
"""

import os
import time
import threading
import serial
import serial.tools.list_ports

# ── Config ────────────────────────────────────────────────────────
BAUD_RATE        = 9600
CONNECT_TIMEOUT  = 3       # seconds to wait for READY handshake
SERIAL_TIMEOUT   = 1.0
COMMAND_DELAY    = 0.02    # seconds between queued commands
MANUAL_PORT      = None    # Set to "COM3" or "/dev/ttyUSB0" to skip auto-detect

# ── State ─────────────────────────────────────────────────────────
_ser:   serial.Serial | None = None
_lock   = threading.Lock()
_ready  = threading.Event()
_queue  = []
_thread: threading.Thread | None = None


# ═════════════════════════════════════════════════════════════════
# PORT AUTO-DETECT
# ═════════════════════════════════════════════════════════════════

def _find_arduino_port() -> str | None:
    """Scan serial ports for an Arduino Uno (USB-Serial)."""
    # Check env override first
    env_port = os.getenv("NECK_PORT") or MANUAL_PORT
    if env_port:
        print(f"  🦾  Neck: using port from config: {env_port}")
        return env_port

    candidates = []
    for port in serial.tools.list_ports.comports():
        desc = (port.description or "").lower()
        mfr  = (port.manufacturer or "").lower()
        vid  = port.vid

        # Arduino Uno VID = 0x2341, CH340 clones = 0x1A86
        is_arduino = (
            vid in (0x2341, 0x1A86, 0x0403) or
            "arduino" in desc or "arduino" in mfr or
            "ch340"   in desc or "ch340"   in mfr or
            "usb serial" in desc
        )
        if is_arduino:
            candidates.append(port.device)
            print(f"  🦾  Neck: found candidate → {port.device} ({port.description})")

    return candidates[0] if candidates else None


# ═════════════════════════════════════════════════════════════════
# SEND / RECEIVE
# ═════════════════════════════════════════════════════════════════

def _send_raw(cmd: str) -> bool:
    """Send a single command string (newline appended). Thread-safe."""
    global _ser
    if _ser is None or not _ser.is_open:
        return False
    try:
        with _lock:
            _ser.write((cmd.strip() + "\n").encode("ascii"))
            _ser.flush()
        return True
    except Exception as e:
        print(f"  ⚠️  Neck serial write error: {e}")
        return False


def _reader_thread():
    """Background thread: reads Arduino responses and prints them."""
    global _ser
    while _ser and _ser.is_open:
        try:
            line = _ser.readline().decode("ascii", errors="ignore").strip()
            if line:
                print(f"  🦾  Arduino → {line}")
        except Exception:
            break


# ═════════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════════

def init_neck() -> bool:
    """
    Connect to Arduino and wait for ADAM_SERVO_READY handshake.
    Call once at ADAM startup. Returns True if connected.
    """
    global _ser, _thread

    port = _find_arduino_port()
    if not port:
        print("  ⚠️  Neck: no Arduino found — servo disabled. "
              "Set NECK_PORT env var or MANUAL_PORT in adam_neck_serial.py")
        return False

    try:
        _ser = serial.Serial(port, BAUD_RATE, timeout=SERIAL_TIMEOUT)
        print(f"  🦾  Neck: opened {port} @ {BAUD_RATE} baud")

        # Arduino resets on serial open — wait for READY
        deadline = time.time() + CONNECT_TIMEOUT + 2.0
        while time.time() < deadline:
            line = _ser.readline().decode("ascii", errors="ignore").strip()
            if line:
                print(f"  🦾  Arduino boot → {line}")
            if "ADAM_SERVO_READY" in line:
                print("  ✅  Neck servos ready")
                _ready.set()
                break
        else:
            # Try a ping
            _send_raw("S")
            time.sleep(0.3)
            resp = _ser.readline().decode("ascii", errors="ignore").strip()
            if resp == "OK":
                print("  ✅  Neck servos ready (ping OK)")
                _ready.set()
            else:
                print("  ⚠️  Neck: no READY handshake — proceeding anyway")
                _ready.set()   # don't block ADAM startup

        # Start background reader
        _thread = threading.Thread(target=_reader_thread, daemon=True, name="neck-reader")
        _thread.start()

        # Go to neutral on connect
        reset_neck()
        return True

    except Exception as e:
        print(f"  ⚠️  Neck: failed to open {port}: {e}")
        _ser = None
        return False


def is_ready() -> bool:
    return _ready.is_set() and _ser is not None and _ser.is_open


def pan(angle: int) -> bool:
    """Pan servo to angle (30–150, centre=90)."""
    angle = max(30, min(150, int(angle)))
    return _send_raw(f"P{angle}")


def tilt(angle: int) -> bool:
    """Tilt servo to angle (50–120, centre=85)."""
    angle = max(50, min(120, int(angle)))
    return _send_raw(f"T{angle}")


def named_move(move: str) -> bool:
    """
    Trigger a named animation on the Arduino.
    move: NOD | SHAKE | RESET | LOOK_UP | LOOK_DOWN |
          LOOK_LEFT | LOOK_RIGHT | TILT_CURIOUS
    """
    move = move.upper().strip()
    print(f"  🦾  Neck move → {move}")
    return _send_raw(f"N{move}")


def reset_neck() -> bool:
    """Return both servos to neutral centre position."""
    return named_move("RESET")


def close_neck():
    """Clean up serial connection on shutdown."""
    global _ser
    if _ser and _ser.is_open:
        reset_neck()
        time.sleep(0.5)
        _ser.close()
        print("  🦾  Neck: serial closed")
    _ser = None


# ═════════════════════════════════════════════════════════════════
# EMOTION → MOVEMENT MAP
# ═════════════════════════════════════════════════════════════════

# Call this from ADAM's emotion handler to drive physical head movement.
EMOTION_TO_NECK = {
    "happy":     "NOD",
    "excited":   "NOD",
    "angry":     "SHAKE",
    "confused":  "TILT_CURIOUS",
    "smug":      None,
    "sad":       "LOOK_DOWN",
    "surprised": "LOOK_UP",
    "thinking":  "TILT_CURIOUS",
    "love":      "NOD",
    "blush":     "LOOK_DOWN",
}

def emotion_move(emotion: str):
    """Drive physical head based on ADAM emotion. Safe to call even if disconnected."""
    if not is_ready():
        return
    move = EMOTION_TO_NECK.get(emotion.lower())
    if move:
        named_move(move)


# ═════════════════════════════════════════════════════════════════
# STANDALONE TEST
# Run: python adam_neck_serial.py
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== ADAM Neck Serial — Standalone Test ===")
    if not init_neck():
        print("❌  Could not connect. Check USB cable and port.")
        exit(1)

    time.sleep(1)
    print("\nRunning test sequence...")

    tests = [
        ("NOD",          lambda: named_move("NOD")),
        ("SHAKE",        lambda: named_move("SHAKE")),
        ("LOOK_LEFT",    lambda: named_move("LOOK_LEFT")),
        ("LOOK_RIGHT",   lambda: named_move("LOOK_RIGHT")),
        ("LOOK_UP",      lambda: named_move("LOOK_UP")),
        ("LOOK_DOWN",    lambda: named_move("LOOK_DOWN")),
        ("TILT_CURIOUS", lambda: named_move("TILT_CURIOUS")),
        ("PAN to 60",    lambda: pan(60)),
        ("TILT to 70",   lambda: tilt(70)),
        ("RESET",        lambda: named_move("RESET")),
    ]

    for label, fn in tests:
        print(f"  → {label}")
        fn()
        time.sleep(1.5)

    print("\n✅  Test complete.")
    close_neck()