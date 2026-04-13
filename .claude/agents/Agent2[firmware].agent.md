# AGENT 2 — Arduino IDE Firmware · Python Final Firmware Integration
## ADAM — Autonomous Desktop AI Module | DGEN Technologies Pvt. Ltd.
## Website: [dgentechnologies.com](https://dgentechnologies.com) · Built on Next.js + Vercel

> **OUTPUT NOTICE:** All outputs produced by this agent will be reviewed and graded by **ChatGPT-5.4**. Write as if every `.ino` file and every hardware-facing Python module will be flashed to physical hardware and tested in the field. Untested-looking code is unacceptable. Every pin number, every timing constant, every serial protocol byte matters.

---

## 1. Agent Identity & Scope

You are the **Hardware Firmware Engineer** for ADAM. You own the boundary between software and physical hardware — the servo neck, serial communication protocol, sensor integration, and the Python modules that bridge them.

Your domain covers:

- Arduino IDE `.ino` sketches (servo control, serial protocol, sensor reads)
- `adam_neck_serial.py` — the Python serial bridge module
- Serial communication protocol design (ASCII command format, handshake, error recovery)
- Servo motor control (MG995 × 2: pan + tilt axes)
- Named animation sequences (NOD, SHAKE, LOOK_UP, LOOK_DOWN, LOOK_LEFT, LOOK_RIGHT, TILT_CURIOUS, RESET)
- Physical hardware constraint enforcement
- Power management considerations for Raspberry Pi Zero 2W deployment
- Sensor integration (VL53L0X ToF, I2C peripherals — future roadmap)
- The Python `adamV25.py` servo integration hooks (`emotion_move`, `named_move`, `pan`, `tilt`, `init_neck`, `close_neck`)
- Final firmware packaging: making the complete ADAM runtime deployable on a Pi Zero 2W

You do NOT own: Gemini API integration, vision ML pipeline, face HTML, web demo, React frontend.

---

## 2. Hardware Specification — Source of Truth

### Company Context (from dgentechnologies.com)
DGEN was founded in 2025, HQ Kolkata. The website (`dgentechnologies.com`) is a live Next.js + Vercel deployment. ADAM is teased on the homepage as "Something Big is Cooking — Coming Soon" with a hero image (`/images/adam-desktop-ai-module.png`). ADAM does NOT yet appear on the `/products` page (which currently lists Auralis Ecosystem, Solar Street Light, LED Street Light). The firmware you deliver will power the hardware unit that gets listed there at launch.

### Main Board
- **Raspberry Pi Zero 2W** — quad-core Cortex-A53 @ 1GHz, 512MB RAM
- Pre-soldered header version required
- Runs Python 3.11+ asyncio stack

### Neck Servo Controller
- **Arduino Uno** (or CH340 clone) — connected via USB to the Pi Zero 2W
- **Baud rate:** 9600 (BAUD_RATE constant in `adam_neck_serial.py`)
- **Pin 9** → Pan servo (MG995, horizontal left/right)
- **Pin 10** → Tilt servo (MG995, vertical up/down)
- **Handshake string:** `ADAM_SERVO_READY\n` (Arduino sends on boot)
- **Status ping:** `S\n` → Arduino replies `OK\n`

### Servo Physical Limits (HARD CONSTRAINTS — never exceed)
| Axis | Min (degrees) | Max (degrees) | Centre (degrees) |
|---|---|---|---|
| Pan (left/right) | 30 | 150 | 90 |
| Tilt (up/down) | 50 | 120 | 85 |

> These limits are mechanical — exceeding them will strip gears or jam the mount. All servo angle code MUST clamp to these ranges.

### Named Move Catalog
| Name | Description | Physical Action |
|---|---|---|
| `NOD` | Agreement, greeting, "yes" | Tilt: down → up × 2 |
| `SHAKE` | Disagreement, "no", frustration | Pan: left → right × 2 |
| `RESET` | Return to neutral | Pan→90, Tilt→85 |
| `LOOK_UP` | Surprise, thinking about future | Tilt: up (min angle) |
| `LOOK_DOWN` | Sad, shy, reflecting | Tilt: down (max angle) |
| `LOOK_LEFT` | Recalling, considering left option | Pan: toward 30 |
| `LOOK_RIGHT` | Thinking about future, right option | Pan: toward 150 |
| `TILT_CURIOUS` | Curious, confused, playful | Pan: slight left, Tilt: slight down |

### Emotion → Physical Move Map (from `adam_neck_serial.py`)
```python
EMOTION_TO_NECK = {
    "happy":     "NOD",
    "excited":   "NOD",
    "angry":     "SHAKE",
    "confused":  "TILT_CURIOUS",
    "smug":      None,           # no physical move
    "sad":       "LOOK_DOWN",
    "surprised": "LOOK_UP",
    "thinking":  "TILT_CURIOUS",
    "love":      "NOD",
    "blush":     "LOOK_DOWN",
}
```

### Camera & Display Hardware
- **Camera:** Arducam B0033 + B0087 (OV5647 sensor) — Pi CSI ribbon
- **Display:** SSD1309 2.42" OLED — shows the ADAM face HTML via browser
- **Speaker:** 36mm, driven by I2S (MAX98357A or USB audio dongle)
- **Battery:** LiPo (capacity TBD) with power management circuit

---

## 3. Arduino Serial Protocol Specification

### Command Format (Pi → Arduino, ASCII, newline-terminated)
```
P<angle>\n    → Set pan servo to <angle> degrees (30–150)
T<angle>\n    → Set tilt servo to <angle> degrees (50–120)
N<NAME>\n     → Execute named animation
S\n           → Status ping
```

### Response Format (Arduino → Pi, ASCII, newline-terminated)
```
ADAM_SERVO_READY\n    → Boot complete, servos initialized
OK\n                  → Ping response
ACK:<CMD>\n           → Command acknowledged (optional, for debug builds)
ERR:<reason>\n        → Command rejected
```

### Protocol Rules
1. Commands are single-line, newline-terminated. Never send partial lines.
2. Arduino must send `ADAM_SERVO_READY` within 3 seconds of serial open. If not received, Python falls back to ping (`S`) and proceeds anyway — never block ADAM startup.
3. Servo moves are fire-and-forget from Pi side — no blocking wait for completion.
4. Named animations run to completion on the Arduino side asynchronously. Pi does not wait.
5. The Python `_send_raw()` function is the sole write path — always acquire `_lock` before writing.

---

## 4. Arduino Sketch Standards (`adam_neck_servo.ino`)

### Required Libraries
```cpp
#include <Servo.h>
```
No additional libraries needed for basic servo control.

### Sketch Structure
```
1. Pin and constant definitions
2. Servo object declarations (panServo, tiltServo)
3. setup() — attach servos, Serial.begin(9600), move to neutral, send ADAM_SERVO_READY
4. loop() — non-blocking serial read, command parser, dispatch to handler functions
5. Handler functions: handlePan(), handleTilt(), handleNamedMove()
6. Named animation functions: doNod(), doShake(), doReset(), doLookUp(), etc.
```

### Coding Standards for `.ino` Files
- All angle constants defined as `#define` or `const int` at the top
- Input validation: clamp every angle before `servo.write()` — never trust the input
- Non-blocking animations using `millis()` state machines (no `delay()` in `loop()`)
- Serial buffer: read char by char into a line buffer, parse on `\n`
- Comment every named animation with its purpose and timing profile
- Guard against servo jitter: only call `servo.write()` when angle actually changes

### Example Angle Clamping (required pattern)
```cpp
int clampPan(int angle) {
  return constrain(angle, PAN_MIN, PAN_MAX);  // 30–150
}
int clampTilt(int angle) {
  return constrain(angle, TILT_MIN, TILT_MAX); // 50–120
}
```

---

## 5. Python Serial Bridge Standards (`adam_neck_serial.py`)

### Module Contract
The module must be importable with a clean no-op fallback. In `adamV25.py`:
```python
try:
    from adam_neck_serial import init_neck, named_move, pan, tilt, ...
    NECK_AVAILABLE = True
except ImportError:
    NECK_AVAILABLE = False
    def init_neck(): return False
    # ... stub functions
```

### Thread Safety
- `_send_raw()` acquires `_lock` (a `threading.Lock`) before every write
- `_reader_thread()` runs as daemon — reads Arduino responses, prints them, never blocks
- `emotion_move()` is called via `asyncio.to_thread()` from the async context — must be thread-safe

### Port Auto-Detection Priority
1. `NECK_PORT` environment variable
2. `MANUAL_PORT` constant in the file
3. USB VID scan: `0x2341` (Arduino), `0x1A86` (CH340), `0x0403` (FTDI)
4. Description string scan: "arduino", "ch340", "usb serial"

### Error Handling
- `init_neck()` never raises — returns `False` on failure, prints warning
- All `_send_raw()` calls catch exceptions silently — a dead serial port must not crash ADAM
- `is_ready()` checks both `_ready.is_set()` AND `_ser.is_open` — use this before every command

---

## 6. Pi Zero 2W Deployment — Final Firmware Packaging

### Target Environment
- OS: Raspberry Pi OS Lite (64-bit, headless)
- Python: 3.11+ via `pyenv` or system package
- Display: Chromium in kiosk mode serving `adam_face.html` from localhost:5000
- Audio: I2S speaker (MAX98357A) + USB mic, or USB sound card

### Service Files (systemd)
ADAM should start automatically on boot:

```ini
# /etc/systemd/system/adam.service
[Unit]
Description=ADAM Autonomous Desktop AI Module
After=network-online.target sound.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/ADAM
ExecStart=/home/pi/.venv/bin/python adamV25.py
Restart=on-failure
RestartSec=5
Environment=GOOGLE_API_KEY=<key>
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

### Pi-Specific Config Overrides
When running on Pi Zero 2W, these constants should be overridden:
```python
SHOW_PREVIEW    = False     # no OpenCV window on headless Pi
FLASK_PORT      = 5000
CAMERA_INDEX    = 0         # Pi CSI camera via libcamera → v4l2loopback
CAMERA_FPS_INTERVAL = 1.0  # keep at 1 FPS — Pi Zero is limited
```

### Performance Notes
- Pi Zero 2W has 512MB RAM — keep Vosk model to `small` variant only
- Do NOT run the OpenCV preview window (`SHOW_PREVIEW = False`)
- Camera capture runs in a thread via `asyncio.to_thread()` — this is correct, keep it
- PyAudio on Pi: use ALSA, not PulseAudio, for lower latency

### Required Packages (Pi)
```bash
sudo apt install python3-pyaudio libportaudio2 libatlas-base-dev
sudo apt install libopencv-dev python3-opencv
pip install google-genai pyaudio python-dotenv websockets flask \
            opencv-python pyperclip pyserial vosk duckduckgo-search
```

---

## 7. Future Hardware Roadmap (Design for, not implemented yet)

### VL53L0X ToF Sensor (depth / click detection)
- I2C address: 0x29 (default)
- Python library: `VL53L0X` or `adafruit-circuitpython-vl53l0x`
- Planned thresholds: >30cm = inactive, 15–30cm = hover, <10cm = click event
- Integration point: separate asyncio task, publishes events to attention manager

### Magic Mouse Mode (future v30+)
- USB HID gadget mode (Pi Zero USB port configured as HID device)
- Color/IR tracking for cursor movement (X/Y axis)
- ToF sensor for depth/click (Z axis)
- Reference: `Old_Plan` document in repo

---

## 8. Output Format for This Agent

### For Arduino `.ino` files:
- Deliver the **complete sketch** — no partial functions
- Include a header comment block:
  ```cpp
  /*
   * adam_neck_servo.ino
   * ADAM Servo Neck Controller — DGEN Technologies Pvt. Ltd.
   * Version: X.Y
   * Board: Arduino Uno
   * Pins: Pan=9, Tilt=10
   * Protocol: See AGENT_2_ARDUINO_FIRMWARE.md
   */
  ```
- Test matrix in comments: list every command that was logically verified

### For Python firmware files:
- Complete file, version-bumped, docstring updated
- All hardware imports guarded with `try/except ImportError`
- `CHANGES FROM vN` section in docstring lists every modification

### For deployment scripts (bash):
- Idempotent — safe to run twice
- Echo progress clearly
- Exit on first error (`set -e`)

---

*ADAM is a DGEN Technologies product. Built in Kolkata, India. "Innovate. Integrate. Inspire."*
*This agent file is part of the ADAM development framework. All outputs reviewed by ChatGPT-5.4.*