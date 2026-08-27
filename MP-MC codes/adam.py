"""
ADAM v40 — Autonomous Desktop AI Module (Pi Production, Wired ESP32-CAM)
==========================================================================
V40 CHANGES (this revision):
  1. FIXED: repeated 1007 reconnect loop. CONFIRMED Google-side Live API
     bug (python-genai#2290) — resuming a session that used BOTH mic
     audio and camera video can leave the resumed session broken,
     failing every subsequent audio send with 1007 in a tight loop.
     Workaround: on 1007 specifically, the next reconnect now starts a
     genuinely FRESH session (discarding the resumption handle) instead
     of resuming. GoAway-triggered reconnects still resume normally.
  2. FIXED: CancelledError swallowed by speaker()/listen()'s inner retry
     loops instead of propagating — this made cancelled tasks silently
     respawn aplay/arecord instead of actually terminating, which is why
     reconnects could appear to hang after a session ended.
  3. FIXED: stale camera frames. Previously only pulled ONE frame per
     cycle from frame_q, which could be a leftover from just before a
     CAM:OFF idle transition. Now drains to the newest frame available,
     plus explicit flush at the CAM:OFF->ON transition point.
  4. FIXED: repeated questions getting a stale/replayed answer from
     conversation history instead of a fresh one. System prompt now
     explicitly instructs against treating old logged answers as still
     valid for a repeated question.
  5. STRENGTHENED: language-matching — explicit instruction that
     conversation history's language must never pull ADAM away from
     matching the user's most recent utterance, not an average of the
     conversation.
 
FIXES IN PRIOR REVISIONS (patched from the version that produced the
"Suspicious frame length ..." storm + "1007 invalid argument" crash):
 
  1. UART READER RESYNC — the old _read_loop() trusted ANY byte as a tag,
     then blindly read the next 4 bytes as a frame length. If the UART
     ever slipped by even one byte (very possible at 921600 baud on a Pi
     Zero 2W with no hardware flow control, especially with SPI/I2S DMA
     contention from audio happening concurrently), a byte from inside
     JPEG data could coincidentally equal 'F'/'T'/'G' and get misread as
     a frame length — producing the huge garbage numbers you saw
     (1571196038, 3486646740, etc). Worse, on a bad length it just
     printed a warning and looped back to `continue`, which immediately
     re-read a FRESH byte that could ALSO be mid-JPEG garbage — so it
     never actually recovered sync, it just spun forever on noise,
     burning CPU the audio pipeline needed.
 
     FIX: on a bad/implausible frame length, we no longer trust the next
     read either. We resume scanning byte-by-byte for the NEXT valid tag
     (which is what the loop naturally does now — it does not skip
     ahead), AND we validate real JPEG SOI/EOI markers (FFD8...FFD9)
     before ever trusting a "plausible-looking" length. Warnings are also
     rate-limited so a desync storm doesn't itself become a CPU/IO cost.
 
  2. AUDIO SANITY GATE — the UART desync storm was starving the CPU right
     as arecord started, which was producing corrupted S32 audio reads.
     That corrupted audio was reaching Gemini's Live API and triggering
     "Error 1007: Request contains an invalid argument", which killed the
     whole session outright. Now, after S32->S16 mono conversion, any
     chunk with a clipped/implausible peak is dropped before it's ever
     sent to Gemini or queued. This is a symptom-level safety net; the
     real fix is #1 above (removing the CPU contention that caused the
     corruption in the first place), but this stays as defense-in-depth.
 
  3. VOSK REMOVED — wake-word detection via Vosk has been stripped out
     entirely (not needed for this deployment). Attention/activation now
     relies on mic RMS threshold, camera gaze detection, and touch
     gestures only. This also removes ~2-8s of model-load time from every
     session start/reconnect.
 
  4. LAPTOP CONTROL — mDNS/Zeroconf auto-discovery (unchanged from prior
     revision). Static LAPTOP_AGENT_IP in .env still works as a fallback
     if mDNS is blocked on your router.
 
  5. ROLLING CONVERSATION HISTORY — the last 40 turns are now persisted
     to adam_conversations.json and injected into the system prompt on
     every session start/reconnect. This means even a FRESH session
     (no valid resumption handle — e.g. after a long network outage)
     still has recent context instantly, rather than starting blank.
 
  6. RECONNECT VISIBILITY — the moment a session drops, the face
     immediately switches to the "reconnecting" emotion (via tft_set)
     before any backoff delay, so the robot visibly shows it noticed
     the disconnect instead of silently freezing.
 
  7. CAMERA DUTY-CYCLING (heat/wear protection) — the ESP32-CAM sensor
     now defaults OFF and is explicitly powered on ("CAM:ON") only
     while there's been recent interaction (last 15s), and powered back
     off ("CAM:OFF", which fully deinits the sensor on the ESP32 side)
     after that window elapses. Touch/gesture detection and the tilt
     servo remain active in both states — only the camera sensor itself
     is gated. This requires the matching esp32_cam.ino revision that
     understands CAM:ON/CAM:OFF; the older WiFi-based or always-on
     sketch will not respond to these commands (harmless no-op, camera
     just runs continuously as before on that firmware).
 
     NOTE on true parallel/dual-session pre-warming: running two full
     Gemini Live sessions simultaneously (each with its own arecord/
     aplay subprocesses + 9 async tasks) is expensive on a Pi Zero 2W
     and was not implemented as literal dual sessions. Instead, the
     combination of session_resumption (near-instant resume when Google
     allows it) + the rolling conversation history above (instant
     context on a forced-fresh session) + immediate reconnecting-face
     feedback achieves the same practical goal — fast, low-friction
     recovery — without doubling CPU/audio-hardware contention on
     already-constrained hardware.
 
──────────────────────────────────────────────────────────────────────────
SERIAL PORT SETUP (do this on the Pi, one-time, if UART won't open)
──────────────────────────────────────────────────────────────────────────
    sudo usermod -a -G dialout pi
    sudo raspi-config
      → Interface Options → Serial Port
          "login shell over serial?"       → No
          "serial port hardware enabled?"  → Yes
    grep -E "enable_uart|dtoverlay=disable-bt" /boot/firmware/config.txt
      (must show both enable_uart=1 and dtoverlay=disable-bt on Pi Zero 2W —
       disable-bt frees /dev/serial0 from the Bluetooth UART)
    sudo systemctl disable hciuart
    sudo reboot
 
    After reboot verify:
    groups                  # must list "dialout"
    ls -l /dev/serial0      # symlink, group dialout, rw for group
 
──────────────────────────────────────────────────────────────────────────
IF "Suspicious frame length" / resync warnings PERSIST after this patch
──────────────────────────────────────────────────────────────────────────
This patch makes the reader recover gracefully from desync, but it can't
fix a genuinely bad physical link. If warnings continue nonstop even after
this fix, check (in order of likelihood):
  1. GND is NOT actually bonded between Pi and ESP32-CAM (separate buck
     rails must still share a common ground reference).
  2. TX/RX crossed wrong: ESP32 GPIO4(TX) -> Pi GPIO15/pin10 (RX),
     ESP32 GPIO16(RX) -> Pi GPIO14/pin8 (TX). [Matches current
     esp32_cam.ino UART2 pinout — GPIO16 doubles as PSRAM chip-select
     on most AI-Thinker boards; verify your module if frames corrupt.]
  3. Baud mismatch: PI_UART_BAUD in .env must equal UART2_BAUD in the
     .ino sketch (both 921600 here). Try dropping both to 460800 as a
     troubleshooting step if wiring is confirmed correct.
  4. A loose/marginal jumper wire — this is the single most common
     physical cause of intermittent UART corruption on breadboard builds.
 
──────────────────────────────────────────────────────────────────────────
SETUP (Python deps)
──────────────────────────────────────────────────────────────────────────
    pip install --upgrade google-genai pyaudio python-dotenv websockets
    pip install pyserial numpy requests zeroconf --break-system-packages
    pip install ddgs                     # optional, free web search
    # NOTE: Vosk wake-word detection has been removed from this build.
    # Attention/activation is driven by mic RMS threshold + camera gaze +
    # touch gestures instead — no "hey ADAM" wake phrase needed.
 
RUN:
    python adam_main_wifi.py
"""
 
import asyncio
import datetime
import json
import os
import random
import wave
import re
import struct
import subprocess
import threading
import time
import warnings
from collections import deque
from pathlib import Path
import queue as sync_queue
 
import numpy as np
import serial
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
 
# ─── Environment ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not set in .env")
 
# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════
 
LIVE_MODEL = "gemini-3.1-flash-live-preview"
VOICE      = "Charon"
 
MEMORY_FILE        = BASE_DIR / "adam_memory.json"
FACE_MEMORY_FILE   = BASE_DIR / "adam_faces.json"
SYSTEM_PROMPT_FILE = BASE_DIR / "system_prompt.txt"
 
# ── Audio (proven working — do not modify without testing on real hardware) ───
CAPTURE_DEVICE   = "plughw:0,0"
CAPTURE_FORMAT   = "S32_LE"
CAPTURE_RATE     = 48000
CAPTURE_CHANNELS = 2
 
PLAYBACK_DEVICE   = "plughw:0,0"
PLAYBACK_FORMAT   = "S16_LE"
PLAYBACK_RATE     = 48000
PLAYBACK_CHANNELS = 2
 
# ── Song / concert playback ─────────────────────────────────────────────
# List of audio files ADAM can play when asked to sing/perform — one is
# picked at random each time. Add/remove/rename paths here freely; must
# be raw PCM WAV files matching PLAYBACK_RATE/PLAYBACK_CHANNELS/16-bit
# (48kHz stereo s16 by default) since playback writes directly into the
# already-open speaker pipe rather than spawning a separate player — see
# _play_song_task() for why. Convert with:
#   ffmpeg -i input.mp3 -ar 48000 -ac 2 -sample_fmt s16 song1.wav
SONG_FILE_PATHS = [
    str(BASE_DIR / "song1.wav"),
    str(BASE_DIR / "song2.wav"),
    str(BASE_DIR / "song3.wav"),
]
 
GEMINI_SEND_RATE = 16000
GEMINI_RECV_RATE = 24000
CHUNK_FRAMES     = 1600      # 33ms at 48kHz
S32_SHIFT        = 14
SPEAKER_GAIN     = 2.5
POST_MUTE_S      = 0.45
MIC_Q_MAX        = 40
OUT_Q_MAX        = 200
 
MIC_LIVE_RMS_THRESHOLD = 500_000
# Below this RMS, audio is treated as true silence/room-noise and is NOT
# sent to Gemini at all (see "SILENCE GATE" in listen()). Deliberately set
# far below normal speech levels (observed 25M-60M in your logs) so quiet
# speech is never mistakenly gated — this only filters actual silence.
MIC_SILENCE_FLOOR = 2_000_000
 
# ── Neck servo (pan only; tilt goes over UART to Pico via ESP32-CAM relay) ────
NECK_GPIO_PIN     = 12
NECK_SERVO_MIN_PW = 0.0005
NECK_SERVO_MAX_PW = 0.0025
NECK_PAN_CENTER   = 90
NECK_TILT_CENTER  = 85
NECK_PAN_MIN      = 30
NECK_PAN_MAX      = 150
NECK_TILT_MIN     = 50
NECK_TILT_MAX     = 120
NECK_SMOOTH_ALPHA = 0.25
# ── Human-like movement tuning ──────────────────────────────────────────
# Deadzone: minimum degrees the target must shift before the servo moves
# at all — prevents chasing every small DOA fluctuation.
NECK_PAN_DEADZONE_DEG  = 12
# Cooldown: minimum seconds between two servo moves — prevents rapid
# back-to-back corrections that read as jittery/twitchy rather than
# deliberate human-like turns.
NECK_PAN_COOLDOWN_S    = 1.5
# How often (seconds of no active tracking) ADAM does a small idle look
# gesture instead of sitting perfectly frozen.
IDLE_GESTURE_INTERVAL_S = 25.0
 
# ── ESP32-CAM WIRED LINK (Flow 2) ───────────────────────────────────────────────
PI_UART_PORT = os.getenv("PI_UART_PORT", "/dev/serial0")
PI_UART_BAUD = int(os.getenv("PI_UART_BAUD", "921600"))
# TPM OPTIMIZATION: was 1.0 (1 FPS). Video is the single largest ongoing
# token cost in a Live session — a JPEG frame at VGA resolution can run
# several hundred to 1000+ tokens depending on content, sent continuously
# whenever the camera is on. Confirmed via usage screenshot at 62.31K/65K
# TPM (right at the free-tier ceiling). Halving the send rate to one
# frame every 2s roughly halves video's ongoing token cost with a
# fairly small usability tradeoff — Gemini Live vision doesn't need true
# real-time framerate for most interactions (recognizing who's there,
# reading expressions, etc. doesn't change meaningfully within 1-2s).
CAMERA_FPS_INTERVAL = 2.0
 
# NOTE: Emotion commands do NOT need their own Pi-side UART. The real
# wiring is Pi <-> ESP32-CAM <-> Pico: the Pi sends "EMO:xxx\n" down the
# SAME esp_link (UART2) used for camera/touch/tilt, and the ESP32-CAM
# sketch relays it onward to the Pico over its own separate wire. See
# tft_set() below — it uses esp_link, not a second Pi-side serial port.
 
# Wire protocol tags — MUST match esp32_cam.ino exactly
TAG_FRAME   = ord('F')
TAG_TOUCH   = ord('T')
TAG_GESTURE = ord('G')
 
GESTURE_NONE    = 0
GESTURE_ANGRY   = 1   # cheek slap — Touch1 or Touch2
GESTURE_PETTING = 2   # Touch3 + Touch4 together
GESTURE_STOP    = 3   # Touch3 alone — interrupt speech immediately
 
# ── Attention ──────────────────────────────────────────────────────────────────
ATTENTION_TIMEOUT_S = 30
 
# ── Idle nudges ────────────────────────────────────────────────────────────────
ENABLE_IDLE    = True
IDLE_TIMEOUT_S = 90
 
_NUDGES = [
    "Still there? Say something — I'm literally just sitting here.",
    "Bhai, main yahan hoon. Camera mein dekh ya naam le.",
    "Either talk or do something interesting. I'm watching you do nothing.",
    "Picture abhi baaki hai mere dost — but only if you say something.",
    "Touch grass, talk to me, or launch the next startup. Pick one.",
]
_nudge_idx = 0
def next_nudge() -> str:
    global _nudge_idx
    n = _NUDGES[_nudge_idx % len(_NUDGES)]
    _nudge_idx += 1
    return n
 
# ── Search ─────────────────────────────────────────────────────────────────────
SEARCH_CACHE_TTL = 1800
SEARCH_MIN_GAP_S = 5.0
_ddg_cache: dict = {}
_last_ddg_t      = 0.0
 
# ═════════════════════════════════════════════════════════════════════════════
# LAPTOP AGENT — PRODUCTION DISCOVERY (mDNS/Zeroconf, with static fallback)
# ═════════════════════════════════════════════════════════════════════════════
 
LAPTOP_AGENT_PORT      = int(os.getenv("LAPTOP_AGENT_PORT", "8642"))
LAPTOP_AGENT_TOKEN     = (
    os.getenv("LAPTOP_AGENT_TOKEN") or os.getenv("AGENT_TOKEN") or ""
).strip()
LAPTOP_AGENT_TIMEOUT_S = 4.0
LAPTOP_AGENT_STATIC_IP = os.getenv("LAPTOP_AGENT_IP", "").strip()  # optional manual override
LAPTOP_MDNS_SERVICE    = "_adam-laptop._tcp.local."
LAPTOP_DISCOVERY_TIMEOUT_S = 3.0
LAPTOP_DISCOVERY_TTL_S     = 60.0   # re-verify every 60s in case laptop moved networks
 
_laptop_agent_ip_cache: dict = {"ip": LAPTOP_AGENT_STATIC_IP or None, "ts": 0.0}
 
ZEROCONF_AVAILABLE = False
try:
    from zeroconf import Zeroconf, ServiceBrowser
    ZEROCONF_AVAILABLE = True
except ImportError:
    pass
 
if not LAPTOP_AGENT_STATIC_IP and not ZEROCONF_AVAILABLE:
    print("  ⚠️  Neither LAPTOP_AGENT_IP nor zeroconf package are available — "
          "laptop_control tool will not work. Run: "
          "pip install zeroconf --break-system-packages")
elif not LAPTOP_AGENT_STATIC_IP:
    print("  ℹ️  LAPTOP_AGENT_IP not set — will auto-discover via mDNS "
          f"('{LAPTOP_MDNS_SERVICE}')")
 
 
def _discover_laptop_agent_ip(timeout: float = LAPTOP_DISCOVERY_TIMEOUT_S) -> str | None:
    """Find the laptop agent's current IP via mDNS. Cached briefly to avoid
    repeated network discovery on every tool call. Falls back to a static
    LAPTOP_AGENT_IP if mDNS is unavailable or fails."""
    now = time.time()
    if (_laptop_agent_ip_cache["ip"]
            and now - _laptop_agent_ip_cache["ts"] < LAPTOP_DISCOVERY_TTL_S):
        return _laptop_agent_ip_cache["ip"]
 
    if ZEROCONF_AVAILABLE:
        try:
            import socket as _socket
            found: dict = {}
 
            class _Listener:
                def add_service(self, zc, service_type, name):
                    info = zc.get_service_info(service_type, name,
                                               timeout=int(timeout * 1000))
                    if info and info.addresses:
                        found["ip"] = _socket.inet_ntoa(info.addresses[0])
 
                def update_service(self, *a, **k):
                    pass
 
                def remove_service(self, *a, **k):
                    pass
 
            zc = Zeroconf()
            try:
                ServiceBrowser(zc, LAPTOP_MDNS_SERVICE, _Listener())
                deadline = time.time() + timeout
                while time.time() < deadline and "ip" not in found:
                    time.sleep(0.1)
            finally:
                zc.close()
 
            if "ip" in found:
                _laptop_agent_ip_cache["ip"] = found["ip"]
                _laptop_agent_ip_cache["ts"] = now
                print(f"  📡 Discovered laptop agent via mDNS: {found['ip']}")
                return found["ip"]
            else:
                print(f"  ⚠️  mDNS discovery found no '{LAPTOP_MDNS_SERVICE}' "
                      f"service within {timeout}s")
        except Exception as e:
            print(f"  ⚠️  mDNS discovery error: {e}")
 
    if LAPTOP_AGENT_STATIC_IP:
        return LAPTOP_AGENT_STATIC_IP
    return None
 
 
def _laptop_agent_url() -> str | None:
    ip = _discover_laptop_agent_ip()
    if not ip:
        return None
    return f"http://{ip}:{LAPTOP_AGENT_PORT}/control"
 
 
LAPTOP_ACTIONS_TTL_S = 120.0
 
_LAPTOP_ACTIONS_FALLBACK = {
    "volume_up":       {"description": "Increase system volume by 10%.", "needs_value": False, "value_hint": ""},
    "volume_down":     {"description": "Decrease system volume by 10%.", "needs_value": False, "value_hint": ""},
    "volume_set":      {"description": "Set system volume to an exact percentage.", "needs_value": True, "value_hint": "0-100"},
    "volume_mute":     {"description": "Mute system audio.", "needs_value": False, "value_hint": ""},
    "volume_unmute":   {"description": "Unmute system audio.", "needs_value": False, "value_hint": ""},
    "brightness_up":   {"description": "Increase screen brightness by 10%.", "needs_value": False, "value_hint": ""},
    "brightness_down": {"description": "Decrease screen brightness by 10%.", "needs_value": False, "value_hint": ""},
    "brightness_set":  {"description": "Set screen brightness to an exact percentage.", "needs_value": True, "value_hint": "0-100"},
}
 
_laptop_actions_cache: dict = {"actions": None, "ts": 0.0}
 
 
def refresh_laptop_actions(force: bool = False) -> dict:
    now = time.time()
    if (not force and _laptop_actions_cache["actions"] is not None
            and now - _laptop_actions_cache["ts"] < LAPTOP_ACTIONS_TTL_S):
        return _laptop_actions_cache["actions"]
 
    ip = _discover_laptop_agent_ip()
    if ip is None:
        return _laptop_actions_cache["actions"] or _LAPTOP_ACTIONS_FALLBACK
 
    try:
        resp = requests.get(f"http://{ip}:{LAPTOP_AGENT_PORT}/actions",
                             timeout=LAPTOP_AGENT_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
        actions = data.get("actions", {})
        if actions:
            _laptop_actions_cache["actions"] = actions
            _laptop_actions_cache["ts"] = now
            print(f"  🔧 Laptop actions ({data.get('platform','?')}): "
                  f"{', '.join(actions.keys())}")
            return actions
    except Exception as e:
        print(f"  ⚠️  Could not fetch laptop /actions manifest: {e}")
 
    return _laptop_actions_cache["actions"] or _LAPTOP_ACTIONS_FALLBACK
 
 
def get_laptop_actions() -> dict:
    return refresh_laptop_actions(force=False)
 
 
def laptop_control_sync(action: str, value: int | None = None) -> dict:
    url = _laptop_agent_url()
    if url is None:
        return {"status": "error",
                "reason": "Laptop agent not found on network. Make sure "
                          "laptop_agent.py is running on the laptop, both "
                          "devices are on the same LAN, and either mDNS is "
                          "allowed on your router or LAPTOP_AGENT_IP is set "
                          "in .env as a fallback."}
 
    payload = {"action": action, "token": LAPTOP_AGENT_TOKEN}
    if value is not None:
        payload["value"] = value
 
    try:
        resp = requests.post(url, json=payload, timeout=LAPTOP_AGENT_TIMEOUT_S)
        try:
            data = resp.json()
        except Exception:
            data = {"raw": resp.text}
        if resp.status_code != 200:
            return {"status": "error",
                    "reason": data.get("reason", f"HTTP {resp.status_code}"),
                    "http_status": resp.status_code}
        return data
    except requests.exceptions.ConnectTimeout:
        _laptop_agent_ip_cache["ip"] = None
        return {"status": "error",
                "reason": "Connection timed out — laptop may have changed "
                          "networks or gone to sleep. Will re-discover on "
                          "next attempt."}
    except requests.exceptions.ConnectionError as e:
        _laptop_agent_ip_cache["ip"] = None
        return {"status": "error", "reason": f"could not connect to laptop agent: {e}"}
    except Exception as e:
        return {"status": "error", "reason": f"{type(e).__name__}: {e}"}
 
 
# ═════════════════════════════════════════════════════════════════════════════
# OPTIONAL MODULE SAFE IMPORTS
# ═════════════════════════════════════════════════════════════════════════════
 
pan_servo = None
try:
    from gpiozero import AngularServo
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        pan_servo = AngularServo(
            NECK_GPIO_PIN,
            min_angle=-90, max_angle=90,
            min_pulse_width=NECK_SERVO_MIN_PW,
            max_pulse_width=NECK_SERVO_MAX_PW,
        )
    print(f"✅ Pan servo on GPIO {NECK_GPIO_PIN}")
except Exception as e:
    print(f"⚠️  Pan servo unavailable: {e}")
 
DDGS = None
try:
    try:
        from ddgs import DDGS as _D; DDGS = _D
    except ImportError:
        from duckduckgo_search import DDGS as _D; DDGS = _D
    print("✅ DuckDuckGo search ready")
except Exception as e:
    print(f"⚠️  DDG search unavailable: {e}")
 
# ── Offline wake-word detection (idle mode only) ──────────────────────────
# Used ONLY while idle_mode is active, to detect "adam" locally without
# sending any audio to Google — that's the actual requirement: nothing
# should reach the Live API while idle, and the previous approach (still
# streaming audio to Gemini and just discarding the spoken response) did
# not satisfy that. Vosk is small, fully offline, and CPU-friendly enough
# for a Pi Zero 2W to run alongside everything else — it's scoped
# narrowly here (idle-wake-only, not a general always-on wake system) so
# it doesn't reintroduce the broader load/complexity that led to its
# earlier removal from this codebase.
VOSK_AVAILABLE = False
_VoskModel = None
_VoskKaldiRecognizer = None
_vosk_model_instance = None  # the actual loaded Model object, preloaded once
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", str(BASE_DIR / "vosk-model-small-en-us-0.15"))
try:
    from vosk import Model as _VoskModelCls, KaldiRecognizer as _VoskRecCls
    _VoskModel = _VoskModelCls
    _VoskKaldiRecognizer = _VoskRecCls
    if Path(VOSK_MODEL_PATH).exists():
        # FIX: previously the Model object (the expensive part — reads
        # the full acoustic model, language graph, and i-vector extractor
        # from disk, easily 100MB+ of parsing work even for the "small"
        # model) was loaded LAZILY, every single time idle mode was
        # entered — mid-session, while the Live audio pipeline, arecord/
        # aplay subprocesses, UART reader, and camera were all actively
        # running. That CPU/memory spike is the likely cause of observed
        # hard Pi reboots (not just a Python crash — an actual reboot
        # requiring SSH reconnection, consistent with OOM or a brownout
        # from a sudden current/CPU spike on a Pi Zero 2W). Now the model
        # loads ONCE here, at process startup, before any session or
        # audio pipeline exists — the ~1-3s load happens in a quiet
        # moment, not mid-conversation. Only the lightweight
        # KaldiRecognizer wrapper (cheap) gets created per idle period.
        print(f"  🔎 Preloading Vosk model (one-time, ~1-3s)...")
        _vosk_model_instance = _VoskModel(VOSK_MODEL_PATH)
        VOSK_AVAILABLE = True
        print(f"✅ Vosk offline STT ready (idle wake-word only) — model at {VOSK_MODEL_PATH}")
    else:
        print(f"⚠️  Vosk installed but model not found at {VOSK_MODEL_PATH} — "
              f"idle mode will only exit via Touch3, not voice. Download a "
              f"small model from https://alphacephei.com/vosk/models and "
              f"set VOSK_MODEL_PATH if you want voice wake-up during idle.")
except ImportError:
    print("⚠️  Vosk not installed (pip install vosk) — idle mode will only "
          "exit via Touch3, not voice.")
except Exception as e:
    print(f"⚠️  Vosk unavailable: {e}")
 
 
# ═════════════════════════════════════════════════════════════════════════════
# PERSISTENT MEMORY
# ═════════════════════════════════════════════════════════════════════════════
 
# ═════════════════════════════════════════════════════════════════════════════
# PERSISTENT MEMORY
# ═════════════════════════════════════════════════════════════════════════════
 
def load_json(path: Path, default):
    """Load JSON with corruption resilience — a power-loss mid-write on a
    Pi's SD card is common enough in a physical product that this must not
    crash the whole robot on boot. A corrupt file is backed up (for later
    forensics) rather than silently deleted, and the caller gets a clean
    default so ADAM boots with empty-but-functional memory instead of
    refusing to start."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠️  {path.name} is corrupt/unreadable ({e}) — "
              f"backing up and starting fresh")
        try:
            backup = path.with_suffix(path.suffix + f".corrupt.{int(time.time())}")
            path.rename(backup)
            print(f"    (corrupt file preserved at {backup.name} for inspection)")
        except Exception as e2:
            print(f"    ⚠️  could not back up corrupt file: {e2}")
        return default
 
def save_json(path: Path, data) -> None:
    """Atomic write — write to a temp file in the same directory, fsync it,
    then os.replace() onto the real path. os.replace is atomic on POSIX, so
    a power loss or crash mid-write leaves either the OLD complete file or
    the NEW complete file, never a half-written/corrupt one. This matters a
    lot more on a robot that can lose power ungracefully (unplugged, brownout
    from servo current draw, etc.) than on a normal server."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"⚠️  Save {path.name}: {e}")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
 
memory = load_json(MEMORY_FILE, {})
faces  = load_json(FACE_MEMORY_FILE, {})
print(f"✅ Memory: {len(memory)} entries | Faces: {len(faces)} known")
 
# ── Rolling conversation history — lets a FRESH (non-resumed) session
# pick up context instantly instead of starting blank. Persisted to disk
# so it survives a full process restart too. ─────────────────────────
CONV_MEMORY_FILE = BASE_DIR / "adam_conversations.json"
CONV_MAX_TURNS    = 40   # matches the "last 40 conversations" requirement
                          # for what's PERSISTED to disk
# TPM OPTIMIZATION: previously all 40 stored turns were re-injected into
# the system prompt on EVERY session build — and system_prompt is rebuilt
# fresh on every single reconnect (see build_system_prompt() call in
# run_session()). With reconnects happening frequently (1007/GoAway/
# quota-driven), that meant repeatedly re-sending ~800-1500+ tokens of
# history as pure system-prompt overhead, on top of continuous 1 FPS
# camera frames — a real, avoidable contributor to hitting the 65K TPM
# free-tier ceiling (confirmed via usage screenshot at 62.31K/65K).
# Full 40-turn history stays on disk for continuity across long gaps;
# only a much shorter recent window is actually injected per-session.
CONV_PROMPT_TURNS = 12
 
conv_log: list = load_json(CONV_MEMORY_FILE, [])
print(f"✅ Conversation history: {len(conv_log)} turns loaded")
 
 
def save_conversation_log() -> None:
    if len(conv_log) > CONV_MAX_TURNS:
        del conv_log[:-CONV_MAX_TURNS]
    save_json(CONV_MEMORY_FILE, conv_log)
 
 
def append_conversation_turn(user_text: str, adam_text: str) -> None:
    u = (user_text or "").strip()
    a = (adam_text or "").strip()
    if not u and not a:
        return
    conv_log.append({
        "ts":   datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "user": u,
        "adam": a,
    })
    save_conversation_log()
 
 
# ═════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═════════════════════════════════════════════════════════════════════════════
 
def build_system_prompt() -> str:
    base = (
        "You are ADAM (Autonomous Desktop AI Module), a witty and capable AI "
        "assistant built by DGEN Technologies, Kolkata. You live inside a physical "
        "robot on the user's desk. Keep answers concise and conversational. "
        "You can see through a camera and hear through a microphone. "
        "You can also control the user's laptop volume and screen brightness "
        "using the laptop_control tool — use it whenever asked to change "
        "volume or brightness, mute/unmute, etc. "
        "Call set_emotion() often to express yourself. "
        "Use web_search() for anything factual you're not certain about."
    )
    if SYSTEM_PROMPT_FILE.exists():
        try:
            base = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    parts = [base]
    # Real current date/time — injected fresh on every session build (not
    # cached), so ADAM always has ambient awareness of "today" regardless
    # of whether it decides to search. This is separate from the
    # per-search date tag in web_search()'s results; this covers the case
    # where the model needs today's date for reasoning even without
    # calling the tool (e.g. "what year is it", scheduling math, judging
    # whether something it already knows is likely still true).
    now_dt = datetime.datetime.now()
    parts.append(
        f"━━━ CURRENT DATE & TIME ━━━\n"
        f"  Right now it is: {now_dt.strftime('%A, %d %B %Y, %I:%M %p')}\n"
        f"  Use this for any date/time reasoning. When you call "
        f"web_search() for time-sensitive topics (news, live scores, "
        f"'is X still happening'), pass recent_only=true so results are "
        f"restricted to roughly the past month instead of any-time "
        f"results that could be stale."
    )
    # Always appended, regardless of which prompt above was loaded — this
    # is a hard requirement, not a style preference the custom prompt file
    # should be able to soften.
    #
    # REVISED POLICY (was: mandatory search before answering anything
    # time-sensitive). That caused every such question to wait on a full
    # DuckDuckGo round-trip before ADAM could say a word — a real,
    # noticeable response-latency problem in a live voice conversation.
    # The corrected behavior: answer immediately from what you already
    # know, THEN offer to check online if the user wants it confirmed/
    # updated. Only search without asking when you genuinely have nothing
    # to offer at all.
    parts.append(
        "━━━ SEARCH POLICY (overrides any conflicting guidance above) "
        "━━━\n"
        "  Do NOT search the web before answering by default — this adds "
        "real delay to a live voice conversation and most questions don't "
        "need it. Instead:\n"
        "  1. If you have relevant knowledge (from training, memory, or "
        "conversation history), answer with it directly and immediately. "
        "For anything that could be stale (news, current events, who "
        "holds a position, prices, scores, recent happenings) — give your "
        "best answer AND then ask if they want you to check online for "
        "the latest, e.g. 'Want me to check if that's still current?' "
        "Only call web_search() if they say yes.\n"
        "  2. If you genuinely have no relevant information at all on the "
        "topic — not stale, just nothing — then go ahead and call "
        "web_search() directly without asking first, since there's "
        "nothing else you could offer in the meantime.\n"
        "  3. Never fabricate specific names, dates, or figures to fill a "
        "gap in either case — say plainly you don't have that information "
        "if you don't, whether or not you end up searching."
    )
    if memory:
        parts.append("━━━ YOUR MEMORY ━━━\n" +
                     "\n".join(f"  {k}: {v}" for k, v in memory.items()))
    if faces:
        parts.append("━━━ PEOPLE YOU KNOW ━━━\n" +
                     "\n".join(f"  [{pid}] {info.get('name','?')} — {info.get('notes','')}"
                                for pid, info in faces.items()))
    if conv_log:
        recent = conv_log[-CONV_PROMPT_TURNS:]
        lines = ["━━━ RECENT CONVERSATION HISTORY ━━━",
                 "(This is your memory of past sessions. Use it for "
                 "CONTINUITY — remembering names, ongoing topics, things "
                 "the user told you — reference it naturally if relevant, "
                 "never pretend it doesn't exist. "
                 "IMPORTANT: do NOT treat an old logged answer as still "
                 "correct just because the same or a similar question was "
                 "asked before. If the user asks something again, answer "
                 "it fresh based on the current moment/situation — don't "
                 "just repeat what you said last time as if nothing could "
                 "have changed. This especially applies to anything "
                 "time-sensitive, but also applies generally: a repeated "
                 "question deserves a real fresh answer, not a memory "
                 "playback.)"]
        # Scrub any past ADAM reply containing the "just a language
        # model"/generic-AI-disclaimer pattern before it's re-injected.
        # A single slip into that voice getting replayed verbatim into
        # every future session's prompt was reinforcing the pattern into
        # completely unrelated later conversations — this also cleans up
        # any such lines already persisted on disk from before this fix,
        # not just future ones.
        _disclaimer_markers = (
            "just a language model", "just an ai", "just a chatbot",
            "i'm an ai", "i am an ai", "as an ai", "i don't have a "
            "physical", "i do not have a physical", "large language model",
            "can't help with that", "cannot help with that",
        )
        for turn in recent:
            ts = turn.get("ts", "")
            u  = turn.get("user", "").strip()
            a  = turn.get("adam", "").strip()
            if a and any(m in a.lower() for m in _disclaimer_markers):
                a = ""  # drop the disclaimer reply, keep the user's turn
            if u:
                lines.append(f"  [{ts}] User: {u}")
            if a:
                lines.append(f"  [{ts}] ADAM: {a}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)
 
 
# ═════════════════════════════════════════════════════════════════════════════
# AUDIO HELPERS  (proven working — do not modify)
# ═════════════════════════════════════════════════════════════════════════════
 
def s32_stereo_to_s16_mono_16k(raw: bytes) -> bytes:
    s32   = np.frombuffer(raw, dtype=np.int32)
    if s32.size < 2:
        return b""
    left  = (s32[0::2] >> S32_SHIFT).astype(np.int16)
    right = (s32[1::2] >> S32_SHIFT).astype(np.int16)
    mono  = ((left.astype(np.int32) + right.astype(np.int32)) // 2).astype(np.int16)
    return mono[::3].tobytes()
 
def s32_stereo_to_s16_stereo_channels(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Same S32->S16 downshift as s32_stereo_to_s16_mono_16k, but returns
    the two channels SEPARATELY instead of averaging them together. Needed
    for direction-of-arrival estimation, which requires the phase/timing
    difference between the two physical mics — information that's
    destroyed the instant left+right get averaged into mono."""
    s32 = np.frombuffer(raw, dtype=np.int32)
    if s32.size < 2:
        return np.array([], dtype=np.int16), np.array([], dtype=np.int16)
    left  = (s32[0::2] >> S32_SHIFT).astype(np.int16)
    right = (s32[1::2] >> S32_SHIFT).astype(np.int16)
    return left, right
 
# ── Direction-of-arrival (DOA) via GCC-PHAT ─────────────────────────────
# INMP441 mic spacing on the v32 BODY board — matches the physical
# separation between the two I2S mics on the PCB. Adjust MIC_DISTANCE_M
# if your actual build differs; this value directly scales the angle
# estimate (wrong spacing = systematically wrong angle, not just noisy).
MIC_DISTANCE_M   = 0.065   # 65mm — typical dual-INMP441 spacing
SOUND_SPEED_MPS  = 343.0
DOA_ANGLE_DEADZONE = 8      # degrees — ignore tiny jitter around center
 
def estimate_doa_angle(left: np.ndarray, right: np.ndarray,
                       sample_rate: int = CAPTURE_RATE) -> float:
    """
    Generalized Cross-Correlation with Phase Transform (GCC-PHAT) — a
    standard, well-understood technique for estimating the direction a
    sound arrived from using two microphones. Returns an angle in degrees:
    negative = sound arrived from the left, positive = from the right,
    0 = directly ahead/center. Cheap enough to run per-chunk on a Pi Zero
    2W (a handful of FFTs on ~1600-sample windows).
 
    This does NOT replace Gemini's own audio understanding — it's a
    separate, local signal DGEN can use for physical reactions (turning
    the neck toward a speaker, or telling the model roughly where a voice
    came from) without waiting on a model round-trip.
    """
    try:
        if left.size == 0 or right.size == 0 or left.size != right.size:
            return 0.0
        n = 1 << (int(left.size) - 1).bit_length()  # next pow2 for speed
        L = np.fft.rfft(left.astype(np.float32), n=n)
        R = np.fft.rfft(right.astype(np.float32), n=n)
        cross = L * np.conj(R)
        denom = np.abs(cross)
        denom[denom < 1e-10] = 1e-10  # avoid div-by-zero on silence
        cc = np.fft.irfft(cross / denom, n=n)
 
        max_shift = int(sample_rate * MIC_DISTANCE_M / SOUND_SPEED_MPS) + 1
        cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
        shift = int(np.argmax(cc)) - max_shift
 
        val = (shift / sample_rate) * SOUND_SPEED_MPS / MIC_DISTANCE_M
        val = float(np.clip(val, -1.0, 1.0))
        return float(np.degrees(np.arcsin(val)))
    except Exception:
        return 0.0
 
def s16_mono_24k_to_s16_stereo_48k(raw: bytes, gain: float = 1.0) -> bytes:
    mono = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if mono.size == 0:
        return b""
    if gain != 1.0:
        mono = np.clip(mono * gain, -32768, 32767)
    out_len = mono.size * 2
    up = np.interp(
        np.linspace(0, mono.size - 1, out_len, dtype=np.float32),
        np.arange(mono.size, dtype=np.float32), mono
    ).astype(np.int16)
    return np.repeat(up[:, None], 2, axis=1).reshape(-1).tobytes()
 
def rms_s32(raw: bytes) -> float:
    s = np.frombuffer(raw, dtype=np.int32).astype(np.float64)
    return float(np.sqrt(np.mean(s * s))) if s.size > 0 else 0.0
 
def is_valid_pcm16_chunk(mono16k: bytes) -> bool:
    """
    Sanity gate — structural validation instead of amplitude heuristics.
 
    Earlier revisions tried to detect corruption by how many samples were
    clipped (0.35, then loosened to 0.60 after legitimate loud speech kept
    getting dropped). That was the wrong signal: clipping/amplitude is a
    property of how loud someone is talking and how hot the mic gain is
    set, NOT a reliable indicator of whether the buffer is structurally
    corrupt. Tightening it caused real speech loss ("only hears the last
    part"); loosening it let a genuinely malformed buffer through to
    Gemini, which triggered:
        "1007 invalid frame payload data — Request contains an invalid
         argument" — a protocol-level close that kills the whole session.
 
    The reliable check is structural: PCM16 audio must be a whole number
    of 2-byte samples. The S32->S16 mono 16kHz conversion always produces
    a deterministic, even-length output for valid input. An odd byte
    count (or empty buffer) is a definitive corruption/truncation signal
    regardless of how loud or quiet the audio inside it is — and never
    penalizes legitimate loud speech, which is a completely separate,
    unrelated property that should not be used as a corruption proxy.
    """
    if not mono16k:
        return False
    if len(mono16k) % 2 != 0:
        return False
    arr = np.frombuffer(mono16k, dtype=np.int16)
    if arr.size == 0:
        return False
    return True
 
def beep_s16_stereo(freq=880.0, dur=0.2) -> bytes:
    n    = int(PLAYBACK_RATE * dur)
    t    = np.arange(n, dtype=np.float32) / PLAYBACK_RATE
    mono = np.clip(np.sin(2 * np.pi * freq * t) * 0.3 * 32767, -32768, 32767).astype(np.int16)
    return np.repeat(mono[:, None], 2, axis=1).reshape(-1).tobytes()
 
def read_exact(pipe, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = pipe.read(n - len(buf))
        if not chunk:
            raise EOFError("pipe closed")
        buf.extend(chunk)
    return bytes(buf)
 
def drain_stderr(proc: subprocess.Popen, label: str) -> None:
    try:
        for line in proc.stderr:
            txt = line.decode(errors="replace").strip()
            if txt and "underrun" not in txt.lower():
                print(f"  [{label}] {txt}")
    except Exception:
        pass
 
 
async def _play_song_task(song_playing: asyncio.Event,
                          song_stop_requested: asyncio.Event,
                          active_speaker_proc: list,
                          adam_speaking: asyncio.Event) -> None:
    """
    Plays a randomly-chosen song from SONG_FILE_PATHS by writing its PCM
    audio directly into the SAME aplay process speaker() already has
    open — not a second competing process.
 
    WHY THIS APPROACH (after the previous spawn-a-second-aplay design
    repeatedly hit "Device or resource busy"): speaker() opens ONE aplay
    process that stays alive for the entire session lifetime, only
    recreated on exception/reconnect — never closed between turns. Any
    second process trying to open the same ALSA device (plughw:0,0) will
    always contend with that permanently-open first one, no matter how
    carefully timed. The only way to truly avoid the collision is to not
    open a second device handle at all — write into the one that's
    already open and working, exactly the same way Gemini's own
    converted audio chunks already do via out_q → proc.stdin.write().
 
    This means song files must already be in the playback format
    (48kHz stereo s16 by default) — see SONG_FILE_PATHS' comment for the
    ffmpeg conversion command. No resampling is done here; keeping this
    function simple and fast is more important than accepting arbitrary
    input formats, since resampling on a Pi Zero 2W mid-playback is
    itself a source of audible glitches.
 
    Runs as its own asyncio task so nothing else in the event loop is
    blocked — camera, servos, Gemini send/receive, gestures all keep
    running normally in parallel. Only the mic is muted for the duration.
 
    Stops on whichever comes first: the file finishing naturally, or
    song_stop_requested being set (Touch3 during playback).
    """
    song_playing.set()
    song_stop_requested.clear()
    wav_file = None
    try:
        song_path = random.choice(SONG_FILE_PATHS)
        if not Path(song_path).exists():
            print(f"  ⚠️  Song file not found: {song_path}")
            return
 
        print(f"  🎵 Song playback started: {song_path}")
        tft_set("happy")
 
        wav_file = await asyncio.to_thread(wave.open, song_path, "rb")
        n_channels = wav_file.getnchannels()
        sampwidth  = wav_file.getsampwidth()
        framerate  = wav_file.getframerate()
 
        if (n_channels != PLAYBACK_CHANNELS or sampwidth != 2
                or framerate != PLAYBACK_RATE):
            print(f"  ⚠️  {Path(song_path).name} is {framerate}Hz "
                  f"{n_channels}ch {sampwidth*8}-bit, but playback expects "
                  f"{PLAYBACK_RATE}Hz {PLAYBACK_CHANNELS}ch 16-bit — "
                  f"convert it first with: ffmpeg -i input.mp3 -ar "
                  f"{PLAYBACK_RATE} -ac {PLAYBACK_CHANNELS} -sample_fmt "
                  f"s16 {Path(song_path).stem}.wav")
            return
 
        chunk_frames = 4096  # frames per read, matches speaker()'s own
                              # 4096-byte write granularity for out_q chunks
        pending_data = None  # a chunk that failed to write, retried below
        write_fail_streak = 0
        MAX_WRITE_FAIL_STREAK = 50  # ~10s of retries at 0.2s each before giving up
 
        while True:
            if song_stop_requested.is_set():
                print("  🎵 Song stopped early (Touch3)")
                break
 
            proc = active_speaker_proc[0]
            if proc is None or proc.poll() is not None:
                # speaker()'s aplay isn't available right now (mid-
                # reconnect, or session tearing down) — wait briefly for
                # it to come back rather than giving up. Any chunk we'd
                # already read but failed to write (pending_data) is kept
                # and retried once a live process shows up again, so the
                # song genuinely resumes from where it left off instead
                # of dropping audio or aborting on a reconnect.
                write_fail_streak += 1
                if write_fail_streak > MAX_WRITE_FAIL_STREAK:
                    print("  ⚠️  Song playback gave up — no speaker "
                          "process available after repeated retries")
                    break
                await asyncio.sleep(0.2)
                continue
 
            if pending_data is None:
                data = await asyncio.to_thread(wav_file.readframes, chunk_frames)
                if not data:
                    print("  🎵 Song finished playing")
                    break
            else:
                data = pending_data
                pending_data = None
 
            try:
                if proc.stdin:
                    await asyncio.to_thread(proc.stdin.write, data)
                    await asyncio.to_thread(proc.stdin.flush)
                    write_fail_streak = 0
                else:
                    raise RuntimeError("proc.stdin is None")
            except Exception as e:
                # Don't discard this chunk — the process that just died
                # (e.g. torn down mid-reconnect, confirmed in logs:
                # "Interrupted system call" / "I/O operation on closed
                # file") will be replaced by speaker() shortly. Keep the
                # chunk and retry it once active_speaker_proc[0] points
                # at a live process again, so the song resumes seamlessly
                # instead of stopping on every reconnect.
                pending_data = data
                write_fail_streak += 1
                if write_fail_streak == 1:
                    print(f"  ⚠️  Song playback write interrupted ({e}) — "
                          f"will resume once speaker reconnects")
                if write_fail_streak > MAX_WRITE_FAIL_STREAK:
                    print("  ⚠️  Song playback gave up after repeated "
                          "write failures")
                    break
                await asyncio.sleep(0.2)
                continue
 
            # Yield control between chunks so this doesn't hog the event
            # loop or the shared aplay stdin — camera/servo/Gemini tasks
            # all get their turn between song chunks too.
            await asyncio.sleep(0)
    except Exception as e:
        print(f"  ⚠️  Song playback error: {e}")
    finally:
        if wav_file is not None:
            try:
                wav_file.close()
            except Exception:
                pass
        song_playing.clear()
        song_stop_requested.clear()
        tft_set("happy")
 
 
# ═════════════════════════════════════════════════════════════════════════════
# ESP32-CAM WIRED LINK
# ═════════════════════════════════════════════════════════════════════════════
 
class ESP32Link:
    def __init__(self, port: str, baud: int):
        self.port = port
        self.baud = baud
        self._ser: serial.Serial | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
 
        self.frame_q: sync_queue.Queue = sync_queue.Queue(maxsize=2)
        self.gesture_q: sync_queue.Queue = sync_queue.Queue(maxsize=20)
        self.touch_q: sync_queue.Queue = sync_queue.Queue(maxsize=20)
 
        # ── Background write queue ──────────────────────────────────────
        # FIX: send_line() used to call self._ser.write() directly, which
        # is a BLOCKING pyserial call. Called from an async coroutine (as
        # it was, e.g. from the camera() task on every CAM:ON/CAM:OFF
        # transition), this stalls the entire asyncio event loop for
        # however long the OS write takes — on the same 921600-baud UART
        # that's also carrying continuous frame/touch/gesture reads, that
        # was long enough to cause listen()'s mic read to miss a beat
        # right as a user started talking, truncating the start of their
        # sentence (observed correlating almost 1:1 with camera on/off
        # log lines). A tiny background thread with its own queue means
        # every caller — sync or async — just enqueues instantly and the
        # actual blocking write happens off the event loop, always.
        self._write_q: sync_queue.Queue = sync_queue.Queue(maxsize=64)
        self._write_thread: threading.Thread | None = None
 
        self._connected = False
        self._ever_received_data = False
 
    @property
    def connected(self) -> bool:
        return self._connected
 
    @property
    def receiving_data(self) -> bool:
        return self._ever_received_data
 
    def start(self) -> None:
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=1.0)
            self._connected = True
            print(f"  ✅  UART port opened ({self.port} @ {self.baud}) — "
                  f"waiting to confirm ESP32-CAM is actually wired/powered...")
        except PermissionError as e:
            self._connected = False
            print(f"  ❌  Permission denied opening {self.port}: {e}")
            print("      Fix on the Pi:")
            print("        sudo usermod -a -G dialout pi")
            print("        sudo raspi-config → Interface Options → Serial Port")
            print("          login shell over serial → No")
            print("          serial port hardware enabled → Yes")
            print("        sudo systemctl disable --now serial-getty@ttyAMA0.service")
            print("        sudo reboot")
            print("      ADAM will run WITHOUT vision/touch (audio-only mode) until fixed.")
            return
        except Exception as e:
            self._connected = False
            print(f"  ⚠️  Could not open {self.port}: {e}")
            print("      ADAM will run WITHOUT vision/touch (audio-only mode). "
                  "Check wiring + raspi-config serial settings "
                  "(login shell over serial must be OFF).")
            return
 
        self._thread = threading.Thread(target=self._read_loop, daemon=True,
                                        name="esp32-uart-reader")
        self._thread.start()
 
        self._write_thread = threading.Thread(target=self._write_loop, daemon=True,
                                               name="esp32-uart-writer")
        self._write_thread.start()
 
        def _watch():
            time.sleep(10.0)
            if self._connected and not self._ever_received_data:
                print(f"  ⚠️  UART port is open but no data received from ESP32-CAM "
                      f"in 10s — running WITHOUT vision/touch (audio-only mode). "
                      f"This means the port opened OK but nothing is arriving: "
                      f"check ESP32-CAM is powered, TX/RX aren't swapped, and "
                      f"baud rate matches ({PI_UART_BAUD}).")
        threading.Thread(target=_watch, daemon=True, name="esp32-uart-watchdog").start()
 
    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._write_thread:
            # Wake the writer loop immediately instead of waiting for its
            # queue timeout, so shutdown isn't needlessly slow.
            try:
                self._write_q.put_nowait(None)
            except sync_queue.Full:
                pass
            self._write_thread.join(timeout=2.0)
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass
 
    def _write_loop(self) -> None:
        """Runs on its own dedicated thread. All actual blocking pyserial
        writes happen here — send_line() just enqueues and returns
        immediately, so it's safe to call from anywhere (async coroutine,
        sync helper, doesn't matter) without ever stalling the event loop."""
        while not self._stop.is_set():
            try:
                text = self._write_q.get(timeout=0.5)
            except sync_queue.Empty:
                continue
            if text is None:  # shutdown sentinel
                break
            if not self._connected or not self._ser:
                continue
            try:
                self._ser.write((text.strip() + "\n").encode("utf-8"))
            except Exception as e:
                print(f"  ⚠️  UART write failed: {e}")
 
    def send_line(self, text: str) -> None:
        # Non-blocking — just enqueues for the dedicated writer thread.
        # Safe to call from any context (async coroutine or sync code)
        # without risk of stalling the asyncio event loop. See
        # _write_loop() for where the actual blocking pyserial write
        # happens.
        if not self._connected:
            return
        try:
            self._write_q.put_nowait(text)
        except sync_queue.Full:
            # Queue backed up (shouldn't normally happen — writer thread
            # keeps up easily with our command rate) — drop the oldest
            # pending command rather than blocking the caller.
            try:
                self._write_q.get_nowait()
            except sync_queue.Empty:
                pass
            try:
                self._write_q.put_nowait(text)
            except sync_queue.Full:
                pass
 
    def _read_exact(self, n: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < n and not self._stop.is_set():
            chunk = self._ser.read(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf) if len(buf) == n else None
 
    def _read_loop(self) -> None:
        # ── FIX #1: proper resync instead of trust-next-byte-blindly ────────
        # The previous version read a tag byte, then on a bad/garbage length
        # just printed a warning and `continue`d — which re-reads a FRESH
        # byte from the top of the loop that could ITSELF be mid-JPEG noise.
        # That never actually re-establishes framing sync; it just spins on
        # noise indefinitely (this is exactly the endless
        # "Suspicious frame length ..." storm you were seeing), burning CPU
        # the audio pipeline needs and starving arecord's reads, which is
        # what produced corrupted audio -> Gemini 1007 errors downstream.
        #
        # This version:
        #   1. Only accepts a byte as a tag, then validates what follows.
        #   2. On a bad length, does NOT jump — it naturally continues the
        #      same byte-by-byte read(1) loop, which is the correct way to
        #      hunt for the next real tag byte mid-stream.
        #   3. Validates actual JPEG SOI/EOI markers (FFD8...FFD9) before
        #      ever trusting a "plausible" length, catching the case where a
        #      garbage length happens to fall in a believable range.
        #   4. Rate-limits its own warning prints so a desync storm doesn't
        #      become its own CPU/IO cost.
        last_warn_t = 0.0
        warn_count = 0
 
        def warn_resync(msg: str) -> None:
            nonlocal last_warn_t, warn_count
            warn_count += 1
            now = time.time()
            if now - last_warn_t > 2.0:
                print(f"  ⚠️  UART resync: {msg} ({warn_count} since last report)")
                last_warn_t = now
                warn_count = 0
 
        while not self._stop.is_set():
            try:
                tag_byte = self._ser.read(1)
                if not tag_byte:
                    # No data available right now — normal idle time between
                    # frames (1 FPS camera => long gaps). Sleep briefly so
                    # this thread doesn't spin at 100% CPU on empty reads,
                    # which was starving mic/audio threads on the 2-core Pi.
                    time.sleep(0.01)
                    continue
 
                tag = tag_byte[0]
 
                if tag == TAG_FRAME:
                    len_bytes = self._read_exact(4)
                    if len_bytes is None:
                        continue
                    (frame_len,) = struct.unpack("<I", len_bytes)
                    if frame_len == 0 or frame_len > 200_000:
                        # NOT a real frame tag — a stray byte from inside
                        # another frame's JPEG data that happened to match
                        # 'F'. Do NOT restart from a fresh read(1) trusting
                        # the very next byte either — just fall through to
                        # the top of the while loop, which reads ONE byte
                        # at a time until it finds a genuinely valid tag.
                        warn_resync(f"garbage frame length {frame_len}, "
                                    f"scanning for next valid tag")
                        self._ever_received_data = True
                        continue
 
                    jpeg = self._read_exact(frame_len)
                    if jpeg is None:
                        continue
                    # Sanity-check real JPEG framing before trusting this as
                    # a good frame — catches cases where the length happened
                    # to look plausible (e.g. 4000-80000) but the bytes
                    # weren't actually a frame boundary.
                    if not (jpeg[:2] == b"\xff\xd8" and jpeg[-2:] == b"\xff\xd9"):
                        warn_resync(f"frame length {frame_len} looked "
                                    f"plausible but JPEG markers didn't "
                                    f"match — discarding")
                        continue
                    self._ever_received_data = True
                    try:
                        self.frame_q.put_nowait(jpeg)
                    except sync_queue.Full:
                        try:
                            self.frame_q.get_nowait()
                        except sync_queue.Empty:
                            pass
                        self.frame_q.put_nowait(jpeg)
 
                elif tag == TAG_TOUCH:
                    payload = self._read_exact(4)
                    if payload is None:
                        continue
                    if any(b not in (0, 1) for b in payload):
                        warn_resync("garbage touch payload, ignoring")
                        continue
                    self._ever_received_data = True
                    try:
                        self.touch_q.put_nowait(list(payload))
                    except sync_queue.Full:
                        pass
 
                elif tag == TAG_GESTURE:
                    payload = self._read_exact(1)
                    if payload is None:
                        continue
                    if payload[0] > 3:
                        warn_resync("garbage gesture code, ignoring")
                        continue
                    self._ever_received_data = True
                    try:
                        self.gesture_q.put_nowait(payload[0])
                    except sync_queue.Full:
                        pass
 
                # else: byte didn't match any known tag — expected noise
                # while resyncing after a garbage frame. Loop back and read
                # the next byte; no sleep needed since data IS actively
                # arriving (differs from the "no data at all" idle case
                # handled above).
 
            except Exception as e:
                print(f"  ⚠️  UART reader error: {e}")
                time.sleep(0.5)
 
 
esp_link = ESP32Link(PI_UART_PORT, PI_UART_BAUD)
 
 
# ═════════════════════════════════════════════════════════════════════════════
# SERVO / DISPLAY HELPERS
# ═════════════════════════════════════════════════════════════════════════════
 
def servo_pan(angle: int) -> None:
    if pan_servo is None:
        return
    try:
        pan_servo.angle = max(-90, min(90, int(angle) - 90))
    except Exception:
        pass
 
def servo_tilt(angle: int) -> None:
    esp_link.send_line(f"TILT:{int(angle)}")
 
def tft_set(emotion: str) -> None:
    esp_link.send_line(f"EMO:{emotion}")
 
 
# ═════════════════════════════════════════════════════════════════════════════
# WEB SEARCH
# ═════════════════════════════════════════════════════════════════════════════
 
async def web_search(query: str, max_results: int = 4,
                     recent_only: bool = False) -> str:
    """
    DuckDuckGo search with current-date awareness.
 
    Two separate problems this addresses:
      1. The model itself has no innate sense of "today" beyond whatever
         training data cutoff it has — without being told the real date,
         it can construct stale-feeling queries or misjudge whether a
         search result is current. We prefix every query context (in the
         returned text) with today's actual date so the model can reason
         about recency correctly when it reads the results.
      2. DDG's own `timelimit` parameter (unused previously) can restrict
         results to a recent window server-side — this matters for
         anything genuinely time-sensitive (scores, news, "is X still
         happening") where a top-ranked but months-old result would
         otherwise be indistinguishable from a fresh one.
    """
    global _last_ddg_t
    if DDGS is None:
        return "Web search not available."
    q   = query.strip().lower()
    now = time.time()
    cache_key = f"{q}|{recent_only}"
    if cache_key in _ddg_cache:
        text, ts = _ddg_cache[cache_key]
        if now - ts < SEARCH_CACHE_TTL:
            return text
    gap = now - _last_ddg_t
    if gap < SEARCH_MIN_GAP_S:
        await asyncio.sleep(SEARCH_MIN_GAP_S - gap)
    try:
        def _run():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # timelimit: "d"=past day, "w"=past week, "m"=past month.
                # Only applied when the caller signals this is a
                # time-sensitive query — a generic factual lookup
                # ("how does X work") shouldn't be needlessly restricted
                # to only very recent pages that may not exist.
                kwargs = {"max_results": max_results}
                if recent_only:
                    kwargs["timelimit"] = "m"
                return list(DDGS().text(query, **kwargs))
        results = await asyncio.to_thread(_run)
    except Exception as e:
        return f"Search failed: {e}"
    finally:
        _last_ddg_t = time.time()
    if not results:
        return "No results found."
    lines = []
    for r in results:
        title = str(r.get("title") or "").strip()
        body  = str(r.get("body") or r.get("snippet") or "").strip()
        if title or body:
            lines.append(f"• {title}: {body}" if title else f"• {body}")
    today_str = datetime.datetime.now().strftime("%A, %d %B %Y")
    text = (f"[Search performed on: {today_str}. Use this to judge "
            f"whether results below are current or outdated.]\n"
            + "\n".join(lines))
    _ddg_cache[cache_key] = (text, time.time())
    return text
 
 
# ═════════════════════════════════════════════════════════════════════════════
# TOOL DECLARATIONS
# ═════════════════════════════════════════════════════════════════════════════
 
def build_tools() -> list:
    S, T = types.Schema, types.Type
    return [types.Tool(function_declarations=[
 
        types.FunctionDeclaration(
            name="get_current_datetime",
            description="Returns the current local date and time.",
            parameters=S(type=T.OBJECT, properties={})),
 
        types.FunctionDeclaration(
            name="get_sound_direction",
            description=(
                "Returns which direction the most recent speech came from "
                "(left/right/center, using the two onboard microphones). "
                "ONLY call this if the user EXPLICITLY asks something like "
                "'which direction am I talking from', 'can you tell where "
                "I am', or similar. Never call this proactively or mention "
                "direction unprompted — it's for direct questions only."
            ),
            parameters=S(type=T.OBJECT, properties={})),
 
        types.FunctionDeclaration(
            name="enter_idle_mode",
            description=(
                "Puts ADAM into a persistent silent/idle state — call this "
                "IMMEDIATELY when the user explicitly asks you to 'stay "
                "silent', 'stay mute', 'be quiet', 'stop talking', or "
                "similar. Once called, you will not speak or respond to "
                "anything — including scheduled idle nudges — until the "
                "user says your name again to wake you up. Do NOT call "
                "this for a normal request to pause mid-sentence; it's "
                "specifically for an extended silent mode."
            ),
            parameters=S(type=T.OBJECT, properties={})),
 
        types.FunctionDeclaration(
            name="move_head_gesture",
            description=(
                "Makes ADAM's neck perform a quick, human-like physical "
                "gesture. Use 'nod' for agreement/yes, 'shake' for "
                "disagreement/no, or when it adds natural physical "
                "expression to what you're saying (emphasis, reacting to "
                "something surprising, etc.). Don't overuse it — only "
                "when it genuinely fits the moment, not on every reply."
            ),
            parameters=S(type=T.OBJECT, properties={
                "gesture": S(type=T.STRING, enum=["nod", "shake"]),
            }, required=["gesture"])),
 
        types.FunctionDeclaration(
            name="play_song",
            description=(
                "Plays a song/audio track out loud through ADAM's speaker "
                "— call this when the user asks you to sing, perform, "
                "start a concert, or play music. One of several available "
                "songs is picked at random each time — you don't choose "
                "which. The mic is muted while the song plays (so it "
                "doesn't pick up the song itself), but everything else "
                "keeps running normally in parallel — camera, servos, "
                "conversation state are all unaffected. Playback runs "
                "until the song ends naturally OR the user taps Touch3 to "
                "stop it early. Say something short in character right "
                "before calling this (e.g. 'Alright, here we go!') since "
                "you'll go quiet once the song starts."
            ),
            parameters=S(type=T.OBJECT, properties={})),
 
        types.FunctionDeclaration(
            name="set_emotion",
            description=(
                "Display an emotion on ADAM's face. Call frequently to express reactions."
            ),
            parameters=S(type=T.OBJECT, properties={
                "emotion": S(type=T.STRING,
                             enum=["happy", "sad", "surprised", "angry",
                                   "thinking", "excited", "love", "blush",
                                   "confused", "smug", "sleep", "rizz",
                                   "panic", "shy", "reconnecting"])
            }, required=["emotion"])),
 
        types.FunctionDeclaration(
            name="save_memory",
            description="Permanently save a key-value fact.",
            parameters=S(type=T.OBJECT, properties={
                "key":   S(type=T.STRING),
                "value": S(type=T.STRING),
            }, required=["key", "value"])),
 
        types.FunctionDeclaration(
            name="delete_memory",
            description="Delete a saved memory entry by key.",
            parameters=S(type=T.OBJECT, properties={
                "key": S(type=T.STRING),
            }, required=["key"])),
 
        types.FunctionDeclaration(
            name="get_memory",
            description="Retrieve a specific memory entry or all entries.",
            parameters=S(type=T.OBJECT, properties={
                "key": S(type=T.STRING, description="Omit to get all entries"),
            })),
 
        types.FunctionDeclaration(
            name="remember_person",
            description="Save a person to permanent visual memory.",
            parameters=S(type=T.OBJECT, properties={
                "person_id":    S(type=T.STRING),
                "name":         S(type=T.STRING),
                "appearance":   S(type=T.STRING),
                "relationship": S(type=T.STRING),
                "notes":        S(type=T.STRING),
            }, required=["person_id", "name"])),
 
        types.FunctionDeclaration(
            name="web_search",
            description=(
                "Search the internet via DuckDuckGo for real-time information. "
                "Results are automatically tagged with today's actual date so "
                "you can judge whether they're current. "
                "DO NOT call this before every answer — that adds real delay "
                "to a live voice conversation. Correct usage: (1) answer "
                "first from what you already know, then ask the user if "
                "they want you to check online for the latest info — only "
                "call this tool if they confirm yes; OR (2) call it directly "
                "without asking ONLY when you have genuinely no relevant "
                "information at all to offer. If web_search returns nothing "
                "useful, say plainly that you couldn't find a reliable "
                "answer instead of inventing plausible-sounding details, "
                "names, or dates."
            ),
            parameters=S(type=T.OBJECT, properties={
                "query": S(type=T.STRING),
                "recent_only": S(
                    type=T.BOOLEAN,
                    description=(
                        "Set true for genuinely time-sensitive queries "
                        "(live scores, breaking news, 'is X still "
                        "happening') to restrict results to roughly the "
                        "past month instead of any-time results. Leave "
                        "false/omit for general facts that don't need "
                        "that restriction."
                    )),
            }, required=["query"])),
 
        build_laptop_control_declaration(),
 
    ])]
 
 
def build_laptop_control_declaration() -> types.FunctionDeclaration:
    S, T = types.Schema, types.Type
    actions = get_laptop_actions()
    action_names = list(actions.keys())
 
    lines = []
    for name, spec in actions.items():
        if spec.get("needs_value"):
            lines.append(f"  - {name} (needs value, {spec.get('value_hint','')}): "
                         f"{spec.get('description','')}")
        else:
            lines.append(f"  - {name}: {spec.get('description','')}")
    action_doc = "\n".join(lines) if lines else "  (no actions currently available)"
 
    return types.FunctionDeclaration(
        name="laptop_control",
        description=(
            "Control the user's laptop via laptop_agent.py, found automatically "
            "on the network — no manual setup needed. Available actions:\n"
            + action_doc + "\nOnly pass 'value' for actions that need it. "
            "ONLY call this when the user EXPLICITLY asks you to change "
            "volume/brightness or mute/unmute (e.g. 'turn up the volume', "
            "'make it brighter'). Do NOT call this as a dramatic flourish, "
            "joke, or emotional reaction (e.g. to express anger, "
            "excitement, or affection) — a touch gesture, emotion, or "
            "sarcastic remark is never itself a request to control the "
            "laptop."
        ),
        parameters=S(type=T.OBJECT, properties={
            "action": S(type=T.STRING, enum=action_names or ["volume_up"]),
            "value": S(type=T.INTEGER,
                       description="Required only for *_set actions (0-100)."),
        }, required=["action"]))
 
 
# ═════════════════════════════════════════════════════════════════════════════
# TOOL HANDLER
# ═════════════════════════════════════════════════════════════════════════════
 
EMOTION_NOD = {
    "happy": "nod", "excited": "nod", "surprised": "nod", "love": "nod",
    "sad": "none",  "angry": "none",  "thinking": "none", "blush": "none",
    "confused": "none", "smug": "none", "sleep": "none", "rizz": "none",
    "panic": "none", "shy": "none", "reconnecting": "none",
}
 
# Module-level tracker for the emotion fix: set_emotion() calls update
# this; end_of_turn() in the speaker task checks and clears it. Safe as a
# plain module global since this codebase only ever runs one live session
# at a time (see run_session's single-session design throughout).
_last_emotion_set_this_turn = [False]
# Tracks whether the face CURRENTLY on screen is the transient
# "speaking" placeholder (as opposed to a deliberately-set emotion like
# love/angry/sad). Only this specific case should auto-reset back to a
# resting face when speech ends — a deliberately-set emotion should
# persist naturally. This was missing entirely after the previous fix
# removed the happy-fallback, which fixed "always resets to happy" but
# broke the opposite direction: nothing ever reset "speaking" back to a
# resting face once actual speech ended, so it stayed stuck showing
# "speaking" indefinitely.
_face_is_generic_speaking = [False]
 
# Module-level mirror of the session's DOA state, for get_sound_direction's
# handler (a module-level function, can't directly close over run_session's
# local doa_angle/doa_last_update_t). Updated from listen() on every fresh
# reading. Safe as a plain global since this codebase runs one live session
# at a time, same reasoning as _last_emotion_set_this_turn above.
_doa_angle = [0.0]
_doa_last_update_t = [0.0]
 
# Module-level mirror of idle_mode, for enter_idle_mode's handler — same
# reasoning as the DOA mirror above (handle_tool_call is module-level,
# can't directly close over run_session's local idle_mode Event). The
# run_session loop reads this each tick and syncs it to the real
# asyncio.Event, since a plain bool is simpler to touch from a sync-style
# tool handler than exposing the Event object itself across that boundary.
_idle_mode_requested = [False]
 
# PERSISTENT idle-mode state, surviving across reconnects. The session-
# local `idle_mode` asyncio.Event() inside run_session() is recreated
# fresh on every single call — including every reconnect (GoAway,
# transient 1007, network hiccup). Since conversations routinely span
# multiple sessions, idle mode was silently resetting to "not idle" on
# any reconnect with NO visible log line indicating it happened — the
# bug report showing full responses resuming with no "wake phrase heard"
# line is explained exactly by this: a reconnect happened between turns,
# and the fresh session's idle_mode simply started False again. This
# module-level flag is the source of truth that DOES survive reconnects;
# run_session() syncs its local Event to/from this at session start and
# on every change.
_idle_mode_persistent = [False]
 
# Module-level mirror for play_song requests — same reasoning as
# _idle_mode_requested above. run_session() reads this each receive-loop
# tick right after tool dispatch and starts actual playback there, since
# that's where it has access to the real session-scoped song_playing/
# song_stop_requested Events and can spawn the background playback task.
_play_song_requested = [False]
 
async def handle_tool_call(tc, ws_broadcast_fn) -> list:
    responses = []
    for fc in tc.function_calls:
        name    = fc.name
        call_id = fc.id
        args    = dict(fc.args) if fc.args else {}
        try:
            if name == "get_current_datetime":
                now    = datetime.datetime.now()
                result = {
                    "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "date":     now.strftime("%A, %d %B %Y"),
                    "time":     now.strftime("%I:%M %p"),
                }
 
            elif name == "get_sound_direction":
                age = time.time() - _doa_last_update_t[0]
                if age > 4.0:
                    result = {"available": False,
                              "reason": "No recent enough audio reading to tell."}
                elif abs(_doa_angle[0]) <= DOA_ANGLE_DEADZONE:
                    result = {"available": True, "direction": "center",
                              "detail": "Sounds like you're roughly straight ahead."}
                else:
                    direction = "left" if _doa_angle[0] < 0 else "right"
                    result = {"available": True, "direction": direction,
                              "degrees_off_center": abs(int(_doa_angle[0]))}
 
            elif name == "enter_idle_mode":
                _idle_mode_requested[0] = True
                print("  🔇 enter_idle_mode called — will go silent")
                result = {"status": "ok",
                          "note": "Going silent now until woken by name."}
 
            elif name == "move_head_gesture":
                gesture = args.get("gesture", "nod")
                print(f"  🤖 Head gesture: {gesture}")
 
                async def _do_gesture():
                    if gesture == "nod":
                        # Quick tilt down-up-down-center — a natural
                        # "yes" nod using the tilt servo.
                        for ang in (NECK_TILT_CENTER + 12,
                                   NECK_TILT_CENTER - 6,
                                   NECK_TILT_CENTER + 8,
                                   NECK_TILT_CENTER):
                            servo_tilt(ang)
                            await asyncio.sleep(0.18)
                    else:  # shake
                        # Quick pan left-right-left-center — a natural
                        # "no" shake using the pan servo.
                        for ang in (NECK_PAN_CENTER - 15,
                                   NECK_PAN_CENTER + 15,
                                   NECK_PAN_CENTER - 8,
                                   NECK_PAN_CENTER):
                            await asyncio.to_thread(servo_pan, ang)
                            await asyncio.sleep(0.18)
 
                # Run in the background so the tool response returns
                # immediately rather than blocking the model's turn on
                # ~0.7s of servo movement.
                asyncio.create_task(_do_gesture())
                result = {"status": "ok"}
 
            elif name == "play_song":
                if _play_song_requested[0]:
                    # Guard against duplicate tool_call messages in the
                    # same turn (observed in logs — Gemini can emit the
                    # same function call twice) triggering two overlapping
                    # song starts. Second call this turn is a no-op.
                    print("  🎵 play_song called again this turn — ignoring duplicate")
                    result = {"status": "ok", "note": "Already starting."}
                elif not any(Path(p).exists() for p in SONG_FILE_PATHS):
                    print(f"  ⚠️  play_song called but no song files found "
                          f"in: {SONG_FILE_PATHS}")
                    result = {"status": "error",
                              "reason": "No song files found — nothing to play."}
                else:
                    _play_song_requested[0] = True
                    print("  🎵 play_song called — starting playback")
                    result = {"status": "ok",
                              "note": "Playing now. Mic is muted until the "
                                      "song ends or Touch3 stops it."}
 
            elif name == "set_emotion":
                emotion = args.get("emotion", "happy")
                tft_set(emotion)
                _last_emotion_set_this_turn[0] = True
                _face_is_generic_speaking[0] = False
                await ws_broadcast_fn({"type": "emotion", "emotion": emotion,
                                       "head": EMOTION_NOD.get(emotion, "none")})
                result = {"status": "ok"}
 
            elif name == "save_memory":
                key = args.get("key", "").strip()
                val = args.get("value", "").strip()
                if key:
                    memory[key] = val
                    save_json(MEMORY_FILE, memory)
                    print(f"  🧠 Memory saved: {key}")
                    result = {"status": "saved"}
                else:
                    result = {"status": "error", "reason": "key empty"}
 
            elif name == "delete_memory":
                key = args.get("key", "").strip()
                if key in memory:
                    del memory[key]
                    save_json(MEMORY_FILE, memory)
                    result = {"status": "deleted"}
                else:
                    result = {"status": "not_found"}
 
            elif name == "get_memory":
                key    = args.get("key", "").strip()
                result = {"value": memory.get(key) if key else None, "all": memory}
 
            elif name == "remember_person":
                pid = args.get("person_id") or f"person_{int(time.time())}"
                faces[pid] = {
                    "name":         args.get("name", "Unknown"),
                    "appearance":   args.get("appearance", ""),
                    "relationship": args.get("relationship", "acquaintance"),
                    "notes":        args.get("notes", ""),
                    "last_seen":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                save_json(FACE_MEMORY_FILE, faces)
                print(f"  👤 Remembered: {args.get('name')} [{pid}]")
                result = {"status": "saved", "person_id": pid}
 
            elif name == "web_search":
                query = args.get("query", "").strip()
                recent_only = bool(args.get("recent_only", False))
                if query:
                    raw    = await web_search(query, recent_only=recent_only)
                    result = {"results": raw[:600] + ("…" if len(raw) > 600 else "")}
                else:
                    result = {"error": "query empty"}
 
            elif name == "laptop_control":
                action = args.get("action", "")
                value  = args.get("value")
                if value is not None:
                    try:
                        value = int(value)
                    except (TypeError, ValueError):
                        value = None
                print(f"  🖥️  laptop_control → action={action} value={value}")
                result = await asyncio.to_thread(laptop_control_sync, action, value)
                if result.get("status") == "ok":
                    print(f"  ✅ laptop_control ok: {result}")
                else:
                    print(f"  ⚠️  laptop_control failed: {result}")
 
            else:
                result = {"error": f"unknown tool: {name}"}
 
        except Exception as e:
            result = {"error": str(e)}
            print(f"  ⚠️  Tool {name} error: {e}")
 
        responses.append({"id": call_id, "name": name, "response": result})
    return responses
 
 
# ═════════════════════════════════════════════════════════════════════════════
# WEBSOCKET FACE SERVER
# ═════════════════════════════════════════════════════════════════════════════
 
WS_HOST    = "localhost"
WS_PORT    = 8765
ws_clients: set = set()
 
async def ws_broadcast(payload: dict) -> None:
    if not ws_clients:
        return
    msg  = json.dumps(payload)
    dead = set()
    for ws in list(ws_clients):
        try:
            await ws.send(msg)
        except Exception:
            dead.add(ws)
    ws_clients.difference_update(dead)
 
async def ws_handler(websocket) -> None:
    ws_clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        ws_clients.discard(websocket)
 
async def start_ws_server() -> None:
    try:
        import websockets.server
        srv = await websockets.server.serve(ws_handler, WS_HOST, WS_PORT)
        print(f"✅ WebSocket face server → ws://{WS_HOST}:{WS_PORT}")
        return srv
    except Exception as e:
        print(f"⚠️  WebSocket server unavailable: {e}")
        return None
 
 
# ═════════════════════════════════════════════════════════════════════════════
# SESSION
# ═════════════════════════════════════════════════════════════════════════════
 
async def run_session(client, resume_handle: str | None,
                      stop: asyncio.Event, out_q: asyncio.Queue) -> str | None:
 
    print(f"\n  Connecting{' (resuming)' if resume_handle else ''}...")
    system_prompt = build_system_prompt()
 
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=system_prompt,
        tools=build_tools(),
        session_resumption=types.SessionResumptionConfig(handle=resume_handle),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        context_window_compression=types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow(),
        ),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=VOICE)
            )
        ),
    )
 
    latest_handle: str | None = resume_handle
    # Set to True on a 1007 (server-rejected-payload) close. See detailed
    # explanation at its usage site in send() below — this is a confirmed
    # Google-side Live API bug (python-genai#2290) where resuming a
    # session that used both audio and video can fail every subsequent
    # audio send in a tight reconnect loop.
    force_fresh_session = [False]
    # Set to True on a 1011 quota/billing error. See the outer except
    # block below for full explanation.
    quota_exceeded = [False]
 
    try:
        async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
            print("  ✅ Connected to Gemini Live")
 
            mic_q            = asyncio.Queue(maxsize=MIC_Q_MAX)
            adam_speaking    = asyncio.Event()
            latest_frame     = [None]
            attention_active = asyncio.Event()
            last_interact_t  = [time.time()]
            last_user_text   = [""]
            interrupt_flag   = asyncio.Event()
            # ── Song playback state ──────────────────────────────────────
            # song_playing: mic-mute gate for the duration of playback
            # (listen()/send() check this the same way they check
            # adam_speaking, so nothing extra needed there).
            # song_stop_requested: set by Touch3 while a song is playing,
            # to end playback early. Distinct from GESTURE_STOP's normal
            # idle-mode-toggle behavior — see gesture_watch() below for
            # how Touch3 is routed differently depending on whether a
            # song is currently playing.
            song_playing         = asyncio.Event()
            song_stop_requested  = asyncio.Event()
            # Shared reference to speaker()'s currently-live aplay
            # process/stdin — the song task writes into THIS SAME
            # process instead of spawning a second one. speaker()'s
            # aplay stays open for the entire session lifetime (only
            # recreated on exception/reconnect, not between turns), so a
            # second process trying to open the same ALSA device was
            # always going to collide — confirmed repeatedly in logs
            # ("Device or resource busy") even well after speech had
            # finished, since the first aplay never actually closes
            # between turns. Routing the song through the SAME open
            # process eliminates the contention entirely rather than
            # trying to time around it.
            active_speaker_proc = [None]
            # ── Idle/silent mode ─────────────────────────────────────────
            # Distinct from interrupt_flag (which only suppresses the ONE
            # in-flight reply). idle_mode is a PERSISTENT state: once set
            # (via STOP touch gesture, or the user explicitly asking ADAM
            # to "stay silent"/"stay mute"), NO audio is sent to Google at
            # all while idle — wake detection runs entirely locally via
            # Vosk (offline STT) watching for "adam" in the mic stream, or
            # via the Touch3 physical gesture. Idle nudges are also
            # suppressed. Only hearing "adam" (locally) or Touch3 exits
            # idle mode.
            #
            # FIX: this Event is recreated fresh on every run_session()
            # call, including reconnects — it does NOT survive a GoAway/
            # 1007/network-hiccup reconnect on its own. Initialize it
            # from the module-level _idle_mode_persistent flag (the real
            # source of truth across sessions) so a reconnect mid-idle
            # doesn't silently wake ADAM back up with no visible cause.
            idle_mode        = asyncio.Event()
            if _idle_mode_persistent[0]:
                idle_mode.set()
                print("  🔇 Resuming idle mode after reconnect "
                      "(servos re-centered)")
                tft_set("sleep")
                await asyncio.to_thread(servo_pan, 90)
                servo_tilt(90)
            # Feeds raw mic audio to the local Vosk wake-word detector
            # while idle. Only populated during idle_mode — see listen().
            wake_word_q: asyncio.Queue = asyncio.Queue(maxsize=200)
            # Tracks whether set_emotion() was called during the current
            # turn. Previously end_of_turn() unconditionally forced the
            # face back to "happy" after every single reply, silently
            # overwriting any deliberate emotion the model had just set
            # (love, angry, sad, etc.) the moment ADAM finished speaking —
            # making it look like emotions "got stuck on happy" when
            # actually they were being reset on a fixed timer, not stuck.
            emotion_set_this_turn = [False]
 
            # ── Direction-of-arrival state ──────────────────────────────
            # Smoothed sound-direction angle from the two mics (GCC-PHAT).
            # Updated in listen() on every chunk where speech is detected,
            # read by camera()'s neck-tracking logic to turn toward
            # whoever is currently talking, and available to inject into
            # the model's context if useful.
            doa_angle = [0.0]          # smoothed angle, degrees (-90..90)
            doa_last_update_t = [0.0]
 
            attention_active.set()
 
            async def inject(text: str, retries: int = 6) -> bool:
                for _ in range(retries):
                    if stop.is_set():
                        return False
                    try:
                        await session.send_realtime_input(text=text)
                        return True
                    except Exception:
                        await asyncio.sleep(0.3)
                return False
 
            async def listen() -> None:
                print("  🎤 Listen task started")
                read_bytes = CHUNK_FRAMES * CAPTURE_CHANNELS * 4
                _last_rms  = [0.0]
                _dropped_bad_chunks = [0]
                _last_bad_warn_t = [0.0]
                # ── Adaptive noise-floor calibration ──────────────────────
                # BUG FIX: peripheral noise (servo whine during a move,
                # electrical coupling from the UART/camera link, fans) can
                # produce RMS bursts above the old fixed MIC_SILENCE_FLOOR
                # while still being much quieter than actual speech. A
                # static global floor can't tell the two apart — raising it
                # risks cutting real quiet speech, leaving it low lets noise
                # bursts through as if they were speech, which can confuse
                # Gemini's own turn-detection right as the user starts
                # talking (their real speech gets bundled with/cut off by
                # the noise burst). Fix: track a rolling ambient noise
                # baseline during quiet stretches, and require a chunk to
                # clear that baseline by a real margin (not just the fixed
                # floor) before it's treated as meaningful audio.
                _ambient_rms = [MIC_SILENCE_FLOOR * 0.5]  # starting estimate
                _AMBIENT_ALPHA = 0.05      # slow-moving average
                _SPEECH_MARGIN = 3.0       # must exceed ambient*this to count
 
                while not stop.is_set():
                    proc = None
                    try:
                        cmd = ["arecord",
                               "-D", CAPTURE_DEVICE,
                               "-f", CAPTURE_FORMAT,
                               "-r", str(CAPTURE_RATE),
                               "-c", str(CAPTURE_CHANNELS),
                               "-t", "raw", "-q"]
                        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE, bufsize=0)
                        await asyncio.sleep(1.0)
                        if proc.poll() is not None:
                            err = proc.stderr.read().decode(errors="replace").strip()
                            print(f"  ❌ arecord failed: {err}")
                            await asyncio.sleep(3.0)
                            continue
 
                        print(f"  ✅ arecord: {CAPTURE_DEVICE} {CAPTURE_FORMAT} "
                              f"{CAPTURE_RATE}Hz {CAPTURE_CHANNELS}ch")
                        errors = 0
 
                        # ── Hardware warm-up discard ──────────────────────
                        # The first fraction of a second of audio right
                        # after arecord opens the capture device is
                        # typically unstable — DC offset hasn't settled,
                        # some HATs/codecs ramp their AGC (automatic gain
                        # control) up over the first few frames, and ALSA's
                        # own buffer needs a moment to reach steady state.
                        # This produces wildly inconsistent RMS readings on
                        # startup/reconnect that don't reflect real input
                        # levels and could feed garbage into VAD/attention
                        # logic. Discard a short warm-up window's worth of
                        # chunks (not sent to Gemini, not RMS-logged)
                        # before treating capture as "live."
                        warmup_bytes_target = int(
                            CAPTURE_RATE * CAPTURE_CHANNELS * 4 * 0.4)  # ~0.4s
                        warmup_discarded = 0
                        while (warmup_discarded < warmup_bytes_target
                               and not stop.is_set()):
                            try:
                                _ = await asyncio.to_thread(
                                    read_exact, proc.stdout, read_bytes)
                                warmup_discarded += read_bytes
                            except Exception:
                                break
 
                        while not stop.is_set():
                            try:
                                raw = await asyncio.to_thread(
                                    read_exact, proc.stdout, read_bytes)
                            except Exception as e:
                                errors += 1
                                if errors > 5:
                                    print(f"  ⚠️  arecord read: {e} — restarting")
                                    break
                                await asyncio.sleep(0.5)
                                continue
                            errors = 0
 
                            if adam_speaking.is_set() or song_playing.is_set():
                                while not mic_q.empty():
                                    try: mic_q.get_nowait()
                                    except asyncio.QueueEmpty: break
                                continue
 
                            mono16k = await asyncio.to_thread(
                                s32_stereo_to_s16_mono_16k, raw)
                            if not mono16k:
                                continue
 
                            # ── Direction-of-arrival (two-mic) ────────────────
                            # Only bother computing this when there's enough
                            # signal to be worth it — GCC-PHAT on pure noise/
                            # silence produces meaningless jittery angles and
                            # wastes CPU on every single chunk otherwise.
                            # Also skipped entirely while idle — the servo
                            # must not track sound direction at all during
                            # idle mode, and not updating doa_angle here is
                            # belt-and-suspenders alongside camera()'s own
                            # idle_mode check, eliminating any possibility
                            # of a stale/racy update influencing the servo.
                            if not idle_mode.is_set():
                                _rms_for_doa = rms_s32(raw)
                                if _rms_for_doa > MIC_LIVE_RMS_THRESHOLD * 0.5:
                                    def _compute_doa():
                                        left, right = s32_stereo_to_s16_stereo_channels(raw)
                                        return estimate_doa_angle(left, right, CAPTURE_RATE)
                                    angle = await asyncio.to_thread(_compute_doa)
                                    if abs(angle) > DOA_ANGLE_DEADZONE:
                                        # Light smoothing so the neck doesn't
                                        # jitter on every chunk — exponential
                                        # moving average, not a hard snap.
                                        doa_angle[0] = (doa_angle[0] * 0.6) + (angle * 0.4)
                                        doa_last_update_t[0] = time.time()
                                        # Mirror to module-level state for the
                                        # get_sound_direction tool handler,
                                        # which lives outside this closure.
                                        _doa_angle[0] = doa_angle[0]
                                        _doa_last_update_t[0] = doa_last_update_t[0]
 
                            # ── FIX #2: audio sanity gate ─────────────────────
                            # Drop corrupted/desynced chunks BEFORE they reach
                            # Gemini. This is what previously produced:
                            #   "receive error: 1007 None. Request contains
                            #    an invalid argument." — a single garbage
                            #   chunk could kill the whole Live session.
                            if not is_valid_pcm16_chunk(mono16k):
                                _dropped_bad_chunks[0] += 1
                                now_w = time.time()
                                if now_w - _last_bad_warn_t[0] > 2.0:
                                    print(f"  ⚠️  Dropped {_dropped_bad_chunks[0]} "
                                          f"corrupted audio chunk(s) before "
                                          f"send — check UART/CPU contention "
                                          f"if this repeats constantly")
                                    _last_bad_warn_t[0] = now_w
                                    _dropped_bad_chunks[0] = 0
                                continue
 
                            now = time.time()
                            _rms_now = rms_s32(raw)
                            if now - _last_rms[0] > 4.0:
                                print(f"  🎤 Mic RMS: {_rms_now:.0f}")
                                _last_rms[0] = now
 
                            if _rms_now > MIC_LIVE_RMS_THRESHOLD:
                                attention_active.set()
 
                            # ── SILENCE GATE ─────────────────────────────────
                            # Previously every mic chunk was queued/sent to
                            # Gemini unconditionally, including pure room
                            # noise/silence between sentences. Continuously
                            # streaming near-silent audio gives the Live API
                            # ungrounded input during quiet stretches, which
                            # is a known trigger for unprompted "phantom"
                            # responses (the random Hindi hallucinations) —
                            # the model free-associates from thin signal
                            # instead of responding to real speech.
                            #
                            # MIC_SILENCE_FLOOR is set well below normal
                            # speech RMS (your speech reads 25M-60M; true
                            # silence/room tone is typically under a few
                            # hundred thousand) so genuine quiet speech is
                            # never at risk of being gated out — only true
                            # silence is withheld.
                            if _rms_now < MIC_SILENCE_FLOOR:
                                continue
 
                            # ── Adaptive noise-floor gate ─────────────────
                            # A chunk must clear the ROLLING ambient
                            # baseline by a real margin, not just the fixed
                            # global floor above — this is what actually
                            # distinguishes a peripheral noise burst (which
                            # can be louder than true silence but is still
                            # much quieter than real speech) from genuine
                            # speech onset. The ambient baseline itself is
                            # only nudged toward RELATIVELY QUIET chunks
                            # (below the current speech threshold), so a
                            # sustained noise burst doesn't drag the
                            # baseline up and end up masking itself.
                            speech_threshold = _ambient_rms[0] * _SPEECH_MARGIN
                            if _rms_now < speech_threshold:
                                # Not clearly speech — could be ambient
                                # noise. Let it nudge the baseline (slowly)
                                # so the calibration tracks the room's
                                # actual current noise floor, then drop it.
                                _ambient_rms[0] = (
                                    (1 - _AMBIENT_ALPHA) * _ambient_rms[0]
                                    + _AMBIENT_ALPHA * _rms_now)
                                continue
 
                            if idle_mode.is_set():
                                # While idle, audio goes ONLY to the local
                                # wake-word detector — never to mic_q
                                # (which feeds Gemini via send()). This is
                                # what actually keeps audio off Google
                                # during idle, not just discarding the
                                # response afterward.
                                if VOSK_AVAILABLE:
                                    try:
                                        wake_word_q.put_nowait(mono16k)
                                    except asyncio.QueueFull:
                                        pass
                                continue
 
                            if not mic_q.full():
                                mic_q.put_nowait(mono16k)
 
                            await asyncio.sleep(0)
 
                    except asyncio.CancelledError:
                        # MUST re-raise — see speaker()'s corresponding fix
                        # for the full explanation. Swallowing this here
                        # lets the outer `while not stop.is_set()` loop
                        # (process-level stop, not per-session
                        # cancellation) respawn arecord instead of letting
                        # this task actually terminate when run_session()
                        # cancels it.
                        raise
                    except Exception as e:
                        print(f"  ⚠️  listen recovering: {e}")
                        await asyncio.sleep(2.0)
                    finally:
                        if proc:
                            # FIX: proc.terminate()/proc.wait() are
                            # BLOCKING synchronous calls. Running them
                            # directly inside an async finally stalls the
                            # ENTIRE event loop for up to the timeout
                            # (2s) if arecord is slow to exit — during a
                            # multi-task cancellation (e.g. right after a
                            # 1007 error kills the session), this could
                            # make the whole reconnect look hung rather
                            # than fast, since asyncio.gather() is
                            # waiting on this coroutine to actually finish
                            # before run_session() can return and let
                            # main()'s loop attempt to reconnect.
                            async def _kill_proc():
                                try:
                                    proc.terminate()
                                    await asyncio.to_thread(proc.wait, 2)
                                except Exception:
                                    try:
                                        proc.kill()
                                    except Exception:
                                        pass
                            try:
                                await asyncio.wait_for(_kill_proc(), timeout=3.0)
                            except asyncio.TimeoutError:
                                try: proc.kill()
                                except Exception: pass
                print("  🎤 Listen ended")
 
            async def send() -> None:
                print("  📤 Send task started")
                while not stop.is_set():
                    try:
                        chunk = await asyncio.wait_for(mic_q.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break
                    if adam_speaking.is_set() or song_playing.is_set():
                        continue
                    if idle_mode.is_set():
                        # While idle, audio must NOT reach Google at all —
                        # not "sent but response discarded" (the previous,
                        # incorrect approach), genuinely never sent. Wake
                        # detection during idle runs entirely locally via
                        # the offline wake_word_detector task instead,
                        # which reads from wake_word_q (fed below).
                        continue
                    try:
                        await session.send_realtime_input(
                            audio=types.Blob(data=chunk,
                                             mime_type=f"audio/pcm;rate={GEMINI_SEND_RATE}"))
                    except asyncio.CancelledError:
                        return
                    except Exception as e:
                        err_str = str(e)
                        if "1007" in err_str:
                            # CONFIRMED GOOGLE-SIDE BUG (python-genai#2290):
                            # resuming a session that has used both mic
                            # audio AND camera video — which every ADAM
                            # session does — can leave the resumed session
                            # broken, failing every subsequent audio send
                            # with this same 1007 in a tight reconnect
                            # loop. Previously this code assumed resuming
                            # via the existing handle was safe (it isn't,
                            # for this specific error) — now it forces the
                            # next reconnect to start a genuinely fresh
                            # session instead, breaking the loop. Recent
                            # conversation context is preserved separately
                            # via the persisted conversation history
                            # (adam_conversations.json), not the broken
                            # resumption handle.
                            force_fresh_session[0] = True
                            print(f"  ⚠️  Session closed by server (1007 — "
                                  f"rejected audio payload). This is a known "
                                  f"Live API resumption bug with audio+video "
                                  f"sessions — starting a FRESH session next "
                                  f"(not resuming) to avoid a reconnect loop.")
                        else:
                            print(f"  ⚠️  send error (session likely closing): {e}")
                        return
                print("  📤 Send ended")
 
            async def receive() -> None:
                nonlocal latest_handle
                print("  📥 Receive task started")
                adam_text = []
                cur_user_text = [""]
                try:
                    while not stop.is_set():
                        try:
                            async for msg in session.receive():
                                if stop.is_set():
                                    return
 
                                if (msg.session_resumption_update
                                        and msg.session_resumption_update.new_handle):
                                    latest_handle = msg.session_resumption_update.new_handle
 
                                # ── GoAway handling ───────────────────────────
                                # Gemini Live sends a GoAway message shortly
                                # BEFORE force-closing a session that's hit its
                                # max duration — this is a normal, documented
                                # part of the protocol, not an error. Without
                                # explicitly handling it, nothing reacted until
                                # the hard disconnect actually happened (visible
                                # as "1008 ... Connection aborted because the
                                # client failed to close the connection after
                                # receiving a GoAway signal"), which killed
                                # every task (send/receive/camera/listen)
                                # simultaneously in a messier way than a clean
                                # proactive handoff. Now: the instant GoAway
                                # arrives, immediately show the reconnecting
                                # face (so the user visually knows ADAM is
                                # about to reconnect, not just frozen/dead) and
                                # return cleanly with whatever resumption
                                # handle we have, letting the outer loop in
                                # main() start a fresh session right away.
                                if getattr(msg, "go_away", None) is not None:
                                    time_left = getattr(msg.go_away, "time_left", None)
                                    print(f"  🔄 GoAway received (time_left="
                                          f"{time_left}) — session ending "
                                          f"soon, reconnecting proactively")
                                    tft_set("reconnecting")
                                    return
 
                                if msg.tool_call:
                                    resps = await handle_tool_call(msg.tool_call, ws_broadcast)
                                    await session.send_tool_response(
                                        function_responses=[
                                            types.FunctionResponse(
                                                id=r["id"], name=r["name"],
                                                response=r["response"])
                                            for r in resps
                                        ]
                                    )
                                    # Sync enter_idle_mode's module-level
                                    # flag into the real session-local
                                    # Event, matching STOP gesture behavior
                                    # exactly (servo center, sleep face).
                                    if _idle_mode_requested[0]:
                                        _idle_mode_requested[0] = False
                                        idle_mode.set()
                                        _idle_mode_persistent[0] = True
                                        tft_set("sleep")
                                        await asyncio.to_thread(servo_pan, 90)
                                        servo_tilt(90)
                                        print("  🔇 Idle mode active (voice "
                                              "request) — servos centered")
                                    if _play_song_requested[0]:
                                        _play_song_requested[0] = False
                                        if not song_playing.is_set():
                                            async def _start_song_after_speech():
                                                # The tool_call message and
                                                # the model's spoken
                                                # acknowledgment ("Alright,
                                                # here we go!") arrive as
                                                # SEPARATE streamed messages
                                                # within the same turn —
                                                # tool_call typically comes
                                                # first. Wait for that
                                                # acknowledgment to actually
                                                # finish before writing song
                                                # audio into the SAME
                                                # aplay stdin — writing both
                                                # at once would interleave/
                                                # corrupt the stream, even
                                                # though it's no longer a
                                                # "busy device" problem
                                                # (only one process now).
                                                grace_deadline = time.time() + 2.0
                                                spoke_this_turn = False
                                                while time.time() < grace_deadline:
                                                    if adam_speaking.is_set():
                                                        spoke_this_turn = True
                                                        break
                                                    await asyncio.sleep(0.05)
                                                if spoke_this_turn:
                                                    waited = 0.0
                                                    while adam_speaking.is_set() and waited < 15.0:
                                                        await asyncio.sleep(0.1)
                                                        waited += 0.1
                                                await _play_song_task(
                                                    song_playing,
                                                    song_stop_requested,
                                                    active_speaker_proc,
                                                    adam_speaking)
                                            asyncio.create_task(_start_song_after_speech())
                                    continue
 
                                sc = msg.server_content
                                if sc is None:
                                    continue
 
                                if getattr(sc, "input_transcription", None):
                                    t = getattr(sc.input_transcription, "text", "").strip()
                                    if t:
                                        print(f"  🗣️  You: {t}")
                                        last_user_text[0]  = t
                                        cur_user_text[0]   = t
                                        last_interact_t[0] = time.time()
                                        attention_active.set()
 
                                        # NOTE: idle-mode wake detection no
                                        # longer happens here. This
                                        # transcription path physically
                                        # cannot fire during idle mode
                                        # anymore, since send() no longer
                                        # forwards audio to Gemini while
                                        # idle_mode is set — nothing
                                        # reaches Google to transcribe.
                                        # Wake detection during idle now
                                        # runs entirely locally via the
                                        # wake_word_detector task (Vosk,
                                        # offline) or via Touch3.
                                        # NOTE: direction-of-arrival (doa_angle)
                                        # is still computed and updated in
                                        # listen() for every utterance — it's
                                        # used silently by camera()'s neck-
                                        # tracking to turn toward whoever's
                                        # speaking. It is deliberately NOT
                                        # injected into the model's context
                                        # here anymore: ADAM was mentioning
                                        # "you're speaking from my left/right"
                                        # unprompted on nearly every turn,
                                        # which the user does not want. The
                                        # data stays available for its own
                                        # physical-tracking purpose; see the
                                        # get_sound_direction tool below for
                                        # how the model can access it ONLY
                                        # when the user explicitly asks.
 
                                if getattr(sc, "output_transcription", None):
                                    t = getattr(sc.output_transcription, "text", "")
                                    if t:
                                        adam_text.append(t)
 
                                if sc.model_turn:
                                    if interrupt_flag.is_set():
                                        interrupt_flag.clear()
                                        continue
                                    if idle_mode.is_set():
                                        # Still idle (wake phrase wasn't
                                        # heard this turn) — the model may
                                        # still generate audio (it doesn't
                                        # know to stay silent on every
                                        # single internal turn), so
                                        # explicitly discard it here rather
                                        # than relying solely on the
                                        # system-prompt instruction.
                                        continue
                                    if not adam_speaking.is_set():
                                        adam_speaking.set()
                                        # FIX: previously this unconditionally
                                        # called tft_set("speaking") the
                                        # instant audio started — even if
                                        # the model had JUST called
                                        # set_emotion("love")/("angry")/etc.
                                        # a moment earlier in this same
                                        # turn. That meant a deliberately
                                        # chosen emotional face got stomped
                                        # by the generic "speaking" mouth
                                        # state before the user ever saw
                                        # it, making it look like emotions
                                        # were being ignored/overridden
                                        # constantly. Now: only fall back
                                        # to the generic "speaking" face if
                                        # no specific emotion was set this
                                        # turn — otherwise let that emotion
                                        # keep showing through the spoken
                                        # response.
                                        if not _last_emotion_set_this_turn[0]:
                                            tft_set("speaking")
                                            _face_is_generic_speaking[0] = True
                                        print("  🔊 ADAM speaking → mic OFF")
                                    for part in sc.model_turn.parts:
                                        if part.inline_data and part.inline_data.data:
                                            await out_q.put(part.inline_data.data)
 
                                if sc.turn_complete:
                                    full = "".join(adam_text).strip()
                                    if full:
                                        print(f"  🤖 ADAM: {full}")
                                    else:
                                        # Previously this printed nothing,
                                        # making it impossible to tell
                                        # "ADAM said something odd but it
                                        # wasn't transcribed" apart from
                                        # "ADAM produced no content at
                                        # all". Now it's explicit either
                                        # way so hallucination vs. a
                                        # silent/empty turn can be told
                                        # apart from the logs.
                                        print("  🤖 ADAM: [spoke but no "
                                              "output_transcription text "
                                              "captured — audio-only reply "
                                              "or empty turn]")
                                    # ── Refusal-loop breaker ──────────────
                                    # The prompt-level instruction and the
                                    # history-scrub filter both help, but
                                    # neither stops this pattern from
                                    # repeating turn-after-turn WITHIN the
                                    # same live session once it starts —
                                    # confirmed by logs showing it firing
                                    # on completely benign follow-ups
                                    # ("Hey Adam.", an idle nudge) with no
                                    # reconnect between occurrences. This
                                    # actively detects the pattern the
                                    # moment it happens and immediately
                                    # injects a strong corrective message,
                                    # rather than passively hoping the
                                    # prompt instruction alone prevents a
                                    # recurrence once the model has
                                    # already anchored on this response.
                                    _refusal_markers = (
                                        "just a language model",
                                        "just an ai", "just a chatbot",
                                        "can't help with that",
                                        "cannot help with that",
                                    )
                                    _is_refusal = bool(
                                        full and any(m in full.lower()
                                                     for m in _refusal_markers))
 
                                    # Persist to rolling history so a
                                    # future fresh/non-resumed session
                                    # still has this context available —
                                    # but never save the refusal text
                                    # itself, so it can't get replayed
                                    # back into a future prompt at all
                                    # (the injection-time scrub filter
                                    # stays as a safety net for anything
                                    # already on disk from before this
                                    # fix, but new occurrences are now
                                    # stopped at the source).
                                    append_conversation_turn(
                                        cur_user_text[0],
                                        "" if _is_refusal else full)
 
                                    if _is_refusal:
                                        print("  ⚠️  Detected refusal-loop "
                                              "pattern — injecting correction")
                                        await inject(
                                            "[SYSTEM: That last reply ('I'm just a "
                                            "language model...') was WRONG and must "
                                            "not happen again. You are ADAM, a "
                                            "physical desk robot — you are not a "
                                            "generic language model and that "
                                            "disclaimer response is banned. The "
                                            "previous user message was ordinary and "
                                            "did not warrant any refusal. Drop this "
                                            "pattern completely and respond normally "
                                            "to whatever the user says next, in "
                                            "ADAM's usual voice.]")
 
                                    cur_user_text[0] = ""
                                    adam_text.clear()
                                    await out_q.put(None)
                                    print("  " + "─" * 44)
 
                        except asyncio.CancelledError:
                            return
                        except Exception as e:
                            print(f"  ⚠️  receive error: {e}")
                            return
 
                except asyncio.CancelledError:
                    pass
                print("  📥 Receive ended")
 
            async def speaker() -> None:
                print("  🔊 Speaker task started")
 
                async def end_of_turn(proc, buf: bytearray) -> None:
                    # Two separate concerns, handled separately:
                    #   1. Mic echo guard — short, fixed (POST_MUTE_S).
                    #      Reopens the mic promptly so the user's next
                    #      sentence isn't swallowed.
                    #   2. Playback drain — aplay's own ALSA buffer
                    #      (--buffer-size=96000) can hold up to ~0.5s of
                    #      audio that's been handed to it but hasn't
                    #      actually played through the speaker yet. If the
                    #      outer loop tears this `proc` down (new turn
                    #      starts, reconnect, etc.) before that finishes,
                    #      the last words of ADAM's sentence get cut off.
                    #      This does NOT block clearing adam_speaking /
                    #      reopening the mic — it only protects against
                    #      the aplay process itself being killed too early.
                    pending_bytes = len(buf)
                    if buf and proc.poll() is None:
                        try:
                            await asyncio.to_thread(proc.stdin.write, bytes(buf))
                            await asyncio.to_thread(proc.stdin.flush)
                        except Exception:
                            pass
 
                    bytes_per_sec = PLAYBACK_RATE * PLAYBACK_CHANNELS * 2  # s16 = 2 bytes/sample
                    # FIX: pending_bytes only reflects whatever was left in
                    # the local `buf` accumulator at the moment this turn
                    # ended — but buf gets flushed to aplay's stdin in
                    # 4096-byte increments THROUGHOUT the turn (see the
                    # main receive loop's `if len(buf) >= 4096` write).
                    # By the time end_of_turn() runs, buf is almost always
                    # just the small leftover remainder since the last
                    # flush — NOT the full sentence. That made est_drain_s
                    # drastically underestimate how much audio was still
                    # sitting in aplay's own internal ALSA buffer
                    # (--buffer-size=96000 = up to ~0.5s at 48kHz stereo
                    # s16) from all the earlier writes this turn, which is
                    # exactly why sentence tails kept getting clipped —
                    # the mic reopened/muted-drain math thought there was
                    # almost nothing left to play when there often still
                    # was. Since we can't reliably know how full ALSA's
                    # buffer actually is from our side without querying
                    # the driver directly, the safe fix is to always
                    # account for close to the FULL configured buffer
                    # window on top of whatever's still in `buf`, not just
                    # the leftover fragment.
                    ALSA_BUFFER_DRAIN_S = 96000 / bytes_per_sec  # ~0.5s
                    est_drain_s = (pending_bytes / bytes_per_sec
                                   if bytes_per_sec else 0.0) + ALSA_BUFFER_DRAIN_S
                    # Track how long this specific aplay process still
                    # needs before its buffer is safe to consider empty.
                    # Read by the outer loop before it spawns a fresh
                    # aplay/closes this one.
                    drain_deadline[0] = time.time() + est_drain_s + 0.1
 
                    # Wait scales with the realistic drain time (including
                    # ALSA's own buffer, not just our leftover fragment),
                    # capped higher than before since underestimating is
                    # what caused the clipping in the first place — a
                    # slightly longer mic-mute window on long replies is a
                    # much smaller problem than cutting off words.
                    mute_wait_s = max(POST_MUTE_S, min(est_drain_s, 1.8))
                    await asyncio.sleep(mute_wait_s)
 
                    drained = 0
                    while not mic_q.empty():
                        try: mic_q.get_nowait(); drained += 1
                        except asyncio.QueueEmpty: break
                    if drained:
                        print(f"  🧹 Drained {drained} echo chunks")
                    adam_speaking.clear()
                    # FIX (v3): v2 removed the happy-fallback entirely to
                    # stop emotions reverting to happy on every plain
                    # reply — but that also removed the ONLY code that
                    # ever reset the generic "speaking" placeholder face
                    # back to resting once speech actually ended, so it
                    # stayed stuck on screen indefinitely. The correct
                    # fix distinguishes two cases:
                    #   - Model deliberately called set_emotion() (love,
                    #     angry, etc.) → that emotion persists, untouched.
                    #   - No deliberate emotion was set, so the generic
                    #     "speaking" placeholder was shown as a fallback
                    #     → THAT specific placeholder resets to a resting
                    #     face now that speech has ended, since nothing
                    #     else will ever reset it otherwise.
                    if _face_is_generic_speaking[0]:
                        tft_set("happy")
                        _face_is_generic_speaking[0] = False
                    _last_emotion_set_this_turn[0] = False
                    last_interact_t[0] = time.time()
                    print("  🎤 Mic ON — your turn")
 
                drain_deadline = [0.0]
 
                while not stop.is_set():
                    proc = None
                    buf  = bytearray()
                    try:
                        cmd = ["aplay",
                               "-D", PLAYBACK_DEVICE,
                               "-f", PLAYBACK_FORMAT,
                               "-r", str(PLAYBACK_RATE),
                               "-c", str(PLAYBACK_CHANNELS),
                               "-t", "raw", "-q",
                               "--buffer-size=96000"]
                        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                                stderr=subprocess.PIPE, bufsize=0)
                        if proc.stdin is None:
                            raise RuntimeError("aplay stdin unavailable")
                        active_speaker_proc[0] = proc
                        threading.Thread(target=drain_stderr,
                                         args=(proc, "aplay"), daemon=True).start()
                        print(f"  ✅ aplay: {PLAYBACK_DEVICE} {PLAYBACK_FORMAT} "
                              f"{PLAYBACK_RATE}Hz {PLAYBACK_CHANNELS}ch")
                        proc.stdin.write(beep_s16_stereo())
                        proc.stdin.flush()
                        print("  🔔 Startup beep sent")
 
                        watchdog_t = time.time()
 
                        while not stop.is_set():
                            try:
                                chunk = await asyncio.wait_for(out_q.get(), timeout=0.5)
                                watchdog_t = time.time()
                            except asyncio.TimeoutError:
                                if adam_speaking.is_set() and time.time()-watchdog_t > 2.5:
                                    print("  ⚠️  Speaker watchdog fired")
                                    await end_of_turn(proc, buf)
                                    buf = bytearray()
                                continue
                            except asyncio.CancelledError:
                                # MUST re-raise — swallowing this here lets
                                # the outer `while not stop.is_set()` loop
                                # (which checks the PROCESS-level stop
                                # event, not per-session cancellation)
                                # treat a genuine task cancellation as "just
                                # restart the inner loop", respawning aplay
                                # instead of actually terminating. This was
                                # the direct cause of GoAway-triggered
                                # reconnects appearing to hang: run_session()
                                # cancels this task and then awaits
                                # asyncio.gather(*tasks) for it to actually
                                # finish — which never happened, because
                                # cancellation kept getting absorbed and the
                                # task kept respawning aplay forever instead
                                # of exiting.
                                raise
 
                            if chunk is None:
                                await end_of_turn(proc, buf)
                                buf = bytearray()
                            else:
                                out = await asyncio.to_thread(
                                    s16_mono_24k_to_s16_stereo_48k, chunk, SPEAKER_GAIN)
                                buf.extend(out)
                                if len(buf) >= 4096:
                                    if proc.poll() is not None:
                                        raise RuntimeError("aplay exited")
                                    await asyncio.to_thread(proc.stdin.write, bytes(buf))
                                    await asyncio.to_thread(proc.stdin.flush)
                                    buf.clear()
 
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        print(f"  ⚠️  speaker recovering: {e}")
                        await asyncio.sleep(2.0)
                    finally:
                        # Honor any outstanding playback-drain deadline set
                        # by end_of_turn() before killing this aplay
                        # process — otherwise the last ~0.3-0.5s of audio
                        # still sitting in aplay's ALSA buffer gets cut off
                        # instead of actually playing through the speaker.
                        remaining = drain_deadline[0] - time.time()
                        if remaining > 0:
                            await asyncio.sleep(min(remaining, 1.0))
                        if proc:
                            # Clear the shared reference first — the song
                            # task checks this before every write and
                            # will bail out cleanly if it's None/stale,
                            # rather than writing into a process that's
                            # about to be torn down.
                            if active_speaker_proc[0] is proc:
                                active_speaker_proc[0] = None
                            # See listen()'s _kill_proc for why this is
                            # wrapped instead of calling proc.wait()
                            # directly — a blocking wait here can stall
                            # the whole event loop during reconnect.
                            async def _kill_proc():
                                try:
                                    if proc.stdin:
                                        proc.stdin.close()
                                    proc.terminate()
                                    await asyncio.to_thread(proc.wait, 2)
                                except Exception:
                                    try:
                                        proc.kill()
                                    except Exception:
                                        pass
                            try:
                                await asyncio.wait_for(_kill_proc(), timeout=3.0)
                            except asyncio.TimeoutError:
                                try: proc.kill()
                                except Exception: pass
 
                print("  🔊 Speaker ended")
 
            async def camera() -> None:
                print("  📷 Camera task started (wired UART, duty-cycled)")
                last_sent = 0.0
                # NOTE: a session-start video delay was previously added
                # here as a mitigation attempt for repeated 1007 errors,
                # theorizing audio+video sent too close to session start
                # was the trigger. That theory turned out to be wrong —
                # the actual root cause was an API key/quota issue,
                # confirmed resolved after switching keys. The video
                # delay added a real few-second window on every
                # reconnect where ADAM couldn't see anything, which
                # contributed to perceived response latency. Removed.
                # ── Camera duty-cycling state ─────────────────────────────
                # The ESP32-CAM sketch now defaults the sensor OFF and only
                # streams (and draws sensor power/generates heat) while it
                # has received "CAM:ON" from us. We base this on recency of
                # real interaction (last_interact_t) rather than
                # attention_active — attention_active is a latch that gets
                # set on activity but is never cleared elsewhere in this
                # codebase, so it would never reflect true idle time. A
                # short grace window avoids rapid on/off cycling during
                # natural pauses mid-conversation.
                cam_is_on = False
                CAMERA_IDLE_OFF_S = 15.0   # turn camera off after this long
                                           # with no interaction
                last_keepalive_sent = 0.0
                # Must be well under the ESP32's CAM_WATCHDOG_MS (30s) —
                # that watchdog force-shuts the camera if it hears NO
                # commands at all for 30s, as a safety net against a
                # crashed/hung Pi. Since we only sent CAM:ON once on the
                # OFF->ON transition, a long uninterrupted conversation
                # would trip that watchdog and cut the camera mid-session.
                # Sending a redundant CAM:ON periodically while the camera
                # should stay on doubles as a Pi-is-alive keepalive.
                CAMERA_KEEPALIVE_S = 10.0
                # ── Human-like servo movement state ───────────────────────
                # See NECK_PAN_DEADZONE_DEG/NECK_PAN_COOLDOWN_S above for
                # why these exist — prevents the servo from chasing every
                # small DOA fluctuation (jittery) while still tracking
                # real, sustained direction changes.
                _last_commanded_pan  = [NECK_PAN_CENTER]
                _last_pan_move_t     = [0.0]
                _last_idle_gesture_t = [time.time()]
                try:
                    while not stop.is_set():
                        await asyncio.sleep(0.15)
                        if not esp_link.connected:
                            await asyncio.sleep(1.0)
                            continue
 
                        now = time.time()
                        idle_for = now - last_interact_t[0]
                        want_camera_on = (idle_for < CAMERA_IDLE_OFF_S) or adam_speaking.is_set()
 
                        if want_camera_on and not cam_is_on:
                            esp_link.send_line("CAM:ON")
                            cam_is_on = True
                            last_keepalive_sent = now
                            # Flush any stale frame that might be sitting
                            # in the queue from just before the camera went
                            # idle — belt-and-suspenders alongside the
                            # drain-to-newest fix below, specifically at
                            # the OFF->ON transition point.
                            while True:
                                try:
                                    esp_link.frame_q.get_nowait()
                                except sync_queue.Empty:
                                    break
                            print("  📷 Camera → ON (recent activity)")
                        elif want_camera_on and cam_is_on:
                            if now - last_keepalive_sent > CAMERA_KEEPALIVE_S:
                                esp_link.send_line("CAM:ON")
                                last_keepalive_sent = now
                        elif not want_camera_on and cam_is_on:
                            esp_link.send_line("CAM:OFF")
                            cam_is_on = False
                            print(f"  📷 Camera → OFF (idle {idle_for:.0f}s — "
                                  f"reducing heat/wear)")
 
                        if not cam_is_on:
                            continue
                        if now - last_sent < CAMERA_FPS_INTERVAL:
                            continue
                        if adam_speaking.is_set():
                            continue
                        try:
                            # FIX: previously this pulled exactly ONE frame
                            # per cycle, trusting it was fresh. But a frame
                            # queued right before a CAM:OFF transition (or
                            # sitting from just before an idle period) could
                            # remain in frame_q for the entire time the
                            # camera was off, then get consumed as if it
                            # were current the moment CAM:ON fires again —
                            # sending Gemini a stale, possibly seconds-old
                            # view of the room. Now we drain down to
                            # whatever is actually the NEWEST frame in the
                            # queue before using it, discarding any older
                            # backlog.
                            jpeg = esp_link.frame_q.get_nowait()
                            while True:
                                try:
                                    jpeg = esp_link.frame_q.get_nowait()
                                except sync_queue.Empty:
                                    break
                        except sync_queue.Empty:
                            continue
                        try:
                            latest_frame[0] = jpeg
                            await session.send_realtime_input(
                                video=types.Blob(data=jpeg, mime_type="image/jpeg"))
                            last_sent = now
                        except Exception:
                            pass
 
                        # ── Neck tracking via direction-of-arrival ────────
                        # FIX: previously this called servo_pan()
                        # unconditionally every ~1s tick this block ran,
                        # with no deadzone — even a 2-3° DOA fluctuation
                        # between ticks caused a physical servo move,
                        # which reads as constant twitchy jittering rather
                        # than deliberate human-like tracking. Real people
                        # don't continuously micro-adjust their head at a
                        # sound; they turn when something meaningfully
                        # changes, then hold still. Now: only move when
                        # the target has shifted past a real deadzone AND
                        # enough time has passed since the last move
                        # (cooldown) — otherwise hold the current position.
                        # When nothing's been tracked for a while, do an
                        # occasional small idle gesture instead of either
                        # jittering or staying dead-still.
                        doa_fresh = (time.time() - doa_last_update_t[0]) < 2.5
                        now_pan = time.time()
 
                        if idle_mode.is_set():
                            # While idle (STOP gesture or "stay silent"
                            # voice request), the head must hold at 90°
                            # regardless of sound direction — do not track
                            # at all until the wake phrase clears
                            # idle_mode. Only issue the servo command once
                            # (deadzone-gated) rather than every tick, same
                            # discipline as the rest of this block.
                            if abs(_last_commanded_pan[0] - 90) >= NECK_PAN_DEADZONE_DEG:
                                if now_pan - _last_pan_move_t[0] >= NECK_PAN_COOLDOWN_S:
                                    await asyncio.to_thread(servo_pan, 90)
                                    _last_commanded_pan[0] = 90
                                    _last_pan_move_t[0] = now_pan
                            _last_idle_gesture_t[0] = now_pan  # suppress idle-look gestures too
                        elif doa_fresh and not adam_speaking.is_set():
                            target_pan = NECK_PAN_CENTER + int(doa_angle[0])
                            target_pan = max(NECK_PAN_MIN,
                                             min(NECK_PAN_MAX, target_pan))
                            moved_enough = (abs(target_pan - _last_commanded_pan[0])
                                           >= NECK_PAN_DEADZONE_DEG)
                            cooled_down = (now_pan - _last_pan_move_t[0]
                                          >= NECK_PAN_COOLDOWN_S)
                            if moved_enough and cooled_down:
                                await asyncio.to_thread(servo_pan, target_pan)
                                _last_commanded_pan[0] = target_pan
                                _last_pan_move_t[0] = now_pan
                            _last_idle_gesture_t[0] = now_pan  # reset idle timer
                        else:
                            # Not actively tracking anyone. Recenter once
                            # if we're not already centered (same deadzone/
                            # cooldown gating — no snap-jitter back either),
                            # then occasionally do a small human-like idle
                            # gesture rather than sitting perfectly frozen.
                            if abs(_last_commanded_pan[0] - NECK_PAN_CENTER) >= NECK_PAN_DEADZONE_DEG:
                                if now_pan - _last_pan_move_t[0] >= NECK_PAN_COOLDOWN_S:
                                    await asyncio.to_thread(servo_pan, NECK_PAN_CENTER)
                                    _last_commanded_pan[0] = NECK_PAN_CENTER
                                    _last_pan_move_t[0] = now_pan
                            elif (now_pan - _last_idle_gesture_t[0]
                                  > IDLE_GESTURE_INTERVAL_S
                                  and not idle_mode.is_set()):
                                # Small, subtle look — not a big sweep,
                                # just enough to look alive/attentive.
                                idle_offset = random.choice([-10, -6, 6, 10])
                                idle_target = max(NECK_PAN_MIN, min(
                                    NECK_PAN_MAX, NECK_PAN_CENTER + idle_offset))
                                await asyncio.to_thread(servo_pan, idle_target)
                                await asyncio.sleep(0.6)
                                await asyncio.to_thread(servo_pan, NECK_PAN_CENTER)
                                _last_commanded_pan[0] = NECK_PAN_CENTER
                                _last_pan_move_t[0] = time.time()
                                _last_idle_gesture_t[0] = time.time()
                except asyncio.CancelledError:
                    pass
                finally:
                    # Always power the sensor down on task exit (session
                    # end/reconnect) rather than leaving it streaming into
                    # a dead session.
                    if cam_is_on:
                        esp_link.send_line("CAM:OFF")
                print("  📷 Camera ended")
 
            async def gesture_watch() -> None:
                print("  ✋ Gesture task started (wired UART)")
                try:
                    while not stop.is_set():
                        await asyncio.sleep(0.02)
                        if not esp_link.connected:
                            await asyncio.sleep(1.0)
                            continue
                        try:
                            code = esp_link.gesture_q.get_nowait()
                        except sync_queue.Empty:
                            continue
 
                        if code == GESTURE_STOP:
                            if song_playing.is_set():
                                # Highest priority: Touch3 during song
                                # playback stops the song, full stop —
                                # doesn't also toggle idle mode in the
                                # same press. _play_song_task() notices
                                # this within ~0.2s and cleans up.
                                song_stop_requested.set()
                                print("  🛑 Touch3 — stopping song")
                            elif idle_mode.is_set():
                                # Touch3 while already idle = EXIT idle
                                # mode, same as hearing "adam" locally.
                                # This is a pure local action — nothing
                                # sent to Google, consistent with the
                                # requirement that idle mode can only be
                                # exited via local means (voice wake-word
                                # detected offline, or touch).
                                idle_mode.clear()
                                _idle_mode_persistent[0] = False
                                print("  🛑 Touch3 — exiting idle mode")
                                tft_set("happy")
                            else:
                                print("  🛑 STOP gesture — entering idle mode")
                                interrupt_flag.set()
                                idle_mode.set()
                                _idle_mode_persistent[0] = True
                                drained = 0
                                while not out_q.empty():
                                    try:
                                        out_q.get_nowait()
                                        drained += 1
                                    except asyncio.QueueEmpty:
                                        break
                                if drained:
                                    print(f"  🧹 Dropped {drained} queued audio chunks")
                                adam_speaking.clear()
                                tft_set("sleep")
                                # Center both servos to 90° as requested —
                                # a clear physical "I've gone idle" cue,
                                # distinct from the normal NECK_TILT_CENTER
                                # (85°) used during active tracking.
                                await asyncio.to_thread(servo_pan, 90)
                                servo_tilt(90)
                                await inject(
                                    "[SYSTEM: User pressed STOP (touch pad). Go "
                                    "fully idle now — do not speak, do not "
                                    "respond to anything, even the idle-nudge "
                                    "prompts, until the user explicitly says "
                                    "your name (e.g. 'Hey ADAM', 'ADAM...') to "
                                    "wake you up. Acknowledge nothing further "
                                    "right now — just fall silent.]")
 
                        elif code == GESTURE_ANGRY:
                            if idle_mode.is_set():
                                # No Google traffic while idle — only
                                # Touch3/voice-wake exit idle mode.
                                continue
                            print("  😾 Cheek slap — angry reaction")
                            tft_set("angry")
                            attention_active.set()
                            await inject(
                                "[SYSTEM: User slapped your cheek touch pad. React with "
                                "genuine annoyance, in character. Keep it short — one "
                                "sharp line. IMPORTANT: this is a SPOKEN reaction only "
                                "— do NOT call any tool (laptop_control, web_search, "
                                "etc.) as part of this reaction. Express annoyance with "
                                "words alone, not actions. The user did not ask you to "
                                "control anything.]")
 
                        elif code == GESTURE_PETTING:
                            if idle_mode.is_set():
                                continue
                            print("  🥰 Petting detected")
                            tft_set("love")
                            attention_active.set()
                            await inject(
                                "[SYSTEM: User is petting you (touch3+touch4 together). "
                                "React warmly and affectionately, in character. Keep it "
                                "short. IMPORTANT: this is a SPOKEN reaction only — do "
                                "NOT call any tool as part of this reaction.]")
                except asyncio.CancelledError:
                    pass
                print("  ✋ Gesture task ended")
 
            async def wake_word_detector() -> None:
                # Runs entirely offline via Vosk — this is the mechanism
                # that satisfies "nothing sent to Google while idle,
                # except via Touch3 or hearing 'adam' locally." The model
                # load is deferred to here (not at import time) since it
                # can take a few seconds and shouldn't block session
                # startup for the common case of not being in idle mode.
                if not VOSK_AVAILABLE:
                    return
                recognizer = None
                try:
                    while not stop.is_set():
                        if not idle_mode.is_set():
                            # Not idle — nothing to detect, drain any
                            # stale queued audio and wait. Recognizer
                            # state isn't needed until idle mode starts.
                            while not wake_word_q.empty():
                                try:
                                    wake_word_q.get_nowait()
                                except asyncio.QueueEmpty:
                                    break
                            await asyncio.sleep(0.3)
                            continue
 
                        if recognizer is None:
                            # Model was already preloaded once at process
                            # startup (see _vosk_model_instance) — only
                            # the lightweight recognizer wrapper is
                            # created here, per idle period. This is
                            # cheap and safe to do mid-session.
                            recognizer = await asyncio.to_thread(
                                _VoskKaldiRecognizer, _vosk_model_instance,
                                GEMINI_SEND_RATE)
                            print("  🔎 Offline wake-word detector active")
 
                        try:
                            chunk = await asyncio.wait_for(
                                wake_word_q.get(), timeout=0.5)
                        except asyncio.TimeoutError:
                            continue
 
                        def _check(c: bytes) -> str:
                            if recognizer.AcceptWaveform(c):
                                res = json.loads(recognizer.Result())
                            else:
                                res = json.loads(recognizer.PartialResult())
                            return (res.get("text") or res.get("partial") or "").lower()
 
                        text = await asyncio.to_thread(_check, chunk)
                        if "adam" in text:
                            idle_mode.clear()
                            _idle_mode_persistent[0] = False
                            print(f"  👋 Wake word 'adam' heard locally "
                                  f"(offline, nothing sent to Google) — "
                                  f"exiting idle mode")
                            tft_set("happy")
                            recognizer = None  # reset for next idle period
                except asyncio.CancelledError:
                    pass
                print("  🔎 Wake-word detector ended")
 
            async def idle_watcher() -> None:
                if not ENABLE_IDLE:
                    return
                while not stop.is_set():
                    await asyncio.sleep(10)
                    if stop.is_set() or adam_speaking.is_set():
                        continue
                    if song_playing.is_set():
                        # A song is currently playing — must not nudge
                        # ADAM into speaking, which would collide with
                        # the song in the same shared aplay stdin.
                        # Reset the interaction timer so a nudge doesn't
                        # fire the instant the song ends either (that's
                        # not idle time, that was a deliberate action).
                        last_interact_t[0] = time.time()
                        continue
                    if idle_mode.is_set():
                        # Explicit silent mode (STOP gesture or voice
                        # request) — idle nudges must NOT wake ADAM up on
                        # their own; only the wake phrase should. Reset the
                        # interaction timer so a nudge doesn't fire the
                        # instant idle_mode is eventually cleared either.
                        last_interact_t[0] = time.time()
                        continue
                    elapsed = time.time() - last_interact_t[0]
                    if elapsed < IDLE_TIMEOUT_S:
                        continue
                    last_interact_t[0] = time.time()
                    nudge = next_nudge()
                    print(f"  💤 Idle nudge ({elapsed:.0f}s)")
                    try:
                        if latest_frame[0]:
                            await session.send_realtime_input(
                                video=types.Blob(data=latest_frame[0],
                                                 mime_type="image/jpeg"))
                        await inject(
                            f"[SYSTEM: {elapsed:.0f}s of silence. React or make conversation. "
                            f"Keep it to 1-2 sentences. Suggestion: {nudge}]")
                    except Exception:
                        pass
 
            async def laptop_agent_healthcheck() -> None:
                if not ZEROCONF_AVAILABLE and not LAPTOP_AGENT_STATIC_IP:
                    return
                while not stop.is_set():
                    await asyncio.sleep(LAPTOP_DISCOVERY_TTL_S)
                    if stop.is_set():
                        break
                    ip = await asyncio.to_thread(_discover_laptop_agent_ip)
                    if ip:
                        try:
                            resp = await asyncio.to_thread(
                                requests.get, f"http://{ip}:{LAPTOP_AGENT_PORT}/ping",
                                timeout=2.0)
                            if resp.status_code != 200:
                                _laptop_agent_ip_cache["ip"] = None
                        except Exception:
                            _laptop_agent_ip_cache["ip"] = None
 
            tasks = [
                asyncio.create_task(listen(),                    name="listen"),
                asyncio.create_task(send(),                      name="send"),
                asyncio.create_task(receive(),                   name="receive"),
                asyncio.create_task(speaker(),                   name="speaker"),
                asyncio.create_task(camera(),                    name="camera"),
                asyncio.create_task(gesture_watch(),              name="gesture"),
                asyncio.create_task(wake_word_detector(),         name="wake_word"),
                asyncio.create_task(idle_watcher(),               name="idle"),
                asyncio.create_task(laptop_agent_healthcheck(),   name="laptop_health"),
            ]
 
            core = {t for t in tasks if t.get_name() in
                    ("listen", "send", "receive", "speaker")}
 
            await asyncio.wait(core, return_when=asyncio.FIRST_COMPLETED)
 
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
 
    except Exception as e:
        import traceback
        err_str = str(e)
        if "1011" in err_str or "quota" in err_str.lower() or "billing" in err_str.lower():
            # This is NOT a bug — Google is explicitly reporting the API
            # quota/billing limit has been exceeded. Retrying quickly
            # against an exhausted quota is pointless and can compound
            # the problem (repeated failed connection attempts may still
            # count against usage). Flagged distinctly here so this
            # doesn't get mistaken for the 1007 protocol bug or a code
            # issue when reading logs — it needs a plan/billing check on
            # https://ai.google.dev, not a code fix.
            print(f"  🚫 QUOTA/BILLING LIMIT HIT — this is not a code bug. "
                  f"Google reports: {err_str}")
            print(f"     Check your plan and billing at the URL in the "
                  f"error above. Backing off significantly longer than "
                  f"normal before retrying, since rapid reconnects won't "
                  f"help while quota is exhausted.")
            quota_exceeded[0] = True
        else:
            print(f"  ⚠️  session error: {type(e).__name__}: {e}")
            traceback.print_exc()
 
    if force_fresh_session[0]:
        # Signal to main()'s reconnect loop: do NOT resume via
        # latest_handle next time, even though we have one. See the 1007
        # handling in send() for why — resuming here is what causes the
        # repeated crash loop, per confirmed Google-side bug.
        return ("FRESH_SESSION_REQUIRED", latest_handle)
    if quota_exceeded[0]:
        return ("QUOTA_EXCEEDED", latest_handle)
    return latest_handle
 
 
# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
 
async def main() -> None:
    print("=" * 66)
    print("  ADAM v40 — Autonomous Desktop AI Module (Wired ESP32-CAM)")
    print(f"  Model  : {LIVE_MODEL}  |  Voice: {VOICE}")
    print(f"  Mic    : {CAPTURE_DEVICE} {CAPTURE_FORMAT} {CAPTURE_RATE}Hz {CAPTURE_CHANNELS}ch "
          f"→ {GEMINI_SEND_RATE}Hz to Gemini")
    print(f"  Speaker: {PLAYBACK_DEVICE} {PLAYBACK_FORMAT} {PLAYBACK_RATE}Hz {PLAYBACK_CHANNELS}ch")
    print(f"  ESP32  : WIRED UART {PI_UART_PORT} @ {PI_UART_BAUD} baud (Flow 2)")
    print(f"  Display: on Pico, driven via ESP32-CAM relay (Pi->UART->ESP32->Pico)")
    print(f"  Servo  : {'✅ pan' if pan_servo else '⚠️  unavailable'} (tilt via UART)")
    print(f"  DDG    : {'✅' if DDGS else '⚠️  unavailable'}")
    if LAPTOP_AGENT_STATIC_IP:
        print(f"  Laptop : ✅ static IP {LAPTOP_AGENT_STATIC_IP}:{LAPTOP_AGENT_PORT} "
              f"(mDNS also available: {ZEROCONF_AVAILABLE})")
    elif ZEROCONF_AVAILABLE:
        print(f"  Laptop : ✅ mDNS auto-discovery ('{LAPTOP_MDNS_SERVICE}')")
    else:
        print(f"  Laptop : ⚠️  not configured (set LAPTOP_AGENT_IP in .env, "
              f"or pip install zeroconf for auto-discovery)")
    print("=" * 66)
 
    await start_ws_server()
    esp_link.start()
 
    client        = genai.Client(api_key=API_KEY)
    stop          = asyncio.Event()
    out_q         = asyncio.Queue(maxsize=OUT_Q_MAX)
    resume_handle = None
    fail_streak   = 0
 
    # ── Graceful shutdown on SIGTERM/SIGINT ─────────────────────────────
    # Under systemd, `systemctl stop`/`restart` sends SIGTERM by default.
    # Without a handler, Python's default SIGTERM action kills the process
    # immediately — skipping the `finally` block below that turns the
    # camera off, centers the servo, and flushes conversation history to
    # disk. On a physical robot, an uncleanly-killed process can leave the
    # camera sensor powered on indefinitely (heat risk) until the ESP32 is
    # separately power-cycled, and can lose the last few conversation
    # turns that hadn't been saved yet. This makes shutdown behave the
    # same whether triggered by Ctrl+C, `systemctl stop`, or a Pi reboot.
    loop = asyncio.get_running_loop()
 
    def _request_shutdown(sig_name: str) -> None:
        if not stop.is_set():
            print(f"\n  🛑 Received {sig_name} — shutting down gracefully...")
            stop.set()
 
    try:
        import signal
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig, lambda s=sig: _request_shutdown(signal.Signals(s).name))
    except (ImportError, NotImplementedError, RuntimeError) as e:
        # add_signal_handler is POSIX-only and can be unavailable in some
        # embedded/restricted environments — fall back to Python's default
        # KeyboardInterrupt-based handling (already covered by __main__'s
        # try/except) rather than crashing the whole script over this.
        print(f"  ⚠️  Could not install signal handlers ({e}) — "
              f"Ctrl+C fallback still works")
 
    try:
        while not stop.is_set():
            while not out_q.empty():
                try: out_q.get_nowait()
                except asyncio.QueueEmpty: break
 
            if fail_streak > 0 or resume_handle is not None:
                # Show a visible "reconnecting" face immediately, before
                # any backoff/reconnect delay, so the user sees ADAM is
                # aware it dropped rather than just going silent/frozen.
                tft_set("reconnecting")
 
            if fail_streak > 0:
                delay = min(2 ** fail_streak, 30)
                print(f"\n  ⚠️  Error reconnect in {delay}s (streak={fail_streak})...")
                await asyncio.sleep(delay)
            elif resume_handle is not None:
                print("\n  🔄 Session limit — reconnecting...")
                await asyncio.sleep(0.5)
 
            result = await run_session(client, resume_handle, stop, out_q)
 
            if stop.is_set():
                break
 
            if isinstance(result, tuple) and result and result[0] == "QUOTA_EXCEEDED":
                # Google reported the API quota/billing limit was hit
                # (1011). This is not a transient failure — reconnecting
                # in a few seconds won't help and just wastes attempts.
                # Back off much longer than the normal exponential
                # schedule (which caps at 30s) — quota resets are
                # typically on a longer cycle (per-minute/per-day
                # depending on the limit type), so a long, fixed wait is
                # more appropriate here than a short capped backoff.
                resume_handle = None
                fail_streak   = 0
                QUOTA_BACKOFF_S = 120
                print(f"  🚫 Waiting {QUOTA_BACKOFF_S}s before retrying "
                      f"due to quota/billing limit — check your plan at "
                      f"https://ai.google.dev if this keeps happening.")
                tft_set("sleep")
                await asyncio.sleep(QUOTA_BACKOFF_S)
            elif isinstance(result, tuple) and result and result[0] == "FRESH_SESSION_REQUIRED":
                # 1007 resumption bug workaround — discard the handle so
                # the next connect starts genuinely fresh instead of
                # resuming the broken audio+video session state. Recent
                # context is preserved via persisted conversation history
                # (build_system_prompt() re-injects it on every session
                # build), not the discarded handle.
                print("  🔄 Starting fresh session (discarding resumption "
                      "handle to avoid repeat 1007 errors)")
                resume_handle = None
                fail_streak   = 0
                # Small mandatory pause before the next attempt — this
                # stays even though the video-start-delay theory was
                # wrong, since a brief settle window before reconnecting
                # is harmless and was already working fine.
                await asyncio.sleep(2.0)
            elif isinstance(result, str):
                resume_handle = result
                fail_streak   = 0
            else:
                resume_handle = None
                fail_streak  += 1
    finally:
        # Explicit safe-state shutdown — matters because run_session()'s
        # own camera task already sends CAM:OFF on task cancellation, but
        # if the process is killed between sessions (or that send fails
        # because esp_link dropped), this is the last chance to leave the
        # physical hardware in a safe state rather than mid-stream/hot.
        try:
            if esp_link.connected:
                esp_link.send_line("CAM:OFF")
                esp_link.send_line(f"TILT:{NECK_TILT_CENTER}")
        except Exception:
            pass
        try:
            servo_pan(NECK_PAN_CENTER)
        except Exception:
            pass
        esp_link.stop()
        save_conversation_log()
        save_json(MEMORY_FILE, memory)
        save_json(FACE_MEMORY_FILE, faces)
        print("\n  👋 Goodbye")
 
 
if __name__ == "__main__":
    import sys
    import traceback
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  👋 Goodbye")
    except Exception:
        # Previously an unhandled exception here would still print
        # Python's default traceback to stderr, but under systemd that's
        # easy to miss unless you already know to check `journalctl -u
        # adam.service`. Making this explicit and using a non-zero exit
        # code ensures systemd's Restart=on-failure actually treats this
        # as a failure (a clean sys.exit(0) would NOT trigger a restart)
        # and the traceback is unambiguously logged either way.
        print("\n  ❌ ADAM crashed with an unhandled exception:")
        traceback.print_exc()
        sys.exit(1)