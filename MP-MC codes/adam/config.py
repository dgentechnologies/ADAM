"""
config.py — ADAM v40 configuration
==============================================================================
All tunable constants, environment loading, and static config live here.
Nothing in this file should import from any other ADAM module — it sits at
the bottom of the dependency graph so everything else can import from it
safely without circular imports.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Environment ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not set in .env")

# ═════════════════════════════════════════════════════════════════════════════
# AI MODEL
# ═════════════════════════════════════════════════════════════════════════════

LIVE_MODEL = "gemini-3.1-flash-live-preview"
VOICE      = "Charon"

# ═════════════════════════════════════════════════════════════════════════════
# FILE PATHS
# ═════════════════════════════════════════════════════════════════════════════

MEMORY_FILE        = BASE_DIR / "adam_memory.json"
FACE_MEMORY_FILE   = BASE_DIR / "adam_faces.json"
SYSTEM_PROMPT_FILE = BASE_DIR / "system_prompt.txt"
CONV_MEMORY_FILE   = BASE_DIR / "adam_conversations.json"

# ═════════════════════════════════════════════════════════════════════════════
# AUDIO — CAPTURE / PLAYBACK (proven working — do not modify without testing
# on real hardware)
# ═════════════════════════════════════════════════════════════════════════════

CAPTURE_DEVICE   = "plughw:0,0"
CAPTURE_FORMAT   = "S32_LE"
CAPTURE_RATE     = 48000
CAPTURE_CHANNELS = 2

PLAYBACK_DEVICE   = "plughw:0,0"
PLAYBACK_FORMAT   = "S16_LE"
PLAYBACK_RATE     = 48000
PLAYBACK_CHANNELS = 2

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

# ═════════════════════════════════════════════════════════════════════════════
# SONG / CONCERT PLAYBACK
# ═════════════════════════════════════════════════════════════════════════════

# List of audio files ADAM can play when asked to sing/perform — one is
# picked at random each time. Add/remove/rename paths here freely; must
# be raw PCM WAV files matching PLAYBACK_RATE/PLAYBACK_CHANNELS/16-bit
# (48kHz stereo s16 by default) since playback writes directly into the
# already-open speaker pipe rather than spawning a separate player. Convert
# with:
#   ffmpeg -i input.mp3 -ar 48000 -ac 2 -sample_fmt s16 song1.wav
SONG_FILE_PATHS = [
    str(BASE_DIR / "song1.wav"),
    str(BASE_DIR / "song2.wav"),
    str(BASE_DIR / "song3.wav"),
]

# ═════════════════════════════════════════════════════════════════════════════
# NECK SERVO (pan only; tilt goes over UART to Pico via ESP32-CAM relay)
# ═════════════════════════════════════════════════════════════════════════════

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

# ═════════════════════════════════════════════════════════════════════════════
# DIRECTION-OF-ARRIVAL (DOA) — dual INMP441 mics on v32 BODY board
# ═════════════════════════════════════════════════════════════════════════════

MIC_DISTANCE_M      = 0.065   # 65mm — typical dual-INMP441 spacing
SOUND_SPEED_MPS      = 343.0
DOA_ANGLE_DEADZONE   = 8      # degrees — ignore tiny jitter around center

# ═════════════════════════════════════════════════════════════════════════════
# ESP32-CAM WIRED LINK (Flow 2)
# ═════════════════════════════════════════════════════════════════════════════

PI_UART_PORT = os.getenv("PI_UART_PORT", "/dev/serial0")
PI_UART_BAUD = int(os.getenv("PI_UART_BAUD", "921600"))

# TPM OPTIMIZATION: was 1.0 (1 FPS). Video is the single largest ongoing
# token cost in a Live session — a JPEG frame at VGA resolution can run
# several hundred to 1000+ tokens depending on content, sent continuously
# whenever the camera is on. Confirmed via usage screenshot at 62.31K/65K
# TPM (right at the free-tier ceiling). Halving the send rate to one
# frame every 2s roughly halves video's ongoing token cost with a fairly
# small usability tradeoff.
CAMERA_FPS_INTERVAL = 2.0

# Wire protocol tags — MUST match esp32_cam.ino exactly
TAG_FRAME   = ord('F')
TAG_TOUCH   = ord('T')
TAG_GESTURE = ord('G')

GESTURE_NONE    = 0
GESTURE_ANGRY   = 1   # cheek slap — Touch1 or Touch2
GESTURE_PETTING = 2   # Touch3 + Touch4 together
GESTURE_STOP    = 3   # Touch3 alone — interrupt speech immediately

# ═════════════════════════════════════════════════════════════════════════════
# ATTENTION / IDLE
# ═════════════════════════════════════════════════════════════════════════════

ATTENTION_TIMEOUT_S = 30

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

# ═════════════════════════════════════════════════════════════════════════════
# SEARCH
# ═════════════════════════════════════════════════════════════════════════════

SEARCH_CACHE_TTL = 1800
SEARCH_MIN_GAP_S = 5.0

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
LAPTOP_ACTIONS_TTL_S       = 120.0

# ═════════════════════════════════════════════════════════════════════════════
# VOSK OFFLINE WAKE-WORD
# ═════════════════════════════════════════════════════════════════════════════

VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", str(BASE_DIR / "vosk-model-small-en-us-0.15"))

# ═════════════════════════════════════════════════════════════════════════════
# CONVERSATION HISTORY
# ═════════════════════════════════════════════════════════════════════════════

CONV_MAX_TURNS    = 40   # max turns persisted to disk
# TPM OPTIMIZATION: system_prompt is rebuilt fresh on every single
# reconnect. Re-injecting the full 40-turn history every time was a real,
# avoidable contributor to hitting the 65K TPM free-tier ceiling
# (confirmed via usage screenshot at 62.31K/65K). Full 40-turn history
# stays on disk for continuity across long gaps; only a much shorter
# recent window is actually injected per-session.
CONV_PROMPT_TURNS = 12

# ═════════════════════════════════════════════════════════════════════════════
# WEBSOCKET FACE SERVER
# ═════════════════════════════════════════════════════════════════════════════

WS_HOST = "localhost"
WS_PORT = 8765