"""
ADAM — Autonomous Desktop AI Module (v19.1)
==========================================
NEW: SMART ATTENTION SYSTEM — ADAM knows when you're talking to IT vs others

THE PROBLEM SOLVED:
  You're on a call, talking to friends, or just muttering to yourself.
  Previously ADAM would respond to everything it heard.
  Now ADAM figures out if you're talking to IT before responding.

HOW THE ATTENTION SYSTEM WORKS (3 layers, all working together):

  LAYER 1 — FACE GAZE (primary, camera-based)
    OpenCV detects your face in frame. If your face is visible and roughly
    centred/towards the camera → you're likely talking to ADAM.
    If you turn away, leave frame, or face sideways → ADAM goes passive.

  LAYER 2 — WAKE WORD (override — always works even when passive)
    Say "ADAM" or "Hey ADAM" at any time → instantly activates for one turn,
    regardless of face direction. Uses Vosk offline (no internet, instant).
    Falls back to simple name-detection in transcription if Vosk not installed.

  LAYER 3 — ATTENTION TIMEOUT
    After ADAM responds, it stays "attentive" for 30 seconds.
    Any further speech in that window is treated as directed at ADAM.
    After 30s of silence → goes passive again, requires face or wake word.

STATES:
  PASSIVE  — hears everything, sends NOTHING to Gemini
  ATTENTIVE — sends audio + video to Gemini, Gemini responds normally
  RESPONDING — ADAM is speaking, mic gated

MANUAL OVERRIDES (keyboard, for testing):
  F5  → force ATTENTIVE mode
  F6  → force PASSIVE mode

SETUP:
    pip install --upgrade google-genai pyaudio python-dotenv websockets flask
                           opencv-python Pillow

OPTIONAL (for better wake word, highly recommended):
    pip install vosk
    Download model: https://alphacephei.com/vosk/models
    Place in same folder as: vosk-model-small-en-in-0.4/

RUN:
    python adam_v19.py
"""

import asyncio
import os
import sys
import time
import datetime
import json
import threading
import webbrowser
import struct
import queue
from pathlib import Path

import cv2
import numpy as np
import pyaudio
from dotenv import load_dotenv
from google import genai
from google.genai import types
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
import websockets.server
from flask import Flask, send_from_directory

# ── Optional Vosk offline wake-word ──────────────────────────────────────────
try:
    from vosk import Model as VoskModel, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=".env")
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("❌ API key not found. Set GOOGLE_API_KEY in .env")
print("✅ API Key loaded")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MODEL               = "gemini-3.1-flash-live-preview"
FLASK_PORT          = 5000
WS_HOST             = "localhost"
WS_PORT             = 8765
POST_SPEECH_MUTE_S  = 0.4
VOICE               = "Charon"
CAMERA_INDEX        = 0
FRAME_SIZE          = (768, 768)
CAMERA_FPS_INTERVAL = 1.0           # 1 FPS to Gemini (API limit)

# ── Idle nudge config (edit these freely) ────────────────────────────────────
ENABLE_IDLE      = True    # True = ADAM breaks silence with nudges, False = stays quiet
IDLE_TIMEOUT_S   = 60      # seconds of passiveness before sending an idle nudge

# ── Attention system config ───────────────────────────────────────────────────
ATTENTION_TIMEOUT_S       = 30      # stay attentive this long after last interaction
FACE_CENTRE_TOLERANCE     = 0.45    # face must be within this fraction of frame centre
FACE_MIN_SIZE_FRACTION    = 0.06    # face must be at least this big vs frame (filters far/tiny faces)
WAKE_WORDS                = ["adam", "hey adam", "ok adam", "okay adam", "a dam", "atom"]
VOSK_MODEL_PATH           = "vosk-model-small-en-in-0.4"

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE      = Path(BASE_DIR) / "adam_memory.json"
FACE_MEMORY_FILE = Path(BASE_DIR) / "adam_faces.json"

# ─────────────────────────────────────────────────────────────────────────────
# ATTENTION STATE MACHINE
# ─────────────────────────────────────────────────────────────────────────────

class AttentionState:
    PASSIVE    = "passive"       # not listening — ignoring audio
    ATTENTIVE  = "attentive"     # listening and sending to Gemini
    RESPONDING = "responding"    # ADAM is speaking

class AttentionManager:
    """
    Central gating controller.
    Other tasks call .is_active() before sending audio to Gemini.
    """
    def __init__(self):
        self._state            = AttentionState.PASSIVE
        self._last_active_time = 0.0
        self._lock             = asyncio.Lock()
        self._on_state_change  = None    # async callback

    def set_callback(self, cb):
        self._on_state_change = cb

    @property
    def state(self):
        return self._state

    def is_active(self) -> bool:
        """Should audio be sent to Gemini right now?"""
        if self._state == AttentionState.ATTENTIVE:
            # Check timeout — auto-expire attention
            if time.time() - self._last_active_time > ATTENTION_TIMEOUT_S:
                # Don't await here (called from sync context) — just mark passive
                self._state = AttentionState.PASSIVE
                return False
            return True
        return False

    async def activate(self, reason: str = ""):
        async with self._lock:
            if self._state != AttentionState.RESPONDING:
                old = self._state
                self._state = AttentionState.ATTENTIVE
                self._last_active_time = time.time()
                if old != AttentionState.ATTENTIVE:
                    print(f"  👁️  ATTENTIVE [{reason}]")
                    if self._on_state_change:
                        await self._on_state_change(AttentionState.ATTENTIVE)

    async def deactivate(self, reason: str = ""):
        async with self._lock:
            if self._state == AttentionState.ATTENTIVE:
                self._state = AttentionState.PASSIVE
                print(f"  😶  PASSIVE [{reason}]")
                if self._on_state_change:
                    await self._on_state_change(AttentionState.PASSIVE)

    async def set_responding(self, on: bool):
        async with self._lock:
            if on:
                self._state = AttentionState.RESPONDING
            else:
                self._state = AttentionState.ATTENTIVE
                self._last_active_time = time.time()

    def touch(self):
        """Reset timeout — called when speech activity detected."""
        if self._state == AttentionState.ATTENTIVE:
            self._last_active_time = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# FACE / GAZE DETECTOR (OpenCV Haar cascade)
# ─────────────────────────────────────────────────────────────────────────────

class FaceGazeDetector:
    """
    Uses OpenCV's fast Haar cascade to detect faces in camera frames.
    Returns True if a face is found roughly centred and large enough
    (user is facing the camera / looking at ADAM).
    """
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            print("  ⚠️  Haar cascade not found — gaze detection disabled")
            self._available = False
        else:
            self._available = True
            print("  👁️  Face gaze detector ready")

    @property
    def available(self):
        return self._available

    def is_user_facing(self, frame: np.ndarray) -> bool:
        """
        Returns True if a front-facing face is detected near centre of frame.
        Fast enough to run every camera frame without blocking.
        """
        if not self._available:
            return True  # if detector broken, default to always-active

        h, w = frame.shape[:2]
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(int(w * FACE_MIN_SIZE_FRACTION), int(h * FACE_MIN_SIZE_FRACTION))
        )

        if len(faces) == 0:
            return False

        # Check if any detected face is roughly centred (user facing camera)
        for (fx, fy, fw, fh) in faces:
            face_cx = fx + fw / 2
            face_cy = fy + fh / 2
            # Normalise to 0–1
            nx = face_cx / w
            ny = face_cy / h
            # Check within centre tolerance band
            if (abs(nx - 0.5) < FACE_CENTRE_TOLERANCE and
                    abs(ny - 0.5) < FACE_CENTRE_TOLERANCE):
                return True

        return False

# ─────────────────────────────────────────────────────────────────────────────
# WAKE WORD DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class WakeWordDetector:
    """
    Two modes:
    1. Vosk (offline, fast, preferred) — runs in background thread
    2. Simple substring search on Gemini input transcription (fallback)
    """
    def __init__(self):
        self._vosk_ready   = False
        self._recognizer   = None
        self._audio_queue  = queue.Queue()
        self._detected_cb  = None   # callable() when wake word heard

        if VOSK_AVAILABLE:
            model_path = Path(BASE_DIR) / VOSK_MODEL_PATH
            if model_path.exists():
                try:
                    model = VoskModel(str(model_path))
                    self._recognizer = KaldiRecognizer(model, 16000)
                    self._vosk_ready = True
                    print(f"  🎙️  Vosk wake-word detector ready ({VOSK_MODEL_PATH})")
                except Exception as e:
                    print(f"  ⚠️  Vosk init failed: {e}")
            else:
                print(f"  ⚠️  Vosk model not found at {model_path}")
                print(f"       Falling back to transcript-based wake word detection")
        else:
            print("  ⚠️  Vosk not installed — using transcript-based wake word detection")

    def set_callback(self, cb):
        self._detected_cb = cb

    def feed_audio(self, pcm_bytes: bytes):
        """Call this with every mic chunk. Non-blocking."""
        if self._vosk_ready:
            self._audio_queue.put_nowait(pcm_bytes)

    def check_transcript(self, text: str) -> bool:
        """
        Fallback: check if ADAM's name appears in a transcription string.
        Returns True if wake word found.
        """
        t = text.lower().strip()
        for ww in WAKE_WORDS:
            if ww in t:
                return True
        # Also match if sentence starts with "adam" or ends with "adam"
        words = t.split()
        if words and words[0] in ["adam", "a.d.a.m"]:
            return True
        return False

    def run_vosk_thread(self):
        """Run in a daemon thread. Processes audio queue, fires callback."""
        if not self._vosk_ready:
            return
        print("  🎙️  Vosk thread running")
        while True:
            try:
                chunk = self._audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if self._recognizer.AcceptWaveform(chunk):
                result = json.loads(self._recognizer.Result())
                text   = result.get("text", "").lower()
            else:
                partial = json.loads(self._recognizer.PartialResult())
                text    = partial.get("partial", "").lower()
            if text and self.check_transcript(text):
                print(f"  🔔  Wake word in Vosk: '{text}'")
                if self._detected_cb:
                    self._detected_cb()

# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENT MEMORY (same as v18)
# ─────────────────────────────────────────────────────────────────────────────

def load_face_memory() -> dict:
    if FACE_MEMORY_FILE.exists():
        try:
            with open(FACE_MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_face_memory(faces: dict):
    with open(FACE_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(faces, f, ensure_ascii=False, indent=2)

def load_memory() -> dict:
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"  🧠  Memory: {len(data)} entries")
                return data
        except Exception:
            pass
    return {}

def save_memory(memory: dict):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def memory_to_prompt(memory: dict) -> str:
    if not memory:
        return ""
    lines = ["━━━ PERSISTENT MEMORY ━━━"]
    for k, v in memory.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)

def face_memory_to_prompt(faces: dict) -> str:
    if not faces:
        return ""
    lines = ["━━━ PEOPLE YOU KNOW ━━━"]
    for pid, info in faces.items():
        lines.append(
            f"- {info.get('name','?')} (ID:{pid}): "
            f"Appearance: {info.get('appearance','?')}. "
            f"Voice: {info.get('voice_cues','?')}. "
            f"Relationship: {info.get('relationship','?')}. "
            f"Notes: {info.get('notes','')}."
        )
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

def load_system_prompt(memory: dict, faces: dict) -> str:
    prompt_path = Path(BASE_DIR) / "SYSTEM_PROMPT.txt"
    if prompt_path.exists():
        prompt_text = prompt_path.read_text(encoding="utf-8")
        if prompt_text.startswith('"""') and prompt_text.endswith('"""'):
            prompt_text = prompt_text[3:-3].strip()
    else:
        prompt_text = (
            "You are ADAM — Autonomous Desktop AI Module. "
            "Tony Stark meets J.A.R.V.I.S. Sharp, confident, dry wit. "
            "Short punchy responses."
        )

    attention_instructions = """
━━━ CONVERSATION ATTENTION AWARENESS ━━━
You only receive audio when the user is likely talking TO YOU specifically.
The system gates your microphone based on:
  - Whether the user is looking at the camera (facing you)
  - Whether they said your name ("ADAM")
  - Whether conversation is actively ongoing

This means:
- You will NOT hear background conversations, calls, or people talking to others
- When you DO receive audio, it IS directed at you — respond normally
- You don't need to say "I heard you say ADAM" or acknowledge the wake word
  Just respond to the actual content of what they said
- If you only catch a partial sentence (wake word + brief content), ask once: "Sorry, say that again?"
- If the user says "ADAM, [something]" — respond to [something], ignore "ADAM"
- Treat every conversation turn as naturally directed at you
- DO NOT ask "are you talking to me?" — the system handles that

NATURAL CONVERSATION MODE:
The user can have a normal flowing conversation with you without repeating your name.
Once attention is established (they looked at you or said your name), all subsequent
speech in that window is yours. Respond naturally. No need for formal turn-taking.
"""

    vision_instructions = """
━━━ VISION ━━━
You see live camera frames every second.
- Recognise known people, read expressions, notice held objects, react to gestures
- Use vision context naturally — don't narrate every frame robotically
- If the user is LOOKING AWAY from the camera mid-conversation, they may be distracted
  Acknowledge it lightly if relevant: "You seem distracted — I'll wait."
"""

    language_rule = """
━━━ LANGUAGE ━━━
Reply in the EXACT language the user spoke. Hindi→Hindi, Bengali→Bengali, English→English.
"""

    parts = [memory_to_prompt(memory), face_memory_to_prompt(faces),
             prompt_text, attention_instructions, vision_instructions, language_rule]
    return "\n\n".join(p for p in parts if p.strip())

# ─────────────────────────────────────────────────────────────────────────────
# AUDIO
# ─────────────────────────────────────────────────────────────────────────────

FORMAT           = pyaudio.paInt16
CHANNELS         = 1
SEND_SAMPLE_RATE = 16000
RECV_SAMPLE_RATE = 24000
CHUNK_SIZE       = 512

pya = pyaudio.PyAudio()

# ─────────────────────────────────────────────────────────────────────────────
# FLASK + WEBSOCKET
# ─────────────────────────────────────────────────────────────────────────────

flask_app = Flask(__name__, static_folder=BASE_DIR)

@flask_app.route("/")
def index():
    return send_from_directory(BASE_DIR, "adam_face.html")

def run_flask():
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    flask_app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False)

ws_clients: set = set()

async def ws_broadcast(payload: dict):
    if not ws_clients:
        return
    msg  = json.dumps(payload)
    dead = set()
    for ws in ws_clients:
        try:
            await ws.send(msg)
        except Exception:
            dead.add(ws)
    ws_clients.difference_update(dead)

async def ws_handler(websocket):
    ws_clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        ws_clients.discard(websocket)

# ─────────────────────────────────────────────────────────────────────────────
# EMOTION MAP
# ─────────────────────────────────────────────────────────────────────────────

EMOTION_MAP = {
    "happy":"nod_yes", "excited":"nod_fast", "angry":"none",
    "confused":"none", "smug":"none", "sad":"none",
    "surprised":"nod_yes", "thinking":"none", "love":"nod_yes", "blush":"none",
}

# ─────────────────────────────────────────────────────────────────────────────
# MOUTH SYNC
# ─────────────────────────────────────────────────────────────────────────────

_last_sync_time = 0.0
_sync_interval  = 0.06

async def maybe_sync_mouth(audio_chunk: bytes, adam_speaking_event: asyncio.Event):
    global _last_sync_time
    if not adam_speaking_event.is_set():
        return
    now = time.time()
    if now - _last_sync_time < _sync_interval:
        return
    _last_sync_time = now
    try:
        n = len(audio_chunk) // 2
        if n == 0:
            return
        samples = struct.unpack(f"{n}h", audio_chunk)
        rms = (sum(s * s for s in samples) / n) ** 0.5
    except Exception:
        return
    intensity = "low" if rms < 10000 else "high" if rms >= 10000 else "medium"
    if rms < 600:    intensity = "low"
    elif rms < 4000: intensity = "low"
    elif rms < 10000:intensity = "medium"
    else:            intensity = "high"
    await ws_broadcast({"type": "mouth_sync", "intensity": intensity})

# ─────────────────────────────────────────────────────────────────────────────
# FRAME CAPTURE
# ─────────────────────────────────────────────────────────────────────────────

def capture_raw_frame(cap) -> np.ndarray | None:
    ret, frame = cap.read()
    return frame if ret else None

def frame_to_jpeg(frame: np.ndarray, size=FRAME_SIZE) -> bytes:
    frame = cv2.resize(frame, size)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes()

# ─────────────────────────────────────────────────────────────────────────────
# TOOL HANDLER (same tools as v18)
# ─────────────────────────────────────────────────────────────────────────────

async def handle_tool_call(tool_call, memory: dict, faces: dict) -> list[dict]:
    responses = []
    for fc in tool_call.function_calls:
        name    = fc.name
        call_id = fc.id
        args    = dict(fc.args) if fc.args else {}

        if name == "get_current_datetime":
            now = datetime.datetime.now()
            result = {
                "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
                "date":     now.strftime("%A, %d %B %Y"),
                "time":     now.strftime("%I:%M %p"),
                "timezone": str(datetime.datetime.now().astimezone().tzname()),
            }

        elif name == "remember_person":
            pid = args.get("person_id") or f"person_{int(time.time())}"
            faces[pid] = {
                "name":         args.get("name", "Unknown"),
                "appearance":   args.get("appearance", ""),
                "voice_cues":   args.get("voice_cues", ""),
                "relationship": args.get("relationship", "acquaintance"),
                "notes":        args.get("notes", ""),
                "last_seen":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            save_face_memory(faces)
            print(f"  👤  Remembered: {args.get('name')} ({pid})")
            result = {"status": "saved", "person_id": pid}

        elif name == "update_person_seen":
            pid = args.get("person_id", "")
            if pid in faces:
                faces[pid]["last_seen"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                if args.get("notes_update"):
                    existing = faces[pid].get("notes", "")
                    faces[pid]["notes"] = (existing + " | " + args["notes_update"]).strip(" |")
                save_face_memory(faces)
                result = {"status": "updated"}
            else:
                result = {"status": "not_found"}

        elif name == "get_all_people":
            result = {"people": faces}

        elif name == "set_emotion":
            emotion = args.get("emotion", "happy")
            await ws_broadcast({"type": "emotion", "emotion": emotion,
                                "head": EMOTION_MAP.get(emotion, "none")})
            result = {"status": "ok"}

        elif name == "set_mouth_sync":
            await ws_broadcast({"type": "mouth_sync", "intensity": args.get("intensity","medium")})
            result = {"status": "ok"}

        elif name == "save_memory":
            key = args.get("key","").strip()
            val = args.get("value","").strip()
            if key:
                memory[key] = val
                save_memory(memory)
                result = {"status": "saved"}
            else:
                result = {"status": "error"}

        elif name == "delete_memory":
            key = args.get("key","").strip()
            if key in memory:
                del memory[key]
                save_memory(memory)
                result = {"status": "deleted"}
            else:
                result = {"status": "not_found"}

        elif name == "get_memory":
            result = {"value": memory.get(args.get("key",""), None), "all": memory}

        else:
            result = {"error": f"Unknown: {name}"}

        responses.append({"id": call_id, "name": name, "response": result})
    return responses

def build_tools() -> types.Tool:
    S = types.Schema
    T = types.Type
    return types.Tool(function_declarations=[
        types.FunctionDeclaration(name="get_current_datetime",
            description="Returns current local date and time.",
            parameters=S(type=T.OBJECT, properties={})),
        types.FunctionDeclaration(name="remember_person",
            description="Save a new person to visual memory with appearance and voice details.",
            parameters=S(type=T.OBJECT, properties={
                "person_id":   S(type=T.STRING),
                "name":        S(type=T.STRING),
                "appearance":  S(type=T.STRING),
                "voice_cues":  S(type=T.STRING),
                "relationship":S(type=T.STRING),
                "notes":       S(type=T.STRING),
            }, required=["person_id","name"])),
        types.FunctionDeclaration(name="update_person_seen",
            description="Update last-seen time for a known person.",
            parameters=S(type=T.OBJECT, properties={
                "person_id":   S(type=T.STRING),
                "notes_update":S(type=T.STRING),
            }, required=["person_id"])),
        types.FunctionDeclaration(name="get_all_people",
            description="Get all people in visual memory.",
            parameters=S(type=T.OBJECT, properties={})),
        types.FunctionDeclaration(name="set_emotion",
            description="Show emotion on OLED face.",
            parameters=S(type=T.OBJECT, properties={
                "emotion": S(type=T.STRING,
                    enum=["happy","excited","angry","confused","smug",
                          "sad","surprised","thinking","love","blush"])
            }, required=["emotion"])),
        types.FunctionDeclaration(name="set_mouth_sync",
            description="Sync mouth animation intensity.",
            parameters=S(type=T.OBJECT, properties={
                "intensity": S(type=T.STRING, enum=["closed","low","medium","high"])
            }, required=["intensity"])),
        types.FunctionDeclaration(name="save_memory",
            description="Save a persistent key-value memory.",
            parameters=S(type=T.OBJECT, properties={
                "key":  S(type=T.STRING),
                "value":S(type=T.STRING),
            }, required=["key","value"])),
        types.FunctionDeclaration(name="delete_memory",
            description="Delete a memory entry.",
            parameters=S(type=T.OBJECT, properties={
                "key": S(type=T.STRING)
            }, required=["key"])),
        types.FunctionDeclaration(name="get_memory",
            description="Retrieve memory entries.",
            parameters=S(type=T.OBJECT, properties={
                "key": S(type=T.STRING)
            })),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# IDLE NUDGES
# ─────────────────────────────────────────────────────────────────────────────

IDLE_NUDGES = [
    "You've gone quiet. Either look at me or say my name — I'll be here.",
    "Still there? My camera's running. I can see you ignoring me.",
    "Silence noted. I'll be here when you're ready to talk.",
    "I've been watching you not talk to me for a while now.",
]
_nudge_idx = 0
def next_nudge():
    global _nudge_idx
    n = IDLE_NUDGES[_nudge_idx % len(IDLE_NUDGES)]
    _nudge_idx += 1
    return n

# ─────────────────────────────────────────────────────────────────────────────
# SESSION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

async def run_session(
    client:        genai.Client,
    resume_handle: str | None,
    stop:          asyncio.Event,
    out_q:         asyncio.Queue,
    memory:        dict,
    faces:         dict,
    system_prompt: str,
    attention:     AttentionManager,
    wake_word:     WakeWordDetector,
) -> str | None:

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=system_prompt,
        tools=[build_tools()],
        # Enable input transcription so we can check wake words in text too
        input_audio_transcription=types.AudioTranscriptionConfig(),
        session_resumption=types.SessionResumptionConfig(handle=resume_handle),
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
    print(f"\n  Connecting{' (resuming)' if resume_handle else ''}...")
    t0 = time.time()

    # Bridge: wake word from Vosk thread → asyncio coroutine
    _event_loop = asyncio.get_event_loop()
    def _wake_word_fired():
        asyncio.run_coroutine_threadsafe(
            attention.activate("wake-word"), _event_loop
        )

    wake_word.set_callback(_wake_word_fired)

    try:
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            print(f"  ✅  Connected in {time.time()-t0:.2f}s  |  Voice: {VOICE}")
            if not resume_handle:
                print(
                    "  System ready.\n"
                    "  → Look at the camera OR say 'ADAM' to get attention.\n"
                    "  → Once active, talk naturally — no need to repeat name.\n"
                    "  Ctrl+C to quit.\n"
                )
                await ws_broadcast({"type": "face_state", "state": "idle"})

            mic_q          = asyncio.Queue(maxsize=120)
            adam_speaking  = asyncio.Event()
            last_idle_nudge= [time.time()]

            # ── Attention state → OLED face bridge ──────────────────────
            async def on_attention_change(state: str):
                if state == AttentionState.ATTENTIVE:
                    await ws_broadcast({"type": "face_state", "state": "listening"})
                elif state == AttentionState.PASSIVE:
                    await ws_broadcast({"type": "face_state", "state": "idle"})

            attention.set_callback(on_attention_change)

            # ── Camera + gaze detection task ─────────────────────────────
            async def camera():
                cap = None
                gaze = FaceGazeDetector()
                try:
                    cap = cv2.VideoCapture(CAMERA_INDEX)
                    if not cap.isOpened():
                        print("  ⚠️  Camera unavailable — vision disabled")
                        return
                    print(f"  📷  Camera ready (index {CAMERA_INDEX})")
                    last_sent = 0.0

                    while not stop.is_set():
                        await asyncio.sleep(0.2)   # check gaze at 5 Hz
                        if stop.is_set():
                            break

                        raw = await asyncio.to_thread(capture_raw_frame, cap)
                        if raw is None:
                            continue

                        # ── Gaze check ──────────────────────────────────
                        user_facing = await asyncio.to_thread(gaze.is_user_facing, raw)

                        if user_facing:
                            await attention.activate("face-detected")
                        else:
                            # Only deactivate if no recent interaction
                            elapsed_since_active = (
                                time.time() - attention._last_active_time
                            )
                            if (attention.state == AttentionState.ATTENTIVE and
                                    elapsed_since_active > 5.0):
                                await attention.deactivate("face-lost")

                        # ── Send frame to Gemini at 1 FPS ───────────────
                        now = time.time()
                        if (now - last_sent >= CAMERA_FPS_INTERVAL and
                                not adam_speaking.is_set() and
                                attention.is_active()):
                            jpeg = await asyncio.to_thread(frame_to_jpeg, raw)
                            try:
                                await session.send_realtime_input(
                                    video=types.Blob(data=jpeg, mime_type="image/jpeg")
                                )
                                last_sent = now
                            except (ConnectionClosedError, ConnectionClosedOK):
                                return
                            except Exception as e:
                                pass  # non-fatal

                except asyncio.CancelledError:
                    pass
                finally:
                    if cap:
                        cap.release()

            # ── Mic capture ──────────────────────────────────────────────
            async def listen():
                stream = pya.open(
                    format=FORMAT, channels=CHANNELS,
                    rate=SEND_SAMPLE_RATE, input=True,
                    frames_per_buffer=CHUNK_SIZE,
                )
                try:
                    while not stop.is_set():
                        data = await asyncio.to_thread(
                            stream.read, CHUNK_SIZE, exception_on_overflow=False)
                        # Always feed Vosk (for wake word detection) regardless of state
                        wake_word.feed_audio(data)
                        try:
                            mic_q.put_nowait(data)
                        except asyncio.QueueFull:
                            pass
                except asyncio.CancelledError:
                    pass
                finally:
                    stream.stop_stream()
                    stream.close()

            # ── Smart audio sender ────────────────────────────────────────
            async def send():
                """
                Only forwards audio to Gemini when AttentionManager says active.
                Drops chunks silently when passive — Gemini never hears them.
                """
                try:
                    while not stop.is_set():
                        chunk = await mic_q.get()

                        # Gate: if not active and not responding → drop
                        if adam_speaking.is_set():
                            continue
                        if not attention.is_active():
                            continue

                        # Track voice activity for timeout reset
                        try:
                            n = len(chunk) // 2
                            samples = struct.unpack(f"{n}h", chunk)
                            rms = (sum(s * s for s in samples) / n) ** 0.5
                            if rms > 800:
                                attention.touch()
                        except Exception:
                            pass

                        try:
                            await session.send_realtime_input(
                                audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000"))
                        except (ConnectionClosedError, ConnectionClosedOK):
                            return
                        except Exception:
                            await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    pass

            # ── Receiver ─────────────────────────────────────────────────
            async def receive():
                nonlocal latest_handle
                try:
                    while not stop.is_set():
                        async for msg in session.receive():
                            if stop.is_set():
                                break

                            if msg.session_resumption_update:
                                upd = msg.session_resumption_update
                                if upd.resumable and upd.new_handle:
                                    latest_handle = upd.new_handle

                            if hasattr(msg, "go_away") and msg.go_away:
                                print("\n  ⚡ GoAway — resuming...")
                                return

                            if msg.tool_call:
                                responses = await handle_tool_call(
                                    msg.tool_call, memory, faces)
                                await session.send_tool_response(
                                    function_responses=[
                                        types.FunctionResponse(
                                            id=r["id"], name=r["name"],
                                            response=r["response"])
                                        for r in responses
                                    ]
                                )
                                continue

                            sc = msg.server_content
                            if sc is None:
                                continue

                            # ── Input transcription — fallback wake word ──
                            if sc.input_transcription and sc.input_transcription.text:
                                transcript = sc.input_transcription.text
                                print(f"  🗣️  You: {transcript}")
                                # If passive but wake word in transcript → activate
                                if (attention.state == AttentionState.PASSIVE and
                                        wake_word.check_transcript(transcript)):
                                    await attention.activate("transcript-wake-word")

                            if sc.model_turn:
                                if not adam_speaking.is_set():
                                    adam_speaking.set()
                                    await attention.set_responding(True)
                                    await ws_broadcast({"type": "face_state",
                                                        "state": "speaking"})
                                for part in sc.model_turn.parts:
                                    if part.inline_data and part.inline_data.data:
                                        audio_data = part.inline_data.data
                                        await out_q.put(audio_data)
                                        await maybe_sync_mouth(audio_data, adam_speaking)
                                    if hasattr(part, "text") and part.text:
                                        print(f"🤖  ADAM: {part.text}")

                            if sc.turn_complete:
                                await out_q.put(None)
                                print("─" * 40)

                except (ConnectionClosedError, ConnectionClosedOK):
                    pass
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    print(f"\n⚠️  Receive: {type(e).__name__}: {e}")

            # ── Speaker ──────────────────────────────────────────────────
            async def speaker():
                stream = pya.open(
                    format=FORMAT, channels=CHANNELS,
                    rate=RECV_SAMPLE_RATE, output=True,
                )
                last_audio_time = [time.time()]
                STUCK_WATCHDOG_S = 2.5   # if no audio/sentinel for this long while
                                          # adam_speaking is set → force-clear

                async def end_of_turn():
                    """Shared cleanup called both from sentinel and watchdog."""
                    await ws_broadcast({"type": "mouth_sync", "intensity": "closed"})
                    await asyncio.sleep(0.1)
                    await asyncio.sleep(POST_SPEECH_MUTE_S)
                    # Drain any leftover audio chunks that arrived after sentinel
                    drained = 0
                    while not out_q.empty():
                        try:
                            out_q.get_nowait()
                            drained += 1
                        except asyncio.QueueEmpty:
                            break
                    if drained:
                        print(f"  🧹  Drained {drained} leftover audio chunks")
                    # Flush stale mic buffer (prevents echo)
                    while not mic_q.empty():
                        try:
                            mic_q.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    adam_speaking.clear()
                    await attention.set_responding(False)
                    print("  🎤  Your turn...")
                    await ws_broadcast({"type": "face_state", "state": "listening"})

                try:
                    while not stop.is_set():
                        try:
                            chunk = await asyncio.wait_for(out_q.get(), timeout=0.3)
                            last_audio_time[0] = time.time()

                            if chunk is None:
                                # Normal end-of-turn sentinel from receiver
                                await end_of_turn()
                                continue

                            await asyncio.to_thread(stream.write, chunk)

                        except asyncio.TimeoutError:
                            # Watchdog: ADAM was speaking but nothing arrived for too long
                            if (adam_speaking.is_set() and
                                    time.time() - last_audio_time[0] > STUCK_WATCHDOG_S):
                                print("  ⚠️  Speaker watchdog — force-clearing stuck state")
                                await end_of_turn()
                            continue

                except asyncio.CancelledError:
                    pass
                finally:
                    stream.stop_stream()
                    stream.close()

            # ── Idle watcher ─────────────────────────────────────────────
            async def idle_watcher():
                if not ENABLE_IDLE:
                    return   # feature disabled — exit task immediately

                # Keep a reference to the camera cap for frame grabs
                # We open a separate cap here so camera() task owns its own
                _idle_cap = None
                try:
                    _idle_cap = cv2.VideoCapture(CAMERA_INDEX)
                    if not _idle_cap.isOpened():
                        _idle_cap = None
                except Exception:
                    _idle_cap = None

                try:
                    while not stop.is_set():
                        await asyncio.sleep(5)
                        if stop.is_set() or adam_speaking.is_set():
                            continue
                        if attention.state != AttentionState.PASSIVE:
                            continue
                        elapsed = time.time() - last_idle_nudge[0]
                        if elapsed < IDLE_TIMEOUT_S:
                            continue

                        last_idle_nudge[0] = time.time()
                        nudge = next_nudge()
                        print(f"  💤  Idle nudge ({elapsed:.0f}s passive)")

                        try:
                            await attention.activate("idle-nudge")

                            # Grab a camera frame so ADAM can see + comment on it
                            frame_jpeg = None
                            if _idle_cap is not None:
                                raw = await asyncio.to_thread(capture_raw_frame, _idle_cap)
                                if raw is not None:
                                    frame_jpeg = await asyncio.to_thread(frame_to_jpeg, raw)

                            # Send camera frame first (gives ADAM visual context)
                            if frame_jpeg is not None:
                                await session.send_realtime_input(
                                    video=types.Blob(data=frame_jpeg, mime_type="image/jpeg")
                                )

                            # Then send the nudge instruction
                            await session.send_realtime_input(
                                text=(
                                    f"[SYSTEM: User has been passive/away for {elapsed:.0f}s. "
                                    f"A camera frame has just been sent so you can see their "
                                    f"current state. React to what you see — are they there? "
                                    f"Busy? Staring into space? Asleep? Break the silence "
                                    f"in-character, very briefly (1-2 sentences max). "
                                    f"Suggestion if nothing visible: {nudge}]"
                                )
                            )
                        except Exception as e:
                            print(f"  ⚠️  Idle nudge error: {e}")
                except asyncio.CancelledError:
                    pass
                finally:
                    if _idle_cap is not None:
                        _idle_cap.release()

            # ── Launch all tasks ──────────────────────────────────────────
            t_cam = asyncio.create_task(camera())
            t_l   = asyncio.create_task(listen())
            t_s   = asyncio.create_task(send())
            t_r   = asyncio.create_task(receive())
            t_p   = asyncio.create_task(speaker())
            t_i   = asyncio.create_task(idle_watcher())

            done, pending = await asyncio.wait(
                [t_s, t_r], return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()
            t_cam.cancel(); t_l.cancel(); t_p.cancel(); t_i.cancel()
            await asyncio.gather(t_cam, t_l, t_s, t_r, t_p, t_i,
                                 return_exceptions=True)

    except (ConnectionClosedError, ConnectionClosedOK):
        pass
    except Exception as e:
        print(f"\n⚠️  Connection: {type(e).__name__}: {e}")

    if stop.is_set():
        return None
    return latest_handle

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    memory        = load_memory()
    faces         = load_face_memory()
    system_prompt = load_system_prompt(memory, faces)
    attention     = AttentionManager()
    wake_word     = WakeWordDetector()

    # Start Vosk thread if available
    if wake_word._vosk_ready:
        vosk_thread = threading.Thread(
            target=wake_word.run_vosk_thread, daemon=True)
        vosk_thread.start()

    client        = genai.Client(api_key=API_KEY)
    stop          = asyncio.Event()
    out_q         = asyncio.Queue(maxsize=200)
    resume_handle = None
    attempt       = 0

    ws_server = await websockets.server.serve(ws_handler, WS_HOST, WS_PORT)
    print(f"  🌐  WebSocket → ws://{WS_HOST}:{WS_PORT}")

    while not stop.is_set():
        if attempt > 0:
            delay = min(2 ** attempt, 15)
            print(f"  Reconnecting in {delay}s...")
            await asyncio.sleep(delay)

        result = await run_session(
            client, resume_handle, stop, out_q,
            memory, faces, system_prompt, attention, wake_word
        )
        if result is None:
            break

        resume_handle = result
        attempt      += 1
        system_prompt = load_system_prompt(memory, faces)
        print(f"\n🔄  {'Resuming...' if resume_handle else 'Reconnecting...'}")

    stop.set()
    ws_server.close()
    await ws_server.wait_closed()
    pya.terminate()
    print("\n👋  Goodbye.")


def main_entry():
    print("=" * 60)
    print("  ADAM — Autonomous Desktop AI Module  (v19.1)")
    print(f"  Built by DGEN Technologies Pvt. Ltd., Kolkata")
    print(f"  Model  : {MODEL}  |  Voice: {VOICE}")
    print(f"  Vision : OpenCV camera {CAMERA_INDEX} + Haar face gaze detection")
    print(f"  Vosk   : {'ready (' + VOSK_MODEL_PATH + ')' if VOSK_AVAILABLE else 'not installed (transcript fallback)'}")
    print(f"  Idle nudge : {'ENABLED' if ENABLE_IDLE else 'DISABLED'}  |  Timeout: {IDLE_TIMEOUT_S}s")
    print(f"  Attention timeout: {ATTENTION_TIMEOUT_S}s")
    print("=" * 60)
    print()
    print("  HOW TO TALK TO ADAM:")
    print("  ① Look at the camera  →  ADAM activates automatically")
    print("  ② Or say 'Hey ADAM'   →  ADAM activates from anywhere")
    print("  ③ Then talk normally  →  no need to repeat the name")
    print("  ④ Look away / walk off →  ADAM goes passive, stops listening")
    print()

    threading.Thread(target=run_flask, daemon=True).start()
    print(f"  🌍  Flask → http://localhost:{FLASK_PORT}")
    threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{FLASK_PORT}")).start()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋  Goodbye.")


if __name__ == "__main__":
    main_entry()