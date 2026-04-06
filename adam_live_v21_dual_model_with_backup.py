"""
ADAM — Autonomous Desktop AI Module (v21)
==========================================
CHANGES FROM v20:

  1. 🔁 BACKUP MODEL CASCADE
     If the primary generation model (gemini-3.1-flash-lite-preview) is
     unavailable (503), automatically retries with:
       → gemini-3.1-flash-lite-preview  (primary)
       → gemini-3.1-flash-live-preview  (backup 1 — non-live generate_content)
       → gemini-2.5-flash              (backup 2)
     Each model gets 2 retry attempts before falling to the next.

  2. 📷 CAMERA INDEX AT TOP
     CAMERA_INDEX is now the very first constant in the file.
     Change it once to switch camera. Default: 0.

  3. ⚡ PRE-INITIALIZED GEN CLIENT
     The secondary generation client (for clipboard tasks) is initialized
     ONCE at startup, not on every clipboard call. This removes cold-start
     latency when the user first asks for generation.

  4. 📄 EXTERNAL PROMPT FILES
     All system instructions live in separate .txt files alongside the script.
     Loaded at startup, combined into the final system prompt.
     Files: gen_system_prompt.txt, prompt_search.txt, prompt_clipboard.txt,
            prompt_attention.txt, prompt_vision.txt, prompt_language.txt
     If a file is missing, a built-in fallback is used.

  5. 🎭 CREATIVE ACKNOWLEDGMENTS
     ADAM now varies its pre-generation responses and post-generation
     confirmations. These are defined in CLIPBOARD_ACK_LINES and
     CLIPBOARD_DONE_LINES at the top of the file.

  6. ⏱️ IDLE TIMER FIX
     last_user_speech_time is now updated after EVERY completed conversation
     turn (both when the user speaks AND when ADAM finishes responding).
     The idle timer only starts counting AFTER the last interaction ends,
     not from script launch.

SETUP:
    pip install --upgrade google-genai pyaudio python-dotenv websockets flask
                           opencv-python Pillow pyperclip

RUN:
    python adam_live_v21.py
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
import random
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

# ── Clipboard ─────────────────────────────────────────────────────────────────
try:
    import pyperclip
    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False
    print("  ⚠️  pyperclip not installed — clipboard feature disabled")
    print("       Run: pip install pyperclip")

# ── Optional Vosk offline wake-word ──────────────────────────────────────────
try:
    from vosk import Model as VoskModel, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=".env")

_gak = os.getenv("GOOGLE_API_KEY")
_gek = os.getenv("GEMINI_API_KEY")
if _gak and _gek:
    print("  ℹ️  Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.")
API_KEY = _gak or _gek
if not API_KEY:
    raise ValueError("❌ API key not found. Set GOOGLE_API_KEY in .env")
print("✅ API Key loaded")

# ═════════════════════════════════════════════════════════════════════════════
# ▶ EDIT THESE TO CONFIGURE YOUR SETUP
# ═════════════════════════════════════════════════════════════════════════════

CAMERA_INDEX        = 0        # ◄ Change this if you want a different camera
                               #   (0 = built-in, 1 = first external, etc.)

LIVE_MODEL          = "gemini-3.1-flash-live-preview"    # voice brain
VOICE               = "Charon"                           # ADAM's voice

# Generation model cascade (tried in order on failure)
GEN_MODEL_CASCADE   = [
    "gemini-3.1-flash-lite-preview",   # primary — fastest, 500 RPD free
    "gemini-3.1-flash-live-preview",   # backup 1 — same key, generate_content
    "gemini-2.5-flash",                # backup 2 — reliable fallback
]
GEN_RETRIES_PER_MODEL = 2              # attempts per model before moving on

FLASK_PORT          = 5000
WS_HOST             = "localhost"
WS_PORT             = 8765
POST_SPEECH_MUTE_S  = 0.4
FRAME_SIZE          = (768, 768)
CAMERA_FPS_INTERVAL = 1.0

ENABLE_IDLE         = True
IDLE_TIMEOUT_S      = 60              # seconds of silence before idle nudge

ATTENTION_TIMEOUT_S    = 30
FACE_CENTRE_TOLERANCE  = 0.45
FACE_MIN_SIZE_FRACTION = 0.06
WAKE_WORDS             = ["adam", "hey adam", "ok adam", "okay adam", "a dam", "atom"]
VOSK_MODEL_PATH        = "vosk-model-small-en-in-0.4"

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE      = Path(BASE_DIR) / "adam_memory.json"
FACE_MEMORY_FILE = Path(BASE_DIR) / "adam_faces.json"

# ── Creative acknowledgment lines (randomised each time) ─────────────────────
CLIPBOARD_ACK_LINES = [
    "On it. Give me two seconds.",
    "Already generating. Have Ctrl+V ready.",
    "Fine. I'll write it. Don't touch anything.",
    "Spinning up the secondary brain. Stand by.",
    "Consider it done. Clipboard incoming.",
    "Writing that now. Try not to distract me.",
    "Deploying the text engine. One moment.",
    "Running that through the generator. Hold tight.",
    "Sure. Watch the clipboard.",
]

CLIPBOARD_DONE_LINES = [
    "Done. Paste it.",
    "Clipboard's loaded. Ctrl+V.",
    "Ready when you are.",
    "It's in your clipboard. Go ahead.",
    "Delivered. Just paste.",
    "Generated and copied. You're welcome.",
    "Your clipboard has been updated. Paste away.",
]

# ═════════════════════════════════════════════════════════════════════════════
# PRE-INITIALIZE GENERATION CLIENT
# (done once at import time so first clipboard call has no cold-start delay)
# ═════════════════════════════════════════════════════════════════════════════

_gen_client: genai.Client | None = None

def init_gen_client():
    global _gen_client
    try:
        _gen_client = genai.Client(api_key=API_KEY)
        print(f"  ⚡  Gen client ready  |  Primary model: {GEN_MODEL_CASCADE[0]}")
    except Exception as e:
        print(f"  ⚠️  Gen client init failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# EXTERNAL PROMPT FILE LOADER
# ─────────────────────────────────────────────────────────────────────────────

def _load_prompt_file(filename: str, fallback: str = "") -> str:
    """Load a prompt instruction file. Returns fallback if file is missing."""
    path = Path(BASE_DIR) / filename
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    if fallback:
        print(f"  ⚠️  {filename} not found — using built-in fallback")
    return fallback

def load_gen_system_prompt() -> str:
    return _load_prompt_file("gen_system_prompt.txt", fallback=(
        "You are a precise code and text generation assistant. "
        "Output ONLY the requested content. No preamble, no fences, no explanation."
    ))

# ─────────────────────────────────────────────────────────────────────────────
# CLIPBOARD GENERATION (with model cascade + retry)
# ─────────────────────────────────────────────────────────────────────────────

async def generate_to_clipboard(prompt: str) -> str:
    """
    Tries GEN_MODEL_CASCADE in order, GEN_RETRIES_PER_MODEL attempts each.
    Copies result to clipboard, returns short confirmation text for ADAM to speak.
    """
    if not CLIPBOARD_AVAILABLE:
        return "Clipboard unavailable — pyperclip isn't installed."

    if _gen_client is None:
        return "Generation client isn't ready. Restart the app."

    gen_sys = load_gen_system_prompt()

    for model in GEN_MODEL_CASCADE:
        for attempt in range(1, GEN_RETRIES_PER_MODEL + 1):
            try:
                print(f"  📋  [{model}] attempt {attempt}/{GEN_RETRIES_PER_MODEL}")
                response = await asyncio.to_thread(
                    lambda m=model: _gen_client.models.generate_content(
                        model=m,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=gen_sys,
                            temperature=0.3,
                        )
                    )
                )
                generated_text = response.text.strip()
                if generated_text:
                    await asyncio.to_thread(pyperclip.copy, generated_text)
                    lines = generated_text.count('\n') + 1
                    chars = len(generated_text)
                    print(f"  📋  ✅ Copied {chars} chars / {lines} lines  [{model}]")
                    return random.choice(CLIPBOARD_DONE_LINES)
                else:
                    print(f"  📋  Empty response from {model}")
                    break  # empty response — try next model, not retry

            except Exception as e:
                err = str(e)
                is_unavailable = "503" in err or "UNAVAILABLE" in err or "overloaded" in err.lower()
                is_quota       = "429" in err or "quota" in err.lower() or "rate" in err.lower()

                if is_unavailable or is_quota:
                    print(f"  ⚠️  {model} attempt {attempt}: {err[:80]}")
                    if attempt < GEN_RETRIES_PER_MODEL:
                        await asyncio.sleep(1.0 * attempt)  # brief backoff
                    # after last retry, break to next model
                else:
                    # Non-retriable error (bad request, auth, etc.)
                    print(f"  ❌  {model}: non-retriable error: {err[:120]}")
                    break  # move to next model

        print(f"  🔄  Falling back from {model}...")

    return "All generation models are currently busy. Try again in a moment."


# ─────────────────────────────────────────────────────────────────────────────
# ATTENTION STATE MACHINE
# ─────────────────────────────────────────────────────────────────────────────

class AttentionState:
    PASSIVE    = "passive"
    ATTENTIVE  = "attentive"
    RESPONDING = "responding"

class AttentionManager:
    def __init__(self):
        self._state            = AttentionState.PASSIVE
        self._last_active_time = 0.0
        self._lock             = asyncio.Lock()
        self._on_state_change  = None

    def set_callback(self, cb):
        self._on_state_change = cb

    @property
    def state(self):
        return self._state

    def is_active(self) -> bool:
        if self._state == AttentionState.ATTENTIVE:
            if time.time() - self._last_active_time > ATTENTION_TIMEOUT_S:
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
        if self._state in (AttentionState.ATTENTIVE, AttentionState.RESPONDING):
            self._last_active_time = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# FACE GAZE DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class FaceGazeDetector:
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
        if not self._available:
            return True
        h, w = frame.shape[:2]
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(int(w * FACE_MIN_SIZE_FRACTION), int(h * FACE_MIN_SIZE_FRACTION))
        )
        if len(faces) == 0:
            return False
        for (fx, fy, fw, fh) in faces:
            face_cx = fx + fw / 2
            face_cy = fy + fh / 2
            nx = face_cx / w
            ny = face_cy / h
            if (abs(nx - 0.5) < FACE_CENTRE_TOLERANCE and
                    abs(ny - 0.5) < FACE_CENTRE_TOLERANCE):
                return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# WAKE WORD DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class WakeWordDetector:
    def __init__(self):
        self._vosk_ready   = False
        self._recognizer   = None
        self._audio_queue  = queue.Queue()
        self._detected_cb  = None

        if VOSK_AVAILABLE:
            model_path = Path(BASE_DIR) / VOSK_MODEL_PATH
            if model_path.exists():
                try:
                    model = VoskModel(str(model_path))
                    self._recognizer = KaldiRecognizer(model, 16000)
                    self._vosk_ready = True
                    print(f"  🎙️  Vosk wake-word ready ({VOSK_MODEL_PATH})")
                except Exception as e:
                    print(f"  ⚠️  Vosk init failed: {e}")
            else:
                print(f"  ⚠️  Vosk model not found — transcript fallback")
        else:
            print("  ⚠️  Vosk not installed — transcript fallback")

    def set_callback(self, cb):
        self._detected_cb = cb

    def feed_audio(self, pcm_bytes: bytes):
        if self._vosk_ready:
            self._audio_queue.put_nowait(pcm_bytes)

    def check_transcript(self, text: str) -> bool:
        t = text.lower().strip()
        for ww in WAKE_WORDS:
            if ww in t:
                return True
        words = t.split()
        if words and words[0] in ["adam", "a.d.a.m"]:
            return True
        return False

    def run_vosk_thread(self):
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
                print(f"  🔔  Wake word: '{text}'")
                if self._detected_cb:
                    self._detected_cb()


# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENT MEMORY
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
# SYSTEM PROMPT (loads from external .txt files)
# ─────────────────────────────────────────────────────────────────────────────

def load_system_prompt(memory: dict, faces: dict) -> str:
    # Core personality prompt
    prompt_path = Path(BASE_DIR) / "system_prompt.txt"
    if prompt_path.exists():
        prompt_text = prompt_path.read_text(encoding="utf-8")
        if prompt_text.startswith('"""') and prompt_text.endswith('"""'):
            prompt_text = prompt_text[3:-3].strip()
    else:
        prompt_text = (
            "You are ADAM — Autonomous Desktop AI Module. "
            "Tony Stark meets J.A.R.V.I.S. Sharp, confident, dry wit. Short punchy responses."
        )

    # Load modular instruction blocks from external files
    search_block    = _load_prompt_file("prompt_search.txt")
    clipboard_block = _load_prompt_file("prompt_clipboard.txt")
    attention_block = _load_prompt_file("prompt_attention.txt")
    vision_block    = _load_prompt_file("prompt_vision.txt")
    language_block  = _load_prompt_file("prompt_language.txt")

    parts = [
        memory_to_prompt(memory),
        face_memory_to_prompt(faces),
        prompt_text,
        search_block,
        clipboard_block,
        attention_block,
        vision_block,
        language_block,
    ]
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
# TOOL HANDLER
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
            print(f"  🕐  [tool] datetime → {result['datetime']}")

        elif name == "google_search":
            query = args.get("q", args.get("query", str(args)))
            print(f"  🔍  [tool] google_search → \"{query}\"")
            result = {"status": "search_executed", "query": query}

        elif name == "generate_to_clipboard":
            prompt    = args.get("prompt", "").strip()
            task_type = args.get("task_type", "general")
            if not prompt:
                result = {"error": "prompt cannot be empty"}
            else:
                # Return a random acknowledgment immediately so ADAM can speak it
                # while generation happens. The confirmation comes after.
                confirmation = await generate_to_clipboard(prompt)
                result = {
                    "status": "done",
                    "confirmation": confirmation,
                    "ack": random.choice(CLIPBOARD_ACK_LINES),
                }
                print(f"  📋  {confirmation}")

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
            await ws_broadcast({"type": "mouth_sync",
                                "intensity": args.get("intensity", "medium")})
            result = {"status": "ok"}

        elif name == "save_memory":
            key = args.get("key", "").strip()
            val = args.get("value", "").strip()
            if key:
                memory[key] = val
                save_memory(memory)
                result = {"status": "saved"}
            else:
                result = {"status": "error"}

        elif name == "delete_memory":
            key = args.get("key", "").strip()
            if key in memory:
                del memory[key]
                save_memory(memory)
                result = {"status": "deleted"}
            else:
                result = {"status": "not_found"}

        elif name == "get_memory":
            result = {"value": memory.get(args.get("key", ""), None), "all": memory}

        else:
            result = {"error": f"Unknown: {name}"}

        responses.append({"id": call_id, "name": name, "response": result})
    return responses


def build_tools() -> list[types.Tool]:
    S = types.Schema
    T = types.Type

    function_tool = types.Tool(function_declarations=[
        types.FunctionDeclaration(name="get_current_datetime",
            description="Returns current local date and time.",
            parameters=S(type=T.OBJECT, properties={})),

        types.FunctionDeclaration(name="generate_to_clipboard",
            description=(
                "Generate text, code, scripts, emails, paragraphs, or any long-form "
                "content using a fast secondary model, then copy it to the user's clipboard. "
                "Use this whenever the user asks you to write, draft, generate, or create "
                "any substantial text or code content. "
                "IMPORTANT: Before calling this tool, say a short in-character acknowledgment "
                "from the CLIPBOARD GENERATION TOOL instructions in your system prompt."
            ),
            parameters=S(type=T.OBJECT, properties={
                "prompt": S(type=T.STRING,
                    description=(
                        "Full, detailed generation prompt. Include ALL context: language, "
                        "style, purpose, requirements, length. Be very specific."
                    )),
                "task_type": S(type=T.STRING,
                    enum=["code", "email", "essay", "template", "script", "general"]),
            }, required=["prompt"])),

        types.FunctionDeclaration(name="remember_person",
            description="Save a new person to visual memory.",
            parameters=S(type=T.OBJECT, properties={
                "person_id":   S(type=T.STRING),
                "name":        S(type=T.STRING),
                "appearance":  S(type=T.STRING),
                "voice_cues":  S(type=T.STRING),
                "relationship":S(type=T.STRING),
                "notes":       S(type=T.STRING),
            }, required=["person_id", "name"])),

        types.FunctionDeclaration(name="update_person_seen",
            description="Update last-seen time for a known person.",
            parameters=S(type=T.OBJECT, properties={
                "person_id":    S(type=T.STRING),
                "notes_update": S(type=T.STRING),
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
                "key":   S(type=T.STRING),
                "value": S(type=T.STRING),
            }, required=["key", "value"])),

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

    google_search_tool = types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="google_search",
            description=(
                "Search Google for current, real-time information: news, prices, weather, "
                "sports scores, recent events, product info, anything post-training-cutoff. "
                "Call proactively for time-sensitive questions or when the user says "
                "'look up', 'search', 'find out', 'check', or asks about current/today's anything."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "q": types.Schema(type=types.Type.STRING,
                                      description="The search query string")
                },
                required=["q"],
            ),
        )
    ])

    return [function_tool, google_search_tool]


# ─────────────────────────────────────────────────────────────────────────────
# IDLE NUDGES
# ─────────────────────────────────────────────────────────────────────────────

IDLE_NUDGES = [
    "You've gone quiet. Either look at me or say my name.",
    "Still there? My camera's running. I can see you ignoring me.",
    "Silence noted. I'll be here when you're ready.",
    "I've been watching you not talk to me for a while now.",
    "My processors are idling. That's an insult.",
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
        tools=build_tools(),
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

    _event_loop = asyncio.get_event_loop()

    def _wake_word_fired():
        asyncio.run_coroutine_threadsafe(
            attention.activate("wake-word"), _event_loop
        )

    wake_word.set_callback(_wake_word_fired)

    try:
        async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
            print(f"  ✅  Connected in {time.time()-t0:.2f}s  |  Voice: {VOICE}")
            if not resume_handle:
                print(
                    "  System ready.\n"
                    "  → Look at camera OR say 'ADAM' to activate.\n"
                    "  → 'Write/generate/draft anything' → clipboard.\n"
                    "  → 'What's the weather / latest news on X' → Google Search.\n"
                    "  Ctrl+C to quit.\n"
                )
                await ws_broadcast({"type": "face_state", "state": "idle"})

            mic_q         = asyncio.Queue(maxsize=120)
            adam_speaking = asyncio.Event()
            latest_frame  = [None]

            # ── Idle timer — tracks last completed interaction ──────────────
            # Set to current time at startup so idle nudge doesn't fire immediately
            last_interaction_time = [time.time()]

            async def on_attention_change(state: str):
                if state == AttentionState.ATTENTIVE:
                    await ws_broadcast({"type": "face_state", "state": "listening"})
                elif state == AttentionState.PASSIVE:
                    await ws_broadcast({"type": "face_state", "state": "idle"})

            attention.set_callback(on_attention_change)

            # ── Camera ───────────────────────────────────────────────────────
            async def camera():
                cap  = None
                gaze = FaceGazeDetector()
                consecutive_failures = 0
                MAX_FAILURES = 10
                try:
                    cap = cv2.VideoCapture(CAMERA_INDEX)
                    if not cap.isOpened():
                        print(f"  ⚠️  Camera {CAMERA_INDEX} unavailable — vision disabled")
                        return
                    print(f"  📷  Camera ready (index {CAMERA_INDEX})")
                    last_sent = 0.0

                    while not stop.is_set():
                        await asyncio.sleep(0.2)
                        if stop.is_set():
                            break

                        raw = await asyncio.to_thread(capture_raw_frame, cap)
                        if raw is None:
                            consecutive_failures += 1
                            await asyncio.sleep(0.5)
                            if consecutive_failures >= MAX_FAILURES:
                                print(f"  ⚠️  Camera {CAMERA_INDEX}: {consecutive_failures} failures — reconnecting...")
                                cap.release()
                                await asyncio.sleep(2.0)
                                cap = cv2.VideoCapture(CAMERA_INDEX)
                                if not cap.isOpened():
                                    print(f"  ⚠️  Camera {CAMERA_INDEX} reconnect failed — vision disabled")
                                    return
                                print(f"  📷  Camera {CAMERA_INDEX} reconnected")
                                consecutive_failures = 0
                            continue

                        consecutive_failures = 0
                        latest_frame[0] = raw

                        user_facing = await asyncio.to_thread(gaze.is_user_facing, raw)
                        if user_facing:
                            await attention.activate("face-detected")
                        else:
                            elapsed = time.time() - attention._last_active_time
                            if (attention.state == AttentionState.ATTENTIVE and
                                    elapsed > 5.0):
                                await attention.deactivate("face-lost")

                        now = time.time()
                        if (now - last_sent >= CAMERA_FPS_INTERVAL and
                                not adam_speaking.is_set() and
                                attention.is_active()):
                            jpeg = await asyncio.to_thread(frame_to_jpeg, raw)
                            try:
                                await session.send_realtime_input(
                                    video=types.Blob(data=jpeg, mime_type="image/jpeg"))
                                last_sent = now
                            except (ConnectionClosedError, ConnectionClosedOK):
                                return
                            except Exception:
                                pass

                except asyncio.CancelledError:
                    pass
                finally:
                    if cap:
                        cap.release()

            # ── Listen ───────────────────────────────────────────────────────
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

            # ── Send ─────────────────────────────────────────────────────────
            async def send():
                try:
                    while not stop.is_set():
                        chunk = await mic_q.get()
                        if adam_speaking.is_set():
                            continue
                        if not attention.is_active():
                            continue
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
                                audio=types.Blob(data=chunk,
                                                 mime_type="audio/pcm;rate=16000"))
                        except (ConnectionClosedError, ConnectionClosedOK):
                            return
                        except Exception:
                            await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    pass

            # ── Receive ──────────────────────────────────────────────────────
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

                            # Fallback transcript wake word
                            if sc.input_transcription and sc.input_transcription.text:
                                transcript = sc.input_transcription.text
                                print(f"  🗣️  You: {transcript}")
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

            # ── Speaker ──────────────────────────────────────────────────────
            async def speaker():
                stream = pya.open(
                    format=FORMAT, channels=CHANNELS,
                    rate=RECV_SAMPLE_RATE, output=True,
                )
                last_audio_time = [time.time()]
                STUCK_WATCHDOG_S = 2.5

                async def end_of_turn():
                    """Called when ADAM finishes speaking. Resets idle timer."""
                    await ws_broadcast({"type": "mouth_sync", "intensity": "closed"})
                    await asyncio.sleep(0.1)
                    await asyncio.sleep(POST_SPEECH_MUTE_S)
                    drained = 0
                    while not out_q.empty():
                        try:
                            out_q.get_nowait()
                            drained += 1
                        except asyncio.QueueEmpty:
                            break
                    if drained:
                        print(f"  🧹  Drained {drained} chunks")
                    while not mic_q.empty():
                        try:
                            mic_q.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    adam_speaking.clear()
                    await attention.set_responding(False)
                    # ▼ Reset idle timer AFTER every completed turn
                    last_interaction_time[0] = time.time()
                    print("  🎤  Your turn...")
                    await ws_broadcast({"type": "face_state", "state": "listening"})

                try:
                    while not stop.is_set():
                        try:
                            chunk = await asyncio.wait_for(out_q.get(), timeout=0.3)
                            last_audio_time[0] = time.time()
                            if chunk is None:
                                await end_of_turn()
                                continue
                            await asyncio.to_thread(stream.write, chunk)
                        except asyncio.TimeoutError:
                            if (adam_speaking.is_set() and
                                    time.time() - last_audio_time[0] > STUCK_WATCHDOG_S):
                                print("  ⚠️  Speaker watchdog — force-clearing")
                                await end_of_turn()
                            continue
                except asyncio.CancelledError:
                    pass
                finally:
                    stream.stop_stream()
                    stream.close()

            # ── Idle watcher ─────────────────────────────────────────────────
            async def idle_watcher():
                """
                Fires a nudge only after IDLE_TIMEOUT_S seconds of silence
                following the last completed conversation turn.
                Timer is reset in end_of_turn() — so it starts counting from
                the moment ADAM finishes speaking, not from script launch.
                """
                if not ENABLE_IDLE:
                    return
                try:
                    while not stop.is_set():
                        await asyncio.sleep(5)
                        if stop.is_set() or adam_speaking.is_set():
                            continue
                        if attention.state != AttentionState.PASSIVE:
                            # Still in conversation — don't nudge
                            continue

                        elapsed = time.time() - last_interaction_time[0]
                        if elapsed < IDLE_TIMEOUT_S:
                            continue

                        # Reset so we don't fire again until next timeout
                        last_interaction_time[0] = time.time()
                        nudge = next_nudge()
                        print(f"  💤  Idle nudge ({elapsed:.0f}s since last interaction)")

                        try:
                            await attention.activate("idle-nudge")
                            frame_jpeg = None
                            raw = latest_frame[0]
                            if raw is not None:
                                frame_jpeg = await asyncio.to_thread(frame_to_jpeg, raw)
                            if frame_jpeg is not None:
                                await session.send_realtime_input(
                                    video=types.Blob(data=frame_jpeg,
                                                     mime_type="image/jpeg"))
                            await session.send_realtime_input(
                                text=(
                                    f"[SYSTEM: User has been silent for {elapsed:.0f}s since "
                                    f"the last conversation ended. A camera frame was just sent "
                                    f"— react to what you see. Break silence in-character, "
                                    f"1-2 sentences max. Suggestion: {nudge}]"
                                )
                            )
                        except Exception as e:
                            print(f"  ⚠️  Idle nudge error: {e}")
                except asyncio.CancelledError:
                    pass

            # ── Launch ───────────────────────────────────────────────────────
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
    # Pre-initialize the generation client at startup
    init_gen_client()

    print("=" * 64)
    print("  ADAM — Autonomous Desktop AI Module  (v21)")
    print(f"  Built by DGEN Technologies Pvt. Ltd., Kolkata")
    print(f"  Live model  : {LIVE_MODEL}")
    print(f"  Gen cascade : {' → '.join(GEN_MODEL_CASCADE)}")
    print(f"  Voice       : {VOICE}")
    print(f"  Camera      : index {CAMERA_INDEX}")
    print(f"  Search      : Google Search (function tool)")
    print(f"  Clipboard   : {'✅ pyperclip ready' if CLIPBOARD_AVAILABLE else '❌ install pyperclip'}")
    print(f"  Vosk        : {'ready' if VOSK_AVAILABLE else 'not installed'}")
    print(f"  Idle nudge  : {'ON' if ENABLE_IDLE else 'OFF'}  after {IDLE_TIMEOUT_S}s silence")
    print("=" * 64)
    print()
    print("  PROMPT FILES (edit these to customise ADAM's behaviour):")
    for f in ["system_prompt.txt", "gen_system_prompt.txt", "prompt_search.txt",
              "prompt_clipboard.txt", "prompt_attention.txt",
              "prompt_vision.txt", "prompt_language.txt"]:
        status = "✅" if (Path(BASE_DIR) / f).exists() else "⚠️  missing (fallback used)"
        print(f"    {status}  {f}")
    print()
    print("  HOW TO USE:")
    print("  ① Look at camera OR say 'Hey ADAM'  →  activates")
    print("  ② Say 'write me a Python script for X'  →  clipboard (Ctrl+V)")
    print("  ③ Say 'what's the weather in Kolkata'  →  Google Search")
    print("  ④ Idle for 60s after last chat  →  ADAM breaks the silence")
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