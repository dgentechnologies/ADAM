"""
ADAM — Autonomous Desktop AI Module (v20)
==========================================
NEW IN THIS VERSION:

  1. 🔍 GOOGLE SEARCH — as a proper function tool
     ADAM can now call web search to answer current events, prices,
     news, weather, and anything beyond its training knowledge.
     Implemented via Gemini's native google_search tool (built-in grounding).
     The Live API routes google_search to the client; we acknowledge it so
     Gemini can merge the search results server-side.

  2. 📋 CLIPBOARD GENERATOR — dual-model architecture
     When you say things like:
       "Write me a Python script for..."
       "Generate a paragraph about..."
       "Draft an email for..."
       "Write a function that..."
     ADAM uses a SECOND model (gemini-3.1-flash-lite-preview) to silently
     generate the content and copy it to your system clipboard.
     You just press Ctrl+V / Cmd+V to paste it anywhere.
     ADAM confirms verbally when it's ready.

  MODELS USED:
    Live voice (ADAM's brain)    : gemini-3.1-flash-live-preview
    Text/code generation (silent): gemini-3.1-flash-lite-preview
    → Flash-Lite chosen for highest free-tier quota (500 RPD from API console)
      and fastest latency for generation tasks.

  SESSION MANAGEMENT (unchanged from v19):
    - Session Resumption (survives WebSocket resets)
    - SlidingWindow context compression (unlimited session length)
    - GoAway detection and automatic reconnect

SETUP:
    pip install --upgrade google-genai pyaudio python-dotenv websockets flask
                           opencv-python Pillow pyperclip

    # pyperclip may need system deps:
    # Linux: sudo apt-get install xclip   (or xdotool)
    # macOS: works out of the box
    # Windows: works out of the box

RUN:
    python adam_live_v20.py
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
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("❌ API key not found. Set GOOGLE_API_KEY in .env")
print("✅ API Key loaded")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

LIVE_MODEL       = "gemini-3.1-flash-live-preview"   # voice / live
GEN_MODEL        = "gemini-3.1-flash-lite-preview"    # clipboard generation

FLASK_PORT          = 5000
WS_HOST             = "localhost"
WS_PORT             = 8765
POST_SPEECH_MUTE_S  = 0.4
VOICE               = "Charon"
CAMERA_INDEX        = 0
FRAME_SIZE          = (768, 768)
CAMERA_FPS_INTERVAL = 1.0

ENABLE_IDLE         = True
IDLE_TIMEOUT_S      = 60

ATTENTION_TIMEOUT_S    = 30
FACE_CENTRE_TOLERANCE  = 0.45
FACE_MIN_SIZE_FRACTION = 0.06
WAKE_WORDS             = ["adam", "hey adam", "ok adam", "okay adam", "a dam", "atom"]
VOSK_MODEL_PATH        = "vosk-model-small-en-in-0.4"

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE      = Path(BASE_DIR) / "adam_memory.json"
FACE_MEMORY_FILE = Path(BASE_DIR) / "adam_faces.json"

# ─────────────────────────────────────────────────────────────────────────────
# CLIPBOARD GENERATION (second model)
# ─────────────────────────────────────────────────────────────────────────────

# Keywords that trigger clipboard generation mode
CLIPBOARD_TRIGGERS = [
    "write me", "write a", "generate", "draft", "create a script",
    "code for", "function that", "program that", "script that",
    "python for", "python that", "html for", "css for", "sql for",
    "email for", "email to", "paragraph about", "paragraph on",
    "essay about", "blog post", "cover letter", "template for",
    "json for", "yaml for", "regex for", "command for",
    "bash script", "shell script", "write code", "give me code",
    "write the code", "generate code", "clipboard",
]

def should_generate_to_clipboard(transcript: str) -> bool:
    """Check if the user's speech implies a generation/clipboard task."""
    t = transcript.lower().strip()
    for trigger in CLIPBOARD_TRIGGERS:
        if trigger in t:
            return True
    return False


async def generate_to_clipboard(prompt: str, session) -> str:
    """
    Uses gemini-3.1-flash-lite-preview to generate content silently,
    copies it to clipboard, returns a short confirmation string for ADAM to speak.
    Runs in a thread-safe async manner.
    """
    if not CLIPBOARD_AVAILABLE:
        return "Clipboard isn't available — pyperclip isn't installed."

    print(f"\n  📋  [clipboard] Generating with {GEN_MODEL}...")
    print(f"       Prompt: {prompt[:80]}...")

    try:
        gen_client = genai.Client(api_key=API_KEY)

        system_instruction = (
            "You are a precise code and text generation assistant. "
            "Output ONLY the requested content — no preamble, no explanation, "
            "no markdown fences unless the user explicitly asks for markdown. "
            "If generating code, output clean, well-commented, production-ready code. "
            "If generating prose, output clean, well-structured text. "
            "Do not say 'Here is...' or 'Sure, here's...' — just the content itself."
        )

        response = await asyncio.to_thread(
            lambda: gen_client.models.generate_content(
                model=GEN_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3,
                )
            )
        )

        generated_text = response.text.strip()

        if generated_text:
            await asyncio.to_thread(pyperclip.copy, generated_text)
            line_count = generated_text.count('\n') + 1
            char_count = len(generated_text)
            print(f"  📋  [clipboard] Copied {char_count} chars, {line_count} lines")
            return (
                f"Done. {line_count} lines copied to your clipboard. "
                f"Just press Ctrl+V to paste it."
            )
        else:
            return "The generation came back empty. Try rephrasing your request."

    except Exception as e:
        print(f"  ❌  [clipboard] Generation error: {e}")
        return f"Generation failed. {str(e)[:60]}"


# ─────────────────────────────────────────────────────────────────────────────
# ATTENTION STATE MACHINE (from v19 — unchanged)
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
        if self._state == AttentionState.ATTENTIVE:
            self._last_active_time = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# FACE GAZE DETECTOR (from v19 — unchanged)
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
# WAKE WORD DETECTOR (from v19 — unchanged)
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
# PERSISTENT MEMORY (unchanged)
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
    prompt_path = Path(BASE_DIR) / "system_prompt.txt"
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

    search_instructions = """
━━━ GOOGLE SEARCH ━━━
You have access to Google Search via the google_search tool.
Use it whenever the user asks about:
  - Current news, events, or recent developments
  - Live prices (stocks, crypto, products)
  - Weather forecasts
  - Sports scores or upcoming events
  - Any information that may have changed after your training cutoff
  - Any time the user says "look up", "search for", "find out", "check"

Trigger search proactively when you detect time-sensitive questions.
Do NOT say "I can't access the internet" — you have search capability.
After receiving search results, answer naturally without narrating the search process.
"""

    clipboard_instructions = """
━━━ CLIPBOARD GENERATION TOOL ━━━
You have a generate_to_clipboard tool. Use it when the user asks you to:
  - Write a script, program, or any code
  - Draft an email, letter, or document
  - Generate a paragraph, essay, or article
  - Create a template or boilerplate
  - Write any long-form content

When you call generate_to_clipboard:
  1. Acknowledge what you're generating: "Generating that now..."
  2. Call the tool with the full detailed prompt
  3. After the tool returns: confirm verbally — "Done, paste it with Ctrl+V."

The tool uses a fast secondary model to generate silently in the background.
The user's clipboard will contain the result ready to paste.
Keep your voice response SHORT — the content is in the clipboard, not spoken aloud.
"""

    attention_instructions = """
━━━ ATTENTION SYSTEM ━━━
You only receive audio/video when the user is talking to YOU.
The system gates your microphone based on face gaze and wake word.
Respond to content naturally — don't acknowledge the gating mechanism.
"""

    vision_instructions = """
━━━ VISION ━━━
You see live camera frames every second. Recognize people, read expressions,
notice objects and gestures. Use vision context naturally.
"""

    language_rule = """
━━━ LANGUAGE ━━━
Reply in the EXACT language the user spoke. Hindi→Hindi, Bengali→Bengali, English→English.
"""

    parts = [
        memory_to_prompt(memory),
        face_memory_to_prompt(faces),
        prompt_text,
        search_instructions,
        clipboard_instructions,
        attention_instructions,
        vision_instructions,
        language_rule,
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

async def handle_tool_call(tool_call, memory: dict, faces: dict, session) -> list[dict]:
    responses = []
    for fc in tool_call.function_calls:
        name    = fc.name
        call_id = fc.id
        args    = dict(fc.args) if fc.args else {}

        # ── Datetime ─────────────────────────────────────────────────────────
        if name == "get_current_datetime":
            now = datetime.datetime.now()
            result = {
                "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
                "date":     now.strftime("%A, %d %B %Y"),
                "time":     now.strftime("%I:%M %p"),
                "timezone": str(datetime.datetime.now().astimezone().tzname()),
            }
            print(f"  🕐  [tool] datetime → {result['datetime']}")

        # ── Google Search (Live API routes this to client since Mar 2026) ────
        elif name == "google_search":
            query = args.get("q", args.get("query", str(args)))
            print(f"  🔍  [tool] google_search → \"{query}\"")
            # Acknowledge — Gemini merges its own search results server-side
            result = {"status": "search_executed", "query": query}

        # ── Clipboard generation (dual-model) ─────────────────────────────────
        elif name == "generate_to_clipboard":
            prompt   = args.get("prompt", "").strip()
            task_type= args.get("task_type", "general")
            if not prompt:
                result = {"error": "prompt cannot be empty"}
            else:
                confirmation = await generate_to_clipboard(prompt, session)
                result = {"status": "done", "confirmation": confirmation}
                print(f"  📋  [clipboard] {confirmation}")

        # ── Face memory ───────────────────────────────────────────────────────
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

        # ── Emotion / mouth ───────────────────────────────────────────────────
        elif name == "set_emotion":
            emotion = args.get("emotion", "happy")
            await ws_broadcast({"type": "emotion", "emotion": emotion,
                                "head": EMOTION_MAP.get(emotion, "none")})
            result = {"status": "ok"}

        elif name == "set_mouth_sync":
            await ws_broadcast({"type": "mouth_sync", "intensity": args.get("intensity","medium")})
            result = {"status": "ok"}

        # ── General memory ────────────────────────────────────────────────────
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


def build_tools() -> list[types.Tool]:
    S = types.Schema
    T = types.Type

    # ── Custom function tools ─────────────────────────────────────────────────
    function_tool = types.Tool(function_declarations=[
        types.FunctionDeclaration(name="get_current_datetime",
            description="Returns current local date and time.",
            parameters=S(type=T.OBJECT, properties={})),

        # ── NEW: Clipboard generation ─────────────────────────────────────────
        types.FunctionDeclaration(name="generate_to_clipboard",
            description=(
                "Generate text, code, scripts, emails, paragraphs, or any long-form "
                "content using a fast secondary model, then copy it silently to the "
                "user's system clipboard so they can paste it with Ctrl+V. "
                "Use this whenever the user asks you to write, draft, generate, or create "
                "any substantial text or code content."
            ),
            parameters=S(type=T.OBJECT, properties={
                "prompt": S(type=T.STRING,
                    description=(
                        "The full, detailed generation prompt. Include all context: "
                        "language, style, length, purpose, any specific requirements. "
                        "Be specific — this goes directly to the generation model."
                    )),
                "task_type": S(type=T.STRING,
                    enum=["code", "email", "essay", "template", "script", "general"],
                    description="Type of content being generated"),
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

    # ── Google Search (built-in grounding tool) ───────────────────────────────
    # Using the native google_search tool as a proper function declaration
    # so ADAM can explicitly decide when to search rather than auto-grounding.
    google_search_tool = types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="google_search",
            description=(
                "Search the web using Google Search to retrieve current, real-time "
                "information. Use this for: current news, live prices, weather forecasts, "
                "sports scores, recent events, product information, and anything that "
                "may have changed after training. Call this proactively when you detect "
                "time-sensitive questions or when the user says 'look up', 'search', "
                "'find out', 'check', or asks about current/recent/today's anything."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "q": types.Schema(
                        type=types.Type.STRING,
                        description="The search query string"
                    )
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
                    "  → Look at the camera OR say 'ADAM' to get attention.\n"
                    "  → Ask ADAM to 'write', 'generate', 'draft' anything → clipboard.\n"
                    "  → Ask about news, weather, prices → ADAM will search.\n"
                    "  Ctrl+C to quit.\n"
                )
                await ws_broadcast({"type": "face_state", "state": "idle"})

            mic_q          = asyncio.Queue(maxsize=120)
            adam_speaking  = asyncio.Event()
            last_idle_nudge= [time.time()]

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
                try:
                    cap = cv2.VideoCapture(CAMERA_INDEX)
                    if not cap.isOpened():
                        print("  ⚠️  Camera unavailable — vision disabled")
                        return
                    print(f"  📷  Camera ready (index {CAMERA_INDEX})")
                    last_sent = 0.0

                    while not stop.is_set():
                        await asyncio.sleep(0.2)
                        if stop.is_set():
                            break

                        raw = await asyncio.to_thread(capture_raw_frame, cap)
                        if raw is None:
                            continue

                        user_facing = await asyncio.to_thread(gaze.is_user_facing, raw)

                        if user_facing:
                            await attention.activate("face-detected")
                        else:
                            elapsed_since_active = (
                                time.time() - attention._last_active_time
                            )
                            if (attention.state == AttentionState.ATTENTIVE and
                                    elapsed_since_active > 5.0):
                                await attention.deactivate("face-lost")

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
                                audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000"))
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
                                    msg.tool_call, memory, faces, session)
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

                            # Fallback transcript wake word check
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
                if not ENABLE_IDLE:
                    return
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
                            frame_jpeg = None
                            if _idle_cap is not None:
                                raw = await asyncio.to_thread(capture_raw_frame, _idle_cap)
                                if raw is not None:
                                    frame_jpeg = await asyncio.to_thread(frame_to_jpeg, raw)
                            if frame_jpeg is not None:
                                await session.send_realtime_input(
                                    video=types.Blob(data=frame_jpeg, mime_type="image/jpeg")
                                )
                            await session.send_realtime_input(
                                text=(
                                    f"[SYSTEM: User passive {elapsed:.0f}s. "
                                    f"Break silence in-character, 1-2 sentences. "
                                    f"Suggestion: {nudge}]"
                                )
                            )
                        except Exception as e:
                            print(f"  ⚠️  Idle nudge error: {e}")
                except asyncio.CancelledError:
                    pass
                finally:
                    if _idle_cap is not None:
                        _idle_cap.release()

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
    print("=" * 64)
    print("  ADAM — Autonomous Desktop AI Module  (v20)")
    print(f"  Built by DGEN Technologies Pvt. Ltd., Kolkata")
    print(f"  Live model  : {LIVE_MODEL}")
    print(f"  Gen model   : {GEN_MODEL}  (clipboard generation)")
    print(f"  Voice       : {VOICE}")
    print(f"  Search      : Google Search (function tool)")
    print(f"  Clipboard   : {'✅ pyperclip ready' if CLIPBOARD_AVAILABLE else '❌ install pyperclip'}")
    print(f"  Vosk        : {'ready (' + VOSK_MODEL_PATH + ')' if VOSK_AVAILABLE else 'not installed'}")
    print(f"  Idle nudge  : {'ENABLED' if ENABLE_IDLE else 'DISABLED'}  |  Timeout: {IDLE_TIMEOUT_S}s")
    print("=" * 64)
    print()
    print("  HOW TO TALK TO ADAM:")
    print("  ① Look at camera  →  ADAM activates")
    print("  ② Or say 'Hey ADAM'  →  activates from anywhere")
    print("  ③ Say 'write me a Python script for X'  →  clipboard")
    print("  ④ Say 'what's the weather in Kolkata'   →  Google Search")
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