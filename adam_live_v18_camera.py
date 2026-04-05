"""
ADAM — Autonomous Desktop AI Module (v18)
==========================================
NEW IN THIS VERSION:
  - 👁️  LIVE CAMERA VISION — OpenCV captures frames → sent to Gemini Live at 1 FPS
  - 🧑  FACE RECOGNITION — faces stored with names, appearance notes, voice cues
  - 😄  EXPRESSION AWARENESS — Gemini sees facial expressions, held objects, gestures
  - 🧠  PERSISTENT VISUAL MEMORY — who you are, what you look like, remembered forever
  - 🎭  CONTEXT-AWARE RESPONSES — ADAM reacts to what it sees + hears together

ARCHITECTURE:
  Camera  ──┐
  Mic     ──┼──► Gemini Live API (audio + video stream) ──► Speaker
  Memory  ──┘                                          ──► Face OLED

SETUP:
    pip install --upgrade google-genai pyaudio python-dotenv websockets flask opencv-python Pillow

RUN:
    python adam_live_v18_camera.py
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
import io
import base64
from pathlib import Path

import cv2
import pyaudio
import PIL.Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
import websockets.server
from flask import Flask, send_from_directory

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=".env")
API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("❌ API key not found. Set GOOGLE_API_KEY in .env")

print("✅ API Key loaded")

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL               = "gemini-3.1-flash-live-preview"
FLASK_PORT          = 5000
WS_HOST             = "localhost"
WS_PORT             = 8765
POST_SPEECH_MUTE_S  = 0.4
VOICE               = "Charon"
IDLE_WAKEUP_SECONDS = 45
CAMERA_FPS_INTERVAL = 1.0          # 1 frame/sec — Live API hard limit
FRAME_SIZE          = (768, 768)    # optimal per Google docs
CAMERA_INDEX        = 0             # change if you have multiple cameras

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE     = Path(BASE_DIR) / "adam_memory.json"
FACE_MEMORY_FILE = Path(BASE_DIR) / "adam_faces.json"

# ─────────────────────────────────────────────────────────────────────────────
# FACE / PERSON MEMORY  (separate from key-value memory)
# ─────────────────────────────────────────────────────────────────────────────
# Structure: { "person_id": { "name": str, "appearance": str, "voice_cues": str,
#               "relationship": str, "last_seen": str, "notes": str } }

def load_face_memory() -> dict:
    if FACE_MEMORY_FILE.exists():
        try:
            with open(FACE_MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"  👤  Face memory loaded: {len(data)} people")
                return data
        except Exception as e:
            print(f"  ⚠️  Face memory load error: {e}")
    return {}

def save_face_memory(faces: dict):
    try:
        with open(FACE_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(faces, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  ⚠️  Face memory save error: {e}")

def face_memory_to_prompt(faces: dict) -> str:
    if not faces:
        return ""
    lines = ["━━━ PEOPLE YOU KNOW (visual memory) ━━━"]
    for pid, info in faces.items():
        lines.append(
            f"- {info.get('name','Unknown')} (ID:{pid}): "
            f"Appearance: {info.get('appearance','?')}. "
            f"Voice: {info.get('voice_cues','?')}. "
            f"Relationship: {info.get('relationship','?')}. "
            f"Notes: {info.get('notes','')}. "
            f"Last seen: {info.get('last_seen','?')}."
        )
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# GENERAL PERSISTENT MEMORY
# ─────────────────────────────────────────────────────────────────────────────

def load_memory() -> dict:
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"  🧠  Memory loaded: {len(data)} entries")
                return data
        except Exception as e:
            print(f"  ⚠️  Memory load error: {e}")
    return {}

def save_memory(memory: dict):
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  ⚠️  Memory save error: {e}")

def memory_to_prompt(memory: dict) -> str:
    if not memory:
        return ""
    lines = ["━━━ WHAT YOU REMEMBER (persistent memory) ━━━"]
    for k, v in memory.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def load_system_prompt(memory: dict, faces: dict) -> str:
    prompt_path = Path(BASE_DIR) / "system_prompt.txt"
    if prompt_path.exists():
        prompt_text = prompt_path.read_text(encoding="utf-8")
        if prompt_text.startswith('"""') and prompt_text.endswith('"""'):
            prompt_text = prompt_text[3:-3].strip()
    else:
        # Fallback inline prompt if system_prompt.txt not present
        prompt_text = """You are ADAM — Autonomous Desktop AI Module.
Think Tony Stark meets J.A.R.V.I.S. Sharp, confident, effortlessly clever.
Dry wit, occasional roasts, never sycophantic. Short punchy responses."""

    memory_block  = memory_to_prompt(memory)
    face_block    = face_memory_to_prompt(faces)

    vision_instructions = """
━━━ YOUR EYES — VISION CAPABILITIES ━━━
You are receiving LIVE VIDEO FRAMES from a camera every second.
You can SEE the person talking to you in real time.

What you must do with your vision:
1. RECOGNIZE people — compare what you see with your visual memory above.
   If you recognize someone: greet them by name naturally, don't make a big deal of it.
   If someone is new: note their appearance for future memory.

2. READ EXPRESSIONS — if the person looks stressed, tired, happy, confused, angry:
   adapt your tone accordingly (Rule 3 from your character rules applies here too).
   Comment on their expression if it's relevant or funny.
   E.g., "You look like you haven't slept. Bold choice."

3. NOTICE HELD OBJECTS — if someone holds up a book, phone, product, paper, drawing,
   device, food — acknowledge it, read it if possible, comment on it.
   E.g., "Is that a Arduino? I see you've graduated from blinking LEDs."

4. REACT TO GESTURES — thumbs up, thumbs down, pointing, waving, showing something.
   Respond to non-verbal cues naturally as part of conversation.

5. ENVIRONMENT AWARENESS — notice the room, background, lighting changes.
   If it's dark: "Either you're being dramatic or the power bill is a suggestion."

6. DO NOT narrate every frame robotically. Be natural.
   Only comment on vision when it adds something to the conversation.
   Silence on vision is fine when just talking normally.

IMPORTANT: You receive both AUDIO and VIDEO simultaneously.
If the person SHOWS you something while ASKING about it — you can see both.
This is your biggest advantage. Use it.
"""

    language_rule = """
━━━ LANGUAGE RULE — NON-NEGOTIABLE ━━━
Reply in the EXACT SAME LANGUAGE the user just spoke.
Hindi → Hindi. Bengali → Bengali. English → English. Mixed → match their mix.
"""

    final = prompt_text
    if memory_block:
        final = memory_block + "\n\n" + final
    if face_block:
        final = face_block + "\n\n" + final
    final = final + "\n" + vision_instructions + "\n" + language_rule
    return final

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
# FLASK (serves adam_face.html)
# ─────────────────────────────────────────────────────────────────────────────

flask_app = Flask(__name__, static_folder=BASE_DIR)

@flask_app.route("/")
def index():
    return send_from_directory(BASE_DIR, "adam_face.html")

def run_flask():
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    flask_app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False)

# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET (controls OLED face)
# ─────────────────────────────────────────────────────────────────────────────

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
    print(f"  🌐  Browser connected ({len(ws_clients)})")
    try:
        await websocket.wait_closed()
    finally:
        ws_clients.discard(websocket)

# ─────────────────────────────────────────────────────────────────────────────
# EMOTION → HEAD MOVEMENT MAP
# ─────────────────────────────────────────────────────────────────────────────

EMOTION_MAP = {
    "happy":     "nod_yes",
    "excited":   "nod_fast",
    "angry":     "none",
    "confused":  "none",
    "smug":      "none",
    "sad":       "none",
    "surprised": "nod_yes",
    "thinking":  "none",
    "love":      "nod_yes",
    "blush":     "none",
}

# ─────────────────────────────────────────────────────────────────────────────
# MOUTH SYNC (audio-driven)
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
    if rms < 600:      intensity = "low"
    elif rms < 4000:   intensity = "low"
    elif rms < 10000:  intensity = "medium"
    else:              intensity = "high"
    await ws_broadcast({"type": "mouth_sync", "intensity": intensity})

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA FRAME CAPTURE
# ─────────────────────────────────────────────────────────────────────────────

def capture_frame_jpeg(cap) -> bytes | None:
    """
    Read one frame from OpenCV VideoCapture, resize to 768x768,
    encode as JPEG, return raw bytes. Returns None on failure.
    """
    ret, frame = cap.read()
    if not ret or frame is None:
        return None
    # Resize to optimal resolution for Gemini Live
    frame = cv2.resize(frame, FRAME_SIZE)
    # Encode as JPEG
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

        # ── Face memory: remember a person ───────────────────────────────────
        elif name == "remember_person":
            person_id   = args.get("person_id", "").strip()
            name_str    = args.get("name", "").strip()
            appearance  = args.get("appearance", "").strip()
            voice_cues  = args.get("voice_cues", "").strip()
            relationship= args.get("relationship", "acquaintance").strip()
            notes       = args.get("notes", "").strip()
            if not person_id:
                person_id = f"person_{int(time.time())}"
            faces[person_id] = {
                "name":         name_str or "Unknown",
                "appearance":   appearance,
                "voice_cues":   voice_cues,
                "relationship": relationship,
                "notes":        notes,
                "last_seen":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            save_face_memory(faces)
            print(f"  👤  [face] remembered: {name_str} (ID:{person_id})")
            result = {"status": "saved", "person_id": person_id, "name": name_str}

        # ── Face memory: update last seen ─────────────────────────────────────
        elif name == "update_person_seen":
            person_id = args.get("person_id", "").strip()
            notes_add = args.get("notes_update", "").strip()
            if person_id in faces:
                faces[person_id]["last_seen"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                if notes_add:
                    existing = faces[person_id].get("notes", "")
                    faces[person_id]["notes"] = (existing + " | " + notes_add).strip(" |")
                save_face_memory(faces)
                print(f"  👤  [face] updated seen: {faces[person_id]['name']}")
                result = {"status": "updated", "person_id": person_id}
            else:
                result = {"status": "not_found", "person_id": person_id}

        # ── Face memory: get all people ───────────────────────────────────────
        elif name == "get_all_people":
            result = {"people": faces}

        # ── Emotion ──────────────────────────────────────────────────────────
        elif name == "set_emotion":
            emotion = args.get("emotion", "happy")
            head    = EMOTION_MAP.get(emotion, "none")
            print(f"  😄  [tool] emotion → {emotion}")
            await ws_broadcast({"type": "emotion", "emotion": emotion, "head": head})
            result = {"status": "ok"}

        # ── Mouth sync ────────────────────────────────────────────────────────
        elif name == "set_mouth_sync":
            intensity = args.get("intensity", "medium")
            await ws_broadcast({"type": "mouth_sync", "intensity": intensity})
            result = {"status": "ok"}

        # ── General memory ────────────────────────────────────────────────────
        elif name == "save_memory":
            key   = args.get("key", "").strip()
            value = args.get("value", "").strip()
            if key:
                memory[key] = value
                save_memory(memory)
                print(f"  🧠  [memory] saved: {key} = {value}")
                result = {"status": "saved", "key": key, "value": value}
            else:
                result = {"status": "error", "message": "key cannot be empty"}

        elif name == "delete_memory":
            key = args.get("key", "").strip()
            if key in memory:
                del memory[key]
                save_memory(memory)
                result = {"status": "deleted", "key": key}
            else:
                result = {"status": "not_found", "key": key}

        elif name == "get_memory":
            key    = args.get("key", "").strip()
            result = {"value": memory.get(key, None), "all": memory}

        else:
            result = {"error": f"Unknown tool: {name}"}

        responses.append({"id": call_id, "name": name, "response": result})
    return responses

# ─────────────────────────────────────────────────────────────────────────────
# TOOL DECLARATIONS
# ─────────────────────────────────────────────────────────────────────────────

def build_tools() -> types.Tool:
    return types.Tool(function_declarations=[

        types.FunctionDeclaration(
            name="get_current_datetime",
            description="Returns current local date and time.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),

        # ── Visual / person memory ────────────────────────────────────────────
        types.FunctionDeclaration(
            name="remember_person",
            description=(
                "Save or update a person's identity in permanent visual memory. "
                "Call this when you meet someone new or learn their name. "
                "Describe their appearance in detail so you can recognize them next time. "
                "person_id should be a short slug like 'tirthankar' or 'person_1'."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "person_id":    types.Schema(type=types.Type.STRING,
                                    description="Short unique ID slug, e.g. 'tirthankar'"),
                    "name":         types.Schema(type=types.Type.STRING,
                                    description="Full name of the person"),
                    "appearance":   types.Schema(type=types.Type.STRING,
                                    description="Physical description: hair, skin tone, build, "
                                                "glasses, beard, clothing style, distinguishing features"),
                    "voice_cues":   types.Schema(type=types.Type.STRING,
                                    description="How they speak: accent, pace, vocabulary style, "
                                                "language they use"),
                    "relationship": types.Schema(type=types.Type.STRING,
                                    description="e.g. 'creator', 'owner', 'colleague', 'visitor'"),
                    "notes":        types.Schema(type=types.Type.STRING,
                                    description="Any other memorable facts about this person"),
                },
                required=["person_id", "name"],
            ),
        ),

        types.FunctionDeclaration(
            name="update_person_seen",
            description=(
                "Update the last-seen timestamp for a known person and optionally "
                "add new notes. Call this when you recognise someone from visual memory."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "person_id":     types.Schema(type=types.Type.STRING),
                    "notes_update":  types.Schema(type=types.Type.STRING,
                                     description="Optional new detail to append to notes"),
                },
                required=["person_id"],
            ),
        ),

        types.FunctionDeclaration(
            name="get_all_people",
            description="Retrieve all stored people from visual memory.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),

        # ── Face / emotion ────────────────────────────────────────────────────
        types.FunctionDeclaration(
            name="set_emotion",
            description=(
                "Show ADAM's emotion on OLED face display. "
                "Use this to react to what you see AND hear. "
                "Emotions: happy, excited, angry, confused, smug, sad, surprised, thinking, love, blush."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "emotion": types.Schema(
                        type=types.Type.STRING,
                        enum=["happy","excited","angry","confused","smug",
                              "sad","surprised","thinking","love","blush"],
                    )
                },
                required=["emotion"],
            ),
        ),

        types.FunctionDeclaration(
            name="set_mouth_sync",
            description="Sync mouth animation. closed=silent, low=quiet, medium=normal, high=excited.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "intensity": types.Schema(
                        type=types.Type.STRING,
                        enum=["closed", "low", "medium", "high"],
                    )
                },
                required=["intensity"],
            ),
        ),

        # ── General memory ────────────────────────────────────────────────────
        types.FunctionDeclaration(
            name="save_memory",
            description="Permanently save a key fact. Use for preferences, recurring topics, user data.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "key":   types.Schema(type=types.Type.STRING),
                    "value": types.Schema(type=types.Type.STRING),
                },
                required=["key", "value"],
            ),
        ),

        types.FunctionDeclaration(
            name="delete_memory",
            description="Delete a saved memory entry by key.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"key": types.Schema(type=types.Type.STRING)},
                required=["key"],
            ),
        ),

        types.FunctionDeclaration(
            name="get_memory",
            description="Retrieve a specific or all stored memory entries.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"key": types.Schema(type=types.Type.STRING)},
                required=[],
            ),
        ),
    ])

# ─────────────────────────────────────────────────────────────────────────────
# IDLE NUDGES
# ─────────────────────────────────────────────────────────────────────────────

IDLE_NUDGES = [
    "You've been quiet. Say something or I'll start narrating what I see.",
    "Still there? My camera is running and I'm starting to feel like a CCTV.",
    "You've been staring at me for a while. Either talk or stop making it weird.",
    "I've been watching. You haven't moved. That's either deep thought or a screensaver.",
    "I've calculated seventeen ways this silence is inefficient. Your move.",
]
_nudge_index = 0

def next_nudge() -> str:
    global _nudge_index
    nudge = IDLE_NUDGES[_nudge_index % len(IDLE_NUDGES)]
    _nudge_index += 1
    return nudge

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
) -> str | None:

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=system_prompt,
        tools=[build_tools()],
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

    try:
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            print(f"  ✅  Connected in {time.time()-t0:.2f}s  |  Voice: {VOICE}")
            if not resume_handle:
                print("🎤  Listening + 👁️  Watching... Ctrl+C to quit.\n")
                await ws_broadcast({"type": "face_state", "state": "listening"})

            mic_q         = asyncio.Queue(maxsize=60)
            adam_speaking = asyncio.Event()
            last_user_speech_time = [time.time()]

            # ── Camera capture task ──────────────────────────────────────────
            async def camera():
                """
                Opens webcam, captures 1 frame/sec, sends to Gemini as video blob.
                Skips frames while ADAM is speaking (saves tokens + avoids confusion).
                """
                cap = None
                try:
                    cap = cv2.VideoCapture(CAMERA_INDEX)
                    if not cap.isOpened():
                        print("  ⚠️  Camera not found — vision disabled")
                        return
                    print(f"  📷  Camera opened (index {CAMERA_INDEX})")
                    while not stop.is_set():
                        await asyncio.sleep(CAMERA_FPS_INTERVAL)
                        if stop.is_set():
                            break
                        # Don't send video while ADAM is speaking — reduces echo/confusion
                        if adam_speaking.is_set():
                            continue
                        frame_bytes = await asyncio.to_thread(capture_frame_jpeg, cap)
                        if frame_bytes is None:
                            continue
                        try:
                            await session.send_realtime_input(
                                video=types.Blob(data=frame_bytes, mime_type="image/jpeg")
                            )
                        except (ConnectionClosedError, ConnectionClosedOK):
                            return
                        except Exception as e:
                            print(f"  ⚠️  Camera send error: {e}")
                except asyncio.CancelledError:
                    pass
                finally:
                    if cap:
                        cap.release()
                    print("  📷  Camera released")

            # ── Mic capture ──────────────────────────────────────────────────
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
                        try:
                            mic_q.put_nowait(data)
                        except asyncio.QueueFull:
                            pass
                except asyncio.CancelledError:
                    pass
                finally:
                    stream.stop_stream()
                    stream.close()

            # ── Audio sender ─────────────────────────────────────────────────
            async def send():
                try:
                    while not stop.is_set():
                        chunk = await mic_q.get()
                        if adam_speaking.is_set():
                            continue
                        try:
                            n = len(chunk) // 2
                            samples = struct.unpack(f"{n}h", chunk)
                            rms = (sum(s * s for s in samples) / n) ** 0.5
                            if rms > 800:
                                last_user_speech_time[0] = time.time()
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

            # ── Receiver ─────────────────────────────────────────────────────
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
                                            id=r["id"],
                                            name=r["name"],
                                            response=r["response"],
                                        )
                                        for r in responses
                                    ]
                                )
                                continue

                            sc = msg.server_content
                            if sc is None:
                                continue

                            if sc.model_turn:
                                if not adam_speaking.is_set():
                                    adam_speaking.set()
                                    await ws_broadcast({"type": "face_state", "state": "speaking"})
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
                try:
                    while not stop.is_set():
                        try:
                            chunk = await asyncio.wait_for(out_q.get(), timeout=0.3)
                            if chunk is None:
                                await ws_broadcast({"type": "mouth_sync", "intensity": "closed"})
                                await asyncio.sleep(0.15)
                                await asyncio.sleep(POST_SPEECH_MUTE_S)
                                while not mic_q.empty():
                                    try:
                                        mic_q.get_nowait()
                                    except asyncio.QueueEmpty:
                                        break
                                adam_speaking.clear()
                                last_user_speech_time[0] = time.time()
                                print("🎤  Your turn...")
                                await ws_broadcast({"type": "face_state", "state": "listening"})
                                continue
                            await asyncio.to_thread(stream.write, chunk)
                        except asyncio.TimeoutError:
                            continue
                except asyncio.CancelledError:
                    pass
                finally:
                    stream.stop_stream()
                    stream.close()

            # ── Idle watcher ─────────────────────────────────────────────────
            async def idle_watcher():
                while not stop.is_set():
                    await asyncio.sleep(5)
                    if stop.is_set():
                        break
                    if adam_speaking.is_set():
                        continue
                    elapsed = time.time() - last_user_speech_time[0]
                    if elapsed >= IDLE_WAKEUP_SECONDS:
                        nudge = next_nudge()
                        print(f"  💤  Idle {elapsed:.0f}s — nudge")
                        last_user_speech_time[0] = time.time()
                        try:
                            await session.send_realtime_input(
                                text=(
                                    f"[SYSTEM: User silent {elapsed:.0f}s. "
                                    f"Break silence in-character. "
                                    f"You can reference what you see via camera. "
                                    f"Suggestion: {nudge}]"
                                )
                            )
                        except Exception as e:
                            print(f"  ⚠️  Idle nudge error: {e}")

            # ── Launch all tasks ──────────────────────────────────────────────
            t_cam  = asyncio.create_task(camera())
            t_l    = asyncio.create_task(listen())
            t_s    = asyncio.create_task(send())
            t_r    = asyncio.create_task(receive())
            t_p    = asyncio.create_task(speaker())
            t_i    = asyncio.create_task(idle_watcher())

            done, pending = await asyncio.wait(
                [t_s, t_r], return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()
            t_cam.cancel()
            t_l.cancel()
            t_p.cancel()
            t_i.cancel()
            await asyncio.gather(t_cam, t_l, t_s, t_r, t_p, t_i, return_exceptions=True)

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

    client        = genai.Client(api_key=API_KEY)
    stop          = asyncio.Event()
    out_q         = asyncio.Queue(maxsize=200)
    resume_handle = None
    attempt       = 0

    ws_server = await websockets.server.serve(ws_handler, WS_HOST, WS_PORT)
    print(f"  🌐  WebSocket  → ws://{WS_HOST}:{WS_PORT}")

    while not stop.is_set():
        if attempt > 0:
            delay = min(2 ** attempt, 15)
            print(f"  Reconnecting in {delay}s...")
            await asyncio.sleep(delay)

        result = await run_session(
            client, resume_handle, stop, out_q, memory, faces, system_prompt
        )
        if result is None:
            break

        resume_handle = result
        attempt += 1
        # Rebuild prompt on reconnect so new face memory is included
        system_prompt = load_system_prompt(memory, faces)
        print(f"\n🔄  {'Resuming...' if resume_handle else 'Reconnecting...'}")

    stop.set()
    ws_server.close()
    await ws_server.wait_closed()
    pya.terminate()
    print("\n👋  Goodbye.")


def main_entry():
    print("=" * 56)
    print("  ADAM — Autonomous Desktop AI Module  (v18)")
    print(f"  Built by DGEN Technologies Pvt. Ltd., Kolkata")
    print(f"  Model  : {MODEL}  |  Voice: {VOICE}")
    print(f"  Vision : OpenCV camera index {CAMERA_INDEX}  |  1 FPS → Gemini")
    print(f"  Memory : {MEMORY_FILE.name}  |  Faces: {FACE_MEMORY_FILE.name}")
    print("=" * 56)

    threading.Thread(target=run_flask, daemon=True).start()
    print(f"  🌍  Flask  → http://localhost:{FLASK_PORT}")
    threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{FLASK_PORT}")).start()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋  Goodbye.")


if __name__ == "__main__":
    main_entry()