"""
ADAM — Autonomous Desktop AI Module (v17)
==========================================
Changes from v16:
  - REMOVED: DuckDuckGo web_search tool (was unreliable)

SETUP:
    pip install --upgrade google-genai pyaudio python-dotenv websockets flask

RUN:
    python fullTEST2.py
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
from pathlib import Path
import pyaudio
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
    raise ValueError(
        "❌ API key not found. Please set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file"
    )

print("✅ API Key loaded successfully")

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL               = "gemini-3.1-flash-live-preview"
FLASK_PORT          = 5000
WS_HOST             = "localhost"
WS_PORT             = 8765
POST_SPEECH_MUTE_S  = 0.4
VOICE               = "Charon"
IDLE_WAKEUP_SECONDS = 45

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = Path(BASE_DIR) / "adam_memory.json"

# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENT MEMORY
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
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

def load_system_prompt(memory: dict) -> str:
    prompt_path = Path(BASE_DIR) / "SYSTEM_PROMPT.txt"
    prompt_text = prompt_path.read_text(encoding="utf-8")
    if prompt_text.startswith('"""') and prompt_text.endswith('"""'):
        prompt_text = prompt_text[3:-3].strip()

    memory_block = memory_to_prompt(memory)

    multilingual_enforcement = """
━━━ LANGUAGE RULE — CRITICAL, NON-NEGOTIABLE ━━━
You MUST reply in the EXACT SAME LANGUAGE the user just spoke.
- If the user speaks Hindi → reply 100% in Hindi (Devanagari or Roman, match their style).
- If the user speaks Bengali → reply in Bengali.
- If the user speaks English → reply in English.
- If the user mixes languages → match their mix.
- NEVER reply in a different language than the one the user used in their LAST message.
- This rule overrides everything else. No exceptions.
"""

    final_prompt = prompt_text
    if memory_block:
        final_prompt = memory_block + "\n\n" + final_prompt
    final_prompt = final_prompt + "\n" + multilingual_enforcement
    return final_prompt

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
# FLASK
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
# WEBSOCKET
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

    if rms < 600:      intensity = "low"
    elif rms < 4000:   intensity = "low"
    elif rms < 10000:  intensity = "medium"
    else:              intensity = "high"

    await ws_broadcast({"type": "mouth_sync", "intensity": intensity})

# ─────────────────────────────────────────────────────────────────────────────
# TOOL HANDLER
# ─────────────────────────────────────────────────────────────────────────────

async def handle_tool_call(tool_call, memory: dict) -> list[dict]:
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

        elif name == "set_emotion":
            emotion = args.get("emotion", "happy")
            head    = EMOTION_MAP.get(emotion, "none")
            print(f"  😄  [tool] emotion → {emotion}")
            await ws_broadcast({"type": "emotion", "emotion": emotion, "head": head})
            result = {"status": "ok"}

        elif name == "set_mouth_sync":
            intensity = args.get("intensity", "medium")
            await ws_broadcast({"type": "mouth_sync", "intensity": intensity})
            result = {"status": "ok"}

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
                print(f"  🧠  [memory] deleted: {key}")
                result = {"status": "deleted", "key": key}
            else:
                result = {"status": "not_found", "key": key}

        elif name == "get_memory":
            key    = args.get("key", "").strip()
            result = {"value": memory.get(key, None), "all": memory}

        else:
            result = {"error": f"Unknown: {name}"}

        responses.append({"id": call_id, "name": name, "response": result})
    return responses

# ─────────────────────────────────────────────────────────────────────────────
# IDLE NUDGES
# ─────────────────────────────────────────────────────────────────────────────

IDLE_NUDGES = [
    "You've been quiet. Say something or I'll start optimizing myself out of boredom.",
    "Still there? My processing power is going to waste, you know.",
    "Idle for too long. Either talk to me or admit you don't actually need me.",
    "My sensors are running. My brain is running. You are... not running apparently.",
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
    system_prompt: str,
) -> str | None:

    function_tool = types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="get_current_datetime",
            description="Returns current local date and time.",
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
        types.FunctionDeclaration(
            name="set_emotion",
            description=(
                "Show ADAM's emotion on his OLED face. "
                "Emotions: happy, excited, angry, confused, smug, sad, surprised, "
                "thinking, love, blush."
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
            description=(
                "Sync mouth animation to speech intensity. "
                "closed=silent, low=quiet, medium=normal, high=loud/excited."
            ),
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
        types.FunctionDeclaration(
            name="save_memory",
            description=(
                "Permanently save a key fact about the user or context. "
                "Use when the user tells you their name, preferences, or any info "
                "worth remembering across sessions."
            ),
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
            description="Delete a previously saved memory entry by key.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"key": types.Schema(type=types.Type.STRING)},
                required=["key"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_memory",
            description="Retrieve a specific memory entry or all stored memories.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={"key": types.Schema(type=types.Type.STRING)},
                required=[],
            ),
        ),
    ])

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=system_prompt,
        tools=[function_tool],
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
                print("🎤  Listening... Ctrl+C to quit.\n")
                await ws_broadcast({"type": "face_state", "state": "listening"})

            mic_q         = asyncio.Queue(maxsize=60)
            adam_speaking = asyncio.Event()
            last_user_speech_time = [time.time()]

            # ── listen ───────────────────────────────────────────────────
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

            # ── send ─────────────────────────────────────────────────────
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

            # ── receive ──────────────────────────────────────────────────
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
                                responses = await handle_tool_call(msg.tool_call, memory)
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

            # ── speaker ──────────────────────────────────────────────────
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

            # ── idle watcher ─────────────────────────────────────────────
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
                        print(f"  💤  Idle {elapsed:.0f}s — sending wakeup nudge")
                        last_user_speech_time[0] = time.time()
                        try:
                            nudge_text = (
                                f"[SYSTEM: The user has been silent for {elapsed:.0f}s. "
                                f"Break the silence with a short, in-character unprompted "
                                f"remark. Suggestion: {nudge}]"
                            )
                            await session.send_realtime_input(text=nudge_text)
                        except Exception as e:
                            print(f"  ⚠️  Idle nudge error: {e}")

            # ── Run all tasks ────────────────────────────────────────────
            t_l = asyncio.create_task(listen())
            t_s = asyncio.create_task(send())
            t_r = asyncio.create_task(receive())
            t_p = asyncio.create_task(speaker())
            t_i = asyncio.create_task(idle_watcher())

            done, pending = await asyncio.wait(
                [t_s, t_r], return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()
            t_l.cancel()
            t_p.cancel()
            t_i.cancel()
            await asyncio.gather(t_l, t_s, t_r, t_p, t_i, return_exceptions=True)

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
    system_prompt = load_system_prompt(memory)

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
            client, resume_handle, stop, out_q, memory, system_prompt
        )
        if result is None:
            break

        resume_handle = result
        attempt += 1
        system_prompt = load_system_prompt(memory)
        print(f"\n🔄  {'Resuming...' if resume_handle else 'Reconnecting...'}")

    stop.set()
    ws_server.close()
    await ws_server.wait_closed()
    pya.terminate()
    print("\n👋  Goodbye.")


def main_entry():
    print("=" * 52)
    print("  ADAM — Autonomous Desktop AI Module  (v17)")
    print(f"  Built by DGEN Technologies Pvt. Ltd., Kolkata")
    print(f"  Model : {MODEL}  |  Voice: {VOICE}")
    print(f"  Idle wakeup: {IDLE_WAKEUP_SECONDS}s")
    print("=" * 52)

    threading.Thread(target=run_flask, daemon=True).start()
    print(f"  🌍  Flask      → http://localhost:{FLASK_PORT}")
    threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{FLASK_PORT}")).start()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋  Goodbye.")


if __name__ == "__main__":
    main_entry()