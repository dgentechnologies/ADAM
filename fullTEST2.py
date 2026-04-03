"""
ADAM — Autonomous Desktop AI Module (v13 — with creator info)
==============================================================
Updates:
  - Full Dgen Technologies company info injected into system prompt
  - Tirthankar Dasgupta (CEO & CTO) credited as ADAM's creator
  - All personality rules, multilingual, mouth sync retained

SETUP:
    pip install --upgrade google-genai pyaudio python-dotenv websockets flask

RUN:
    python adam_v13.py
"""

import asyncio
import os
import sys
import time
import datetime
import json
import threading
import webbrowser
from pathlib import Path
import pyaudio
import struct
from dotenv import load_dotenv
from google import genai
from google.genai import types
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
import websockets.server
from flask import Flask, send_from_directory

load_dotenv()

API_KEY = (
    os.environ.get("GOOGLE_API_KEY")
    or os.environ.get("GEMINI_API_KEY")
    or "PASTE_YOUR_KEY_HERE"
)

MODEL              = "gemini-3.1-flash-live-preview"
FLASK_PORT         = 5000
WS_HOST            = "localhost"
WS_PORT            = 8765
POST_SPEECH_MUTE_S = 0.4
VOICE              = "Charon"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_system_prompt() -> str:
    prompt_path = Path(BASE_DIR) / "SYSTEM_PROMPT.txt"
    prompt_text = prompt_path.read_text(encoding="utf-8")
    if prompt_text.startswith('"""') and prompt_text.endswith('"""'):
        return prompt_text[3:-3].strip()
    return prompt_text


SYSTEM_PROMPT = load_system_prompt()

FORMAT           = pyaudio.paInt16
CHANNELS         = 1
SEND_SAMPLE_RATE = 16000
RECV_SAMPLE_RATE = 24000
CHUNK_SIZE       = 512

pya = pyaudio.PyAudio()

# ── Flask ─────────────────────────────────────────────────────────────────────

flask_app = Flask(__name__, static_folder=BASE_DIR)

@flask_app.route("/")
def index():
    return send_from_directory(BASE_DIR, "adam_face.html")

def run_flask():
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    flask_app.run(host="0.0.0.0", port=FLASK_PORT, debug=False, use_reloader=False)

# ── WebSocket ─────────────────────────────────────────────────────────────────

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

# ── Emotion → head movement ───────────────────────────────────────────────────

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

# ── Mouth sync from audio energy ─────────────────────────────────────────────

_last_sync_time = 0.0
_sync_interval  = 0.08   # max one sync broadcast per 80ms

async def maybe_sync_mouth(audio_chunk: bytes):
    global _last_sync_time
    now = time.time()
    if now - _last_sync_time < _sync_interval:
        return
    _last_sync_time = now
    samples = struct.unpack(f"{len(audio_chunk)//2}h", audio_chunk)
    if not samples:
        return
    rms = (sum(s*s for s in samples) / len(samples)) ** 0.5
    if rms < 800:       intensity = "closed"
    elif rms < 4000:    intensity = "low"
    elif rms < 12000:   intensity = "medium"
    else:               intensity = "high"
    await ws_broadcast({"type": "mouth_sync", "intensity": intensity})

# ── Tool handler ──────────────────────────────────────────────────────────────

async def handle_tool_call(tool_call) -> list[dict]:
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

        else:
            result = {"error": f"Unknown: {name}"}

        responses.append({"id": call_id, "name": name, "response": result})
    return responses

# ── Session runner ────────────────────────────────────────────────────────────

async def run_session(
    client:        genai.Client,
    resume_handle: str | None,
    stop:          asyncio.Event,
    out_q:         asyncio.Queue,
) -> str | None:

    tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name="get_current_datetime",
                description="Returns current local date and time.",
                parameters=types.Schema(type=types.Type.OBJECT, properties={}),
            ),
            types.FunctionDeclaration(
                name="set_emotion",
                description=(
                    "Show ADAM's emotion on his OLED face. "
                    "happy, excited, angry, confused, smug, sad, surprised, "
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
        ])
    ]

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=SYSTEM_PROMPT,
        tools=tools,
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

            async def listen():
                stream = pya.open(format=FORMAT, channels=CHANNELS,
                                  rate=SEND_SAMPLE_RATE, input=True,
                                  frames_per_buffer=CHUNK_SIZE)
                try:
                    while not stop.is_set():
                        data = await asyncio.to_thread(
                            stream.read, CHUNK_SIZE, exception_on_overflow=False)
                        try: mic_q.put_nowait(data)
                        except asyncio.QueueFull: pass
                except asyncio.CancelledError: pass
                finally: stream.stop_stream(); stream.close()

            async def send():
                try:
                    while not stop.is_set():
                        chunk = await mic_q.get()
                        if adam_speaking.is_set(): continue
                        try:
                            await session.send_realtime_input(
                                audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000"))
                        except (ConnectionClosedError, ConnectionClosedOK): return
                        except Exception: await asyncio.sleep(0.01)
                except asyncio.CancelledError: pass

            async def receive():
                nonlocal latest_handle
                try:
                    while not stop.is_set():
                        async for msg in session.receive():
                            if stop.is_set(): break
                            if msg.session_resumption_update:
                                upd = msg.session_resumption_update
                                if upd.resumable and upd.new_handle:
                                    latest_handle = upd.new_handle
                            if hasattr(msg, "go_away") and msg.go_away:
                                print("\n  ⚡ GoAway — resuming...")
                                return
                            if msg.tool_call:
                                responses = await handle_tool_call(msg.tool_call)
                                await session.send_tool_response(
                                    function_responses=[
                                        types.FunctionResponse(
                                            id=r["id"], name=r["name"], response=r["response"])
                                        for r in responses
                                    ]
                                )
                                continue
                            sc = msg.server_content
                            if sc is None: continue
                            if sc.model_turn:
                                adam_speaking.set()
                                await ws_broadcast({"type": "face_state", "state": "speaking"})
                                for part in sc.model_turn.parts:
                                    if part.inline_data and part.inline_data.data:
                                        audio_data = part.inline_data.data
                                        await out_q.put(audio_data)
                                        await maybe_sync_mouth(audio_data)
                                    if hasattr(part, "text") and part.text:
                                        print(f"🤖  ADAM: {part.text}")
                            if sc.turn_complete:
                                await out_q.put(None)
                                await ws_broadcast({"type": "mouth_sync", "intensity": "closed"})
                                print("─" * 40)
                except (ConnectionClosedError, ConnectionClosedOK): pass
                except asyncio.CancelledError: pass
                except Exception as e:
                    print(f"\n⚠️  Receive: {type(e).__name__}: {e}")

            async def speaker():
                stream = pya.open(format=FORMAT, channels=CHANNELS,
                                  rate=RECV_SAMPLE_RATE, output=True)
                try:
                    while not stop.is_set():
                        try:
                            chunk = await asyncio.wait_for(out_q.get(), timeout=0.3)
                            if chunk is None:
                                await asyncio.sleep(0.15)
                                await asyncio.sleep(POST_SPEECH_MUTE_S)
                                while not mic_q.empty():
                                    try: mic_q.get_nowait()
                                    except asyncio.QueueEmpty: break
                                adam_speaking.clear()
                                print("🎤  Your turn...")
                                await ws_broadcast({"type": "face_state", "state": "listening"})
                                continue
                            await asyncio.to_thread(stream.write, chunk)
                        except asyncio.TimeoutError: continue
                except asyncio.CancelledError: pass
                finally: stream.stop_stream(); stream.close()

            t_l = asyncio.create_task(listen())
            t_s = asyncio.create_task(send())
            t_r = asyncio.create_task(receive())
            t_p = asyncio.create_task(speaker())

            done, pending = await asyncio.wait([t_s, t_r], return_when=asyncio.FIRST_COMPLETED)
            for t in pending: t.cancel()
            t_l.cancel(); t_p.cancel()
            await asyncio.gather(t_l, t_s, t_r, t_p, return_exceptions=True)

    except (ConnectionClosedError, ConnectionClosedOK): pass
    except Exception as e:
        print(f"\n⚠️  Connection: {type(e).__name__}: {e}")

    if stop.is_set(): return None
    return latest_handle

# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
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
        result = await run_session(client, resume_handle, stop, out_q)
        if result is None: break
        resume_handle = result
        attempt += 1
        print(f"\n🔄  {'Resuming...' if resume_handle else 'Reconnecting...'}")

    stop.set()
    ws_server.close()
    await ws_server.wait_closed()
    pya.terminate()
    print("\n👋  Goodbye.")


def main_entry():
    if API_KEY == "PASTE_YOUR_KEY_HERE":
        print("❌  No API key.  $env:GOOGLE_API_KEY = 'your_key'")
        sys.exit(1)
    print("=" * 52)
    print("  ADAM — Autonomous Desktop AI Module  (v13)")
    print(f"  Built by DGEN Technologies Pvt. Ltd., Kolkata")
    print(f"  Model : {MODEL}  |  Voice: {VOICE}")
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