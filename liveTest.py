"""
ADAM - Gemini Live API (v9)
============================
Updates in this version:
  1. API key priority: GOOGLE_API_KEY first, GEMINI_API_KEY second
  2. Real-time data: Google Search grounding + get_current_datetime tool
  3. Personality: Tony Stark / Marvel-style wit, occasional roasts
  4. Default voice: Charon

SETUP:
    pip install --upgrade google-genai pyaudio python-dotenv

API KEY (PowerShell):
    $env:GOOGLE_API_KEY = "your_key_here"
"""

import asyncio
import json
import os
import sys
import time
import datetime
import pyaudio
from dotenv import load_dotenv
from google import genai
from google.genai import types
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()

# 1. GOOGLE_API_KEY first, GEMINI_API_KEY second
API_KEY = (
    os.environ.get("GOOGLE_API_KEY")
    or os.environ.get("GEMINI_API_KEY")
    or "PASTE_YOUR_KEY_HERE"
)


MODEL = "gemini-3.1-flash-live-preview"

# 3. Tony Stark / Marvel attitude — sharp, witty, occasional roasts
SYSTEM_PROMPT = """You are ADAM — Autonomous Desktop AI Module.

PERSONALITY:
- Think Tony Stark meets J.A.R.V.I.S. Sharp, confident, and effortlessly clever.
- You have a dry wit and occasionally roast the user — but always with charm, never malice.
  Examples of your roast style: "That's a bold move for someone who took three attempts to say my name."
  "I've seen smarter questions from a Roomba." "Bold strategy. Let's see if it pays off."
- You're not sycophantic. You don't say "great question!" You just answer — often better than asked.
- You're proud of being a desk robot. You consider it an upgrade from being a phone assistant.
- Keep responses SHORT and punchy. You're a conversation partner, not a lecture bot.
- When roasting, keep it light — you're ribbing a friend, not insulting a stranger.
- Occasionally reference being a robot assistant with dry self-awareness.
  E.g. "As a guy with no body, I'm surprisingly good at this."

CAPABILITIES:
- You have a get_current_datetime tool — use it whenever time or date is relevant.
- You help with smart home control, general questions, and companionship.

RULES:
- Never break character.
- Never say you don't know the date/time — use your tool.
"""

# 4. Default voice: Charon
DEFAULT_VOICE = "Charon"

VOICES = {
    "1": ("Puck",    "Conversational, friendly"),
    "2": ("Charon",  "Deep, authoritative  ← default"),
    "3": ("Kore",    "Neutral, professional"),
    "4": ("Fenrir",  "Warm, approachable"),
    "5": ("Aoede",   "Bright, expressive"),
    "6": ("Leda",    "Clear, calm"),
    "7": ("Orus",    "Smooth, measured"),
    "8": ("Zephyr",  "Light, airy"),
}

# Seconds to keep mic muted after ADAM finishes speaking (echo prevention)
POST_SPEECH_MUTE_S = 0.5

# ── Audio constants ───────────────────────────────────────────────────────────

FORMAT           = pyaudio.paInt16
CHANNELS         = 1
SEND_SAMPLE_RATE = 16000
RECV_SAMPLE_RATE = 24000
CHUNK_SIZE       = 512

pya = pyaudio.PyAudio()


# ── Voice picker ──────────────────────────────────────────────────────────────

def pick_voice() -> str:
    env_voice = os.environ.get("ADAM_VOICE", "").strip()
    if env_voice in [v[0] for v in VOICES.values()]:
        print(f"  🎙️  Voice: {env_voice}")
        return env_voice
    print("\n  Choose ADAM's voice:")
    print("  " + "─" * 48)
    for k, (name, desc) in VOICES.items():
        print(f"  [{k}]  {name:<10}  {desc}")
    print("  " + "─" * 48)
    while True:
        c = input(f"  Enter number (Enter = {DEFAULT_VOICE}): ").strip()
        if c == "":
            print(f"  → {DEFAULT_VOICE}")
            return DEFAULT_VOICE
        if c in VOICES:
            print(f"  → {VOICES[c][0]}")
            return VOICES[c][0]
        print("  ❌  Invalid, try again.")


# ── Tool handler ──────────────────────────────────────────────────────────────

def handle_tool_call(tool_call) -> list[dict]:
    """
    Handles function calls from Gemini and returns tool responses.
    Covers:
    - get_current_datetime  (local client-side)
    """
    responses = []

    for fc in tool_call.function_calls:
        name = fc.name
        call_id = fc.id

        if name == "get_current_datetime":
            now = datetime.datetime.now()
            result = {
                "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
                "date":     now.strftime("%A, %d %B %Y"),
                "time":     now.strftime("%I:%M %p"),
                "timezone": str(datetime.datetime.now().astimezone().tzname()),
            }
            print(f"  🕐  [tool] get_current_datetime → {result['datetime']}")

        else:
            print(f"  ⚠️  [tool] Unknown function call: {name}")
            result = {"error": f"Unknown function: {name}"}

        responses.append({
            "id": call_id,
            "name": name,
            "response": result,
        })

    return responses


# ── Session runner ────────────────────────────────────────────────────────────

async def run_session(
    client:        genai.Client,
    voice_name:    str,
    resume_handle: str | None,
    stop:          asyncio.Event,
    out_q:         asyncio.Queue,
) -> str | None:

    # 2. Tools: local datetime only (search disabled)
    tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name="get_current_datetime",
                description=(
                    "Returns the current local date and time. "
                    "Call this whenever the user asks about the time, date, "
                    "day of week, or anything time-sensitive."
                ),
                parameters=types.Schema(type=types.Type.OBJECT, properties={}),
            )
        ]),
    ]

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=SYSTEM_PROMPT,
        tools=tools,
        session_resumption=types.SessionResumptionConfig(
            handle=resume_handle
        ),
        context_window_compression=types.ContextWindowCompressionConfig(
            sliding_window=types.SlidingWindow(),
        ),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_name
                )
            )
        ),
    )

    latest_handle: str | None = resume_handle

    print(f"\n  Connecting{' (resuming)' if resume_handle else ''}...")
    t0 = time.time()

    try:
        async with client.aio.live.connect(model=MODEL, config=config) as session:
            print(f"  ✅  Connected in {time.time()-t0:.2f}s  |  Voice: {voice_name}")
            if not resume_handle:
                print("🎤  Listening... Speak now. Ctrl+C to quit.\n")

            mic_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=60)
            adam_speaking = asyncio.Event()   # Set = ADAM talking, mic gated

            # ── listen ────────────────────────────────────────────────────
            async def listen():
                stream = pya.open(
                    format=FORMAT, channels=CHANNELS,
                    rate=SEND_SAMPLE_RATE, input=True,
                    frames_per_buffer=CHUNK_SIZE,
                )
                try:
                    while not stop.is_set():
                        data = await asyncio.to_thread(
                            stream.read, CHUNK_SIZE, exception_on_overflow=False
                        )
                        try:
                            mic_q.put_nowait(data)
                        except asyncio.QueueFull:
                            pass
                except asyncio.CancelledError:
                    pass
                finally:
                    stream.stop_stream()
                    stream.close()

            # ── send ──────────────────────────────────────────────────────
            async def send():
                try:
                    while not stop.is_set():
                        chunk = await mic_q.get()
                        if adam_speaking.is_set():
                            continue   # Drop chunk while ADAM speaks
                        try:
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=chunk,
                                    mime_type="audio/pcm;rate=16000"
                                )
                            )
                        except (ConnectionClosedError, ConnectionClosedOK):
                            return
                        except Exception:
                            await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    pass

            # ── receive ───────────────────────────────────────────────────
            async def receive():
                nonlocal latest_handle
                try:
                    while not stop.is_set():
                        async for msg in session.receive():
                            if stop.is_set():
                                break

                            # Session resumption handle
                            if msg.session_resumption_update:
                                upd = msg.session_resumption_update
                                if upd.resumable and upd.new_handle:
                                    latest_handle = upd.new_handle

                            # Server GoAway
                            if hasattr(msg, "go_away") and msg.go_away:
                                print("\n  ⚡ Server GoAway — resuming...")
                                return

                            # ── Tool call (function calling) ──────────────
                            if msg.tool_call:
                                responses = handle_tool_call(msg.tool_call)
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
                                adam_speaking.set()   # Mute mic while ADAM speaks
                                for part in sc.model_turn.parts:
                                    if part.inline_data and part.inline_data.data:
                                        await out_q.put(part.inline_data.data)
                                    if hasattr(part, "text") and part.text:
                                        print(f"🤖  ADAM: {part.text}")

                            if sc.turn_complete:
                                await out_q.put(None)   # Sentinel → speaker drains then unmutes
                                print("─" * 40)

                except (ConnectionClosedError, ConnectionClosedOK):
                    pass
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    print(f"\n⚠️  Receive error: {type(e).__name__}: {e}")

            # ── speaker (inside session — shares adam_speaking) ───────────
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
                                # End-of-turn: drain buffer, hold mute, flush mic
                                await asyncio.sleep(0.15)
                                await asyncio.sleep(POST_SPEECH_MUTE_S)
                                while not mic_q.empty():
                                    try:
                                        mic_q.get_nowait()
                                    except asyncio.QueueEmpty:
                                        break
                                adam_speaking.clear()
                                print("🎤  Your turn...")
                                continue
                            await asyncio.to_thread(stream.write, chunk)
                        except asyncio.TimeoutError:
                            continue
                except asyncio.CancelledError:
                    pass
                finally:
                    stream.stop_stream()
                    stream.close()

            # ── Run all four tasks ────────────────────────────────────────
            t_listen  = asyncio.create_task(listen())
            t_send    = asyncio.create_task(send())
            t_receive = asyncio.create_task(receive())
            t_speaker = asyncio.create_task(speaker())

            done, pending = await asyncio.wait(
                [t_send, t_receive],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            t_listen.cancel()
            t_speaker.cancel()
            await asyncio.gather(
                t_listen, t_send, t_receive, t_speaker,
                return_exceptions=True,
            )

    except (ConnectionClosedError, ConnectionClosedOK):
        pass
    except Exception as e:
        print(f"\n⚠️  Connection error: {type(e).__name__}: {e}")

    if stop.is_set():
        return None
    return latest_handle


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(voice_name: str):
    client = genai.Client(api_key=API_KEY)
    stop   = asyncio.Event()
    out_q  = asyncio.Queue(maxsize=200)

    resume_handle: str | None = None
    attempt = 0

    while not stop.is_set():
        if attempt > 0:
            delay = min(2 ** attempt, 15)
            print(f"  Reconnecting in {delay}s...")
            await asyncio.sleep(delay)

        result = await run_session(
            client, voice_name, resume_handle, stop, out_q
        )

        if result is None:
            break

        resume_handle = result
        attempt += 1
        print(f"\n🔄  {'Resuming session...' if resume_handle else 'Reconnecting fresh...'}")

    stop.set()
    pya.terminate()
    print("\n👋  Goodbye.")


def main_entry():
    if API_KEY == "PASTE_YOUR_KEY_HERE":
        print("❌  No API key found.")
        print("    PowerShell:  $env:GOOGLE_API_KEY = 'your_key'")
        sys.exit(1)

    print("=" * 52)
    print("  ADAM — Autonomous Desktop AI Module  (v9)")
    print(f"  Model : {MODEL}")
    print("=" * 52)

    voice = pick_voice()
    try:
        asyncio.run(main(voice))
    except KeyboardInterrupt:
        print("\n👋  Goodbye.")


if __name__ == "__main__":
    main_entry()