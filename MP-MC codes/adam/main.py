"""
main.py — ADAM v40 entrypoint
==============================================================================
Wires together all the split-out modules and runs the top-level reconnect
loop. This file should contain almost no logic of its own — just startup
sequencing, the reconnect/backoff loop, and graceful shutdown.

Run:
    python main.py
"""

import asyncio
import signal
import sys
import time
import traceback

from config import (
    LIVE_MODEL,
    VOICE,
    CAPTURE_DEVICE, CAPTURE_FORMAT, CAPTURE_RATE, CAPTURE_CHANNELS,
    PLAYBACK_DEVICE, PLAYBACK_FORMAT, PLAYBACK_RATE, PLAYBACK_CHANNELS,
    PI_UART_PORT, PI_UART_BAUD,
    NECK_TILT_CENTER, NECK_PAN_CENTER,
    OUT_Q_MAX,
    API_KEY,
)
from hardware import pan_servo, servo_pan
from esp32_link import esp_link
from memory_store import save_conversation_log, save_json, memory, faces, MEMORY_FILE, FACE_MEMORY_FILE
from ws_server import start_ws_server
from session import run_session, tft_set

from google import genai

# DDGS import only needed here for the startup banner
try:
    from web_search import DDGS
except Exception:
    DDGS = None

# Zeroconf/static-IP flags only needed here for the startup banner
from laptop_agent_client import (
    ZEROCONF_AVAILABLE,
    LAPTOP_AGENT_STATIC_IP,
    LAPTOP_AGENT_PORT,
    LAPTOP_MDNS_SERVICE,
)


async def main() -> None:
    print("=" * 66)
    print("  ADAM v40 — Autonomous Desktop AI Module (Wired ESP32-CAM)")
    print(f"  Model  : {LIVE_MODEL}  |  Voice: {VOICE}")
    print(f"  Mic    : {CAPTURE_DEVICE} {CAPTURE_FORMAT} {CAPTURE_RATE}Hz {CAPTURE_CHANNELS}ch")
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
    out_q: asyncio.Queue = asyncio.Queue(maxsize=OUT_Q_MAX)
    resume_handle = None
    fail_streak   = 0

    # ── Graceful shutdown on SIGTERM/SIGINT ─────────────────────────────
    # Under systemd, `systemctl stop`/`restart` sends SIGTERM by default.
    # Without a handler, Python's default SIGTERM action kills the process
    # immediately — skipping the `finally` block below that turns the
    # camera off, centers the servo, and flushes conversation history to
    # disk.
    loop = asyncio.get_running_loop()

    def _request_shutdown(sig_name: str) -> None:
        if not stop.is_set():
            print(f"\n  🛑 Received {sig_name} — shutting down gracefully...")
            stop.set()

    try:
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
                # (1011). Reconnecting quickly won't help — back off much
                # longer than the normal exponential schedule.
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
                # resuming the broken audio+video session state.
                print("  🔄 Starting fresh session (discarding resumption "
                      "handle to avoid repeat 1007 errors)")
                resume_handle = None
                fail_streak   = 0
                await asyncio.sleep(2.0)
            elif isinstance(result, str):
                resume_handle = result
                fail_streak   = 0
            else:
                resume_handle = None
                fail_streak  += 1
    finally:
        # Explicit safe-state shutdown — run_session()'s own camera task
        # already sends CAM:OFF on task cancellation, but if the process
        # is killed between sessions (or that send fails because esp_link
        # dropped), this is the last chance to leave the physical
        # hardware in a safe state rather than mid-stream/hot.
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
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n  👋 Goodbye")
    except Exception:
        # Ensures systemd's Restart=on-failure actually treats this as a
        # failure (a clean sys.exit(0) would NOT trigger a restart) and
        # the traceback is unambiguously logged either way.
        print("\n  ❌ ADAM crashed with an unhandled exception:")
        traceback.print_exc()
        sys.exit(1)