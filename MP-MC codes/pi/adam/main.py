"""
main.py — ADAM v41 entrypoint
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

from heartbeat import clear_heartbeat, record_heartbeat
clear_heartbeat()
record_heartbeat(status="booting", mic_rms=0.0, zero_run=0)

from config import (
    LIVE_MODEL,
    VOICE,
    CAPTURE_DEVICE, CAPTURE_FORMAT, CAPTURE_RATE, CAPTURE_CHANNELS,
    GEMINI_SEND_RATE,
    PLAYBACK_DEVICE, PLAYBACK_FORMAT, PLAYBACK_RATE, PLAYBACK_CHANNELS,
    PI_UART_PORT, PI_UART_BAUD,
    NECK_TILT_CENTER, NECK_PAN_CENTER,
    OUT_Q_MAX,
    API_KEY,
    SYNC_HOST, SYNC_PORT, SYNC_TOKEN,
)
from hardware import pan_servo, servo_pan
from esp32_link import esp_link
from memory_store import save_conversation_log, save_json, memory, faces, MEMORY_FILE, FACE_MEMORY_FILE
from ws_server import start_ws_server
from sync_api import start_sync_api, stop_sync_api
import touch_controls
import laptop_pairing
from session import run_session, tft_set
from mic_calibrate import calibrate as calibrate_mic
from heartbeat import clear_heartbeat, record_heartbeat

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
    print("  ADAM v41 — Autonomous Desktop AI Module (Wired ESP32-CAM)")
    print(f"  Model  : {LIVE_MODEL}  |  Voice: {VOICE}")
    print(f"  Mic    : {CAPTURE_DEVICE} {CAPTURE_FORMAT} {CAPTURE_RATE}Hz {CAPTURE_CHANNELS}ch "
          f"→ {GEMINI_SEND_RATE}Hz to Gemini")
    print(f"  Speaker: {PLAYBACK_DEVICE} {PLAYBACK_FORMAT} {PLAYBACK_RATE}Hz {PLAYBACK_CHANNELS}ch")
    print(f"  ESP32  : WIRED UART {PI_UART_PORT} @ {PI_UART_BAUD} baud (Flow 2)")
    print(f"  Display: on Pico, driven via ESP32-CAM relay (Pi->UART->ESP32->Pico)")
    print(f"  Servo  : {'✅ pan' if pan_servo else '⚠️  unavailable'} (tilt via UART)")
    print(f"  Sync   : http://{SYNC_HOST}:{SYNC_PORT} "
          f"{'[read-write]' if SYNC_TOKEN else '[read-only — set SYNC_TOKEN]'}")
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

    clear_heartbeat()
    record_heartbeat(status="starting", mic_rms=0.0, zero_run=0)

    await start_ws_server()

    try:
        await laptop_pairing.initialize_laptop_pairing()
    except Exception:
        print("  Desktop pairing could not start; connect again after checking local storage.")

    # One touch controller for the whole process, so laptop shortcuts continue
    # through Gemini reconnects. Disk and laptop I/O run outside the event loop.
    try:
        await touch_controls.initialize_touch_controls(esp_link)
    except Exception:
        print("  Touch shortcuts could not start; original gestures remain active.")

    # The companion app's data API (schedules/todos/memories/conversations).
    # Started here, beside the WS server and for the same reason: both are
    # inbound listeners, and this one must exist BEFORE the reconnect loop
    # below so the PC app can read the Pi's lists while ADAM is offline or
    # waiting out a backoff. Deliberately non-fatal — the app losing its data
    # view must never cost ADAM its startup, and the voice face channel is
    # unaffected either way.
    try:
        await start_sync_api()
    except Exception as e:
        print(f"  ⚠️  Sync API failed to start ({e}) — PC app will show its "
              f"cached copy only")

    # The scheduler ticker. Started HERE, outside the reconnect loop below,
    # and exactly once per process (master prompt §0.4): a reconnect must
    # never start a second ticker, or every alarm would fire twice, then
    # three times, for as long as the Wi-Fi stayed flaky. start_scheduler()
    # is idempotent as a second line of defence, but the placement is the
    # actual guarantee.
    #
    # It also runs before the Gemini client exists, which is deliberate —
    # alarms are a local, offline feature and must keep working through an
    # API outage. Firing only queues an announcement; session.py picks it up
    # whenever there is a live session to deliver it through.
    try:
        from scheduler import start_scheduler
        await start_scheduler()
    except Exception as e:
        # A broken scheduler costs ADAM its alarms, never its startup.
        print(f"  ⚠️  Scheduler unavailable ({e}) — alarms and todos are off")

    esp_link.start()

    # Measure the microphone path before anything depends on it. This runs
    # HERE — after esp_link so the face can react, before the Gemini client
    # exists — for two reasons. It needs the ALSA devices to itself: listen()
    # and speaker() each hold a long-lived arecord/aplay for the whole session,
    # and a second opener of plughw:0,0 reliably hits "Device or resource
    # busy" (the same collision that shaped song_playback's design). And the
    # values it produces — the noise floor the gate compares every chunk
    # against, and which microphone feeds the speech path — have to be right
    # BEFORE the first word is captured, not corrected a minute into the
    # conversation.
    #
    # Deliberately not fatal: calibrate() returns None on any problem and
    # leaves every existing setting untouched. A failed measurement must cost
    # ADAM its tuning, never its startup.
    try:
        calibrate_mic()
    except Exception as e:
        print(f"  ⚠️  startup calibration failed ({e}) — continuing with "
              f"the previously learned mic settings")

    client        = genai.Client(api_key=API_KEY)
    stop          = asyncio.Event()
    out_q: asyncio.Queue = asyncio.Queue(maxsize=OUT_Q_MAX)
    resume_handle = None
    fail_streak   = 0
    # Counted separately from fail_streak: a local DNS/socket fault is not
    # an API fault and must not ride the 2**n schedule up to 30s of
    # deafness, nor throw away the resumption handle. See the
    # NETWORK_TRANSIENT branch below.
    net_streak    = 0

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

            if fail_streak > 0 or net_streak > 0 or resume_handle is not None:
                # Show a visible "reconnecting" face immediately, before
                # any backoff/reconnect delay, so the user sees ADAM is
                # aware it dropped rather than just going silent/frozen.
                tft_set("reconnecting")

            if fail_streak > 0:
                delay = min(2 ** fail_streak, 30)
                print(f"\n  ⚠️  Error reconnect in {delay}s (streak={fail_streak})...")
                await asyncio.sleep(delay)
            elif net_streak > 0:
                # Flat, short retry. At boot the resolver typically starts
                # answering within a few seconds of wlan0 associating, so
                # a linear 2/4/6/8s schedule gets ADAM online roughly a
                # minute earlier than 2**n would, and keeps the handle.
                delay = min(2.0 * net_streak, 8.0)
                print(f"\n  📡 Waiting {delay:.0f}s for the network "
                      f"(attempt {net_streak}) — conversation context kept")
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
                net_streak    = 0
                QUOTA_BACKOFF_S = 120
                print(f"  🚫 Waiting {QUOTA_BACKOFF_S}s before retrying "
                      f"due to quota/billing limit — check your plan at "
                      f"https://ai.google.dev if this keeps happening.")
                tft_set("sleep")
                await asyncio.sleep(QUOTA_BACKOFF_S)
            elif isinstance(result, tuple) and result and result[0] == "NETWORK_TRANSIENT":
                # DNS/socket failure on this side — the API was never
                # reached, so result[1] is still a valid handle. Keep it
                # and retry on the flat schedule above instead of
                # escalating fail_streak.
                resume_handle = result[1]
                fail_streak   = 0
                net_streak   += 1
            elif isinstance(result, tuple) and result and result[0] == "FRESH_SESSION_REQUIRED":
                # 1007 resumption bug workaround — discard the handle so
                # the next connect starts genuinely fresh instead of
                # resuming the broken audio+video session state.
                print("  🔄 Starting fresh session (discarding resumption "
                      "handle to avoid repeat 1007 errors)")
                resume_handle = None
                fail_streak   = 0
                net_streak    = 0
                await asyncio.sleep(2.0)
            elif isinstance(result, str):
                resume_handle = result
                fail_streak   = 0
                net_streak    = 0
            else:
                resume_handle = None
                fail_streak  += 1
                net_streak    = 0
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
        try:
            await touch_controls.stop_touch_controls()
        except Exception:
            pass
        try:
            await stop_sync_api()
        except Exception:
            pass
        save_conversation_log()
        save_json(MEMORY_FILE, memory)
        save_json(FACE_MEMORY_FILE, faces)
        clear_heartbeat()
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
