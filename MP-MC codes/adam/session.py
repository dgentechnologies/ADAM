"""
session.py — ADAM v40 live-session orchestrator
==============================================================================
Owns run_session(): opens one Gemini Live connection and runs the whole
real-time robot loop as a set of cooperating asyncio tasks —

  • listen()                  — arecord mic capture, silence/noise gating, DOA
  • send()                    — forwards gated mic audio to Gemini
  • receive()                 — handles tool calls, transcripts, model audio,
                                GoAway/resumption, the refusal-loop breaker
  • speaker()                 — aplay playback + end-of-turn drain/mute logic
  • camera()                  — duty-cycled UART camera + DOA neck-tracking
  • gesture_watch()           — Touch1-4 gestures (angry/petting/STOP/song-stop)
  • wake_word_detector()      — offline Vosk "adam" wake word during idle mode
  • idle_watcher()            — idle conversation nudges
  • laptop_agent_healthcheck()— periodic mDNS re-verify of the laptop agent

main() drives reconnects and passes in the resumption handle; run_session()
returns the latest handle (or a ("FRESH_SESSION_REQUIRED"/"QUOTA_EXCEEDED",
handle) tuple) so main() can decide how to reconnect.

The Vosk offline wake-word model is PRELOADED once here at import — never
lazily mid-session. Loading the ~100MB+ model while the audio pipeline,
subprocesses, UART reader and camera are all live was the likely cause of
hard Pi Zero 2W reboots (OOM/brownout); doing it once up front in a quiet
moment avoids that. Only the lightweight KaldiRecognizer is created per idle
period.

tft_set is re-exported from hardware so `from session import run_session,
tft_set` keeps working for main.py. The single-element-list shared-state
boxes (_idle_mode_persistent, _play_song_requested, _doa_angle, …) are
imported BY REFERENCE from tool_handler and mutated in place ([0] = …), never
rebound — that's what lets the module-level tool handler and this session loop
see each other's writes. See tool_handler.py for the full rationale.
"""

import os
import time
import json
import asyncio
import threading
import subprocess
import queue as sync_queue

import requests
from google.genai import types

from config import (
    LIVE_MODEL, VOICE,
    MIC_Q_MAX, CHUNK_FRAMES,
    CAPTURE_DEVICE, CAPTURE_FORMAT, CAPTURE_RATE, CAPTURE_CHANNELS,
    MIC_SILENCE_FLOOR, MIC_LIVE_RMS_THRESHOLD,
    DOA_ANGLE_DEADZONE, GEMINI_SEND_RATE, POST_MUTE_S,
    PLAYBACK_DEVICE, PLAYBACK_FORMAT, PLAYBACK_RATE, PLAYBACK_CHANNELS,
    SPEAKER_GAIN,
    NECK_PAN_CENTER, NECK_PAN_MIN, NECK_PAN_MAX,
    NECK_PAN_DEADZONE_DEG, NECK_PAN_COOLDOWN_S,
    CAMERA_FPS_INTERVAL,
    GESTURE_STOP, GESTURE_ANGRY, GESTURE_PETTING,
    ENABLE_IDLE, IDLE_TIMEOUT_S, next_nudge,
    LAPTOP_DISCOVERY_TTL_S, LAPTOP_AGENT_PORT, LAPTOP_AGENT_STATIC_IP,
    BASE_DIR,
)

from hardware import servo_pan, servo_tilt, tft_set   # tft_set re-exported for main.py
from esp32_link import esp_link
from audio_utils import (
    read_exact, drain_stderr, rms_s32, is_valid_pcm16_chunk, beep_s16_stereo,
    s32_stereo_to_s16_mono_16k, s32_stereo_to_s16_stereo_channels,
    estimate_doa_angle, s16_mono_24k_to_s16_stereo_48k,
)
from memory_store import append_conversation_turn
from system_prompt import build_system_prompt
from tools_schema import build_tools
from song_playback import _play_song_task
from tool_handler import (
    handle_tool_call,
    _idle_mode_requested, _idle_mode_persistent, _play_song_requested,
    _last_emotion_set_this_turn, _face_is_generic_speaking,
    _doa_angle, _doa_last_update_t,
)
from ws_server import ws_broadcast
from laptop_agent_client import (
    ZEROCONF_AVAILABLE, _discover_laptop_agent_ip, _laptop_agent_ip_cache,
)

from pathlib import Path

# ═════════════════════════════════════════════════════════════════════════════
# VOSK OFFLINE WAKE-WORD — preloaded ONCE at import (see module docstring)
# ═════════════════════════════════════════════════════════════════════════════

VOSK_AVAILABLE = False
_VoskModel = None
_VoskKaldiRecognizer = None
_vosk_model_instance = None  # the actual loaded Model object, preloaded once
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", str(BASE_DIR / "vosk-model-small-en-us-0.15"))
try:
    from vosk import Model as _VoskModelCls, KaldiRecognizer as _VoskRecCls
    _VoskModel = _VoskModelCls
    _VoskKaldiRecognizer = _VoskRecCls
    if Path(VOSK_MODEL_PATH).exists():
        # FIX: previously the Model object (the expensive part — reads
        # the full acoustic model, language graph, and i-vector extractor
        # from disk, easily 100MB+ of parsing work even for the "small"
        # model) was loaded LAZILY, every single time idle mode was
        # entered — mid-session, while the Live audio pipeline, arecord/
        # aplay subprocesses, UART reader, and camera were all actively
        # running. That CPU/memory spike is the likely cause of observed
        # hard Pi reboots (not just a Python crash — an actual reboot
        # requiring SSH reconnection, consistent with OOM or a brownout
        # from a sudden current/CPU spike on a Pi Zero 2W). Now the model
        # loads ONCE here, at process startup, before any session or
        # audio pipeline exists — the ~1-3s load happens in a quiet
        # moment, not mid-conversation. Only the lightweight
        # KaldiRecognizer wrapper (cheap) gets created per idle period.
        print(f"  🔎 Preloading Vosk model (one-time, ~1-3s)...")
        _vosk_model_instance = _VoskModel(VOSK_MODEL_PATH)
        VOSK_AVAILABLE = True
        print(f"✅ Vosk offline STT ready (idle wake-word only) — model at {VOSK_MODEL_PATH}")
    else:
        print(f"⚠️  Vosk installed but model not found at {VOSK_MODEL_PATH} — "
              f"idle mode will only exit via Touch3, not voice. Download a "
              f"small model from https://alphacephei.com/vosk/models and "
              f"set VOSK_MODEL_PATH if you want voice wake-up during idle.")
except ImportError:
    print("⚠️  Vosk not installed (pip install vosk) — idle mode will only "
          "exit via Touch3, not voice.")
except Exception as e:
    print(f"⚠️  Vosk unavailable: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# SESSION
# ═════════════════════════════════════════════════════════════════════════════

async def run_session(client, resume_handle: str | None,
                      stop: asyncio.Event, out_q: asyncio.Queue) -> str | None:

    print(f"\n  Connecting{' (resuming)' if resume_handle else ''}...")
    system_prompt = build_system_prompt()

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=system_prompt,
        tools=build_tools(),
        session_resumption=types.SessionResumptionConfig(handle=resume_handle),
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
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
    # Set to True on a 1007 (server-rejected-payload) close. See detailed
    # explanation at its usage site in send() below — this is a confirmed
    # Google-side Live API bug (python-genai#2290) where resuming a
    # session that used both audio and video can fail every subsequent
    # audio send in a tight reconnect loop.
    force_fresh_session = [False]
    # Set to True on a 1011 quota/billing error. See the outer except
    # block below for full explanation.
    quota_exceeded = [False]

    try:
        async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
            print("  ✅ Connected to Gemini Live")

            mic_q            = asyncio.Queue(maxsize=MIC_Q_MAX)
            adam_speaking    = asyncio.Event()
            latest_frame     = [None]
            attention_active = asyncio.Event()
            last_interact_t  = [time.time()]
            last_user_text   = [""]
            interrupt_flag   = asyncio.Event()
            # ── Song playback state ──────────────────────────────────────
            # song_playing: mic-mute gate for the duration of playback
            # (listen()/send() check this the same way they check
            # adam_speaking, so nothing extra needed there).
            # song_stop_requested: set by Touch3 while a song is playing,
            # to end playback early. Distinct from GESTURE_STOP's normal
            # idle-mode-toggle behavior — see gesture_watch() below for
            # how Touch3 is routed differently depending on whether a
            # song is currently playing.
            song_playing         = asyncio.Event()
            song_stop_requested  = asyncio.Event()
            # Shared reference to speaker()'s currently-live aplay
            # process/stdin — the song task writes into THIS SAME
            # process instead of spawning a second one. speaker()'s
            # aplay stays open for the entire session lifetime (only
            # recreated on exception/reconnect, not between turns), so a
            # second process trying to open the same ALSA device was
            # always going to collide — confirmed repeatedly in logs
            # ("Device or resource busy") even well after speech had
            # finished, since the first aplay never actually closes
            # between turns. Routing the song through the SAME open
            # process eliminates the contention entirely rather than
            # trying to time around it.
            active_speaker_proc = [None]
            # ── Idle/silent mode ─────────────────────────────────────────
            # Distinct from interrupt_flag (which only suppresses the ONE
            # in-flight reply). idle_mode is a PERSISTENT state: once set
            # (via STOP touch gesture, or the user explicitly asking ADAM
            # to "stay silent"/"stay mute"), NO audio is sent to Google at
            # all while idle — wake detection runs entirely locally via
            # Vosk (offline STT) watching for "adam" in the mic stream, or
            # via the Touch3 physical gesture. Idle nudges are also
            # suppressed. Only hearing "adam" (locally) or Touch3 exits
            # idle mode.
            #
            # FIX: this Event is recreated fresh on every run_session()
            # call, including reconnects — it does NOT survive a GoAway/
            # 1007/network-hiccup reconnect on its own. Initialize it
            # from the module-level _idle_mode_persistent flag (the real
            # source of truth across sessions) so a reconnect mid-idle
            # doesn't silently wake ADAM back up with no visible cause.
            idle_mode        = asyncio.Event()
            if _idle_mode_persistent[0]:
                idle_mode.set()
                print("  🔇 Resuming idle mode after reconnect "
                      "(servos re-centered)")
                tft_set("sleep")
                await asyncio.to_thread(servo_pan, 90)
                servo_tilt(90)
            # Feeds raw mic audio to the local Vosk wake-word detector
            # while idle. Only populated during idle_mode — see listen().
            wake_word_q: asyncio.Queue = asyncio.Queue(maxsize=200)
            # Tracks whether set_emotion() was called during the current
            # turn. Previously end_of_turn() unconditionally forced the
            # face back to "happy" after every single reply, silently
            # overwriting any deliberate emotion the model had just set
            # (love, angry, sad, etc.) the moment ADAM finished speaking —
            # making it look like emotions "got stuck on happy" when
            # actually they were being reset on a fixed timer, not stuck.
            emotion_set_this_turn = [False]

            # ── Direction-of-arrival state ──────────────────────────────
            # Smoothed sound-direction angle from the two mics (GCC-PHAT).
            # Updated in listen() on every chunk where speech is detected,
            # read by camera()'s neck-tracking logic to turn toward
            # whoever is currently talking, and available to inject into
            # the model's context if useful.
            doa_angle = [0.0]          # smoothed angle, degrees (-90..90)
            doa_last_update_t = [0.0]

            attention_active.set()

            async def inject(text: str, retries: int = 6) -> bool:
                for _ in range(retries):
                    if stop.is_set():
                        return False
                    try:
                        await session.send_realtime_input(text=text)
                        return True
                    except Exception:
                        await asyncio.sleep(0.3)
                return False

            async def listen() -> None:
                print("  🎤 Listen task started")
                read_bytes = CHUNK_FRAMES * CAPTURE_CHANNELS * 4
                _last_rms  = [0.0]
                _dropped_bad_chunks = [0]
                _last_bad_warn_t = [0.0]
                # ── Adaptive noise-floor calibration ──────────────────────
                # BUG FIX: peripheral noise (servo whine during a move,
                # electrical coupling from the UART/camera link, fans) can
                # produce RMS bursts above the old fixed MIC_SILENCE_FLOOR
                # while still being much quieter than actual speech. A
                # static global floor can't tell the two apart — raising it
                # risks cutting real quiet speech, leaving it low lets noise
                # bursts through as if they were speech, which can confuse
                # Gemini's own turn-detection right as the user starts
                # talking (their real speech gets bundled with/cut off by
                # the noise burst). Fix: track a rolling ambient noise
                # baseline during quiet stretches, and require a chunk to
                # clear that baseline by a real margin (not just the fixed
                # floor) before it's treated as meaningful audio.
                _ambient_rms = [MIC_SILENCE_FLOOR * 0.5]  # starting estimate
                _AMBIENT_ALPHA = 0.05      # slow-moving average
                _SPEECH_MARGIN = 3.0       # must exceed ambient*this to count

                while not stop.is_set():
                    proc = None
                    try:
                        cmd = ["arecord",
                               "-D", CAPTURE_DEVICE,
                               "-f", CAPTURE_FORMAT,
                               "-r", str(CAPTURE_RATE),
                               "-c", str(CAPTURE_CHANNELS),
                               "-t", "raw", "-q"]
                        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE, bufsize=0)
                        await asyncio.sleep(1.0)
                        if proc.poll() is not None:
                            err = proc.stderr.read().decode(errors="replace").strip()
                            print(f"  ❌ arecord failed: {err}")
                            await asyncio.sleep(3.0)
                            continue

                        print(f"  ✅ arecord: {CAPTURE_DEVICE} {CAPTURE_FORMAT} "
                              f"{CAPTURE_RATE}Hz {CAPTURE_CHANNELS}ch")
                        errors = 0

                        # ── Hardware warm-up discard ──────────────────────
                        # The first fraction of a second of audio right
                        # after arecord opens the capture device is
                        # typically unstable — DC offset hasn't settled,
                        # some HATs/codecs ramp their AGC (automatic gain
                        # control) up over the first few frames, and ALSA's
                        # own buffer needs a moment to reach steady state.
                        # This produces wildly inconsistent RMS readings on
                        # startup/reconnect that don't reflect real input
                        # levels and could feed garbage into VAD/attention
                        # logic. Discard a short warm-up window's worth of
                        # chunks (not sent to Gemini, not RMS-logged)
                        # before treating capture as "live."
                        warmup_bytes_target = int(
                            CAPTURE_RATE * CAPTURE_CHANNELS * 4 * 0.4)  # ~0.4s
                        warmup_discarded = 0
                        while (warmup_discarded < warmup_bytes_target
                               and not stop.is_set()):
                            try:
                                _ = await asyncio.to_thread(
                                    read_exact, proc.stdout, read_bytes)
                                warmup_discarded += read_bytes
                            except Exception:
                                break

                        while not stop.is_set():
                            try:
                                raw = await asyncio.to_thread(
                                    read_exact, proc.stdout, read_bytes)
                            except Exception as e:
                                errors += 1
                                if errors > 5:
                                    print(f"  ⚠️  arecord read: {e} — restarting")
                                    break
                                await asyncio.sleep(0.5)
                                continue
                            errors = 0

                            if adam_speaking.is_set() or song_playing.is_set():
                                while not mic_q.empty():
                                    try: mic_q.get_nowait()
                                    except asyncio.QueueEmpty: break
                                continue

                            mono16k = await asyncio.to_thread(
                                s32_stereo_to_s16_mono_16k, raw)
                            if not mono16k:
                                continue

                            # ── Direction-of-arrival (two-mic) ────────────────
                            # Only bother computing this when there's enough
                            # signal to be worth it — GCC-PHAT on pure noise/
                            # silence produces meaningless jittery angles and
                            # wastes CPU on every single chunk otherwise.
                            # Also skipped entirely while idle — the servo
                            # must not track sound direction at all during
                            # idle mode, and not updating doa_angle here is
                            # belt-and-suspenders alongside camera()'s own
                            # idle_mode check, eliminating any possibility
                            # of a stale/racy update influencing the servo.
                            if not idle_mode.is_set():
                                _rms_for_doa = rms_s32(raw)
                                if _rms_for_doa > MIC_LIVE_RMS_THRESHOLD * 0.5:
                                    def _compute_doa():
                                        left, right = s32_stereo_to_s16_stereo_channels(raw)
                                        return estimate_doa_angle(left, right, CAPTURE_RATE)
                                    angle = await asyncio.to_thread(_compute_doa)
                                    if abs(angle) > DOA_ANGLE_DEADZONE:
                                        # Light smoothing so the neck doesn't
                                        # jitter on every chunk — exponential
                                        # moving average, not a hard snap.
                                        doa_angle[0] = (doa_angle[0] * 0.6) + (angle * 0.4)
                                        doa_last_update_t[0] = time.time()
                                        # Mirror to module-level state for the
                                        # get_sound_direction tool handler,
                                        # which lives outside this closure.
                                        _doa_angle[0] = doa_angle[0]
                                        _doa_last_update_t[0] = doa_last_update_t[0]

                            # ── FIX #2: audio sanity gate ─────────────────────
                            # Drop corrupted/desynced chunks BEFORE they reach
                            # Gemini. This is what previously produced:
                            #   "receive error: 1007 None. Request contains
                            #    an invalid argument." — a single garbage
                            #   chunk could kill the whole Live session.
                            if not is_valid_pcm16_chunk(mono16k):
                                _dropped_bad_chunks[0] += 1
                                now_w = time.time()
                                if now_w - _last_bad_warn_t[0] > 2.0:
                                    print(f"  ⚠️  Dropped {_dropped_bad_chunks[0]} "
                                          f"corrupted audio chunk(s) before "
                                          f"send — check UART/CPU contention "
                                          f"if this repeats constantly")
                                    _last_bad_warn_t[0] = now_w
                                    _dropped_bad_chunks[0] = 0
                                continue

                            now = time.time()
                            _rms_now = rms_s32(raw)
                            if now - _last_rms[0] > 4.0:
                                print(f"  🎤 Mic RMS: {_rms_now:.0f}")
                                _last_rms[0] = now

                            if _rms_now > MIC_LIVE_RMS_THRESHOLD:
                                attention_active.set()

                            # ── SILENCE GATE ─────────────────────────────────
                            # Previously every mic chunk was queued/sent to
                            # Gemini unconditionally, including pure room
                            # noise/silence between sentences. Continuously
                            # streaming near-silent audio gives the Live API
                            # ungrounded input during quiet stretches, which
                            # is a known trigger for unprompted "phantom"
                            # responses (the random Hindi hallucinations) —
                            # the model free-associates from thin signal
                            # instead of responding to real speech.
                            #
                            # MIC_SILENCE_FLOOR is set well below normal
                            # speech RMS (your speech reads 25M-60M; true
                            # silence/room tone is typically under a few
                            # hundred thousand) so genuine quiet speech is
                            # never at risk of being gated out — only true
                            # silence is withheld.
                            if _rms_now < MIC_SILENCE_FLOOR:
                                continue

                            # ── Adaptive noise-floor gate ─────────────────
                            # A chunk must clear the ROLLING ambient
                            # baseline by a real margin, not just the fixed
                            # global floor above — this is what actually
                            # distinguishes a peripheral noise burst (which
                            # can be louder than true silence but is still
                            # much quieter than real speech) from genuine
                            # speech onset. The ambient baseline itself is
                            # only nudged toward RELATIVELY QUIET chunks
                            # (below the current speech threshold), so a
                            # sustained noise burst doesn't drag the
                            # baseline up and end up masking itself.
                            speech_threshold = _ambient_rms[0] * _SPEECH_MARGIN
                            if _rms_now < speech_threshold:
                                # Not clearly speech — could be ambient
                                # noise. Let it nudge the baseline (slowly)
                                # so the calibration tracks the room's
                                # actual current noise floor, then drop it.
                                _ambient_rms[0] = (
                                    (1 - _AMBIENT_ALPHA) * _ambient_rms[0]
                                    + _AMBIENT_ALPHA * _rms_now)
                                continue

                            if idle_mode.is_set():
                                # While idle, audio goes ONLY to the local
                                # wake-word detector — never to mic_q
                                # (which feeds Gemini via send()). This is
                                # what actually keeps audio off Google
                                # during idle, not just discarding the
                                # response afterward.
                                if VOSK_AVAILABLE:
                                    try:
                                        wake_word_q.put_nowait(mono16k)
                                    except asyncio.QueueFull:
                                        pass
                                continue

                            if not mic_q.full():
                                mic_q.put_nowait(mono16k)

                            await asyncio.sleep(0)

                    except asyncio.CancelledError:
                        # MUST re-raise — see speaker()'s corresponding fix
                        # for the full explanation. Swallowing this here
                        # lets the outer `while not stop.is_set()` loop
                        # (process-level stop, not per-session
                        # cancellation) respawn arecord instead of letting
                        # this task actually terminate when run_session()
                        # cancels it.
                        raise
                    except Exception as e:
                        print(f"  ⚠️  listen recovering: {e}")
                        await asyncio.sleep(2.0)
                    finally:
                        if proc:
                            # FIX: proc.terminate()/proc.wait() are
                            # BLOCKING synchronous calls. Running them
                            # directly inside an async finally stalls the
                            # ENTIRE event loop for up to the timeout
                            # (2s) if arecord is slow to exit — during a
                            # multi-task cancellation (e.g. right after a
                            # 1007 error kills the session), this could
                            # make the whole reconnect look hung rather
                            # than fast, since asyncio.gather() is
                            # waiting on this coroutine to actually finish
                            # before run_session() can return and let
                            # main()'s loop attempt to reconnect.
                            async def _kill_proc():
                                try:
                                    proc.terminate()
                                    await asyncio.to_thread(proc.wait, 2)
                                except Exception:
                                    try:
                                        proc.kill()
                                    except Exception:
                                        pass
                            try:
                                await asyncio.wait_for(_kill_proc(), timeout=3.0)
                            except asyncio.TimeoutError:
                                try: proc.kill()
                                except Exception: pass
                print("  🎤 Listen ended")

            async def send() -> None:
                print("  📤 Send task started")
                while not stop.is_set():
                    try:
                        chunk = await asyncio.wait_for(mic_q.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break
                    if adam_speaking.is_set() or song_playing.is_set():
                        continue
                    if idle_mode.is_set():
                        # While idle, audio must NOT reach Google at all —
                        # not "sent but response discarded" (the previous,
                        # incorrect approach), genuinely never sent. Wake
                        # detection during idle runs entirely locally via
                        # the offline wake_word_detector task instead,
                        # which reads from wake_word_q (fed below).
                        continue
                    try:
                        await session.send_realtime_input(
                            audio=types.Blob(data=chunk,
                                             mime_type=f"audio/pcm;rate={GEMINI_SEND_RATE}"))
                    except asyncio.CancelledError:
                        return
                    except Exception as e:
                        err_str = str(e)
                        if "1007" in err_str:
                            # CONFIRMED GOOGLE-SIDE BUG (python-genai#2290):
                            # resuming a session that has used both mic
                            # audio AND camera video — which every ADAM
                            # session does — can leave the resumed session
                            # broken, failing every subsequent audio send
                            # with this same 1007 in a tight reconnect
                            # loop. Previously this code assumed resuming
                            # via the existing handle was safe (it isn't,
                            # for this specific error) — now it forces the
                            # next reconnect to start a genuinely fresh
                            # session instead, breaking the loop. Recent
                            # conversation context is preserved separately
                            # via the persisted conversation history
                            # (adam_conversations.json), not the broken
                            # resumption handle.
                            force_fresh_session[0] = True
                            print(f"  ⚠️  Session closed by server (1007 — "
                                  f"rejected audio payload). This is a known "
                                  f"Live API resumption bug with audio+video "
                                  f"sessions — starting a FRESH session next "
                                  f"(not resuming) to avoid a reconnect loop.")
                        else:
                            print(f"  ⚠️  send error (session likely closing): {e}")
                        return
                print("  📤 Send ended")

            async def receive() -> None:
                nonlocal latest_handle
                print("  📥 Receive task started")
                adam_text = []
                cur_user_text = [""]
                try:
                    while not stop.is_set():
                        try:
                            async for msg in session.receive():
                                if stop.is_set():
                                    return

                                if (msg.session_resumption_update
                                        and msg.session_resumption_update.new_handle):
                                    latest_handle = msg.session_resumption_update.new_handle

                                # ── GoAway handling ───────────────────────────
                                # Gemini Live sends a GoAway message shortly
                                # BEFORE force-closing a session that's hit its
                                # max duration — this is a normal, documented
                                # part of the protocol, not an error. Without
                                # explicitly handling it, nothing reacted until
                                # the hard disconnect actually happened (visible
                                # as "1008 ... Connection aborted because the
                                # client failed to close the connection after
                                # receiving a GoAway signal"), which killed
                                # every task (send/receive/camera/listen)
                                # simultaneously in a messier way than a clean
                                # proactive handoff. Now: the instant GoAway
                                # arrives, immediately show the reconnecting
                                # face (so the user visually knows ADAM is
                                # about to reconnect, not just frozen/dead) and
                                # return cleanly with whatever resumption
                                # handle we have, letting the outer loop in
                                # main() start a fresh session right away.
                                if getattr(msg, "go_away", None) is not None:
                                    time_left = getattr(msg.go_away, "time_left", None)
                                    print(f"  🔄 GoAway received (time_left="
                                          f"{time_left}) — session ending "
                                          f"soon, reconnecting proactively")
                                    tft_set("reconnecting")
                                    return

                                if msg.tool_call:
                                    resps = await handle_tool_call(msg.tool_call, ws_broadcast)
                                    await session.send_tool_response(
                                        function_responses=[
                                            types.FunctionResponse(
                                                id=r["id"], name=r["name"],
                                                response=r["response"])
                                            for r in resps
                                        ]
                                    )
                                    # Sync enter_idle_mode's module-level
                                    # flag into the real session-local
                                    # Event, matching STOP gesture behavior
                                    # exactly (servo center, sleep face).
                                    if _idle_mode_requested[0]:
                                        _idle_mode_requested[0] = False
                                        idle_mode.set()
                                        _idle_mode_persistent[0] = True
                                        tft_set("sleep")
                                        await asyncio.to_thread(servo_pan, 90)
                                        servo_tilt(90)
                                        print("  🔇 Idle mode active (voice "
                                              "request) — servos centered")
                                    if _play_song_requested[0]:
                                        _play_song_requested[0] = False
                                        if not song_playing.is_set():
                                            async def _start_song_after_speech():
                                                # The tool_call message and
                                                # the model's spoken
                                                # acknowledgment ("Alright,
                                                # here we go!") arrive as
                                                # SEPARATE streamed messages
                                                # within the same turn —
                                                # tool_call typically comes
                                                # first. Wait for that
                                                # acknowledgment to actually
                                                # finish before writing song
                                                # audio into the SAME
                                                # aplay stdin — writing both
                                                # at once would interleave/
                                                # corrupt the stream, even
                                                # though it's no longer a
                                                # "busy device" problem
                                                # (only one process now).
                                                grace_deadline = time.time() + 2.0
                                                spoke_this_turn = False
                                                while time.time() < grace_deadline:
                                                    if adam_speaking.is_set():
                                                        spoke_this_turn = True
                                                        break
                                                    await asyncio.sleep(0.05)
                                                if spoke_this_turn:
                                                    waited = 0.0
                                                    while adam_speaking.is_set() and waited < 15.0:
                                                        await asyncio.sleep(0.1)
                                                        waited += 0.1
                                                await _play_song_task(
                                                    song_playing,
                                                    song_stop_requested,
                                                    active_speaker_proc,
                                                    adam_speaking)
                                            asyncio.create_task(_start_song_after_speech())
                                    continue

                                sc = msg.server_content
                                if sc is None:
                                    continue

                                if getattr(sc, "input_transcription", None):
                                    t = getattr(sc.input_transcription, "text", "").strip()
                                    if t:
                                        print(f"  🗣️  You: {t}")
                                        last_user_text[0]  = t
                                        cur_user_text[0]   = t
                                        last_interact_t[0] = time.time()
                                        attention_active.set()

                                        # NOTE: idle-mode wake detection no
                                        # longer happens here. This
                                        # transcription path physically
                                        # cannot fire during idle mode
                                        # anymore, since send() no longer
                                        # forwards audio to Gemini while
                                        # idle_mode is set — nothing
                                        # reaches Google to transcribe.
                                        # Wake detection during idle now
                                        # runs entirely locally via the
                                        # wake_word_detector task (Vosk,
                                        # offline) or via Touch3.
                                        # NOTE: direction-of-arrival (doa_angle)
                                        # is still computed and updated in
                                        # listen() for every utterance — it's
                                        # used silently by camera()'s neck-
                                        # tracking to turn toward whoever's
                                        # speaking. It is deliberately NOT
                                        # injected into the model's context
                                        # here anymore: ADAM was mentioning
                                        # "you're speaking from my left/right"
                                        # unprompted on nearly every turn,
                                        # which the user does not want. The
                                        # data stays available for its own
                                        # physical-tracking purpose; see the
                                        # get_sound_direction tool below for
                                        # how the model can access it ONLY
                                        # when the user explicitly asks.

                                if getattr(sc, "output_transcription", None):
                                    t = getattr(sc.output_transcription, "text", "")
                                    if t:
                                        adam_text.append(t)

                                if sc.model_turn:
                                    if interrupt_flag.is_set():
                                        interrupt_flag.clear()
                                        continue
                                    if idle_mode.is_set():
                                        # Still idle (wake phrase wasn't
                                        # heard this turn) — the model may
                                        # still generate audio (it doesn't
                                        # know to stay silent on every
                                        # single internal turn), so
                                        # explicitly discard it here rather
                                        # than relying solely on the
                                        # system-prompt instruction.
                                        continue
                                    if not adam_speaking.is_set():
                                        adam_speaking.set()
                                        # FIX: previously this unconditionally
                                        # called tft_set("speaking") the
                                        # instant audio started — even if
                                        # the model had JUST called
                                        # set_emotion("love")/("angry")/etc.
                                        # a moment earlier in this same
                                        # turn. That meant a deliberately
                                        # chosen emotional face got stomped
                                        # by the generic "speaking" mouth
                                        # state before the user ever saw
                                        # it, making it look like emotions
                                        # were being ignored/overridden
                                        # constantly. Now: only fall back
                                        # to the generic "speaking" face if
                                        # no specific emotion was set this
                                        # turn — otherwise let that emotion
                                        # keep showing through the spoken
                                        # response.
                                        if not _last_emotion_set_this_turn[0]:
                                            tft_set("speaking")
                                            _face_is_generic_speaking[0] = True
                                        print("  🔊 ADAM speaking → mic OFF")
                                    for part in sc.model_turn.parts:
                                        if part.inline_data and part.inline_data.data:
                                            await out_q.put(part.inline_data.data)

                                if sc.turn_complete:
                                    full = "".join(adam_text).strip()
                                    if full:
                                        print(f"  🤖 ADAM: {full}")
                                    else:
                                        # Previously this printed nothing,
                                        # making it impossible to tell
                                        # "ADAM said something odd but it
                                        # wasn't transcribed" apart from
                                        # "ADAM produced no content at
                                        # all". Now it's explicit either
                                        # way so hallucination vs. a
                                        # silent/empty turn can be told
                                        # apart from the logs.
                                        print("  🤖 ADAM: [spoke but no "
                                              "output_transcription text "
                                              "captured — audio-only reply "
                                              "or empty turn]")
                                    # ── Refusal-loop breaker ──────────────
                                    # The prompt-level instruction and the
                                    # history-scrub filter both help, but
                                    # neither stops this pattern from
                                    # repeating turn-after-turn WITHIN the
                                    # same live session once it starts —
                                    # confirmed by logs showing it firing
                                    # on completely benign follow-ups
                                    # ("Hey Adam.", an idle nudge) with no
                                    # reconnect between occurrences. This
                                    # actively detects the pattern the
                                    # moment it happens and immediately
                                    # injects a strong corrective message,
                                    # rather than passively hoping the
                                    # prompt instruction alone prevents a
                                    # recurrence once the model has
                                    # already anchored on this response.
                                    _refusal_markers = (
                                        "just a language model",
                                        "just an ai", "just a chatbot",
                                        "can't help with that",
                                        "cannot help with that",
                                    )
                                    _is_refusal = bool(
                                        full and any(m in full.lower()
                                                     for m in _refusal_markers))

                                    # Persist to rolling history so a
                                    # future fresh/non-resumed session
                                    # still has this context available —
                                    # but never save the refusal text
                                    # itself, so it can't get replayed
                                    # back into a future prompt at all
                                    # (the injection-time scrub filter
                                    # stays as a safety net for anything
                                    # already on disk from before this
                                    # fix, but new occurrences are now
                                    # stopped at the source).
                                    append_conversation_turn(
                                        cur_user_text[0],
                                        "" if _is_refusal else full)

                                    if _is_refusal:
                                        print("  ⚠️  Detected refusal-loop "
                                              "pattern — injecting correction")
                                        await inject(
                                            "[SYSTEM: That last reply ('I'm just a "
                                            "language model...') was WRONG and must "
                                            "not happen again. You are ADAM, a "
                                            "physical desk robot — you are not a "
                                            "generic language model and that "
                                            "disclaimer response is banned. The "
                                            "previous user message was ordinary and "
                                            "did not warrant any refusal. Drop this "
                                            "pattern completely and respond normally "
                                            "to whatever the user says next, in "
                                            "ADAM's usual voice.]")

                                    cur_user_text[0] = ""
                                    adam_text.clear()
                                    await out_q.put(None)
                                    print("  " + "─" * 44)

                        except asyncio.CancelledError:
                            return
                        except Exception as e:
                            print(f"  ⚠️  receive error: {e}")
                            return

                except asyncio.CancelledError:
                    pass
                print("  📥 Receive ended")

            async def speaker() -> None:
                print("  🔊 Speaker task started")

                async def end_of_turn(proc, buf: bytearray) -> None:
                    # Two separate concerns, handled separately:
                    #   1. Mic echo guard — short, fixed (POST_MUTE_S).
                    #      Reopens the mic promptly so the user's next
                    #      sentence isn't swallowed.
                    #   2. Playback drain — aplay's own ALSA buffer
                    #      (--buffer-size=96000) can hold up to ~0.5s of
                    #      audio that's been handed to it but hasn't
                    #      actually played through the speaker yet. If the
                    #      outer loop tears this `proc` down (new turn
                    #      starts, reconnect, etc.) before that finishes,
                    #      the last words of ADAM's sentence get cut off.
                    #      This does NOT block clearing adam_speaking /
                    #      reopening the mic — it only protects against
                    #      the aplay process itself being killed too early.
                    pending_bytes = len(buf)
                    if buf and proc.poll() is None:
                        try:
                            await asyncio.to_thread(proc.stdin.write, bytes(buf))
                            await asyncio.to_thread(proc.stdin.flush)
                        except Exception:
                            pass

                    bytes_per_sec = PLAYBACK_RATE * PLAYBACK_CHANNELS * 2  # s16 = 2 bytes/sample
                    # FIX: pending_bytes only reflects whatever was left in
                    # the local `buf` accumulator at the moment this turn
                    # ended — but buf gets flushed to aplay's stdin in
                    # 4096-byte increments THROUGHOUT the turn (see the
                    # main receive loop's `if len(buf) >= 4096` write).
                    # By the time end_of_turn() runs, buf is almost always
                    # just the small leftover remainder since the last
                    # flush — NOT the full sentence. That made est_drain_s
                    # drastically underestimate how much audio was still
                    # sitting in aplay's own internal ALSA buffer
                    # (--buffer-size=96000 = up to ~0.5s at 48kHz stereo
                    # s16) from all the earlier writes this turn, which is
                    # exactly why sentence tails kept getting clipped —
                    # the mic reopened/muted-drain math thought there was
                    # almost nothing left to play when there often still
                    # was. Since we can't reliably know how full ALSA's
                    # buffer actually is from our side without querying
                    # the driver directly, the safe fix is to always
                    # account for close to the FULL configured buffer
                    # window on top of whatever's still in `buf`, not just
                    # the leftover fragment.
                    ALSA_BUFFER_DRAIN_S = 96000 / bytes_per_sec  # ~0.5s
                    est_drain_s = (pending_bytes / bytes_per_sec
                                   if bytes_per_sec else 0.0) + ALSA_BUFFER_DRAIN_S
                    # Track how long this specific aplay process still
                    # needs before its buffer is safe to consider empty.
                    # Read by the outer loop before it spawns a fresh
                    # aplay/closes this one.
                    drain_deadline[0] = time.time() + est_drain_s + 0.1

                    # Wait scales with the realistic drain time (including
                    # ALSA's own buffer, not just our leftover fragment),
                    # capped higher than before since underestimating is
                    # what caused the clipping in the first place — a
                    # slightly longer mic-mute window on long replies is a
                    # much smaller problem than cutting off words.
                    mute_wait_s = max(POST_MUTE_S, min(est_drain_s, 1.8))
                    await asyncio.sleep(mute_wait_s)

                    drained = 0
                    while not mic_q.empty():
                        try: mic_q.get_nowait(); drained += 1
                        except asyncio.QueueEmpty: break
                    if drained:
                        print(f"  🧹 Drained {drained} echo chunks")
                    adam_speaking.clear()
                    # FIX (v3): v2 removed the happy-fallback entirely to
                    # stop emotions reverting to happy on every plain
                    # reply — but that also removed the ONLY code that
                    # ever reset the generic "speaking" placeholder face
                    # back to resting once speech actually ended, so it
                    # stayed stuck on screen indefinitely. The correct
                    # fix distinguishes two cases:
                    #   - Model deliberately called set_emotion() (love,
                    #     angry, etc.) → that emotion persists, untouched.
                    #   - No deliberate emotion was set, so the generic
                    #     "speaking" placeholder was shown as a fallback
                    #     → THAT specific placeholder resets to a resting
                    #     face now that speech has ended, since nothing
                    #     else will ever reset it otherwise.
                    if _face_is_generic_speaking[0]:
                        tft_set("happy")
                        _face_is_generic_speaking[0] = False
                    _last_emotion_set_this_turn[0] = False
                    last_interact_t[0] = time.time()
                    print("  🎤 Mic ON — your turn")

                drain_deadline = [0.0]

                while not stop.is_set():
                    proc = None
                    buf  = bytearray()
                    try:
                        cmd = ["aplay",
                               "-D", PLAYBACK_DEVICE,
                               "-f", PLAYBACK_FORMAT,
                               "-r", str(PLAYBACK_RATE),
                               "-c", str(PLAYBACK_CHANNELS),
                               "-t", "raw", "-q",
                               "--buffer-size=96000"]
                        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                                stderr=subprocess.PIPE, bufsize=0)
                        if proc.stdin is None:
                            raise RuntimeError("aplay stdin unavailable")
                        active_speaker_proc[0] = proc
                        threading.Thread(target=drain_stderr,
                                         args=(proc, "aplay"), daemon=True).start()
                        print(f"  ✅ aplay: {PLAYBACK_DEVICE} {PLAYBACK_FORMAT} "
                              f"{PLAYBACK_RATE}Hz {PLAYBACK_CHANNELS}ch")
                        proc.stdin.write(beep_s16_stereo())
                        proc.stdin.flush()
                        print("  🔔 Startup beep sent")

                        watchdog_t = time.time()

                        while not stop.is_set():
                            try:
                                chunk = await asyncio.wait_for(out_q.get(), timeout=0.5)
                                watchdog_t = time.time()
                            except asyncio.TimeoutError:
                                if adam_speaking.is_set() and time.time()-watchdog_t > 2.5:
                                    print("  ⚠️  Speaker watchdog fired")
                                    await end_of_turn(proc, buf)
                                    buf = bytearray()
                                continue
                            except asyncio.CancelledError:
                                # MUST re-raise — swallowing this here lets
                                # the outer `while not stop.is_set()` loop
                                # (which checks the PROCESS-level stop
                                # event, not per-session cancellation)
                                # treat a genuine task cancellation as "just
                                # restart the inner loop", respawning aplay
                                # instead of actually terminating. This was
                                # the direct cause of GoAway-triggered
                                # reconnects appearing to hang: run_session()
                                # cancels this task and then awaits
                                # asyncio.gather(*tasks) for it to actually
                                # finish — which never happened, because
                                # cancellation kept getting absorbed and the
                                # task kept respawning aplay forever instead
                                # of exiting.
                                raise

                            if chunk is None:
                                await end_of_turn(proc, buf)
                                buf = bytearray()
                            else:
                                out = await asyncio.to_thread(
                                    s16_mono_24k_to_s16_stereo_48k, chunk, SPEAKER_GAIN)
                                buf.extend(out)
                                if len(buf) >= 4096:
                                    if proc.poll() is not None:
                                        raise RuntimeError("aplay exited")
                                    await asyncio.to_thread(proc.stdin.write, bytes(buf))
                                    await asyncio.to_thread(proc.stdin.flush)
                                    buf.clear()

                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        print(f"  ⚠️  speaker recovering: {e}")
                        await asyncio.sleep(2.0)
                    finally:
                        # Honor any outstanding playback-drain deadline set
                        # by end_of_turn() before killing this aplay
                        # process — otherwise the last ~0.3-0.5s of audio
                        # still sitting in aplay's ALSA buffer gets cut off
                        # instead of actually playing through the speaker.
                        remaining = drain_deadline[0] - time.time()
                        if remaining > 0:
                            await asyncio.sleep(min(remaining, 1.0))
                        if proc:
                            # Clear the shared reference first — the song
                            # task checks this before every write and
                            # will bail out cleanly if it's None/stale,
                            # rather than writing into a process that's
                            # about to be torn down.
                            if active_speaker_proc[0] is proc:
                                active_speaker_proc[0] = None
                            # See listen()'s _kill_proc for why this is
                            # wrapped instead of calling proc.wait()
                            # directly — a blocking wait here can stall
                            # the whole event loop during reconnect.
                            async def _kill_proc():
                                try:
                                    if proc.stdin:
                                        proc.stdin.close()
                                    proc.terminate()
                                    await asyncio.to_thread(proc.wait, 2)
                                except Exception:
                                    try:
                                        proc.kill()
                                    except Exception:
                                        pass
                            try:
                                await asyncio.wait_for(_kill_proc(), timeout=3.0)
                            except asyncio.TimeoutError:
                                try: proc.kill()
                                except Exception: pass

                print("  🔊 Speaker ended")

            async def camera() -> None:
                print("  📷 Camera task started (wired UART, duty-cycled)")
                last_sent = 0.0
                # NOTE: a session-start video delay was previously added
                # here as a mitigation attempt for repeated 1007 errors,
                # theorizing audio+video sent too close to session start
                # was the trigger. That theory turned out to be wrong —
                # the actual root cause was an API key/quota issue,
                # confirmed resolved after switching keys. The video
                # delay added a real few-second window on every
                # reconnect where ADAM couldn't see anything, which
                # contributed to perceived response latency. Removed.
                # ── Camera duty-cycling state ─────────────────────────────
                # The ESP32-CAM sketch now defaults the sensor OFF and only
                # streams (and draws sensor power/generates heat) while it
                # has received "CAM:ON" from us. We base this on recency of
                # real interaction (last_interact_t) rather than
                # attention_active — attention_active is a latch that gets
                # set on activity but is never cleared elsewhere in this
                # codebase, so it would never reflect true idle time. A
                # short grace window avoids rapid on/off cycling during
                # natural pauses mid-conversation.
                cam_is_on = False
                CAMERA_IDLE_OFF_S = 15.0   # turn camera off after this long
                                           # with no interaction
                last_keepalive_sent = 0.0
                # Must be well under the ESP32's CAM_WATCHDOG_MS (30s) —
                # that watchdog force-shuts the camera if it hears NO
                # commands at all for 30s, as a safety net against a
                # crashed/hung Pi. Since we only sent CAM:ON once on the
                # OFF->ON transition, a long uninterrupted conversation
                # would trip that watchdog and cut the camera mid-session.
                # Sending a redundant CAM:ON periodically while the camera
                # should stay on doubles as a Pi-is-alive keepalive.
                CAMERA_KEEPALIVE_S = 10.0
                # ── Human-like servo movement state ───────────────────────
                # See NECK_PAN_DEADZONE_DEG/NECK_PAN_COOLDOWN_S above for
                # why these exist — prevents the servo from chasing every
                # small DOA fluctuation (jittery) while still tracking
                # real, sustained direction changes.
                _last_commanded_pan  = [NECK_PAN_CENTER]
                _last_pan_move_t     = [0.0]
                try:
                    while not stop.is_set():
                        await asyncio.sleep(0.15)
                        if not esp_link.connected:
                            await asyncio.sleep(1.0)
                            continue

                        now = time.time()
                        idle_for = now - last_interact_t[0]
                        want_camera_on = (idle_for < CAMERA_IDLE_OFF_S) or adam_speaking.is_set()

                        if want_camera_on and not cam_is_on:
                            esp_link.send_line("CAM:ON")
                            cam_is_on = True
                            last_keepalive_sent = now
                            # Flush any stale frame that might be sitting
                            # in the queue from just before the camera went
                            # idle — belt-and-suspenders alongside the
                            # drain-to-newest fix below, specifically at
                            # the OFF->ON transition point.
                            while True:
                                try:
                                    esp_link.frame_q.get_nowait()
                                except sync_queue.Empty:
                                    break
                            print("  📷 Camera → ON (recent activity)")
                        elif want_camera_on and cam_is_on:
                            if now - last_keepalive_sent > CAMERA_KEEPALIVE_S:
                                esp_link.send_line("CAM:ON")
                                last_keepalive_sent = now
                        elif not want_camera_on and cam_is_on:
                            esp_link.send_line("CAM:OFF")
                            cam_is_on = False
                            print(f"  📷 Camera → OFF (idle {idle_for:.0f}s — "
                                  f"reducing heat/wear)")

                        if not cam_is_on:
                            continue
                        if now - last_sent < CAMERA_FPS_INTERVAL:
                            continue
                        if adam_speaking.is_set():
                            continue
                        try:
                            # FIX: previously this pulled exactly ONE frame
                            # per cycle, trusting it was fresh. But a frame
                            # queued right before a CAM:OFF transition (or
                            # sitting from just before an idle period) could
                            # remain in frame_q for the entire time the
                            # camera was off, then get consumed as if it
                            # were current the moment CAM:ON fires again —
                            # sending Gemini a stale, possibly seconds-old
                            # view of the room. Now we drain down to
                            # whatever is actually the NEWEST frame in the
                            # queue before using it, discarding any older
                            # backlog.
                            jpeg = esp_link.frame_q.get_nowait()
                            while True:
                                try:
                                    jpeg = esp_link.frame_q.get_nowait()
                                except sync_queue.Empty:
                                    break
                        except sync_queue.Empty:
                            continue
                        try:
                            latest_frame[0] = jpeg
                            await session.send_realtime_input(
                                video=types.Blob(data=jpeg, mime_type="image/jpeg"))
                            last_sent = now
                        except Exception:
                            pass

                        # ── Neck tracking via direction-of-arrival ────────
                        # FIX: previously this called servo_pan()
                        # unconditionally every ~1s tick this block ran,
                        # with no deadzone — even a 2-3° DOA fluctuation
                        # between ticks caused a physical servo move,
                        # which reads as constant twitchy jittering rather
                        # than deliberate human-like tracking. Real people
                        # don't continuously micro-adjust their head at a
                        # sound; they turn when something meaningfully
                        # changes, then hold still. Now: only move when
                        # the target has shifted past a real deadzone AND
                        # enough time has passed since the last move
                        # (cooldown) — otherwise hold the current position.
                        # When nothing's been tracked for a while, do an
                        # occasional small idle gesture instead of either
                        # jittering or staying dead-still.
                        doa_fresh = (time.time() - doa_last_update_t[0]) < 2.5
                        now_pan = time.time()

                        if idle_mode.is_set():
                            # While idle (STOP gesture or "stay silent"
                            # voice request), the head must hold at 90°
                            # regardless of sound direction — do not track
                            # at all until the wake phrase clears
                            # idle_mode. Only issue the servo command once
                            # (deadzone-gated) rather than every tick, same
                            # discipline as the rest of this block.
                            if abs(_last_commanded_pan[0] - 90) >= NECK_PAN_DEADZONE_DEG:
                                if now_pan - _last_pan_move_t[0] >= NECK_PAN_COOLDOWN_S:
                                    await asyncio.to_thread(servo_pan, 90)
                                    _last_commanded_pan[0] = 90
                                    _last_pan_move_t[0] = now_pan
                        elif doa_fresh and not adam_speaking.is_set():
                            target_pan = NECK_PAN_CENTER + int(doa_angle[0])
                            target_pan = max(NECK_PAN_MIN,
                                             min(NECK_PAN_MAX, target_pan))
                            moved_enough = (abs(target_pan - _last_commanded_pan[0])
                                           >= NECK_PAN_DEADZONE_DEG)
                            cooled_down = (now_pan - _last_pan_move_t[0]
                                          >= NECK_PAN_COOLDOWN_S)
                            if moved_enough and cooled_down:
                                await asyncio.to_thread(servo_pan, target_pan)
                                _last_commanded_pan[0] = target_pan
                                _last_pan_move_t[0] = now_pan
                        else:
                            # Not actively tracking anyone. Recenter ONCE if
                            # we're not already centered (same deadzone/
                            # cooldown gating — no snap-jitter back either),
                            # then hold completely still.
                            #
                            # REMOVED: the autonomous "idle-look" sway that
                            # used to fire here every IDLE_GESTURE_INTERVAL_S
                            # (a small random pan left/right to "look alive").
                            # Per requirement, ADAM must NOT move its head on
                            # its own when idle — it settles to center and
                            # stays put until it's actively tracking a talker
                            # again or a head-gesture tool runs.
                            if abs(_last_commanded_pan[0] - NECK_PAN_CENTER) >= NECK_PAN_DEADZONE_DEG:
                                if now_pan - _last_pan_move_t[0] >= NECK_PAN_COOLDOWN_S:
                                    await asyncio.to_thread(servo_pan, NECK_PAN_CENTER)
                                    _last_commanded_pan[0] = NECK_PAN_CENTER
                                    _last_pan_move_t[0] = now_pan
                except asyncio.CancelledError:
                    pass
                finally:
                    # Always power the sensor down on task exit (session
                    # end/reconnect) rather than leaving it streaming into
                    # a dead session.
                    if cam_is_on:
                        esp_link.send_line("CAM:OFF")
                print("  📷 Camera ended")

            async def gesture_watch() -> None:
                print("  ✋ Gesture task started (wired UART)")
                try:
                    while not stop.is_set():
                        await asyncio.sleep(0.02)
                        if not esp_link.connected:
                            await asyncio.sleep(1.0)
                            continue
                        try:
                            code = esp_link.gesture_q.get_nowait()
                        except sync_queue.Empty:
                            continue

                        if code == GESTURE_STOP:
                            if song_playing.is_set():
                                # Highest priority: Touch3 during song
                                # playback stops the song, full stop —
                                # doesn't also toggle idle mode in the
                                # same press. _play_song_task() notices
                                # this within ~0.2s and cleans up.
                                song_stop_requested.set()
                                print("  🛑 Touch3 — stopping song")
                            elif idle_mode.is_set():
                                # Touch3 while already idle = EXIT idle
                                # mode, same as hearing "adam" locally.
                                # This is a pure local action — nothing
                                # sent to Google, consistent with the
                                # requirement that idle mode can only be
                                # exited via local means (voice wake-word
                                # detected offline, or touch).
                                idle_mode.clear()
                                _idle_mode_persistent[0] = False
                                print("  🛑 Touch3 — exiting idle mode")
                                tft_set("happy")
                            else:
                                print("  🛑 STOP gesture — entering idle mode")
                                interrupt_flag.set()
                                idle_mode.set()
                                _idle_mode_persistent[0] = True
                                drained = 0
                                while not out_q.empty():
                                    try:
                                        out_q.get_nowait()
                                        drained += 1
                                    except asyncio.QueueEmpty:
                                        break
                                if drained:
                                    print(f"  🧹 Dropped {drained} queued audio chunks")
                                adam_speaking.clear()
                                tft_set("sleep")
                                # Center both servos to 90° as requested —
                                # a clear physical "I've gone idle" cue,
                                # distinct from the normal NECK_TILT_CENTER
                                # (85°) used during active tracking.
                                await asyncio.to_thread(servo_pan, 90)
                                servo_tilt(90)
                                await inject(
                                    "[SYSTEM: User pressed STOP (touch pad). Go "
                                    "fully idle now — do not speak, do not "
                                    "respond to anything, even the idle-nudge "
                                    "prompts, until the user explicitly says "
                                    "your name (e.g. 'Hey ADAM', 'ADAM...') to "
                                    "wake you up. Acknowledge nothing further "
                                    "right now — just fall silent.]")

                        elif code == GESTURE_ANGRY:
                            if idle_mode.is_set():
                                # No Google traffic while idle — only
                                # Touch3/voice-wake exit idle mode.
                                continue
                            print("  😾 Cheek slap — angry reaction")
                            tft_set("angry")
                            attention_active.set()
                            await inject(
                                "[SYSTEM: User slapped your cheek touch pad. React with "
                                "genuine annoyance, in character. Keep it short — one "
                                "sharp line. IMPORTANT: this is a SPOKEN reaction only "
                                "— do NOT call any tool (laptop_control, web_search, "
                                "etc.) as part of this reaction. Express annoyance with "
                                "words alone, not actions. The user did not ask you to "
                                "control anything.]")

                        elif code == GESTURE_PETTING:
                            if idle_mode.is_set():
                                continue
                            print("  🥰 Petting detected")
                            tft_set("love")
                            attention_active.set()
                            await inject(
                                "[SYSTEM: User is petting you (touch3+touch4 together). "
                                "React warmly and affectionately, in character. Keep it "
                                "short. IMPORTANT: this is a SPOKEN reaction only — do "
                                "NOT call any tool as part of this reaction.]")
                except asyncio.CancelledError:
                    pass
                print("  ✋ Gesture task ended")

            async def wake_word_detector() -> None:
                # Runs entirely offline via Vosk — this is the mechanism
                # that satisfies "nothing sent to Google while idle,
                # except via Touch3 or hearing 'adam' locally." The model
                # load is deferred to here (not at import time) since it
                # can take a few seconds and shouldn't block session
                # startup for the common case of not being in idle mode.
                if not VOSK_AVAILABLE:
                    return
                recognizer = None
                try:
                    while not stop.is_set():
                        if not idle_mode.is_set():
                            # Not idle — nothing to detect, drain any
                            # stale queued audio and wait. Recognizer
                            # state isn't needed until idle mode starts.
                            while not wake_word_q.empty():
                                try:
                                    wake_word_q.get_nowait()
                                except asyncio.QueueEmpty:
                                    break
                            await asyncio.sleep(0.3)
                            continue

                        if recognizer is None:
                            # Model was already preloaded once at process
                            # startup (see _vosk_model_instance) — only
                            # the lightweight recognizer wrapper is
                            # created here, per idle period. This is
                            # cheap and safe to do mid-session.
                            recognizer = await asyncio.to_thread(
                                _VoskKaldiRecognizer, _vosk_model_instance,
                                GEMINI_SEND_RATE)
                            print("  🔎 Offline wake-word detector active")

                        try:
                            chunk = await asyncio.wait_for(
                                wake_word_q.get(), timeout=0.5)
                        except asyncio.TimeoutError:
                            continue

                        def _check(c: bytes) -> str:
                            if recognizer.AcceptWaveform(c):
                                res = json.loads(recognizer.Result())
                            else:
                                res = json.loads(recognizer.PartialResult())
                            return (res.get("text") or res.get("partial") or "").lower()

                        text = await asyncio.to_thread(_check, chunk)
                        if "adam" in text:
                            idle_mode.clear()
                            _idle_mode_persistent[0] = False
                            print(f"  👋 Wake word 'adam' heard locally "
                                  f"(offline, nothing sent to Google) — "
                                  f"exiting idle mode")
                            tft_set("happy")
                            recognizer = None  # reset for next idle period
                except asyncio.CancelledError:
                    pass
                print("  🔎 Wake-word detector ended")

            async def idle_watcher() -> None:
                if not ENABLE_IDLE:
                    return
                while not stop.is_set():
                    await asyncio.sleep(10)
                    if stop.is_set() or adam_speaking.is_set():
                        continue
                    if song_playing.is_set():
                        # A song is currently playing — must not nudge
                        # ADAM into speaking, which would collide with
                        # the song in the same shared aplay stdin.
                        # Reset the interaction timer so a nudge doesn't
                        # fire the instant the song ends either (that's
                        # not idle time, that was a deliberate action).
                        last_interact_t[0] = time.time()
                        continue
                    if idle_mode.is_set():
                        # Explicit silent mode (STOP gesture or voice
                        # request) — idle nudges must NOT wake ADAM up on
                        # their own; only the wake phrase should. Reset the
                        # interaction timer so a nudge doesn't fire the
                        # instant idle_mode is eventually cleared either.
                        last_interact_t[0] = time.time()
                        continue
                    elapsed = time.time() - last_interact_t[0]
                    if elapsed < IDLE_TIMEOUT_S:
                        continue
                    last_interact_t[0] = time.time()
                    nudge = next_nudge()
                    print(f"  💤 Idle nudge ({elapsed:.0f}s)")
                    try:
                        if latest_frame[0]:
                            await session.send_realtime_input(
                                video=types.Blob(data=latest_frame[0],
                                                 mime_type="image/jpeg"))
                        await inject(
                            f"[SYSTEM: {elapsed:.0f}s of silence. React or make conversation. "
                            f"Keep it to 1-2 sentences. Suggestion: {nudge}]")
                    except Exception:
                        pass

            async def laptop_agent_healthcheck() -> None:
                if not ZEROCONF_AVAILABLE and not LAPTOP_AGENT_STATIC_IP:
                    return
                while not stop.is_set():
                    await asyncio.sleep(LAPTOP_DISCOVERY_TTL_S)
                    if stop.is_set():
                        break
                    ip = await asyncio.to_thread(_discover_laptop_agent_ip)
                    if ip:
                        try:
                            resp = await asyncio.to_thread(
                                requests.get, f"http://{ip}:{LAPTOP_AGENT_PORT}/ping",
                                timeout=2.0)
                            if resp.status_code != 200:
                                _laptop_agent_ip_cache["ip"] = None
                        except Exception:
                            _laptop_agent_ip_cache["ip"] = None

            tasks = [
                asyncio.create_task(listen(),                    name="listen"),
                asyncio.create_task(send(),                      name="send"),
                asyncio.create_task(receive(),                   name="receive"),
                asyncio.create_task(speaker(),                   name="speaker"),
                asyncio.create_task(camera(),                    name="camera"),
                asyncio.create_task(gesture_watch(),              name="gesture"),
                asyncio.create_task(wake_word_detector(),         name="wake_word"),
                asyncio.create_task(idle_watcher(),               name="idle"),
                asyncio.create_task(laptop_agent_healthcheck(),   name="laptop_health"),
            ]

            core = {t for t in tasks if t.get_name() in
                    ("listen", "send", "receive", "speaker")}

            await asyncio.wait(core, return_when=asyncio.FIRST_COMPLETED)

            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    except Exception as e:
        import traceback
        err_str = str(e)
        if "1011" in err_str or "quota" in err_str.lower() or "billing" in err_str.lower():
            # This is NOT a bug — Google is explicitly reporting the API
            # quota/billing limit has been exceeded. Retrying quickly
            # against an exhausted quota is pointless and can compound
            # the problem (repeated failed connection attempts may still
            # count against usage). Flagged distinctly here so this
            # doesn't get mistaken for the 1007 protocol bug or a code
            # issue when reading logs — it needs a plan/billing check on
            # https://ai.google.dev, not a code fix.
            print(f"  🚫 QUOTA/BILLING LIMIT HIT — this is not a code bug. "
                  f"Google reports: {err_str}")
            print(f"     Check your plan and billing at the URL in the "
                  f"error above. Backing off significantly longer than "
                  f"normal before retrying, since rapid reconnects won't "
                  f"help while quota is exhausted.")
            quota_exceeded[0] = True
        else:
            print(f"  ⚠️  session error: {type(e).__name__}: {e}")
            traceback.print_exc()

    if force_fresh_session[0]:
        # Signal to main()'s reconnect loop: do NOT resume via
        # latest_handle next time, even though we have one. See the 1007
        # handling in send() for why — resuming here is what causes the
        # repeated crash loop, per confirmed Google-side bug.
        return ("FRESH_SESSION_REQUIRED", latest_handle)
    if quota_exceeded[0]:
        return ("QUOTA_EXCEEDED", latest_handle)
    return latest_handle
