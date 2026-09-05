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
  • wake_word_detector()      — offline Vosk: "adam" wake word during idle
                                mode, and a spoken stop phrase during songs
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
import socket        # only for the gaierror/herror classes in the network-fault classifier below
import subprocess
import collections
import statistics   # median() for the VAD sustain window — impulse-immune by construction
import queue as sync_queue

import requests
from google.genai import types

from config import (
    LIVE_MODEL, VOICE, STT_LANGUAGE_CODES,
    MIC_Q_MAX, CHUNK_FRAMES,
    CAPTURE_DEVICE, CAPTURE_FORMAT, CAPTURE_RATE, CAPTURE_CHANNELS,
    MIC_SILENCE_FLOOR,
    MIC_SPEECH_MARGIN, MIC_AMBIENT_INIT, MIC_AMBIENT_MAX,
    MIC_VAD_RELEASE_RATIO, MIC_VAD_HANGOVER_S, MIC_VAD_PREROLL_S,
    MIC_VAD_HOLD_MARGIN, MIC_VAD_MAX_OPEN_S, MIC_VAD_ABS_MAX_OPEN_S,
    MIC_VAD_OPEN_MARGIN, MIC_VAD_MAX_HOLD_RATIO,
    MIC_NOISE_LEARN_COOLDOWN_S, MIC_WARMUP_S, MIC_STATS_S, MIC_VAD_ONSET_CHUNKS,
    MIC_VAD_ONSET_WINDOW,
    MIC_DEAD_STREAM_S, MIC_DEAD_AFTER_PLAY_S, MIC_DEAD_AFTER_PLAY_WINDOW_S,
    MIC_VAD_SUSTAIN_S,
    MIC_ADAPTIVE,
    DOA_ANGLE_DEADZONE, GEMINI_SEND_RATE, POST_MUTE_S,
    MIC_ECHO_GUARD_S, MIC_ECHO_GUARD_MARGIN,
    PLAYBACK_DEVICE, PLAYBACK_FORMAT, PLAYBACK_RATE, PLAYBACK_CHANNELS,
    SPEAKER_GAIN, SPEAKER_IDLE_CLOSE_S, SPEAKER_START_DELAY_US,
    SPEAKER_DRAIN_ALLOWANCE_S,
    NECK_PAN_CENTER, NECK_PAN_MIN, NECK_PAN_MAX,
    NECK_PAN_DEADZONE_DEG, NECK_PAN_COOLDOWN_S,
    CAMERA_FPS_INTERVAL,
    GESTURE_STOP, GESTURE_ANGRY, GESTURE_PETTING,
    ENABLE_IDLE, IDLE_TIMEOUT_S, IDLE_MAX_S, next_nudge,
    LAPTOP_DISCOVERY_TTL_S, LAPTOP_AGENT_PORT, LAPTOP_AGENT_STATIC_IP,
    BASE_DIR,
)

from hardware import servo_pan, servo_tilt, tft_set, servo_moving   # tft_set re-exported for main.py
from esp32_link import esp_link
from audio_utils import (
    read_exact, write_all, drain_stderr, rms_pcm16, is_valid_pcm16_chunk,
    beep_s16_stereo, spk_clip_samples, spk_total_samples,
    s32_stereo_to_s16_mono_16k, s32_stereo_to_s16_stereo_channels,
    denoise_16k, denoise_reset, denoise_db,
    estimate_doa_angle, s16_mono_24k_to_s16_stereo_48k,
    AdaptiveGate,
)
from memory_store import append_conversation_turn
from system_prompt import build_system_prompt
from tools_schema import build_tools
from song_playback import _play_song_task
from tool_handler import (
    handle_tool_call,
    _idle_mode_requested, _idle_mode_persistent, _play_song_requested,
    _idle_since,
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
        # LANGUAGE HINTS, not full auto-detection. An empty
        # AudioTranscriptionConfig() means "score this audio against every
        # language you know and return the best match", which is how accented
        # Hindi fragments came back as Portuguese and Spanish. Naming the
        # languages actually spoken removes those candidates entirely. See
        # STT_LANGUAGE_CODES in config.py for the measurement and the trade.
        # Only the INPUT side is constrained — the output transcription is of
        # ADAM's own speech, which the model already knows the language of.
        input_audio_transcription=types.AudioTranscriptionConfig(
            language_codes=STT_LANGUAGE_CODES or None),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        # MANUAL TURN BOUNDARIES — the fix for "I spoke three times before
        # ADAM answered, and it answered all three at once."
        #
        # By default the Live API runs its OWN server-side VAD over the audio
        # we stream, and decides the user's turn has ended when it HEARS
        # enough silence. That is fundamentally incompatible with what
        # listen() does: listen() gates the stream, so during silence we send
        # NOTHING AT ALL. The server therefore never receives the silence it
        # is waiting for — from its point of view the utterance simply never
        # ends, and the next time our gate reopens that audio is appended to
        # the same turn. Measured live, three separate gate openings ~30s
        # apart came back as ONE transcript,
        #   "Hey, hello madam. Hey. Hello madam."
        # i.e. three utterances concatenated, answered once, late. The delay
        # was never network latency or model speed; nothing had told Gemini
        # the user had stopped talking.
        #
        # So: disable the server VAD and send the boundaries ourselves.
        # listen()'s Schmitt-trigger gate is already a better VAD for this
        # room than a remote one working through a hard-gated stream can be,
        # and it knows the instant speech ends — MIC_VAD_HANGOVER_S (1.0s)
        # after the level drops, comfortably above the 500ms minimum Google
        # documents for a client-side end-of-speech threshold. send() emits
        # activity_start on the first chunk of an utterance and activity_end
        # the moment the gate closes, which is what makes the reply start
        # immediately instead of whenever the server gives up waiting.
        #
        # Consequence to keep in mind: manual mode has no server-side
        # pre-speech buffer, so the audio BEFORE speech onset must come from
        # us. It already does — MIC_VAD_PREROLL_S of ringed pre-onset chunks
        # is flushed into mic_q on open, and it is flushed AFTER
        # activity_start, so none of it lands outside the activity window
        # where it would be discarded.
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=True,
            ),
        ),
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
    # Set to True when the failure was LOCAL networking, not the API:
    # DNS lookup failed, no route, connection refused/reset. On this Pi
    # 28 of 39 reconnects in one boot were
    # `socket.gaierror: [Errno -3] Temporary failure in name resolution`,
    # all clustered in the first minute — wlan0 had a link but the
    # resolver was not answering yet, despite the unit's
    # Wants=network-online.target. Those are indistinguishable from a
    # real API failure to main()'s reconnect loop, which means each one
    # costs an exponential backoff (up to 30s of total deafness) AND
    # discards the resumption handle, so ADAM forgets the conversation
    # over a single dropped UDP query. Flagged separately so it can be
    # retried quickly with the handle intact.
    network_transient = [False]

    # END-OF-TURN MARKER pushed through mic_q by listen() and consumed by
    # send(). A sentinel in the audio queue rather than an Event or a second
    # queue because ORDER is the whole point: activity_end has to reach
    # Gemini AFTER the last audio chunk of the utterance it closes. Anything
    # out-of-band races the audio still sitting in mic_q and would truncate
    # the final word.
    ACTIVITY_END = object()

    try:
        async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
            print("  ✅ Connected to Gemini Live")

            mic_q            = asyncio.Queue(maxsize=MIC_Q_MAX)
            adam_speaking    = asyncio.Event()
            # Wall-clock instant the mic reopens after ADAM finishes speaking.
            # The room's reverb tail outlives the mute window: measured live, the
            # first unmuted chunk after a reply read RMS 2,693 against an open_th
            # of 2,300 and briefly opened the gate on ADAM's own voice. Feeding
            # that back to Gemini is how a model starts answering itself. The gate
            # uses this to demand extra margin for MIC_ECHO_GUARD_S rather than
            # staying muted longer, so a fast human reply is still captured.
            mic_open_t       = [0.0]
            latest_frame     = [None]
            attention_active = asyncio.Event()
            last_interact_t  = [time.time()]
            # last_interact_t is "something happened", and a NUDGE counts as
            # something — so it cannot answer "did the USER say anything?".
            # These two can. See the guard on _idle_mode_requested below:
            # ADAM must not be allowed to talk itself into idle mode by
            # nudging an empty room and then reacting to the silence.
            last_user_turn_t = [0.0]
            last_nudge_t     = [0.0]
            last_user_text   = [""]
            interrupt_flag   = asyncio.Event()
            # ── Song playback state ──────────────────────────────────────
            # song_playing: mic-mute gate for the duration of playback
            # (listen()/send() check this the same way they check
            # adam_speaking, so nothing extra needed there).
            # song_stop_requested: set by Touch3 while a song is playing —
            # or by wake_word_detector() hearing a stop phrase offline — to
            # end playback early. Distinct from GESTURE_STOP's normal
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
            # ── AMP HISS GUARD ───────────────────────────────────────────
            # True whenever aplay owns the playback device, plus a stamp of
            # when it last let go. Both live here, in run_session's scope,
            # because listen() and speaker() are nested in it — no module
            # global, no import between them.
            #
            # This exists because of a measured self-deafening loop, and it
            # is the direct cause of "i am talking but adam is not
            # responding". The voiceHAT's class-D amplifier hisses into the
            # mic for as long as the PCM device is OPEN, not just while
            # audio is flowing: with aplay up, the room's measured noise bed
            # rose from p50 ~1,558 to ~2,470 post-filter RMS, +4 dB, with
            # nothing playing. _read_and_convert() mutes the mic while
            # adam_speaking is set, so the loud part is already excluded —
            # but SPEAKER_IDLE_CLOSE_S keeps the device open for 2.5s AFTER
            # the reply ends, and every one of those ~75 chunks was being
            # fed to AdaptiveGate.observe(). The gate is a percentile of a
            # 45s window, so it faithfully learned the amplifier as the
            # room: open_th climbed to 2,989 while this user's quietest
            # measured speech is 2,357. ADAM had raised its own hearing
            # threshold above its owner's voice, one reply at a time, and
            # every reply made the next one harder to hear.
            #
            # The floor must therefore be learned only from air, never from
            # ADAM's own electronics. Accepted tradeoff: through a 3-minute
            # song observe() is starved and the estimate goes stale. A stale
            # floor is recoverable (MIC_FLOOR_FALL is the fast direction, so
            # it re-converges within seconds of the ring refilling); a floor
            # inflated above the user's voice is not recoverable at all,
            # because nothing the user can say will reopen the gate.
            amp_open    = [False]
            amp_quiet_t = [0.0]
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
            # When the CURRENT idle period began, for IDLE_MAX_S below. A
            # module-level box, not a session-local, so a reconnect does not
            # restart the clock: the log that motivated this shows idle mode
            # surviving a reconnect and then never ending.
            if _idle_mode_persistent[0]:
                idle_mode.set()
                if not _idle_since[0]:
                    _idle_since[0] = time.time()
                print("  🔇 Resuming idle mode after reconnect "
                      "(servos re-centered)")
                tft_set("sleep")
                await asyncio.to_thread(servo_pan, 90)
                servo_tilt(90)
            else:
                _idle_since[0] = 0.0
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
                # Every level constant in config.py was calibrated with
                # adam.service STOPPED, because the diagnostics need exclusive
                # use of the capture device. That is not the operating
                # condition: live, aplay holds the class-D amp enabled for the
                # whole session, the camera duty-cycles, Vosk and the WebSocket
                # server are resident, and the CPU is loaded. The old print here
                # sampled ONE chunk every 4s — 1 in 120 — which can show the
                # median but structurally cannot show the tail, and the tail is
                # what opens the gate. 122 opens in one 30-minute run against a
                # floor that a stopped-service measurement said was never
                # reached in 30s of quiet is the discrepancy that needs it.
                # So: accumulate every chunk's RMS and report the DISTRIBUTION
                # once per window, with the gate's own counters beside it.
                _rms_hist: list[float] = []
                _win_opens = [0]
                _win_sent  = [0]
                # ONSET QUORUM. _onset_win holds one 1/0 verdict per chunk for
                # the last MIC_VAD_ONSET_WINDOW chunks while the gate is shut;
                # the gate opens when MIC_VAD_ONSET_CHUNKS of them passed, in
                # any order. It replaced a consecutive-run counter, which
                # ordinary speech could not satisfy — an unvoiced consonant or
                # a stop closure reset it to zero, so "Hey ADAM" never got 5
                # clean chunks in a row and the gate never opened. See the
                # MIC_VAD_ONSET_WINDOW block in config.py.
                # _win_blocked counts onset attempts that collected passing
                # chunks but decayed back to zero without reaching the quorum:
                # transients rejected on duration, i.e. what the test is
                # actually buying, measured rather than assumed.
                _onset_win   = collections.deque(
                    maxlen=max(MIC_VAD_ONSET_CHUNKS, MIC_VAD_ONSET_WINDOW))
                _win_blocked = [0]
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
                _ambient_rms = [MIC_AMBIENT_INIT]   # filtered-int16 RMS units
                # Asymmetric adaptation. Falling toward a quieter room is safe
                # and should be quick; RISING is how the estimator gets poisoned
                # (see below), so it rises 10x slower than it falls.
                _AMBIENT_ALPHA_DOWN = 0.10          # ~0.3s at 30 chunks/s
                _AMBIENT_ALPHA_UP   = 0.01          # ~3.3s
                # Decaying MAXIMUM of the same non-speech chunks. Because the
                # average above is deliberately biased toward the quiet end, it
                # under-reads the room: measured quiet chunks spanned 1,450-1,990
                # while the average settled near 1,450-1,500. A threshold sized
                # off the average alone therefore sits below real noise peaks and
                # the gate latches on them. 0.999/chunk ~= a 33s bleed-down, long
                # enough to remember an intermittent noise between utterances.
                _noise_peak        = [MIC_AMBIENT_INIT]
                _NOISE_PEAK_DECAY  = 0.999

                # ── VAD state machine ─────────────────────────────────────
                # _vad_open  : gate currently passing audio through
                # _vad_last_loud_t : last time level cleared the RELEASE
                #              threshold — hangover is measured from here
                # _vad_last_strong_t : last time level cleared the OPEN
                #              threshold — the latch watchdog is measured from
                #              here, so a steady noise that only ever clears the
                #              lower hold threshold cannot pin the gate open
                # _preroll   : ring buffer of the most recent pre-onset chunks,
                #              flushed into mic_q when the gate opens so the
                #              attack of the first word isn't lost
                _vad_open          = [False]
                _vad_last_loud_t   = [0.0]
                _vad_last_strong_t = [0.0]
                _vad_opened_t      = [0.0]   # when the current open run began
                # When the gate last went shut. The noise trackers must ignore
                # MIC_NOISE_LEARN_COOLDOWN_S of audio after this instant: the
                # chunks immediately following a close are the reverb tail and
                # trailing consonants of the utterance that just ended, and
                # learning them as "room noise" is a positive feedback loop that
                # walks the gate up out of the speech range. Initialised in the
                # future by the arecord warm-up, so the very first chunks teach
                # the trackers without being judged by them.
                _vad_closed_t      = [0.0]
                _refractory_until  = [0.0]   # gate cannot reopen before this
                # Previous chunk's gate state, so the FALLING edge can be
                # detected in one place. Every close path below (hangover,
                # soft watchdog, hard watchdog, and the mute branch) has to
                # emit exactly one end-of-turn marker, and edge-detecting
                # here is the only way to guarantee that without repeating
                # the push at four sites and eventually missing one.
                _gate_prev         = [False]
                # Consecutive chunks of DIGITAL SILENCE (RMS ~0). See the
                # dead-capture watchdog in the read loop.
                _dead_run          = [0]
                _chunks_per_s    = max(1.0, CAPTURE_RATE / float(CHUNK_FRAMES))
                _preroll_n       = max(0, int(round(MIC_VAD_PREROLL_S * _chunks_per_s)))
                _preroll         = collections.deque(maxlen=_preroll_n or 1)
                _dead_limit      = max(1, int(round(MIC_DEAD_STREAM_S
                                                    * _chunks_per_s)))
                # Shorter fuse for the window right after a playback close,
                # where the wedge is not a mystery but the known consequence of
                # tearing down the shared I2S device. See
                # MIC_DEAD_AFTER_PLAY_S.
                _dead_limit_amp  = max(1, int(round(MIC_DEAD_AFTER_PLAY_S
                                                    * _chunks_per_s)))
                # SUSTAIN WINDOW — the fix for the gate latching OPEN forever.
                #
                # Measured live, four consecutive 10s windows with the gate
                # stuck open and no "🤫 Speech ended" between them:
                #   p50 1705 p90 2895 p99 5448 max 9970 | open≥1899 hold≥1731
                #   p50 1632 p90 1831 p99 2394 max 2558 | opens 0 sent 301
                # The MEDIAN was 1631-1705, i.e. BELOW hold≥1731, so on level
                # the gate should have released — but the hangover was armed off
                # the INSTANTANEOUS chunk, and p90 1831-2895 means 10-25% of
                # noise chunks clear hold_th. One such chunk every 4-10 chunks
                # (0.13-0.33s) resets a 0.8s hangover, so it never expires. The
                # soft latch watchdog failed the same way: p99 clears open_th,
                # so _vad_last_strong_t kept being refreshed too, and only the
                # 45s absolute watchdog could break out — 45s of deafness at a
                # time ("sometimes it is not at all listening to me"). Under
                # manual activity detection it is worse than deafness: no gate
                # close means no activity_end, so Gemini is never told the turn
                # ended and never replies at all.
                #
                # So the SUSTAIN decisions (keep the gate open, and "something
                # speech-loud happened recently") now read a rolling median
                # instead of one chunk. A median over ~0.5s cannot be moved by
                # an impulse by construction — that needs >50% of the window —
                # while connected speech sits above hold_th for far more than
                # half of any half-second. Intra-word stops are 50-150ms, well
                # under the window, so they cannot release the gate either.
                # ATTACK stays on the instantaneous chunk plus the onset
                # quorum, so opening is as fast as it ever was.
                _sustain_n       = max(3, int(round(MIC_VAD_SUSTAIN_S
                                                    * _chunks_per_s)))
                _lvl_win         = collections.deque(maxlen=_sustain_n)
                # ADAPTIVE GATE — the production-ready replacement for the
                # hand-tuned absolute thresholds above. It learns THIS room's
                # noise floor from a low percentile of a long window (immune to
                # the speech it measures, unlike the EMA trackers, which is why
                # they needed a MIC_AMBIENT_MAX clamp that then became the very
                # thing capping them below a noisy room's floor), and it carries
                # a second, LEVEL-INDEPENDENT vote from the spectral shape of
                # each chunk — flatness plus a low/high band ratio, with the
                # flatness threshold itself learned from the room's own noise
                # bed — so a room whose noise sits ON TOP of speech level is
                # still separable, and a room nobody measured still works. See
                # the ADAPTIVE SPEECH GATE block in audio_utils.py for the full
                # rationale and the measurements that forced it.
                #
                # Constructed per session, but its learned floor persists to
                # MIC_FLOOR_STATE_PATH and is reloaded on start, so a reconnect
                # or a service restart does not throw the room away and go
                # through another cold warm-up.
                _agate           = AdaptiveGate(_chunks_per_s)
                # DOA rate limit — see the GCC-PHAT block in the loop below.
                _last_doa_t   = [0.0]
                _DOA_MIN_GAP_S = 0.25

                def _read_and_convert(pipe, nbytes):
                    """Read one chunk AND band-limit it in a single worker-thread
                    hop. Returns (raw_s32, mono16k, mono16k_nr) — mono16k is
                    None while the mic is muted, so the FIR work is skipped at
                    exactly the moment the speaker task needs the CPU.

                    mono16k     — what the GATE sees: unmodified, so every
                                  threshold and learned statistic keeps the
                                  meaning it was tuned with.
                    mono16k_nr  — what GEMINI and VOSK see: noise-suppressed.
                                  Measured SNR in this room is only ~+6 dB,
                                  which the gate handles and a recogniser does
                                  not. Denoising runs on every chunk, including
                                  the ones the gate rejects, because its noise
                                  estimator needs the continuous stream — that
                                  is also why it lives here and not at the
                                  queue, where gate-closed chunks never arrive.
                    """
                    raw = read_exact(pipe, nbytes)
                    if adam_speaking.is_set():
                        return raw, None, None
                    # song_playing used to mute here too. It no longer does,
                    # because the OFFLINE stop-word detector needs audio during a
                    # song: the songs are 182-215s long and the only other way to
                    # stop one is the Touch3 gesture, which arrives over the
                    # ESP32-CAM UART — dead whenever the camera link is down. That
                    # left "adam is not responding to anything, just started
                    # playing the song": three and a half minutes of deafness with
                    # no way out. The audio still never reaches Gemini; the song
                    # branch in the gate below routes it to Vosk and nowhere else.
                    mono16k = s32_stereo_to_s16_mono_16k(raw)
                    return raw, mono16k, denoise_16k(mono16k)

                while not stop.is_set():
                    proc = None
                    try:
                        cmd = ["arecord",
                               "-D", CAPTURE_DEVICE,
                               "-f", CAPTURE_FORMAT,
                               "-r", str(CAPTURE_RATE),
                               "-c", str(CAPTURE_CHANNELS),
                               "-t", "raw",
                               "--buffer-size=48000"]
                        # --buffer-size is in FRAMES, so 48000 = 1.0s of slack
                        # between arecord and this loop. The default is a few
                        # hundred ms, and the journal showed
                        #   [arecord] overrun!!! (at least 2097.350 ms long)
                        # — once the reader stalls past the buffer, ALSA throws
                        # away everything that arrived in the meantime, so the
                        # user's speech is gone before any gate can see it. A
                        # bigger buffer converts a scheduling hiccup from lost
                        # audio into harmless latency.
                        #
                        # No "-q" either — it would hide those overrun warnings,
                        # which are indistinguishable from "the mic didn't hear
                        # me" without the message.
                        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                                stderr=subprocess.PIPE, bufsize=0)
                        await asyncio.sleep(1.0)
                        if proc.poll() is not None:
                            err = proc.stderr.read().decode(errors="replace").strip()
                            print(f"  ❌ arecord failed: {err}")
                            await asyncio.sleep(3.0)
                            continue

                        # Only start draining stderr AFTER the startup check
                        # above — that check reads proc.stderr directly to
                        # report why arecord died, and a drain thread started
                        # earlier would race it and swallow the message.
                        threading.Thread(target=drain_stderr,
                                         args=(proc, "arecord"), daemon=True).start()

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

                        # Hold the gate shut while the trackers meet the room.
                        # They start at MIC_AMBIENT_INIT — a guess, not a
                        # measurement — so without this the first chunk is
                        # judged against a cold threshold. Live, that is exactly
                        # what happened: chunk ~1 read 2,460 against a cold
                        # open_th of 2,430, the gate opened on nothing, and
                        # because the trackers freeze while it is open only the
                        # 45s absolute-cap watchdog could break the latch. The
                        # refractory branch still runs the tracker updates, so
                        # this window is spent measuring, not idling.
                        _refractory_until[0] = time.time() + MIC_WARMUP_S
                        _vad_closed_t[0]     = 0.0

                        while not stop.is_set():
                            try:
                                raw, mono16k, mono16k_nr = await asyncio.to_thread(
                                    _read_and_convert, proc.stdout, read_bytes)
                            except Exception as e:
                                errors += 1
                                if errors > 5:
                                    print(f"  ⚠️  arecord read: {e} — restarting")
                                    break
                                await asyncio.sleep(0.5)
                                continue
                            errors = 0

                            if mono16k is None:
                                # Muted (ADAM speaking). The pipe is still being
                                # drained every iteration — that
                                # is what prevents the ALSA capture overruns
                                # documented below — but nothing is filtered,
                                # metered or queued.
                                #
                                # The gate is also forced SHUT, not left as it
                                # was. Nothing in this branch touches the VAD
                                # state, so an open gate used to survive the
                                # whole reply and reappear as "open" on the first
                                # unmuted chunk — with a _vad_last_loud_t stamped
                                # seconds ago. Two consequences, both seen live:
                                # the echo guard is keyed on `not _vad_open`, so
                                # it silently did not apply at the one moment it
                                # exists for, and the first post-reply chunk
                                # printed "🤫 Speech ended" for an utterance that
                                # had finished before ADAM even started talking.
                                if _vad_open[0]:
                                    _vad_open[0]     = False
                                    _vad_closed_t[0] = time.time()
                                # The 16 kHz stream genuinely stops here, so the
                                # denoiser's overlap-add buffer must not splice
                                # the audio from before ADAM's reply onto the
                                # audio after it. Its noise estimate survives —
                                # same room, and re-learning it would leave the
                                # moment right after a reply unprocessed, which
                                # is exactly when the user speaks next.
                                denoise_reset()
                                while not mic_q.empty():
                                    try: mic_q.get_nowait()
                                    except asyncio.QueueEmpty: break
                                _preroll.clear()
                                # No end-of-turn marker here even though the
                                # gate just closed: the queue is being drained
                                # anyway, so a marker pushed now would be
                                # thrown away by this same branch on the next
                                # muted chunk. send() closes a dangling
                                # activity window off adam_speaking /
                                # song_playing directly, which covers this
                                # path. Clearing _gate_prev keeps the edge
                                # detector from firing a stale marker when
                                # the mic unmutes.
                                _gate_prev[0] = False
                                continue

                            # Converted in the SAME worker thread that did the
                            # read (see _read_and_convert), not a second
                            # asyncio.to_thread hop. Each hop costs a thread-pool
                            # dispatch plus two context switches, and this loop
                            # runs 30x/second; with the old two-hop version the
                            # log showed
                            #   [arecord] overrun!!! (at least 2097.350 ms long)
                            # i.e. ALSA discarded 2.1 SECONDS of captured audio
                            # because this loop drained the pipe too slowly.
                            # Whole sentences vanished before any gate saw them,
                            # which reads exactly like "ADAM can't hear me".
                            if not mono16k:
                                continue

                            # ── Song playing: OFFLINE stop-word only ──────────
                            # A song is 182-215s of aplay on the same I2S device,
                            # and until now the mic was muted for all of it, so
                            # ADAM could not be interrupted by voice at all. The
                            # documented escape was the Touch3 gesture, which
                            # arrives over the ESP32-CAM UART — and that link
                            # reports "no data received … audio-only mode" often
                            # enough that the practical answer was "wait three
                            # minutes". Hence the report: "not responding to
                            # anything, just started playing the song".
                            #
                            # This branch is placed BEFORE all level metering on
                            # purpose. The mic hears the speaker through the same
                            # PCB, so feeding music into _ambient_rms /
                            # _noise_peak would ratchet the thresholds up to
                            # MIC_AMBIENT_MAX and leave the gate deaf for a good
                            # while AFTER the song ended — trading a 3-minute
                            # problem for a longer one. Nothing here touches the
                            # trackers, the gate, or mic_q; the audio goes to the
                            # local Vosk recogniser and nowhere else, so no song
                            # audio is sent to Google either.
                            #
                            # Honest limitation: recognition happens while the
                            # song is playing loudly into the same enclosure and
                            # there is no echo cancellation, so a stop phrase is
                            # not guaranteed to be heard on the first try. The
                            # phrase is deliberately two words ("adam stop" /
                            # "stop the song") to keep music transients from
                            # tripping it, and it can simply be repeated.
                            if song_playing.is_set():
                                if _vad_open[0]:
                                    _vad_open[0]     = False
                                    _vad_closed_t[0] = time.time()
                                _gate_prev[0] = False
                                _preroll.clear()
                                while not mic_q.empty():
                                    try: mic_q.get_nowait()
                                    except asyncio.QueueEmpty: break
                                if VOSK_AVAILABLE:
                                    try:
                                        # Deliberately the RAW chunk, not the
                                        # denoised one. The interfering sound
                                        # here is music, which is anything but
                                        # stationary, so the minimum-statistics
                                        # noise estimate is meaningless against
                                        # it — and a wrong estimate suppresses
                                        # the stop phrase along with the song.
                                        wake_word_q.put_nowait(mono16k)
                                    except asyncio.QueueFull:
                                        pass
                                continue

                            # Level metering happens on the FILTERED audio, in
                            # int16 units. rms_s32(raw) — what every gate below
                            # used to use — measures the raw S32 capture, which
                            # on this hardware is ~85% out-of-band rumble/hiss:
                            # it reads 68M-108M in a SILENT room, so no gate
                            # calibrated for speech could ever fire and pure
                            # room noise was streamed to Gemini nonstop. See
                            # rms_pcm16() and the LEVEL GATES block in config.
                            now      = time.time()
                            _rms_now = rms_pcm16(mono16k)

                            # DEAD-CAPTURE WATCHDOG — the fix for "and after
                            # this it stopped listening", permanently.
                            #
                            # Observed live: immediately after a
                            # "🔇 Playback idle" close, every subsequent chunk
                            # came back as EXACT digital silence and stayed
                            # that way until the service was restarted:
                            #   📊 Mic 10s: p50 0 p90 0 p99 0 max 0 | ...
                            #   (repeating, ambient decaying 1144 → 188 → 0)
                            # arecord was still alive and still handing us
                            # full-size chunks, so nothing in the existing
                            # error path could notice — read_exact never
                            # failed, `errors` never incremented, and the
                            # stats line dutifully reported that the room was
                            # perfectly quiet. ADAM was stone deaf and
                            # cheerful about it.
                            #
                            # The trigger is the shared I2S clock domain: the
                            # voiceHAT is ONE soundcard serving both capture
                            # and playback, so tearing the playback stream
                            # down can leave the capture DMA running but
                            # feeding zeros. SPEAKER_IDLE_CLOSE_S made that
                            # teardown routine rather than once-per-session,
                            # which is what turned a latent race into
                            # something reproducible. speaker()'s teardown is
                            # now graceful (EOF, not SIGTERM) to stop
                            # provoking it — but a capture path that can
                            # silently die MUST also be able to notice and
                            # recover on its own, whatever the cause.
                            #
                            # An INMP441 always has self-noise; a true 0 is
                            # impossible from live hardware. So a sustained
                            # run of it is unambiguous, and the cure is the
                            # one that already exists: break, let the finally
                            # below reap arecord, and let the outer loop
                            # respawn it.
                            if _rms_now < 1.0:
                                _dead_run[0] += 1
                                # Two fuses, one detector. Inside the window
                                # after a playback close the cause is KNOWN, so
                                # waiting the full MIC_DEAD_STREAM_S there just
                                # donates the seconds in which the user replies
                                # to a stream of zeros.
                                _recent_close = (now - amp_quiet_t[0]
                                                 < MIC_DEAD_AFTER_PLAY_WINDOW_S)
                                _lim = (_dead_limit_amp if _recent_close
                                        else _dead_limit)
                                if _dead_run[0] >= _lim:
                                    print(f"  ⚠️  Capture DEAD — "
                                          f"{_dead_run[0] / _chunks_per_s:.1f}s "
                                          f"of exact digital silence from "
                                          f"arecord (voiceHAT I2S capture "
                                          f"wedged"
                                          f"{', playback closed ' if _recent_close else ''}"
                                          f"{f'{now - amp_quiet_t[0]:.1f}s ago' if _recent_close else ''}"
                                          f"). Restarting arecord.")
                                    break
                            else:
                                _dead_run[0] = 0

                            _rms_hist.append(_rms_now)
                            # Rolling median of the last MIC_VAD_SUSTAIN_S of
                            # audio. Fed here — after the dead-capture check and
                            # before any gate reads it — so every unmuted chunk
                            # contributes exactly once. statistics.median on a
                            # 15-element deque is ~15us on a Pi Zero 2 W, against
                            # a 33ms chunk period.
                            _lvl_win.append(_rms_now)
                            _lvl_sus = statistics.median(_lvl_win)
                            # ADAPTIVE GATE, fed on every unmuted chunk, with
                            # exactly ONE exception (the amp guard below):
                            #
                            # observe() must see speech too. That sounds wrong
                            # and is the whole point: it keeps a low PERCENTILE
                            # of a 45s window, and speech never occupies the
                            # bottom 20% of 45 seconds, so the estimate is
                            # immune to it by construction. The old EMA had to
                            # be defended from speech with a cooldown, a
                            # near-baseline guard and a hard MIC_AMBIENT_MAX
                            # clamp — and that clamp is what capped it below a
                            # noisy room's real floor.
                            #
                            # shape_ok() must see every chunk because it also
                            # maintains the rolling shape window that the HOLD
                            # test reads. It is cheap (one 1024-point rFFT,
                            # 1.02 ms measured against a 33.3 ms budget).
                            #
                            # Note what is NOT used here: an "was there speech
                            # recently" timer. At this feature's real false
                            # positive rate on noise (~5-10% per chunk) a 0.5 s
                            # memory reads true ~79% of the time on noise alone,
                            # so it could never let the gate CLOSE — and with
                            # Gemini's manual activity detection a gate that
                            # cannot close means no reply at all. The hold test
                            # is a FRACTION of the sustain window instead.
                            #
                            # THE ONE EXCEPTION — the amp guard. observe() is
                            # skipped while the playback device is open, and
                            # for MIC_ECHO_GUARD_S after it closes, because
                            # what the mic hears then is the voiceHAT's own
                            # amplifier (+4 dB of hiss, measured with nothing
                            # playing) rather than the room. Learning that
                            # walked open_th up to 2,989 against a user whose
                            # quietest speech is 2,357 — see the AMP HISS
                            # GUARD note in run_session for the full trace.
                            # The settle window matters as much as the open
                            # one: the device is closed by then, but the ALSA
                            # teardown transient is still in the capture path,
                            # and the whole point is to only ever learn air.
                            #
                            # shape_ok() is NOT skipped, and that asymmetry is
                            # deliberate. Its per-chunk features cannot be
                            # poisoned by hiss, and its rolling window has to
                            # stay contiguous for the HOLD fraction to mean
                            # what it says. Hiss is also exactly what it is
                            # best at rejecting: broadband noise reads flatness
                            # ~1.0 against a 0.35 limit and lo/hi ~0.15
                            # against a 0.60 floor, so it fails both halves of
                            # the shape test on its own merits. That is what
                            # keeps the stale-floor window from turning into
                            # false opens: during it the hiss does clear
                            # open_th, and the shape vote is the only reason
                            # the gate stays shut.
                            #
                            # Its ONE piece of adaptation — the learned
                            # flatness threshold — is fed separately, through
                            # learn_noise, and that flag repeats the amp guard
                            # for the same reason observe() has it: the
                            # threshold is meant to describe the ROOM. A chunk
                            # only teaches it when the gate is shut, the level
                            # is under the open threshold, and the amplifier is
                            # off — i.e. when this code already believes the
                            # chunk is silence. Speech cannot teach it, and
                            # neither can ADAM's own hiss.
                            _amp_hot = (amp_open[0]
                                        or now - amp_quiet_t[0]
                                            < MIC_ECHO_GUARD_S)
                            if not _amp_hot:
                                _agate.observe(_rms_now)
                            _shape_now  = _agate.shape_ok(
                                mono16k,
                                learn_noise=(not _amp_hot
                                             and not _vad_open[0]
                                             and _agate.ready
                                             and _rms_now < _agate.open_th))
                            _shape_hold = _agate.shape_hold_ok()
                            if now - _last_rms[0] > MIC_STATS_S:
                                # Print the LIVE thresholds, not the static
                                # floor: the floor is only one term of
                                # max(floor, ambient*margin, peak*margin), and it
                                # was the adaptive terms that silently drifted —
                                # ambient rising out of reach of the user's voice,
                                # then hold_th sinking below the noise. Showing
                                # both numbers actually being compared, plus the
                                # gate state, makes either failure visible in one
                                # line instead of inferable from behaviour.
                                _th = max(MIC_SILENCE_FLOOR,
                                          _ambient_rms[0] * MIC_SPEECH_MARGIN,
                                          _noise_peak[0] * MIC_VAD_OPEN_MARGIN)
                                _hth = min(_th * MIC_VAD_MAX_HOLD_RATIO,
                                           max(_th * MIC_VAD_RELEASE_RATIO,
                                               _noise_peak[0] * MIC_VAD_HOLD_MARGIN))
                                if MIC_ADAPTIVE and _agate.ready:
                                    _th  = _agate.open_th
                                    _hth = _agate.hold_th
                                _h = sorted(_rms_hist)
                                _n = len(_h)

                                def _pct(p: float) -> float:
                                    return _h[min(_n - 1, int(p * _n))] if _n else 0.0

                                # p99 next to open_th is the whole point: if p99
                                # is ABOVE open_th while nobody is talking, the
                                # gate is being opened by the room, and no amount
                                # of hangover tuning will produce a transcript.
                                #
                                # IDLE / SONG are printed because their absence
                                # cost hours of misdiagnosis: in idle mode every
                                # chunk goes to the offline recogniser and NOTHING
                                # to Gemini, so the line read "opens 1 sent 0" for
                                # minutes on end and looked exactly like a broken
                                # mic. It was ADAM obeying an "be quiet" it had
                                # overheard from a phone call. The state that
                                # explains `sent 0` now appears on the same line as
                                # `sent 0`.
                                _mode = ("IDLE" if idle_mode.is_set()
                                         else "SONG" if song_playing.is_set()
                                         else "OPEN" if _vad_open[0] else "shut")
                                # "+AMP" means the floor estimate is FROZEN
                                # because the playback device is open. Printed
                                # for the same reason IDLE/SONG are: a frozen
                                # floor is a legitimate state with a visible
                                # symptom (the floor number stops moving), and
                                # without this the only way to tell it apart
                                # from a wedged gate is to read the source.
                                if _amp_hot:
                                    _mode += "+AMP"
                                _floor_s = (f"floor {_agate.floor:.0f}"
                                            f"{'' if _agate.ready else '?'} "
                                            f"flat {_agate.flat:.2f}"
                                            f"/{_agate.flat_max:.2f} "
                                            f"lohi {_agate.lohi:.2f} "
                                            f"shp {100*_agate.shape_frac:.0f}%"
                                            if MIC_ADAPTIVE else
                                            f"ambient {_ambient_rms[0]:.0f} "
                                            f"peak {_noise_peak[0]:.0f}")
                                _nr_db = denoise_db()
                                _nr_s = ("" if _nr_db is None
                                         else f"nr {_nr_db:+.1f}dB | ")
                                print(f"  📊 Mic {MIC_STATS_S:.0f}s: p50 {_pct(0.50):.0f} "
                                      f"p90 {_pct(0.90):.0f} p99 {_pct(0.99):.0f} "
                                      f"max {(_h[-1] if _n else 0):.0f} | "
                                      f"open≥{_th:.0f} hold≥{_hth:.0f} | "
                                      f"{_floor_s} | "
                                      f"opens {_win_opens[0]} sent {_win_sent[0]} | "
                                      f"blocked {_win_blocked[0]} | "
                                      f"{_nr_s}"
                                      f"{_mode}")
                                _rms_hist.clear()
                                _win_opens[0] = 0
                                _win_sent[0] = 0
                                _win_blocked[0] = 0
                                _last_rms[0] = now

                            # ── Direction-of-arrival: see the block after the
                            # VAD gate below. It used to run HERE, on every
                            # chunk whose RMS cleared a separate (now deleted)
                            # MIC_LIVE_RMS_THRESHOLD * 0.5 = 1,400 — below the
                            # ambient the room was believed to have, so the
                            # "only when it's worth it" guard never once engaged
                            # and GCC-PHAT's FFTs ran
                            # 30x/second forever. That was the CPU cost behind
                            # both the capture overruns and the playback
                            # underruns. It now runs only on real speech, and at
                            # a rate the neck can actually use.

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

                            # attention_active is driven by the VAD gate below,
                            # not by a bare RMS comparison. It used to fire on
                            # `_rms_now > MIC_LIVE_RMS_THRESHOLD` (2,800) — a
                            # second, higher bar than the gate that decides what
                            # Gemini actually hears. Speech landing between the
                            # two counted as audio worth sending but NOT as
                            # "someone is talking", so the 90s idle timer kept
                            # running and the nudge talked straight over the
                            # user. Observed live: the gate opened at RMS 5273
                            # and `🔊 ADAM speaking` followed in the same
                            # instant. One gate, one meaning — the constant is
                            # gone, not merely unused.

                            # ── SILENCE / NOISE-FLOOR GATE ───────────────────
                            # Previously every mic chunk was queued/sent to
                            # Gemini unconditionally, including pure room
                            # noise/silence between sentences. Continuously
                            # streaming near-silent audio gives the Live API
                            # ungrounded input during quiet stretches, which
                            # is a known trigger for unprompted "phantom"
                            # responses (the random Hindi hallucinations) —
                            # the model free-associates from thin signal
                            # instead of responding to real speech. It also
                            # leaves Gemini's own turn detection with no
                            # endpoint to latch onto, so genuine speech never
                            # gets answered: the "ADAM talks but can't hear
                            # me" symptom.
                            #
                            # A chunk must clear BOTH the fixed floor (a
                            # backstop, in filtered-int16 RMS units) and the
                            # ROLLING ambient baseline by a real margin. The
                            # rolling part is what distinguishes a peripheral
                            # noise burst — louder than true silence, still
                            # much quieter than speech — from actual speech
                            # onset, and it lets ADAM adapt to whatever room
                            # it's in instead of relying on one hand-tuned
                            # number.
                            #
                            # ORDERING FIX: the baseline update used to sit
                            # BELOW the fixed-floor `continue`, so the very
                            # chunks that represent true silence — the only
                            # honest evidence of the room's noise floor —
                            # returned before ever reaching it. The baseline
                            # therefore never moved off its initial guess.
                            #
                            # HYSTERESIS + HANGOVER + PRE-ROLL: a bare
                            # per-chunk comparison shreds speech at its own
                            # internal gaps and deletes its attack; see the
                            # VAD block in config.py for the measurement that
                            # forced this. `open_th` gates the START of an
                            # utterance, `hold_th` (lower) keeps it open, and
                            # the hangover window keeps it open through short
                            # dips even below that.
                            # Thresholds come from the trackers as they stood at
                            # the END of the previous chunk, and the trackers are
                            # updated further down only for chunks this gate
                            # judged to be non-speech. That ordering matters: a
                            # chunk must never be allowed to raise the noise
                            # estimate that is about to classify it.
                            #
                            # open_th ALSO respects the measured noise PEAK, not
                            # just the average. The averaged ambient estimate
                            # falls fast and rises slowly (deliberately — that is
                            # what stops speech poisoning it), so it settles near
                            # the LOWER envelope of room noise, materially below
                            # the loudest noise chunks. Sizing thresholds off it
                            # alone therefore puts them under peaks that are
                            # still just noise.
                            #
                            # The peak multiplier here is MIC_VAD_OPEN_MARGIN and
                            # the one in hold_th below is MIC_VAD_HOLD_MARGIN, and
                            # they must stay DIFFERENT. Sharing one constant is
                            # what killed the hysteresis: both lines reduced to
                            # `peak * 1.05` whenever the peak term won, so hold_th
                            # came out exactly equal to open_th and the Schmitt
                            # trigger degenerated into a plain threshold. Live it
                            # logged 2356/2356, 2764/2764, 2876/2876, 2914/2914,
                            # 3007/3007 — and with no gap left, every mid-word dip
                            # closed the gate, so 20+ consecutive bursts reached
                            # Gemini as ~1s slices with ~1s holes between them and
                            # not one produced a transcript.
                            open_th = max(MIC_SILENCE_FLOOR,
                                          _ambient_rms[0] * MIC_SPEECH_MARGIN,
                                          _noise_peak[0] * MIC_VAD_OPEN_MARGIN)
                            # hold_th must stay ABOVE the room's noise peak.
                            # Deriving it purely as a fraction of open_th put it
                            # at 2300*0.72 = 1,656 — BELOW the measured quiet
                            # room (1,450-1,990), so every chunk of silence
                            # re-armed the hangover and the gate could never
                            # close. That fed Gemini an unbroken noise bed and it
                            # answered with hallucinated transcripts in random
                            # languages ("안녕하세요", "luego") — the exact
                            # phantom-response failure the gate exists to prevent.
                            # Observed live: opened at 2,447, then held on 1,674 /
                            # 1,930 / 1,770 for ~40s of pure room noise.
                            #
                            # Note what this concedes: measured mid-word dips
                            # (1,776 / 1,913 / 2,008) sit INSIDE the quiet-room
                            # range, so NO threshold can separate a dip from
                            # silence. The hangover timer, not the hold
                            # threshold, is what has to bridge them — and it can,
                            # because it is a duration test rather than a level
                            # test. hold_th's only job is to be low enough to
                            # ride out sustained-but-quieter speech and high
                            # enough that noise alone cannot renew it.
                            hold_th = max(open_th * MIC_VAD_RELEASE_RATIO,
                                          _noise_peak[0] * MIC_VAD_HOLD_MARGIN)
                            # Strictly below open_th, by construction. min(.., open_th)
                            # was not enough: it permits hold_th == open_th, which is
                            # a Schmitt trigger with no gap at all.
                            hold_th = min(hold_th, open_th * MIC_VAD_MAX_HOLD_RATIO)

                            # ── ADAPTIVE OVERRIDE ─────────────────────────────
                            # Everything above is the ORIGINAL absolute-threshold
                            # path, kept intact and reachable (MIC_ADAPTIVE=0) as
                            # a one-variable rollback. When adaptive is on, the
                            # three thresholds come from the learned floor
                            # instead, because the absolute path cannot ship to
                            # customers:
                            #
                            #   • MIC_SILENCE_FLOOR is one hand-measured number
                            #     from ONE room. In this room, live: floor p50
                            #     1872-1923 against open≥1800 — the noise was
                            #     already over the gate. The printed remedy was
                            #     "echo 'MIC_SILENCE_FLOOR=…' >> ~/adam/.env",
                            #     i.e. asking the owner of the product to edit a
                            #     dotfile over SSH. That is not a shipping
                            #     product.
                            #   • MIC_AMBIENT_MAX cannot be raised to cover a
                            #     noisier room: MIC_AMBIENT_MAX x
                            #     MIC_SPEECH_MARGIN must stay under the quietest
                            #     measured speech (2357), which caps it at 1746.
                            #     A room whose floor is 1900 is therefore
                            #     unreachable BY CONSTRUCTION, and pushing the
                            #     clamp up would put open_th above real speech
                            #     and deafen ADAM outright.
                            #   • And the gap that remains is only ~2 dB (floor
                            #     1900 vs quietest speech 2357), so NO level
                            #     threshold whatsoever separates this room's
                            #     noise from this user's voice. That is why the
                            #     shape vote below is not a refinement but the
                            #     load-bearing part: it is the only term that is
                            #     independent of level.
                            #
                            # strong_th is the level at which ADAM opens WITHOUT
                            # the shape vote, so a shout, a clap-to-get-attention
                            # or a voice the VAD mis-scores is never ignored.
                            if MIC_ADAPTIVE and _agate.ready:
                                open_th   = _agate.open_th
                                hold_th   = _agate.hold_th
                                strong_th = _agate.strong_th
                                _vote_req = True
                            else:
                                # Legacy path, bit-for-bit: no shape vote is
                                # required and "strong" is unreachable, so the
                                # decision below reduces to exactly the level
                                # comparison it always was. A rollback has to be
                                # a real rollback to be worth keeping.
                                strong_th = float("inf")
                                _vote_req = False

                            # ECHO GUARD. For a short window after the mic
                            # reopens, the room is still ringing with ADAM's own
                            # last sentence. Measured live: the first unmuted
                            # chunk after a reply read 2,693 against open_th
                            # 2,300, opening the gate on ADAM's own voice — which
                            # is how a model ends up answering itself. Demanding
                            # extra margin here, instead of simply muting for
                            # longer, is what keeps a fast human reply audible:
                            # the audio still flows into the pre-roll buffer, so
                            # nothing is discarded, it just takes a genuinely
                            # louder chunk to declare speech.
                            if (not _vad_open[0]
                                    and now - mic_open_t[0] < MIC_ECHO_GUARD_S):
                                open_th *= MIC_ECHO_GUARD_MARGIN
                                # strong_th has to move with it. It is an OR
                                # alternative to the shape vote, and ADAM's own
                                # recorded voice passes a speech-shape test
                                # trivially — leaving strong_th unscaled would
                                # hand the echo a level-only bypass around the
                                # very guard this block is.
                                strong_th *= MIC_ECHO_GUARD_MARGIN

                            # ── ONE DECISION, TWO INDEPENDENT VOTES ───────────
                            # LEVEL: instantaneous chunk to ATTACK (fast, and
                            # duration-tested by the onset run below), rolling
                            # median to SUSTAIN (an impulse cannot move a median,
                            # which is what stopped intermittent noise pinning
                            # the gate open forever).
                            #
                            # SHAPE: the spectral verdict, which knows nothing
                            # about level — flatness plus a low/high band ratio
                            # over 120-6,800 Hz (see AdaptiveGate.shape_ok).
                            # NOT webrtcvad: it was tried here first and
                            # measured on this hardware it called 100.0% of
                            # this room's noise frames "speech" at
                            # aggressiveness 0, 1 and 2, and 98.6% at 3, so as
                            # a second vote it was a constant. This is the term
                            # that makes the gate work in a room whose noise is
                            # as loud as the user's voice — the measured 2 dB
                            # case above. Requiring
                            # BOTH votes to HOLD is deliberate and is the second
                            # half of the "ADAM never answers" fix: under manual
                            # activity detection the gate's falling edge is the
                            # only activity_end Gemini ever gets, so a gate held
                            # open by a noise bed does not merely waste
                            # bandwidth, it means no reply is ever generated.
                            # With the shape vote in the hold condition, a noise
                            # bed stops refreshing the hangover even while it is
                            # still loud, the hangover expires, and the turn
                            # closes.
                            #
                            # The two branches use DIFFERENT shape statistics
                            # on purpose. Opening reads this chunk's strict
                            # verdict, so every one of the
                            # MIC_VAD_ONSET_CHUNKS chunks in the onset run has
                            # to pass on its own — that run requirement is what
                            # takes the shape test's residual 4.3% per-chunk
                            # false pass on noise down to zero false opens.
                            # Holding reads the FRACTION of the sustain window
                            # that passed with a little flatness slack, which
                            # rides out consonants without ever being true on a
                            # steady noise bed.
                            if _vad_open[0]:
                                _gate_pass = (_lvl_sus >= hold_th
                                              and (not _vote_req or _shape_hold
                                                   or _lvl_sus >= strong_th))
                            else:
                                _gate_pass = (_rms_now >= open_th
                                              and (not _vote_req or _shape_now
                                                   or _rms_now >= strong_th))

                            if servo_moving.is_set():
                                # The head is physically turning, or has just
                                # been released and is still settling. Measured
                                # post-filter RMS: 4,658-4,666 while energised
                                # and still 1,697 / 1,931 for the two seconds
                                # after detach, against a 1,039-1,245 baseline
                                # and an open_th of 1,800 (adam/_floorcal.py,
                                # adam/_servodecay.py). Treated as neither speech
                                # nor silence: it may not OPEN the gate, and it
                                # may not teach the trackers what the room sounds
                                # like. If the gate is already open (DOA only
                                # turns the head toward someone who is talking)
                                # the hangover is refreshed so the move cannot
                                # truncate the utterance in progress, and the
                                # audio still flows through — muting here would
                                # punch a hole in the user's sentence.
                                if _vad_open[0]:
                                    _vad_last_loud_t[0] = now
                                    _vad_last_strong_t[0] = now
                                _onset_win.clear()
                            elif now < _refractory_until[0]:
                                _onset_win.clear()  # forced-shut window, see watchdogs
                            elif _gate_pass:
                                # ATTACK (gate shut) reads the instantaneous
                                # chunk — fast, and already duration-tested by
                                # the onset run below. SUSTAIN (gate open) reads
                                # the rolling median, so intermittent room noise
                                # can no longer hold the gate open forever; see
                                # the SUSTAIN WINDOW note where _lvl_win is
                                # built. Both are ANDed with the shape vote when
                                # MIC_ADAPTIVE is on — see the two-votes block.
                                _vad_last_loud_t[0] = now
                                attention_active.set()
                                if (_lvl_sus if _vad_open[0]
                                        else _rms_now) >= open_th:
                                    # Same split for the soft latch watchdog's
                                    # "strong" stamp: while open it must mean
                                    # SUSTAINED speech-level audio, otherwise a
                                    # p99 impulse every few seconds keeps
                                    # renewing it and MIC_VAD_MAX_OPEN_S can
                                    # never fire.
                                    _vad_last_strong_t[0] = now
                                if not _vad_open[0]:
                                    # ONSET CONFIRMATION — a DURATION test on top
                                    # of the level test.
                                    #
                                    # adam/_hpcal.py established that the residual
                                    # noise cannot be filtered away: after the
                                    # sub-100 Hz rumble is gone what is left is
                                    # broadband hiss sitting on the speech band, and
                                    # no band-pass candidate beat the current chain
                                    # (best 1.67x vs 1.68x, -0.1 dB). But the same
                                    # measurement showed a band-pass dropping the
                                    # noise p50 by 30% while dropping its p99 by
                                    # only 17%. Stationary noise would move both by
                                    # the same factor; a tail that survives
                                    # band-limiting and sits 1.67x over its own
                                    # median is IMPULSIVE — clicks, creaks, taps,
                                    # bearing ticks.
                                    #
                                    # An impulse is one or two 33 ms chunks long.
                                    # The shortest useful utterance is tens of
                                    # chunks. So the level threshold was being set
                                    # by the wrong constraint: pinned high purely to
                                    # stay above transients, which cost every speech
                                    # chunk underneath it — and the measured
                                    # intra-word dips (1776/1913/2008) sit exactly
                                    # there. A DURATION test rejects transients
                                    # instead, which frees the level threshold to
                                    # come down.
                                    #
                                    # A QUORUM, not a run: N of the last
                                    # MIC_VAD_ONSET_WINDOW chunks, gaps allowed.
                                    # The consecutive version of this test is what
                                    # made ADAM look deaf — real speech is full of
                                    # chunks that legitimately fail, an unvoiced
                                    # /h/ or /s/ (broadband, flat, fails shape) and
                                    # a stop closure (2-3 chunks of near-silence,
                                    # fails level), and either one reset the run to
                                    # zero. "Hey ADAM" does not contain 5 clean
                                    # chunks in a row at conversational volume, so
                                    # the run never completed and nothing was ever
                                    # sent. The quorum keeps the duration test that
                                    # rejects impulses (one chunk cannot make three)
                                    # and drops the contiguity requirement that
                                    # speech cannot meet.
                                    #
                                    # The delay is free: MIC_VAD_PREROLL_S of audio
                                    # from BEFORE the gate opened is already ringed
                                    # and gets flushed on open, so the window costs
                                    # latency in the DECISION, not audio.
                                    _onset_win.append(1)
                                    if sum(_onset_win) >= MIC_VAD_ONSET_CHUNKS:
                                        _vad_open[0] = True
                                        _vad_last_strong_t[0] = now
                                        _vad_opened_t[0] = now
                                        _win_opens[0] += 1
                                        print(f"  🎙️  Speech detected "
                                              f"(RMS {_rms_now:.0f} ≥ {open_th:.0f}"
                                              f"{f', {sum(_onset_win)}/{len(_onset_win)} chunks' if MIC_VAD_ONSET_CHUNKS > 1 else ''})")
                                        _onset_win.clear()
                            elif _vad_open[0] and (now - _vad_last_loud_t[0]
                                                   ) > MIC_VAD_HANGOVER_S:
                                _vad_open[0] = False
                                _vad_closed_t[0] = now
                                _onset_win.clear()
                                print("  🤫 Speech ended")
                            else:
                                # A chunk that failed. It does NOT reset the
                                # quorum — it just ages out of the window like
                                # any other, which is the whole point: a
                                # consonant or a stop closure inside an
                                # utterance must not undo the chunks around it.
                                #
                                # A window that collected passes but decayed
                                # back to nothing without reaching the quorum is
                                # exactly the transient this rule exists to
                                # reject, so count that moment — not every quiet
                                # chunk, which would just count silence.
                                # `blocked` is then how much the duration test
                                # is actually buying, measured live rather than
                                # assumed.
                                if not _vad_open[0]:
                                    _had = sum(_onset_win) > 0
                                    _onset_win.append(0)
                                    if _had and sum(_onset_win) == 0:
                                        _win_blocked[0] += 1

                            # LATCH WATCHDOGS. The trackers freeze while the gate
                            # is open (by design — speech must not teach them
                            # what silence sounds like), which means a latched
                            # gate is self-sustaining: no update, so no threshold
                            # movement, so no escape. Bound it in TIME instead.
                            #
                            # Soft: nothing has re-cleared the OPEN threshold for
                            # MAX_OPEN_S, so only the lower hold threshold is being
                            # met. That is not speech; force it shut and resync
                            # both estimates upward to the level that fooled them,
                            # so the room's new, louder floor is learned in one
                            # step and the gate stops reopening on it.
                            if _vad_open[0] and (now - _vad_last_strong_t[0]
                                                 ) > MIC_VAD_MAX_OPEN_S:
                                _vad_open[0] = False
                                _vad_closed_t[0] = now
                                _ambient_rms[0] = min(MIC_AMBIENT_MAX,
                                                      max(_ambient_rms[0], _rms_now * 0.9))
                                _noise_peak[0] = min(MIC_AMBIENT_MAX,
                                                     max(_noise_peak[0], _rms_now))
                                print(f"  🤫 Gate forced shut after "
                                      f"{MIC_VAD_MAX_OPEN_S:.0f}s with no chunk over "
                                      f"{open_th:.0f} — treating RMS {_rms_now:.0f} as "
                                      f"room noise, not speech")
                            # Hard: open CONTINUOUSLY for ABS_MAX_OPEN_S while
                            # chunks keep clearing open_th. The soft watchdog
                            # cannot catch this — the level really is above the
                            # open threshold — so the only remaining explanation is
                            # a room whose noise floor now sits above open_th
                            # itself. Deliberately does NOT resync the legacy
                            # trackers: at these levels that would drag open_th up
                            # past normal speech and deafen ADAM completely.
                            # Instead it caps how much noise reaches Gemini per
                            # cycle. A genuine 45s unbroken monologue also trips
                            # this; it costs the refractory window and the gate
                            # reopens on the next chunk.
                            #
                            # The message used to end by telling the user to run
                            # `echo 'MIC_SILENCE_FLOOR=…' >> ~/adam/.env` over SSH.
                            # For a product that ships to people who did not write
                            # it, that instruction IS the bug: it means the unit
                            # cannot recover from a noisy room by itself. Under
                            # MIC_ADAPTIVE it now recovers by itself — the 45s of
                            # loud audio is already in the floor window, so the
                            # learned floor and open_th are rising as a consequence
                            # of the same event that printed this — and the line
                            # reports that rather than handing out homework.
                            elif _vad_open[0] and (now - _vad_opened_t[0]
                                                   ) > MIC_VAD_ABS_MAX_OPEN_S:
                                _vad_open[0] = False
                                _vad_closed_t[0] = now
                                _refractory_until[0] = now + 1.5
                                if MIC_ADAPTIVE:
                                    print(f"  ⚠️  Mic gate open "
                                          f"{MIC_VAD_ABS_MAX_OPEN_S:.0f}s "
                                          f"continuously at RMS {_rms_now:.0f} "
                                          f"(open≥{open_th:.0f}, learned floor "
                                          f"{_agate.floor:.0f}, flat "
                                          f"{_agate.flat:.2f}, shape "
                                          f"{100*_agate.shape_frac:.0f}%"
                                          f") — this room is loud "
                                          f"and speech-like; adapting upward "
                                          f"automatically, no action needed")
                                else:
                                    print(f"  ⚠️  Mic gate open "
                                          f"{MIC_VAD_ABS_MAX_OPEN_S:.0f}s "
                                          f"continuously at RMS {_rms_now:.0f} "
                                          f"(open≥{open_th:.0f}) — this room's "
                                          f"noise floor is ABOVE the gate, so "
                                          f"Gemini is being fed noise. Set "
                                          f"MIC_ADAPTIVE=1 (default) to have "
                                          f"ADAM learn this room by itself, or "
                                          f"raise the fixed floor: echo "
                                          f"'MIC_SILENCE_FLOOR={_rms_now * 1.1:.0f}'"
                                          f" >> ~/adam/.env && restart adam")

                            # ── TURN BOUNDARY ─────────────────────────────
                            # Falling edge of the gate = the user stopped
                            # talking = end of their turn. With the server's
                            # own VAD disabled (see realtime_input_config in
                            # the connect config) this marker is the ONLY
                            # thing that tells Gemini to start answering, so
                            # it is the difference between a reply that
                            # begins now and one that begins whenever the
                            # server happens to give up waiting for silence
                            # that a gated stream never delivers.
                            #
                            # Pushed through mic_q so it lands strictly after
                            # the last audio chunk of the utterance, and
                            # pushed UNCONDITIONALLY: a dropped end marker
                            # leaves the turn open forever, which is worse
                            # than dropping audio, so make room by discarding
                            # the oldest chunk rather than skipping it.
                            if _gate_prev[0] and not _vad_open[0]:
                                while mic_q.full():
                                    try: mic_q.get_nowait()
                                    except asyncio.QueueEmpty: break
                                mic_q.put_nowait(ACTIVITY_END)
                            _gate_prev[0] = _vad_open[0]

                            if not _vad_open[0]:
                                # Non-speech. Two jobs: teach the noise trackers
                                # what this room sounds like, and keep the chunk
                                # as pre-roll for whatever comes next (dropping it
                                # outright is what deleted word onsets before).
                                #
                                # Ambient = smoothed level; falls fast, rises 10x
                                # slower and only toward chunks near the current
                                # baseline. Without that guard, talking that sits
                                # just under the threshold drags the baseline up
                                # and takes the threshold with it, progressively
                                # locking out the very speaker it should be
                                # listening to — observed live as ambient climbing
                                # 1,816 -> 2,395 while the user talked.
                                #
                                # Noise peak = decaying maximum over the same
                                # chunks. It jumps instantly to any new loud noise
                                # chunk and bleeds back down over ~30s, so the
                                # thresholds track the loudest thing the room does
                                # rather than its average. The same near-baseline
                                # guard applies, so a stray word can lift it by at
                                # most 50%.
                                #
                                # LEARN COOLDOWN. "Not open" is not the same as
                                # "room noise": the chunks right after a close are
                                # the reverb tail, the trailing consonant and the
                                # inter-word gap of the utterance that just ended,
                                # and the ones right after an unmute are ADAM's own
                                # tail. Feeding those in is a positive feedback
                                # loop, and it is what made ADAM go deaf partway
                                # through a conversation: peak walked 1,800 ->
                                # 2,244 -> 2,632 -> 2,864 on speech tails alone,
                                # taking open_th from 2,300 to 3,007 — above most
                                # of the 2,400-5,500 speech range — and every step
                                # up produced more chatter to learn from.
                                _learning = (
                                    now - _vad_closed_t[0] > MIC_NOISE_LEARN_COOLDOWN_S
                                    and now - mic_open_t[0] > MIC_NOISE_LEARN_COOLDOWN_S
                                    and not servo_moving.is_set()
                                    # Same amp guard as the adaptive floor. The
                                    # unmute cooldown above is 1.0s but the
                                    # playback device stays open for
                                    # SPEAKER_IDLE_CLOSE_S = 2.5s, so without
                                    # this these trackers spend the difference
                                    # learning the amplifier. MIC_AMBIENT_MAX
                                    # caps the damage, which is exactly the
                                    # clamp that made this path unusable in a
                                    # loud room — so fix the input instead of
                                    # leaning on the clamp.
                                    and not _amp_hot)
                                if _learning and _rms_now < _ambient_rms[0] * 1.5:
                                    _a = (_AMBIENT_ALPHA_DOWN
                                          if _rms_now < _ambient_rms[0]
                                          else _AMBIENT_ALPHA_UP)
                                    _ambient_rms[0] = min(
                                        MIC_AMBIENT_MAX,
                                        (1 - _a) * _ambient_rms[0] + _a * _rms_now)
                                    _noise_peak[0] = min(
                                        MIC_AMBIENT_MAX,
                                        max(_rms_now, _noise_peak[0] * _NOISE_PEAK_DECAY))
                                elif _learning:
                                    # Too loud to be learned as a level, but the
                                    # peak must still be allowed to DECAY here or a
                                    # room that got quieter never lets the gate back
                                    # down: every chunk over ambient*1.5 skipped the
                                    # decay entirely, so one loud event could hold
                                    # the peak — and the thresholds — up for as long
                                    # as the noise kept recurring.
                                    _noise_peak[0] *= _NOISE_PEAK_DECAY
                                # Pre-roll holds the DENOISED chunks, because
                                # its only consumer is mic_q — the gate has
                                # already had its look at this chunk above.
                                # denoise_16k() returns b"" while its first
                                # frame fills, and those must not be queued as
                                # empty audio blobs.
                                # Keep pre-roll unattenuated so word onset is pristine
                                if mono16k:
                                    _preroll.append(mono16k)
                                continue

                            if idle_mode.is_set():
                                # While idle, audio goes ONLY to the local
                                # wake-word detector — never to mic_q
                                # (which feeds Gemini via send()). This is
                                # what actually keeps audio off Google
                                # during idle, not just discarding the
                                # response afterward.
                                _preroll.clear()
                                if VOSK_AVAILABLE and mono16k_nr:
                                    try:
                                        # Denoised here, unlike the song branch:
                                        # idle-room noise is stationary, which is
                                        # the case the suppressor is built for,
                                        # and "adam" has to be picked out of it
                                        # by a small offline model.
                                        wake_word_q.put_nowait(mono16k_nr)
                                    except asyncio.QueueFull:
                                        pass
                                continue

                            # ── Direction-of-arrival (two-mic GCC-PHAT) ───────
                            # Reached only when the VAD gate is OPEN, i.e. there
                            # is actually speech to localise — GCC-PHAT on room
                            # noise returns a meaningless jittery angle anyway,
                            # so the old RMS-based guard was buying nothing even
                            # when it did fire. Rate-limited on top of that: the
                            # neck already has a 12° deadzone and a 1.5s
                            # cooldown (NECK_PAN_DEADZONE_DEG /
                            # NECK_PAN_COOLDOWN_S), so direction updates faster
                            # than a few per second are discarded downstream.
                            # Skipped entirely while idle — the servo must not
                            # track sound during idle mode, belt-and-suspenders
                            # alongside camera()'s own idle_mode check.
                            if now - _last_doa_t[0] > _DOA_MIN_GAP_S:
                                _last_doa_t[0] = now

                                def _compute_doa(_raw=raw):
                                    left, right = s32_stereo_to_s16_stereo_channels(_raw)
                                    return estimate_doa_angle(left, right, CAPTURE_RATE)

                                angle = await asyncio.to_thread(_compute_doa)
                                if abs(angle) > DOA_ANGLE_DEADZONE:
                                    # Light smoothing so the neck doesn't jitter
                                    # — exponential moving average, not a snap.
                                    doa_angle[0] = (doa_angle[0] * 0.6) + (angle * 0.4)
                                    doa_last_update_t[0] = time.time()
                                    # Mirror to module-level state for the
                                    # get_sound_direction tool handler, which
                                    # lives outside this closure.
                                    _doa_angle[0] = doa_angle[0]
                                    _doa_last_update_t[0] = doa_last_update_t[0]

                            # Flush pre-roll first so Gemini hears the word from
                            # its true beginning, then the current chunk.
                            while _preroll and not mic_q.full():
                                mic_q.put_nowait(_preroll.popleft())
                                _win_sent[0] += 1
                            _preroll.clear()

                            if not mic_q.full() and mono16k_nr:
                                mic_q.put_nowait(mono16k_nr)
                                _win_sent[0] += 1

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
                # Whether an activity window (a user turn) is currently open
                # on the wire. The server's own VAD is disabled, so this
                # bracket is what defines "the user is talking" — audio sent
                # outside it is discarded by the API, and a window left open
                # means Gemini keeps waiting instead of answering.
                activity_open = [False]

                async def _end_activity() -> None:
                    """Close the user's turn if one is open. Idempotent.

                    Swallows send failures on purpose: this is called from
                    the mute/idle guards as well as from the explicit
                    marker, and a lost boundary must not kill the send task
                    — the flag is cleared first so the next utterance opens
                    a fresh window regardless.
                    """
                    if not activity_open[0]:
                        return
                    activity_open[0] = False
                    try:
                        await session.send_realtime_input(
                            activity_end=types.ActivityEnd())
                    except Exception as e:
                        print(f"  ⚠️  activity_end failed: {e}")

                while not stop.is_set():
                    try:
                        chunk = await asyncio.wait_for(mic_q.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        # Nothing queued for a second. If ADAM started
                        # talking (or went idle) while a window was still
                        # open, close it here: listen()'s mute branch drains
                        # mic_q, so the marker that would have closed it may
                        # never arrive, and this is the only place left that
                        # can notice.
                        if (adam_speaking.is_set() or song_playing.is_set()
                                or idle_mode.is_set()):
                            await _end_activity()
                        continue
                    except asyncio.CancelledError:
                        break
                    if chunk is ACTIVITY_END:
                        await _end_activity()
                        continue
                    if adam_speaking.is_set() or song_playing.is_set():
                        await _end_activity()
                        continue
                    if idle_mode.is_set():
                        # While idle, audio must NOT reach Google at all —
                        # not "sent but response discarded" (the previous,
                        # incorrect approach), genuinely never sent. Wake
                        # detection during idle runs entirely locally via
                        # the offline wake_word_detector task instead,
                        # which reads from wake_word_q (fed below).
                        await _end_activity()
                        continue
                    try:
                        if not activity_open[0]:
                            # First audio of a new utterance. Must precede
                            # the audio itself — including the pre-roll,
                            # which listen() flushes into mic_q ahead of the
                            # live chunk, so it lands inside the window
                            # rather than being dropped outside it.
                            await session.send_realtime_input(
                                activity_start=types.ActivityStart())
                            activity_open[0] = True
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
                                    #
                                    # …but only if a HUMAN asked. Observed
                                    # live, with nobody in the room and no
                                    # user speech at all:
                                    #     💤 Idle nudge (90s)
                                    #     🔇 enter_idle_mode called
                                    #     🔇 Idle mode active (voice request)
                                    # ADAM nudged an empty room, got silence
                                    # back, and read its own unanswered nudge
                                    # as "they want me to be quiet" — then
                                    # went deaf to everything but the wake
                                    # word for IDLE_MAX_S. Leave a shipped
                                    # unit alone for IDLE_TIMEOUT_S and it
                                    # mutes itself; walk up and talk to it and
                                    # it ignores you. That is indistinguish-
                                    # able from a broken microphone, and it is
                                    # the likeliest explanation for "ADAM is
                                    # constantly mis-hearing everything".
                                    #
                                    # A nudge is ADAM initiating contact.
                                    # Silence in reply means "nobody is here",
                                    # never "be quiet". So the request is
                                    # honoured only when the user has spoken
                                    # since the last nudge — which is exactly
                                    # the case where they really did ask.
                                    if _idle_mode_requested[0]:
                                        _idle_mode_requested[0] = False
                                        if not ENABLE_IDLE:
                                            # ENABLE_IDLE=0 is the operator
                                            # saying "never go deaf on me".
                                            # The timeout path already honours
                                            # it; the TOOL path has to as well,
                                            # or Gemini can still put ADAM to
                                            # sleep after mishearing a phone
                                            # call, and the one switch that is
                                            # supposed to rule idle mode out as
                                            # the cause of silence would not.
                                            print("  🙉 enter_idle_mode ignored"
                                                  " — ENABLE_IDLE=0")
                                        elif last_user_turn_t[0] <= last_nudge_t[0]:
                                            print("  🙉 enter_idle_mode ignored"
                                                  " — no one has spoken since"
                                                  " the last idle nudge, so"
                                                  " this is ADAM answering"
                                                  " itself, not a request")
                                        else:
                                            idle_mode.set()
                                            _idle_mode_persistent[0] = True
                                            _idle_since[0] = time.time()
                                            tft_set("sleep")
                                            await asyncio.to_thread(servo_pan, 90)
                                            servo_tilt(90)
                                            print(f"  🔇 Idle mode active (voice "
                                                  f"request) — servos centered; "
                                                  f"say \"adam\" to wake, Touch3, "
                                                  f"or wait {IDLE_MAX_S/60:.0f} min")
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
                                        last_user_turn_t[0] = time.time()
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
                            await asyncio.to_thread(
                                write_all, proc.stdin, bytes(buf),
                                PLAYBACK_CHANNELS * 2)
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
                    # How much audio to assume is still sitting inside aplay's
                    # own ALSA ring buffer, on top of whatever was left in `buf`.
                    # This used to be written `96000 / bytes_per_sec`, mixing the
                    # FRAME count from --buffer-size with a BYTES-per-second
                    # divisor; `aplay -v` reports the driver actually grants
                    # 62400 frames (1.3s), so neither the literal nor the "~0.5s"
                    # in the comments above described this card. The 0.5s the old
                    # arithmetic happened to produce is what stopped sentence
                    # tails being clipped in practice, so it is preserved as an
                    # explicit, measured, tunable value rather than corrected into
                    # a 1.3s worst case — that would also add up to 0.8s to
                    # mute_wait_s below, i.e. keep the mic shut that much longer
                    # after every reply.
                    ALSA_BUFFER_DRAIN_S = SPEAKER_DRAIN_ALLOWANCE_S
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
                    # Report how much of the turn exceeded int16 full scale. This
                    # used to warn about "audible distortion" and tell the user to
                    # lower SPEAKER_GAIN and raise the ALSA volume instead — the
                    # first half was right, the second half impossible: amixer
                    # exposes no controls at all on this card. With SPEAKER_GAIN at
                    # 1.0 Gemini's samples cannot exceed full scale by
                    # construction, so anything reported here is resampler
                    # overshoot and is soft-limited rather than flat-topped. A
                    # non-trivial number means SPEAKER_GAIN has been raised in .env
                    # past what the ceiling allows, and the distortion the user
                    # described ("gain goes high, lots of noise") is back.
                    _tot = spk_total_samples[0]
                    if _tot:
                        _pct = 100.0 * spk_clip_samples[0] / _tot
                        if _pct > 0.05:
                            print(f"  🔉 {_pct:.1f}% of samples over full scale at "
                                  f"SPEAKER_GAIN={SPEAKER_GAIN} — soft-limited, not "
                                  f"clipped, but this is where distortion comes "
                                  f"from. Lower SPEAKER_GAIN in .env.")
                        spk_clip_samples[0] = 0
                        spk_total_samples[0] = 0
                    adam_speaking.clear()
                    mic_open_t[0] = time.time()   # arms the echo guard
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
                # False after the very first aplay spawn. Two jobs: the startup
                # beep must only ever play once, and every LATER spawn must be
                # lazy — see SPEAKER_IDLE_CLOSE_S in config.py for why the
                # device is not simply held open for the session lifetime.
                first_open = [True]

                while not stop.is_set():
                    proc = None
                    buf  = bytearray()
                    pending = None
                    try:
                        if not first_open[0]:
                            # The playback device is CLOSED right now, and the
                            # amplifier with it, which is the only condition in
                            # which the mic sees its real 1082 noise floor
                            # instead of 1726. Do not reopen it speculatively;
                            # wait for something that actually needs to be
                            # heard.
                            while not stop.is_set():
                                if song_playing.is_set():
                                    # The song task writes straight into
                                    # active_speaker_proc[0], bypassing out_q,
                                    # so it would otherwise wait ~10s for a
                                    # process nothing was going to spawn.
                                    break
                                try:
                                    chunk = await asyncio.wait_for(out_q.get(),
                                                                   timeout=0.25)
                                except asyncio.TimeoutError:
                                    continue
                                if chunk is None:
                                    continue      # stray end-of-turn marker
                                pending = chunk
                                break
                            if stop.is_set():
                                break
                        cmd = ["aplay",
                               "-D", PLAYBACK_DEVICE,
                               "-f", PLAYBACK_FORMAT,
                               "-r", str(PLAYBACK_RATE),
                               "-c", str(PLAYBACK_CHANNELS),
                               "-t", "raw",
                               "--buffer-size=96000",
                               "--period-size=4800",
                               f"--start-delay={SPEAKER_START_DELAY_US}"]
                        # --start-delay and --period-size are worth 600ms of the
                        # reply latency, measured rather than reasoned:
                        #
                        #   aplay -v, OLD args (--buffer-size=96000 only)
                        #     buffer_size 48000  period_size 24000
                        #     start_threshold 48000        <-- 1.00s
                        #   aplay -v, THESE args
                        #     buffer_size 62400  period_size 4800
                        #     start_threshold 19200        <-- 0.40s
                        #
                        # start_threshold is how much audio ALSA insists on
                        # holding before it starts the stream, and aplay derives
                        # it from --start-delay: with the default 0 it becomes
                        # the whole buffer. So the first second of every reply
                        # was being accumulated in silence before the speaker
                        # made a sound. Note the driver also clamped the
                        # requested 96000-frame buffer to 48000 — the old
                        # comments claiming "~0.5s of slack at 96000" were
                        # describing a buffer this card never granted.
                        #
                        # SPEAKER_IDLE_CLOSE_S closes the device between turns,
                        # so this was not a once-per-session cost: every single
                        # reply paid it. Dropping the period from 24000 to 4800
                        # also stops the writer from stalling in 0.5s lumps.
                        # NOTE: deliberately no "-q". aplay gates its
                        # "underrun!!! (at least N ms long)" message on
                        # !quiet_mode, so -q suppressed the one piece of
                        # evidence that distinguishes broken DSP from the
                        # speaker simply being starved of data. drain_stderr()
                        # rate-limits and summarises them.
                        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                                stderr=subprocess.PIPE, bufsize=0)
                        if proc.stdin is None:
                            raise RuntimeError("aplay stdin unavailable")
                        active_speaker_proc[0] = proc
                        # The amp starts hissing when the DEVICE opens, which
                        # is here — not when the first sample is written. Set
                        # before the startup beep for that reason. Cleared in
                        # the finally below, after the process is actually
                        # reaped. See the AMP HISS GUARD note in run_session.
                        amp_open[0] = True
                        threading.Thread(
                            target=drain_stderr,
                            args=(proc, "aplay"),
                            # An underrun that lands while nothing is playing and
                            # the drain deadline has passed is the device sitting
                            # idle waiting for SPEAKER_IDLE_CLOSE_S, which is an
                            # XRUN by ALSA's definition and inaudible. Only
                            # underruns during actual playback are worth alarm.
                            kwargs={"benign_underrun": lambda: (
                                not adam_speaking.is_set()
                                and not song_playing.is_set()
                                and time.time() > drain_deadline[0])},
                            daemon=True).start()
                        print(f"  ✅ aplay: {PLAYBACK_DEVICE} {PLAYBACK_FORMAT} "
                              f"{PLAYBACK_RATE}Hz {PLAYBACK_CHANNELS}ch")
                        if first_open[0]:
                            write_all(proc.stdin, beep_s16_stereo(),
                                      PLAYBACK_CHANNELS * 2)
                            proc.stdin.flush()
                            print("  🔔 Startup beep sent")
                            first_open[0] = False
                        if pending is not None:
                            # The chunk that woke us; converted here so the
                            # inner loop's normal 4096-byte flush ships it.
                            buf.extend(await asyncio.to_thread(
                                s16_mono_24k_to_s16_stereo_48k, pending,
                                SPEAKER_GAIN))
                            pending = None

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
                                # IDLE CLOSE. Nothing is playing, nothing is
                                # queued, the ALSA buffer has drained and no
                                # song is running: release the device so the
                                # amplifier stops hissing into the mic. Every
                                # guard here is load-bearing — closing while
                                # adam_speaking would clip a reply, closing
                                # before drain_deadline would cut its tail, and
                                # closing during a song would strand the song
                                # task's writes.
                                elif (not adam_speaking.is_set()
                                        and not song_playing.is_set()
                                        and not buf
                                        and time.time() > drain_deadline[0]
                                        and time.time() - watchdog_t
                                            > SPEAKER_IDLE_CLOSE_S):
                                    print(f"  🔇 Playback idle "
                                          f"{SPEAKER_IDLE_CLOSE_S:.1f}s — closing "
                                          f"the device so the amp stops raising "
                                          f"the mic floor")
                                    break
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
                                    # write_all(), not proc.stdin.write():
                                    # aplay is spawned with bufsize=0, so
                                    # proc.stdin is a raw _io.FileIO whose
                                    # write() may accept FEWER bytes than
                                    # offered and report how many. Ignoring
                                    # that count silently drops the tail of
                                    # the chunk; if the dropped count is not a
                                    # multiple of 4 the stream de-aligns and
                                    # every following int16 has its low and
                                    # high bytes swapped — a ×256 error, i.e.
                                    # full-scale buzz, that persists for the
                                    # rest of the session because nothing
                                    # re-syncs a PCM pipe. See write_all() in
                                    # audio_utils.py.
                                    await asyncio.to_thread(
                                        write_all, proc.stdin, bytes(buf),
                                        PLAYBACK_CHANNELS * 2)
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
                                # GRACEFUL FIRST, SIGNAL ONLY AS A FALLBACK.
                                # aplay exits by itself on stdin EOF, after
                                # flushing what it still holds. The old code
                                # closed stdin and immediately SIGTERM'd,
                                # which raced that flush and produced this
                                # pair in every log:
                                #   [aplay] Aborted by signal Terminated...
                                #   ⚠️  [aplay] 1 buffer underrun(s)
                                # Since SPEAKER_IDLE_CLOSE_S this teardown
                                # runs after every reply instead of once a
                                # session, and on the voiceHAT — ONE I2S
                                # device shared by capture and playback —
                                # yanking the playback stream mid-DMA is what
                                # left arecord handing out digital silence
                                # forever (see the dead-capture watchdog in
                                # listen()). Let it close itself; escalate
                                # only if it won't.
                                try:
                                    if proc.stdin:
                                        try:
                                            proc.stdin.close()
                                        except Exception:
                                            pass
                                    try:
                                        await asyncio.to_thread(proc.wait, 1.5)
                                        return          # clean exit on EOF
                                    except Exception:
                                        pass
                                    proc.terminate()
                                    await asyncio.to_thread(proc.wait, 2)
                                except Exception:
                                    try:
                                        proc.kill()
                                    except Exception:
                                        pass
                            try:
                                # 4.0 not 3.0: the graceful path above can
                                # legitimately spend 1.5s waiting for EOF
                                # plus 2s waiting out a SIGTERM.
                                await asyncio.wait_for(_kill_proc(), timeout=4.0)
                            except asyncio.TimeoutError:
                                try: proc.kill()
                                except Exception: pass
                        # Falling edge of the amp guard. Unconditional and
                        # last: if the Popen above raised, amp_open may have
                        # been left set, and a stuck-True flag would starve
                        # the floor estimate for the rest of the session.
                        # Stamped only now, after _kill_proc has waited for
                        # the process to actually exit, because the device is
                        # still open — and still hissing — until it does.
                        amp_open[0]    = False
                        amp_quiet_t[0] = time.time()

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
                                _idle_since[0] = 0.0
                                # A deliberate touch IS a user turn, so a
                                # following enter_idle_mode is a real request.
                                last_user_turn_t[0] = time.time()
                                print("  🛑 Touch3 — exiting idle mode")
                                tft_set("happy")
                            else:
                                print("  🛑 STOP gesture — entering idle mode")
                                interrupt_flag.set()
                                idle_mode.set()
                                _idle_mode_persistent[0] = True
                                _idle_since[0] = time.time()
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
                                mic_open_t[0] = time.time()
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
                # Runs entirely offline via Vosk. Two duties, both local:
                #   1. idle_mode    → hear "adam" and wake up. This is the
                #      mechanism that satisfies "nothing sent to Google while
                #      idle, except via Touch3 or hearing 'adam' locally."
                #   2. song_playing → hear a stop phrase, so a 182-215s song
                #      can be interrupted by VOICE. Previously the only stop
                #      was the Touch3 gesture, which needs the ESP32-CAM UART
                #      link, and the mic was muted for the whole song anyway.
                # The model load is deferred to here (not at import time)
                # since it can take a few seconds and shouldn't block session
                # startup for the common case of neither duty being active.
                if not VOSK_AVAILABLE:
                    return
                recognizer = None
                duty       = None       # "idle" | "song" — see the reset below
                try:
                    while not stop.is_set():
                        want = ("song" if song_playing.is_set()
                                else "idle" if idle_mode.is_set() else None)
                        if want is None:
                            # Neither duty active — nothing to detect, drain
                            # any stale queued audio and wait. Recognizer
                            # state isn't needed until a duty starts.
                            while not wake_word_q.empty():
                                try:
                                    wake_word_q.get_nowait()
                                except asyncio.QueueEmpty:
                                    break
                            recognizer = None
                            duty       = None
                            await asyncio.sleep(0.3)
                            continue

                        if recognizer is None or duty != want:
                            # Model was already preloaded once at process
                            # startup (see _vosk_model_instance) — only the
                            # lightweight recognizer wrapper is created here,
                            # per duty period. This is cheap and safe to do
                            # mid-session. It is rebuilt on every duty CHANGE
                            # too: Kaldi carries decoder state across chunks,
                            # and state accumulated from three minutes of
                            # music is exactly what should not be sitting in
                            # the decoder when it goes back to listening for
                            # a wake word.
                            recognizer = await asyncio.to_thread(
                                _VoskKaldiRecognizer, _vosk_model_instance,
                                GEMINI_SEND_RATE)
                            duty = want
                            print("  🔎 Offline recogniser active — "
                                  + ("listening for a stop phrase"
                                     if want == "song" else "wake word 'adam'"))

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
                        if not text:
                            continue

                        if duty == "song":
                            # TWO words required, in either order, so that a
                            # drum hit or a sung syllable cannot stop the
                            # music on its own. Vosk's partial text
                            # accumulates within an utterance, so "adam stop",
                            # "stop the song" and "stop the music" all land in
                            # a single window.
                            if "stop" in text and ("adam" in text
                                                   or "song" in text
                                                   or "music" in text):
                                song_stop_requested.set()
                                print(f"  🛑 Stop phrase heard locally in "
                                      f"{text!r} (offline) — stopping the song")
                                recognizer = None
                                duty       = None
                            continue

                        if "adam" in text:
                            idle_mode.clear()
                            _idle_mode_persistent[0] = False
                            _idle_since[0] = 0.0
                            # Saying the wake word IS a user turn.
                            last_user_turn_t[0] = time.time()
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
                        # …but idle is not allowed to last forever. See
                        # IDLE_MAX_S in config: while idle, nothing reaches
                        # Gemini at all, and both documented exits can fail
                        # in a noisy room, which looks exactly like a dead
                        # unit. Coming back is silent — no greeting, no
                        # nudge — so an unattended ADAM does not start
                        # talking to an empty room.
                        if IDLE_MAX_S > 0 and _idle_since[0]:
                            _held = time.time() - _idle_since[0]
                            if _held >= IDLE_MAX_S:
                                idle_mode.clear()
                                _idle_mode_persistent[0] = False
                                _idle_since[0] = 0.0
                                tft_set("neutral")
                                print(f"  🔔 Idle mode ended automatically "
                                      f"after {_held/60:.1f} min "
                                      f"(IDLE_MAX_S) — listening again")
                        continue
                    elapsed = time.time() - last_interact_t[0]
                    if elapsed < IDLE_TIMEOUT_S:
                        continue
                    last_interact_t[0] = time.time()
                    last_nudge_t[0]    = time.time()
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
        elif isinstance(e, (socket.gaierror, socket.herror, ConnectionError,
                            TimeoutError, asyncio.TimeoutError)) or any(
                s in err_str.lower() for s in
                ("temporary failure in name resolution", "getaddrinfo",
                 "name or service not known", "network is unreachable",
                 "no route to host", "connection reset by peer",
                 "connection refused", "errno -3", "errno -2")):
            # Local network fault, not an API fault. Say so plainly and
            # do NOT dump a traceback — at boot this fires ~28 times in
            # under a minute and burying the real errors under 28 stack
            # traces is how the actual cause got missed for a whole run.
            print(f"  📡 network not ready ({type(e).__name__}: {err_str}) — "
                  f"local DNS/socket issue, not the Gemini API")
            network_transient[0] = True
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
    if network_transient[0]:
        # Handle deliberately passed back: the session was never reached,
        # so it is still valid and the conversation can resume once the
        # resolver answers.
        return ("NETWORK_TRANSIENT", latest_handle)
    return latest_handle
