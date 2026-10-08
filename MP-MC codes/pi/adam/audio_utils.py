"""
audio_utils.py — ADAM v40 audio DSP helpers
==============================================================================
Pure signal-processing helpers with no side effects and no hardware handles:

  • S32 stereo (ALSA capture) → S16 mono 16 kHz  (what Gemini wants to hear),
    band-limited on the way: anti-alias low-pass before the 48k→16k decimation
    and a de-rumble high-pass after it. See the MIC BAND-LIMITING CHAIN block
    below for the measurements that made both necessary. This one helper is
    stateful (filter tails carried across chunks); everything else is pure.
  • AdaptiveGate — learns the room's own noise floor and answers "is this
    speech?" without any per-room configuration. See its docstring.
  • S32 stereo → separate L/R S16 channels        (needed for DOA — averaging
                                                    to mono destroys the phase
                                                    difference between mics)
  • estimate_doa_angle()  — direction-of-arrival via GCC-PHAT
  • S16 mono 24 kHz (Gemini output) → S16 stereo 48 kHz (what the speaker wants)
  • rms_s32 / rms_pcm16 / is_valid_pcm16_chunk — level metering + sanity gates
  • beep_s16_stereo — local UI beep
  • read_exact / drain_stderr — subprocess pipe helpers

Tuning constants (S32_SHIFT, MIC_HP_HZ/MIC_LP_HZ, MIC_DISTANCE_M, sample rates)
come from config so there's a single source of truth for the wiring/audio
parameters.
"""

import collections
import json
import math
import os
import subprocess
import threading
import time

import numpy as np

from config import (
    S32_SHIFT,
    CAPTURE_RATE,
    GEMINI_SEND_RATE,
    PLAYBACK_RATE,
    MIC_HP_HZ,
    MIC_LP_HZ,
    MIC_LP_STOP_HZ,
    MIC_CHANNEL,
    MIC_CH_WINDOW_S, MIC_CH_MIN_S, MIC_CH_LIVE_DR_DB,
    MIC_CH_DEAD_DR_DB, MIC_CH_DEAD_MARGIN_DB,
    MIC_CH_DECIDE_EVERY_S, MIC_CH_STATE_PATH, MIC_CH_STATE_MAX_AGE_S,
    MIC_FLOOR_WINDOW_S,
    MIC_FLOOR_PERCENTILE,
    MIC_FLOOR_MIN_S,
    MIC_FLOOR_RISE,
    MIC_FLOOR_FALL,
    MIC_FLOOR_STATE_PATH,
    MIC_FLOOR_STATE_MAX_AGE_S,
    MIC_CAL_STATE_PATH,
    MIC_CAL_GAIN_DEADBAND_DB,
    MIC_FLOOR_SAVE_EVERY_S,
    MIC_BAND_LO_HZ,
    MIC_BAND_HI_HZ,
    MIC_OPEN_RATIO,
    MIC_OPEN_STRONG,
    MIC_OPEN_MIN,
    MIC_HOLD_RATIO,
    MIC_HOLD_MAX_RATIO,
    MIC_CAND_RATIO,
    MIC_CAND_SHAPE_FRAC,
    MIC_VAD_BACKEND,
    MIC_VAD_AGGRESSIVENESS,
    MIC_VAD_FRAME_MS,
    MIC_VAD_SUSTAIN_S,
    MIC_SHAPE_FLAT_MAX,
    MIC_SHAPE_FLAT_SLACK,
    MIC_SHAPE_RATIO_MIN,
    MIC_SHAPE_HOLD_FRAC,
    MIC_SHAPE_ADAPT,
    MIC_SHAPE_FLAT_CEIL,
    MIC_SHAPE_FLAT_MARGIN,
    MIC_SHAPE_FLAT_PCTL,
    MIC_DISTANCE_M,
    SOUND_SPEED_MPS,
    SPEAKER_LIMITER_KNEE,
    ENABLE_AEC,
    AEC_DELAY_MS,
    AEC_FILTER_LEN_MS,
    ENABLE_EXPANDER,
    EXPANDER_FLOOR_DB,
    MIC_NR,
    MIC_NR_FRAME,
    MIC_NR_OVERSUB,
    MIC_NR_SMOOTH,
    MIC_NR_FLOOR_DB,
    MIC_NR_NOISE_S,
    SONG_BARGE_WINDOW_S,
    SONG_BARGE_RATIO,
    SONG_BARGE_HOLDS,
    SONG_BARGE_MIN_N,
    SONG_BARGE_LAG_MIN_MS,
    SONG_BARGE_LAG_MAX_MS,
    SONG_BARGE_STALL_S,
)

# ═════════════════════════════════════════════════════════════════════════════
# RNNOISE — optional deep-learning noise suppressor (Priority-2 audio fix)
# ═════════════════════════════════════════════════════════════════════════════
# Mozilla's RNNoise model uses a GRU network to compute per-band gains.  When
# voice is present the gains are near 1.0 (untouched); when only noise is
# present the gains are near 0.0.  Unlike spectral subtraction it cannot
# mangle voice: the network was trained on thousands of hours of human speech
# and has learned what speech looks like.  CPU cost on Pi Zero 2W is ~2-4 ms
# per 33 ms chunk — affordable.
#
# Install:  pip install rnnoise-python
# If the package is absent the flag stays False and the pipeline is unchanged.
try:
    import rnnoise                          # type: ignore
    _rnn = rnnoise.RNNoise()
    RNNOISE_AVAILABLE = True
except Exception:
    _rnn = None
    RNNOISE_AVAILABLE = False

_RNN_FRAME = 160          # RNNoise's fixed frame size: exactly 10 ms @ 16 kHz


def denoise_rnn(pcm16: bytes) -> bytes:
    """Apply RNNoise to S16 PCM at 16 kHz.  Must be called from a thread
    (not the asyncio event loop) because rnnoise releases the GIL but the
    call itself is synchronous.

    rnnoise.RNNoise.process_frame() requires EXACTLY 160 samples (= 320
    bytes) at a time.  We loop over the input in 160-sample windows and
    accumulate; the tail (< 160 samples) is passed through unchanged so we
    never alter the chunk length or introduce latency.

    The AdaptiveGate in session.py sees the UN-denoised RMS (measured before
    this call) so it keeps learning the real room noise floor.  Only the
    audio forwarded to Gemini gets RNNoise applied.
    """
    if not RNNOISE_AVAILABLE or not pcm16:
        return pcm16
    arr = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)
    out = np.empty_like(arr)
    n = len(arr)
    i = 0
    while i + _RNN_FRAME <= n:
        frame = arr[i : i + _RNN_FRAME]
        denoised = _rnn.process_frame(frame)          # returns float32 array
        out[i : i + _RNN_FRAME] = denoised
        i += _RNN_FRAME
    if i < n:                                          # tail — copy unchanged
        out[i:] = arr[i:]
    return np.clip(out, -32768, 32767).astype(np.int16).tobytes()


# ═════════════════════════════════════════════════════════════════════════════
# DOWNWARD EXPANDER  (Priority-3 audio fix)
# ═════════════════════════════════════════════════════════════════════════════
# This is NOT a frequency filter.  It works on RMS level:
#   * When RMS >= speech_thr  -> gain = 1.0  (voice passes untouched)
#   * When RMS <= noise_ceil  -> gain = floor_gain  (~0.032 at -30 dB)
#   * In between              -> smooth linear ramp
# The result: room silence is attenuated ~30 dB, conversational speech passes
# at unity gain, low-level background noise is attenuated 10-20 dB.
# No frequency content is altered — only the envelope.
#
# Changes vs. a hypothetical -16 dB version:
#   floor_db  = -30.0  -> silence floor 22 dB deeper
#   threshold = ambient_rms * 1.20  -> gate opens sooner on speech
#   release   = 0.90 coefficient   -> ~3x faster tail decay than 0.95

class _NoiseExpander:
    """Downward expander applied to S16 16kHz mono audio after the FIR chain,
    before the audio reaches Gemini.  The AdaptiveGate in session.py sees the
    UN-expanded RMS so it keeps learning the true room floor.

    SERVO GUARD: ambient_rms learning is suppressed while the pan servo is
    moving (hardware.servo_moving is set).  Without this guard, the structural
    vibration the servo injects into the mic PCB inflates the p25 ambient
    estimate, raises speech_thr, and then genuine quiet speech falls below the
    expanded threshold and gets attenuated — the exact opposite of what the
    expander is supposed to do.  The servo guard is the expander-side analogue
    of the amp_open guard on the AdaptiveGate floor in session.py.

    The import of servo_moving is done lazily inside process() to avoid a
    circular import at module load time (hardware.py → esp32_link.py → config.py
    → audio_utils.py would form a cycle if audio_utils imported hardware at the
    top level).
    """

    def __init__(self, floor_db: float = EXPANDER_FLOOR_DB, hangover_chunks: int = 12, preroll_chunks: int = 2) -> None:
        self.floor_gain = 10.0 ** (floor_db / 20.0)   # ~0.20 at -14 dB
        self.ambient_rms = 400.0                        # warm start
        self.current_gain = self.floor_gain
        self.hangover_chunks = hangover_chunks
        self.hangover_left = 0
        self.preroll_n = preroll_chunks
        self.preroll_buf = collections.deque(maxlen=self.preroll_n)
        self.is_open = False
        self._rms_acc: list = []
        self._servo_moving = None   # resolved lazily on first call

    def reset(self) -> None:
        """Reset expander state on turn transition or speaker start."""
        self.hangover_left = 0
        self.is_open = False
        self.current_gain = self.floor_gain
        self.preroll_buf.clear()

    def _get_servo_moving(self):
        """Lazy import of hardware.servo_moving to avoid circular imports."""
        if self._servo_moving is None:
            try:
                from hardware import servo_moving as _sm  # noqa: PLC0415
                self._servo_moving = _sm
            except Exception:
                # hardware.py unavailable (unit test, no GPIO) — treat as
                # servo-not-moving so the expander works normally.
                import threading
                self._servo_moving = threading.Event()  # always clear
        return self._servo_moving

    def process(self, chunk: bytes | np.ndarray, is_speech: bool | None = None) -> list[bytes] | bytes:
        if not ENABLE_EXPANDER:
            return chunk
        if chunk is None or len(chunk) == 0:
            return chunk
        is_bytes = isinstance(chunk, (bytes, bytearray))
        raw_bytes = chunk if is_bytes else chunk.tobytes()
        arr = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(arr ** 2))) if arr.size else 0.0

        # Update ambient estimate from the quietest recent chunks (~5 s window).
        servo_moving = self._get_servo_moving()
        if not servo_moving.is_set():
            self._rms_acc.append(rms)
            if len(self._rms_acc) > 150:
                self._rms_acc.pop(0)
            if self._rms_acc:
                sorted_rms = sorted(self._rms_acc)
                pct25 = sorted_rms[max(0, len(sorted_rms) // 4)]
                self.ambient_rms = max(80.0, 0.92 * self.ambient_rms + 0.08 * pct25)

        # Adaptive speech decision: caller can pass is_speech directly (from AdaptiveGate)
        if is_speech is None:
            speech_thr = self.ambient_rms * 1.20
            is_speech = rms >= speech_thr

        out_chunks: list[bytes] = []

        if is_speech:
            self.hangover_left = self.hangover_chunks
            if not self.is_open:
                # Speech onset detected! Flush pre-roll buffer at FULL 1.0 GAIN
                # so initial consonant onsets (/t/, /k/, /p/, /dʒ/) are 100%
                # preserved without truncation.
                while self.preroll_buf:
                    old_c = self.preroll_buf.popleft()
                    out_chunks.append(old_c)
                self.is_open = True
                self.current_gain = 1.0
        elif self.hangover_left > 0:
            self.hangover_left -= 1
            self.is_open = True
        else:
            self.is_open = False

        target = 1.0 if self.is_open else self.floor_gain

        # Instant attack on speech, smooth release on silence
        if target > self.current_gain:
            self.current_gain = target
        else:
            self.current_gain = 0.85 * self.current_gain + 0.15 * target

        if self.is_open:
            out_chunks.append(raw_bytes)
        else:
            # Apply the SMOOTHED gain, not floor_gain. Slamming straight to
            # the floor puts a gain step at the end of every utterance, and a
            # step is a broadband transient — exactly the kind of event a
            # server-side VAD reads as the start of a turn. The 0.85/0.15
            # release above reaches the floor over ~200 ms instead, which also
            # keeps a trailing fricative the gate under-called.
            if len(self.preroll_buf) >= self.preroll_n:
                old_c = self.preroll_buf.popleft()
                old_arr = np.frombuffer(old_c, dtype=np.int16).astype(np.float32)
                gated = np.clip(old_arr * self.current_gain, -32768, 32767).astype(np.int16).tobytes()
                out_chunks.append(gated)
            self.preroll_buf.append(raw_bytes)

        return out_chunks


# Module-level expander instance (stateful — carries tail across chunks)
_noise_expander = _NoiseExpander(floor_db=EXPANDER_FLOOR_DB)


# ═════════════════════════════════════════════════════════════════════════════
# ACOUSTIC ECHO CANCELLATION (AEC) — Optional Full-Duplex (Priority-4 audio fix)
# ═════════════════════════════════════════════════════════════════════════════
# Uses speexdsp (or pywebrtc) to dynamically subtract ADAM's loudspeaker audio
# from the microphone capture in real time.
#
# Install on Pi:
#   sudo apt install -y libspeexdsp-dev
#   pip install speexdsp
#
# When speexdsp is installed and ENABLE_AEC=1, ADAM operates in true full-duplex:
#   • POST_MUTE_S drops from 0.45s to 0.05s (near zero), phone-call style.
#   • ADAM can process user speech while speaking (barge-in / interruption).
#
# If the library is absent or ENABLE_AEC=0, AEC_AVAILABLE stays False and
# microphone audio passes through unchanged with zero performance penalty.

try:
    from speexdsp import EchoCanceller       # type: ignore
    SPEEX_AVAILABLE = True
except Exception:
    EchoCanceller = None
    SPEEX_AVAILABLE = False


class AcousticEchoCanceller:
    """Acoustic Echo Canceller using speexdsp. Maintains a circular delay
    buffer of speaker playback audio and subtracts acoustic echo from the
    microphone signal frame-by-frame."""

    def __init__(self, sample_rate: int = 16000, frame_size: int = 160,
                 filter_len_ms: int = AEC_FILTER_LEN_MS,
                 delay_ms: int = AEC_DELAY_MS) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.bytes_per_frame = frame_size * 2
        self.delay_ms = delay_ms
        self.delay_samples = int(sample_rate * (delay_ms / 1000.0))
        self.delay_bytes = self.delay_samples * 2
        self.filter_length = int(sample_rate * (filter_len_ms / 1000.0))

        self.ref_buffer = bytearray()
        self.canceller = None
        self.enabled = ENABLE_AEC and SPEEX_AVAILABLE
        self.is_playing = False
        self._lock = threading.Lock()

        if self.enabled and EchoCanceller is not None:
            try:
                self.canceller = EchoCanceller.create(
                    self.frame_size, self.filter_length, self.sample_rate)
                print(f"  ✅ AEC active: speexdsp (frame={self.frame_size}, "
                      f"filter={self.filter_length}, delay={delay_ms}ms)")
            except Exception as e:
                print(f"  ⚠️  AEC initialization failed: {e}")
                self.canceller = None
                self.enabled = False
        elif ENABLE_AEC and not SPEEX_AVAILABLE:
            print("  ℹ️  ENABLE_AEC=1 requested but speexdsp is not installed.")
        elif not ENABLE_AEC:
            print("  ℹ️  Half-duplex mode: speaker active → mic MUTED (zero self-echo)")

    @property
    def is_available(self) -> bool:
        return self.enabled and self.canceller is not None

    def start_playback(self) -> None:
        """Called when speaker playback begins."""
        with self._lock:
            self.is_playing = True
            # Pre-pad with delay_bytes of silence so reference matches speaker delay
            self.ref_buffer = bytearray(b"\x00" * self.delay_bytes)

    def feed_playback(self, pcm_mono_16k: bytes) -> None:
        """Feed loudspeaker reference audio (16kHz mono S16) into the AEC delay line."""
        if not self.is_available or not pcm_mono_16k:
            return
        with self._lock:
            if not self.is_playing:
                self.is_playing = True
                self.ref_buffer = bytearray(b"\x00" * self.delay_bytes)
            self.ref_buffer.extend(pcm_mono_16k)
            max_bytes = self.delay_bytes + self.sample_rate * 2 * 3
            if len(self.ref_buffer) > max_bytes:
                del self.ref_buffer[:-max_bytes]

    def feed_playback_48k_stereo(self, pcm_stereo_48k: bytes) -> None:
        """Feed 48kHz stereo S16 audio (as written to aplay) into the AEC delay line.
        Automatically downsamples to 16kHz mono (decimating 3:1 on left channel)."""
        if not self.is_available or not pcm_stereo_48k:
            return
        ref16k = np.frombuffer(pcm_stereo_48k, dtype=np.int16)[0::6].tobytes()
        self.feed_playback(ref16k)

    def stop_playback(self) -> None:
        """Called when speaker playback ends or is interrupted."""
        with self._lock:
            self.is_playing = False
            self.ref_buffer.clear()

    def process(self, mic_pcm16: bytes) -> bytes:
        """Process microphone audio chunk through AEC against delayed speaker reference."""
        if not self.is_available or self.canceller is None or not mic_pcm16:
            return mic_pcm16

        out = bytearray()
        silence_frame = b"\x00" * self.bytes_per_frame
        n = len(mic_pcm16)
        i = 0

        while i + self.bytes_per_frame <= n:
            mic_frame = mic_pcm16[i : i + self.bytes_per_frame]

            with self._lock:
                if self.is_playing and len(self.ref_buffer) >= self.bytes_per_frame:
                    ref_frame = bytes(self.ref_buffer[:self.bytes_per_frame])
                    del self.ref_buffer[:self.bytes_per_frame]
                else:
                    ref_frame = silence_frame
                    if not self.is_playing:
                        self.ref_buffer.clear()

            try:
                cleaned_frame = self.canceller.process(mic_frame, ref_frame)
                out.extend(cleaned_frame)
            except Exception:
                out.extend(mic_frame)

            i += self.bytes_per_frame

        if i < n:
            out.extend(mic_pcm16[i:])

        return bytes(out)

    def reset(self) -> None:
        """Reset canceller state upon speaker turn transitions."""
        with self._lock:
            self.is_playing = False
            self.ref_buffer.clear()
            if self.enabled and EchoCanceller is not None:
                try:
                    self.canceller = EchoCanceller.create(
                        self.frame_size, self.filter_length, self.sample_rate)
                except Exception:
                    pass


# Module-level AEC instance
_aec_canceller = AcousticEchoCanceller()
AEC_AVAILABLE = _aec_canceller.is_available


# ═════════════════════════════════════════════════════════════════════════════
# SPECTRAL NOISE SUPPRESSOR (WOLA) — +7.6 dB in-band SNR improvement
# ═════════════════════════════════════════════════════════════════════════════

class _NoiseSuppressor:
    """Streaming single-channel WOLA minimum-statistics spectral subtraction denoiser
    for int16 mono at GEMINI_SEND_RATE."""

    def __init__(self, frame: int, oversub: float, floor_db: float,
                 noise_s: float, rate: int, smooth: float = MIC_NR_SMOOTH) -> None:
        self._n   = max(64, int(frame) & ~1)          # even
        self._h   = self._n // 2                      # COLA hop for sqrt-Hann
        hann      = np.hanning(self._n + 1)[:self._n]
        self._w   = np.sqrt(hann).astype(np.float32)
        self._in   = np.zeros(0, dtype=np.float32)
        self._acc  = np.zeros(self._n, dtype=np.float32)
        nb         = self._n // 2 + 1
        self._pwr  = np.zeros(nb, dtype=np.float32)
        self._gain = np.ones(nb, dtype=np.float32)
        self._floor   = float(10.0 ** (floor_db / 20.0))
        self._oversub = float(oversub)
        self._alpha   = min(0.99, max(0.0, float(smooth)))
        frames_per_s   = float(rate) / float(self._h)
        self._sub_len  = max(1, int(round(noise_s * frames_per_s / 4.0)))
        self._subs     = collections.deque(maxlen=4)
        self._cur_min  = None
        self._sub_n    = 0
        self._primed   = False
        lo_bin = max(1, int(round(300.0 * self._n / float(rate))))
        hi_bin = min(nb, int(round(3400.0 * self._n / float(rate))) + 1)
        self._band = slice(lo_bin, hi_bin)
        self._db   = 0.0

    def reset(self) -> None:
        self._in  = np.zeros(0, dtype=np.float32)
        self._acc = np.zeros(self._n, dtype=np.float32)
        self._gain[:] = 1.0

    def _noise_est(self, pwr: np.ndarray) -> np.ndarray:
        self._cur_min = (pwr.copy() if self._cur_min is None
                         else np.minimum(self._cur_min, pwr))
        self._sub_n += 1
        if self._sub_n >= self._sub_len:
            self._subs.append(self._cur_min)
            self._cur_min = None
            self._sub_n   = 0
            if len(self._subs) == self._subs.maxlen:
                self._primed = True
        est = self._cur_min
        for s in self._subs:
            est = s if est is None else np.minimum(est, s)
        return est

    def process(self, pcm: bytes) -> bytes:
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        if x.size == 0:
            return pcm
        self._in = np.concatenate((self._in, x)) if self._in.size else x
        out = []
        while self._in.size >= self._n:
            spec = np.fft.rfft(self._in[:self._n] * self._w)
            pwr  = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
            self._pwr = (self._alpha * self._pwr + (1.0 - self._alpha) * pwr)
            noise = self._noise_est(self._pwr)
            if self._primed:
                clean = np.maximum(pwr - self._oversub * noise, 0.0)
                g     = np.sqrt(clean / np.maximum(pwr, 1e-9))
                np.maximum(g, self._floor, out=g)
                g[1:-1] = (g[:-2] + g[1:-1] + g[2:]) / 3.0
                alpha = np.where(g > self._gain, 0.15, 0.65).astype(np.float32)
                g = alpha * self._gain + (1.0 - alpha) * g
                self._gain = g.astype(np.float32)
                spec *= self._gain
                self._db = 20.0 * math.log10(
                    max(float(self._gain[self._band].mean()), 1e-6))
            y = np.fft.irfft(spec, self._n).astype(np.float32) * self._w
            self._acc += y
            out.append(self._acc[:self._h].copy())
            self._acc = np.concatenate(
                (self._acc[self._h:], np.zeros(self._h, dtype=np.float32)))
            self._in = self._in[self._h:]
        if not out:
            return pcm
        y = np.concatenate(out)
        return np.clip(y, -32768, 32767).astype(np.int16).tobytes()

_mic_nr = (_NoiseSuppressor(MIC_NR_FRAME, MIC_NR_OVERSUB, MIC_NR_FLOOR_DB,
                            MIC_NR_NOISE_S, GEMINI_SEND_RATE)
           if MIC_NR else None)

def denoise_16k(pcm: bytes) -> bytes:
    return pcm if _mic_nr is None else _mic_nr.process(pcm)

def denoise_reset() -> None:
    if _mic_nr is not None:
        _mic_nr.reset()


# ═════════════════════════════════════════════════════════════════════════════
# MIC BAND-LIMITING CHAIN  (48kHz stereo S32  -> 16kHz mono S16 for Gemini)
# ═════════════════════════════════════════════════════════════════════════════
# Measured on this build's INMP441 pair in a QUIET room (adam/_specdiag.py):
#
#   band                L mic    R mic       <- % of total captured energy
#   below 300 Hz        71.9%    12.7%
#   300-3400 Hz         13.6%    15.2%       <- the only part that is speech
#   above 8 kHz         11.4%    53.5%
#
# So ~85% of what the mics produce when nobody is talking is out-of-band junk:
# the left mic is rumble-dominated, the right is hiss-dominated. Two concrete
# problems came out of that, and this chain fixes both:
#
#  1. ALIASING (the serious one). The old code went straight from 48kHz to
#     16kHz with `mono[::3]` — plain decimation, NO anti-alias low-pass. Taking
#     every 3rd sample folds everything above 8 kHz down into 0-8 kHz, so the
#     right mic's 53% HF hiss landed *on top of* the speech band and could not
#     be separated afterwards. That is a large part of why Gemini kept
#     mis-transcribing (English coming back as random other languages).
#     Fix: FIR low-pass at MIC_LP_HZ BEFORE decimating.
#
#  2. RUMBLE. 72% of the left mic's energy is under 300 Hz — inaudible as
#     speech, but it dominates RMS, eats int16 headroom, and drags the level
#     gates around. Fix: linear-phase high-pass at MIC_HP_HZ.
#
# Both filters keep STATE ACROSS CHUNKS (the tails below). Filtering each 33ms
# chunk independently would restart the filter every chunk and inject a
# discontinuity at every boundary — a ~30Hz tick train, which is exactly the
# kind of artefact we are trying to remove.

DECIM = max(1, CAPTURE_RATE // GEMINI_SEND_RATE)      # 48000/16000 = 3

def _design_lowpass(fc: float, fs: float, ntaps: int) -> np.ndarray:
    """Windowed-sinc (Hamming) low-pass, unity DC gain. fc is the -6 dB point;
    the transition band straddles it, so ntaps has to be chosen from where the
    stopband must START, not from where the corner is (see _lp_taps_for)."""
    n = np.arange(ntaps, dtype=np.float64) - (ntaps - 1) / 2.0
    h = np.sinc(2.0 * fc / fs * n) * np.hamming(ntaps)
    return (h / h.sum()).astype(np.float32)


def _lp_taps_for(f_pass: float, f_stop: float, fs: float) -> int:
    """Odd tap count whose Hamming transition band fits inside f_pass..f_stop.

    A Hamming-windowed sinc has a transition width of about 3.3*fs/ntaps
    between the passband and the -53 dB stopband. Solving for ntaps and forcing
    it odd keeps the filter linear-phase with an integer group delay, which is
    what lets the decimator below stay sample-accurate.
    """
    width = max(1.0, float(f_stop) - float(f_pass))
    return max(31, int(math.ceil(3.3 * fs / width)) | 1)


# Passband edge, stopband edge, and the -6 dB corner half way between them.
# 6800 -> 8000 needs 132 taps at 48 kHz, so this is 133 rather than the 63 it
# used to be. The polyphase decimation below computes only the 1-in-3 outputs
# that survive, so the real cost is 533x133 MACs per 33 ms chunk: ~2.1 MMAC/s,
# which numpy hands to BLAS and the Pi Zero 2 W does not notice.
_LP_TAPS = _lp_taps_for(MIC_LP_HZ, MIC_LP_STOP_HZ, CAPTURE_RATE)
_LP_FIR  = _design_lowpass((MIC_LP_HZ + MIC_LP_STOP_HZ) * 0.5,
                           CAPTURE_RATE, _LP_TAPS)
# Reversed copy: a dot product against a forward-ordered sliding window equals
# a convolution only if the kernel is flipped. _LP_FIR is symmetric so this is
# the same array, but relying on that silently would break the moment the
# window function or design method changes.
_LP_FIR_R = _LP_FIR[::-1].copy()

def _design_biquad_hp(fc: float, fs: float) -> tuple[float, float, float, float, float]:
    """Design a 2nd-order Butterworth high-pass filter via bilinear transform.
    Maximally flat passband (0.00 dB ripple), steep 12 dB/octave rolloff,
    completely eliminating the comb-filter distortion of moving-average subtraction."""
    w = math.tan(math.pi * fc / fs)
    w2 = w * w
    sqrt2 = math.sqrt(2.0)
    norm = 1.0 / (1.0 + sqrt2 * w + w2)
    b0 = norm
    b1 = -2.0 * norm
    b2 = norm
    a1 = 2.0 * (w2 - 1.0) * norm
    a2 = (1.0 - sqrt2 * w + w2) * norm
    return b0, b1, b2, a1, a2


class _BiquadHP:
    """Direct Form II Transposed biquad filter for clean, transparent high-pass."""
    def __init__(self, fc: float = 80.0, fs: float = 16000.0) -> None:
        self.b0, self.b1, self.b2, self.a1, self.a2 = _design_biquad_hp(fc, fs)
        self.s1 = 0.0
        self.s2 = 0.0
        self._primed = False

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        y = np.empty_like(x, dtype=np.float32)
        b0, b1, b2 = self.b0, self.b1, self.b2
        a1, a2 = self.a1, self.a2
        if not self._primed:
            # Pre-charge states with the initial DC offset to eliminate startup step transient
            x0 = float(x[0])
            self.s1 = -b0 * x0
            self.s2 = b2 * x0
            self._primed = True
        s1, s2 = self.s1, self.s2
        for i in range(len(x)):
            xi = float(x[i])
            yi = b0 * xi + s1
            s1 = b1 * xi - a1 * yi + s2
            s2 = b2 * xi - a2 * yi
            y[i] = yi
        self.s1 = s1
        self.s2 = s2
        return y


class _MicChain:
    """Stateful 48k stereo -> 16k mono band-limited converter."""

    def __init__(self) -> None:
        self._lp_tail   = None
        self._dec_phase = 0
        self._hp        = _BiquadHP(fc=MIC_HP_HZ, fs=GEMINI_SEND_RATE)

    def process(self, mono48: np.ndarray) -> np.ndarray:
        if mono48.size == 0:
            return np.empty(0, dtype=np.float32)
        # Pre-fill tail with initial DC value instead of zeros to avoid FIR edge discontinuity
        if self._lp_tail is None:
            self._lp_tail = np.full(_LP_TAPS - 1, mono48[0], dtype=np.float32)
        # ── anti-alias low-pass + decimation, fused (polyphase) ─────────────
        buf = np.concatenate((self._lp_tail, mono48))
        self._lp_tail = buf[-(_LP_TAPS - 1):].copy()
        win = np.lib.stride_tricks.sliding_window_view(buf, _LP_TAPS)
        lp = win[self._dec_phase::DECIM] @ _LP_FIR_R

        self._dec_phase = (self._dec_phase - int(mono48.size)) % DECIM

        # ── 2nd-order Butterworth high-pass: kills DC and 26Hz ripple without comb notches
        return self._hp.process(lp)


_mic_chain = _MicChain()


# ── MIC CHANNEL LIVENESS — which physical mic feeds the speech path ────
# MEASURED on this unit, 2026-09-07, by comparing 6 s of silence against 6 s
# of loud speech through the robot's own speaker. The one-off probe that
# produced this table has been deleted; the table itself is the finding, and
# mic_check.py re-confirms the live end of it. Per-channel response to sound,
# dB per band:
#
#            100-300  300-1k   1k-2k  2k-3.4k  3.4-5k   5-8k
#     L        +35.1   +31.0   +35.4    +32.7   +27.2   +22.4   responds
#     R         +0.2    -0.0    -0.1     -0.0    +0.0    +0.0   DEAD
#
# The right channel does not respond to sound AT ALL, and its noise is 7.5 dB
# LOUDER than the live left mic's. L<->R magnitude-squared coherence is 0.012
# for speech — exactly the 1/n_frames floor of the estimator, i.e. genuinely
# zero, which two mics in one head sharing one sound field cannot be. This is
# consistent with a single INMP441 wired as L and the SD line floating during
# the R word slot. It is a hardware fault on the Vero board; the software
# must not depend on it being fixed, and must pick it up automatically if it
# is.
#
# Averaging that dead channel into the live one — which is what the old
# default MIC_CHANNEL=mix did — costs, per band:
#
#     +4.6  +7.7  +18.3  +21.1  +21.2  +21.4 dB of SNR
#
# worst exactly in the consonant band. THAT is "ADAM mis-hears everything":
# vowels stay above the bed so "Hello" survives, every consonant cue is
# buried so "ADAM" comes back "madam", and it is intermittent because the
# result sits right at the decision threshold. For scale, the best suppressor
# benchmarked on this hardware buys 9 dB and the shipped WOLA one buys 3.4.
#
# WHY THE OLD CODE CHOSE WRONG, so it is not repeated: it justified mixing
# with "R alone had a post-filter noise floor of p50 1498 against 804 for the
# L+R mix — 5.4 dB WORSE in band", which is precisely what a dead noisy R
# measures. L ALONE WAS NEVER MEASURED. And its `auto` criterion was ADC
# clipping, which never happens on this unit either (7.4 dB of headroom, 0
# saturated samples in 12 s), so `auto` always latched `mix`.
#
# THE CRITERION NOW: a channel wired to a working mic RESPONDS TO SOUND — its
# own level rises far above its own quiet baseline when someone talks. A dead
# channel is stationary, because bus noise does not care about the room. So
# liveness is the dynamic range of each channel's own first-difference RMS
# over a rolling window, 20*log10(p99/p20):
#
#     L  baseline 9.1e6  peak 2.2e8  ->  +27.9 dB
#     R  baseline 1.6e8  peak 1.9e8  ->   +1.7 dB
#
# a 26 dB separation either side of the 8 dB threshold. The first difference
# is used because it is one vector op, removes DC completely (L carries a
# large DC offset) and emphasises the band the decision matters for. No
# absolute level and no room constant appears anywhere: every quantity is
# that channel's own statistic, which is what makes this safe to ship on
# units whose mics are fine.
#
# The verdict is STICKY and one-way per channel — a channel that has proven
# it hears cannot be un-proven by a quiet room — and it is persisted, so only
# the first run on a new unit pays the learning window. DOA is unaffected: it
# reads the channels separately via s32_stereo_to_s16_stereo_channels(), and
# on this unit it cannot work at all until the right mic is repaired.
class _MicChannelLiveness:
    """Decide mix/left/right from each channel's own response to sound."""

    def __init__(self) -> None:
        self.forced = MIC_CHANNEL != "auto"
        self.mode = MIC_CHANNEL if self.forced else "mix"
        chunks = max(1, int(round(MIC_CH_WINDOW_S * CAPTURE_RATE / 1600.0)))
        self._hist: list[collections.deque] = [
            collections.deque(maxlen=chunks), collections.deque(maxlen=chunks)]
        self._min_n = max(1, int(round(MIC_CH_MIN_S * CAPTURE_RATE / 1600.0)))
        self._every = max(1, int(round(MIC_CH_DECIDE_EVERY_S
                                       * CAPTURE_RATE / 1600.0)))
        self._n = 0
        self.seen_live = [False, False]
        self.seen_dead = [False, False]
        self.dr = [0.0, 0.0]
        self._announced = None
        if not self.forced:
            self._load()

    # ── persistence: the fault does not heal between runs ──────────────
    def _load(self) -> None:
        try:
            with open(MIC_CH_STATE_PATH) as f:
                st = json.load(f)
            if time.time() - float(st.get("t", 0)) > MIC_CH_STATE_MAX_AGE_S:
                return
            sl = st.get("seen_live")
            if not (isinstance(sl, list) and len(sl) == 2):
                return
            self.seen_live = [bool(sl[0]), bool(sl[1])]
            sd = st.get("seen_dead")
            if isinstance(sd, list) and len(sd) == 2:
                self.seen_dead = [bool(sd[0]), bool(sd[1])]
            self.dr = [float(x) for x in st.get("dr", [0.0, 0.0])]
            self._apply()

            def _tag(i):
                if self.seen_live[i]:
                    return "live"
                return "DEAF" if self.seen_dead[i] else "unproven"

            print(f"  🎙️  Resuming learned mic channel {self.mode.upper()} "
                  f"(L {_tag(0)} {self.dr[0]:+.1f} dB, "
                  f"R {_tag(1)} {self.dr[1]:+.1f} dB)")
            self._announced = self.mode
        except Exception:
            pass

    def _save(self) -> None:
        try:
            tmp = f"{MIC_CH_STATE_PATH}.tmp"
            with open(tmp, "w") as f:
                json.dump({"t": time.time(), "seen_live": self.seen_live,
                           "seen_dead": self.seen_dead,
                           "dr": self.dr, "mode": self.mode}, f)
            os.replace(tmp, MIC_CH_STATE_PATH)
        except Exception:
            pass

    def _apply(self) -> None:
        live_l, live_r = self.seen_live
        if live_l and live_r:
            self.mode = "mix"
        elif live_l:
            self.mode = "left"
        elif live_r:
            self.mode = "right"
        elif self.seen_dead[1]:
            self.mode = "left"         # R proven deaf relative to L
        elif self.seen_dead[0]:
            self.mode = "right"
        else:
            self.mode = "mix"          # undecided: keep both until proven

    def apply_calibration(self, live: list, dead: list, snr_db: list) -> None:
        """Install the verdict from startup tone calibration.

        This is the SAME state the passive test builds (seen_live/seen_dead/dr)
        reached by a better route. The passive test has to wait for the room to
        be lively enough to clear MIC_CH_LIVE_DR_DB and then infer response
        from the spread between the 20th and 99th percentiles of ambient; the
        calibration plays a known chime and measures each channel's SNR against
        it directly. Same question, a controlled stimulus instead of a hopeful
        one, and an answer in three seconds instead of MIC_CH_MIN_S of luck.

        Seeding rather than freezing: the passive learner keeps running on top
        of this. That is deliberate — a mic that comes loose an hour into a
        session must still be demoted, and only the passive path is watching
        then. Calibration decides where ADAM STARTS; the passive test remains
        the thing that notices if it changes mid-run.
        """
        prev = self.mode
        self.seen_live = [bool(live[0]), bool(live[1])]
        self.seen_dead = [bool(dead[0]), bool(dead[1])]
        self.dr = [float(snr_db[0]), float(snr_db[1])]
        self._apply()
        if self.mode != prev:
            # Same reasoning as in observe(): which channels feed the path sets
            # the level the floor describes, so a channel change invalidates it.
            _reset_mic_floor(f"calibration: mic channel {prev} → {self.mode}")
        self._announced = self.mode
        names = ["left", "right"]
        deaf = [n.upper() for n, ok in zip(names, self.seen_live) if not ok]
        print(f"  🎙️  Mic channel → {self.mode.upper()} "
              f"(calibrated: L {self.dr[0]:+.1f} dB, R {self.dr[1]:+.1f} dB "
              f"tone SNR)")
        if len(deaf) == 1:
            print(f"     ⚠️  the {deaf[0]} mic did not hear ADAM's own chime — "
                  f"it is contributing noise only and has been dropped from "
                  f"the speech path. Check that mic's wiring; DOA/direction "
                  f"sensing cannot work until it does.")
        self._save()

    def observe(self, l: np.ndarray, r: np.ndarray) -> None:
        """One capture chunk. Cheap: two diffs and two dot products."""
        if self.forced:
            return
        for i, ch in ((0, l), (1, r)):
            d = np.diff(ch)
            self._hist[i].append(
                float(math.sqrt(float(np.dot(d, d)) / max(d.size, 1))))
        self._n += 1
        if self._n % self._every or len(self._hist[0]) < self._min_n:
            return

        before = (list(self.seen_live), list(self.seen_dead))
        prev = self.mode
        for i in (0, 1):
            v = np.fromiter(self._hist[i], dtype=np.float64)
            lo, hi = np.percentile(v, [20, 99])
            self.dr[i] = 20.0 * math.log10(max(hi, 1.0) / max(lo, 1.0))
            if self.dr[i] >= MIC_CH_LIVE_DR_DB:
                self.seen_live[i] = True
        # A channel can also be condemned on purely RELATIVE evidence, which
        # matters because the absolute test above needs the room to be lively
        # enough to clear MIC_CH_LIVE_DR_DB. Measured on this unit from 30 s of
        # ambient with nobody speaking: L +9.4 dB, R +1.2 dB — it happened to
        # clear 8 dB, but only just, and a quieter room would have left the
        # path in "mix" and thrown away 11.7 dB (see the block above). A
        # channel that sits still WHILE THE OTHER ONE MOVES is not hearing the
        # room, whatever the absolute numbers are, so it can be dropped from a
        # silent capture with no speech at all.
        for i in (0, 1):
            j = 1 - i
            if (self.dr[i] < MIC_CH_DEAD_DR_DB
                    and self.dr[j] - self.dr[i] >= MIC_CH_DEAD_MARGIN_DB):
                self.seen_dead[i] = True
            elif self.dr[i] >= MIC_CH_DEAD_DR_DB:
                self.seen_dead[i] = False      # it moved: no longer condemned
        # Demotion, so a mic that fails in the field self-corrects: only ever
        # on RELATIVE evidence — a channel that has gone quiet WHILE THE OTHER
        # ONE IS RESPONDING is dead, whereas both going quiet is just a quiet
        # room and must not change anything.
        for i in (0, 1):
            j = 1 - i
            if (self.seen_live[i] and self.dr[i] < MIC_CH_LIVE_DR_DB
                    and self.dr[j] >= MIC_CH_LIVE_DR_DB):
                self.seen_live[i] = False

        self._apply()
        if self.mode != prev:
            # The learned noise floor is an int16 LEVEL, and that level depends
            # on which channels feed the path — dropping a channel changes it
            # by many dB at once. Carrying the old floor across the switch
            # would leave the gate calibrated for a signal that no longer
            # exists, so it is discarded and relearned (~1.5 s).
            _reset_mic_floor(f"mic channel {prev} → {self.mode}")
        decided = any(self.seen_live) or any(self.seen_dead)
        if self.mode != self._announced and decided:
            self._announced = self.mode
            names = ["left", "right"]
            dead = [n for n, s in zip(names, self.seen_live) if not s]
            print(f"  🎙️  Mic channel → {self.mode.upper()}: "
                  f"dynamic range L {self.dr[0]:+.1f} dB / R {self.dr[1]:+.1f} "
                  f"dB over the last {MIC_CH_WINDOW_S:.0f}s "
                  f"(≥{MIC_CH_LIVE_DR_DB:.0f} dB = responds to sound, or "
                  f"≥{MIC_CH_DEAD_MARGIN_DB:.0f} dB below the other channel "
                  f"= deaf)")
            if len(dead) == 1:
                print(f"     ⚠️  the {dead[0].upper()} mic does not respond to "
                      f"sound — it is contributing noise only, so it has been "
                      f"dropped from the speech path. Check that mic's wiring "
                      f"on the Vero board; DOA/direction sensing cannot work "
                      f"until it does. Set MIC_CHANNEL=mix in .env to "
                      f"override.")
        elif self._announced is None:
            self._announced = self.mode      # still undecided: record quietly
        if (list(self.seen_live), list(self.seen_dead)) != before:
            self._save()


def _reset_mic_floor(why: str) -> None:
    """Discard the learned noise floor. Resolved from globals() at call time
    because the gate instance is created further down this module than the
    channel selector that needs to invalidate it."""
    g = globals().get("_adaptive_gate")
    if g is not None:
        g.reset_floor(why)


_mic_live = _MicChannelLiveness()
_mic_ch_mode = [_mic_live.mode]          # kept for callers that read the mode


# ── MIC SCALE (the old S32_SHIFT, made measurable) ──────────────────────
# The divisor applied between the 32-bit I2S word and the 16-bit sample that
# the gate, the expander and Gemini all see. S32_SHIFT is only its STARTING
# value now; mic_calibrate.calibrate() overwrites it at boot from a measured
# peak, because a fixed shift is the wrong shape of answer to this question.
#
# The evidence that it is the wrong shape: this project has set the shift to
# 13, 14, 15 and 16 at different times, once per hardware or room change,
# each time by hand and each time documented as the fix (see
# docs/development_log.md and docs/mic_speaker_issues.md A5, which ends with
# "DO NOT DO THIS"). Both failure directions are recorded there. Too much
# gain clips: at shift 14 the log records "over 2,100 clipped samples per
# 2-second window", and hard clipping "generates harsh odd harmonics,
# flattening vowels and destroying consonant differentiation (/p/, /b/,
# /s/)" — which is what mishearing IS. Too little gain buries speech: at
# shift 16 with the wrong channel selected the log records speech at "RMS
# 535 to 738".
#
# The correct value is not a constant of the software, it is a property of
# the body, the amplifier and the room the unit is standing in — so it is
# measured at startup against a known stimulus rather than guessed here.
# A float divisor rather than a bit-shift because the answer is continuous;
# the arithmetic was already a float divide, so this costs nothing.
_mic_scale = [float(1 << S32_SHIFT)]


def set_mic_scale(divisor: float, why: str = "",
                  reset_floor: bool = True) -> None:
    """Set the 32-to-16 bit divisor. Called by the startup calibration.

    reset_floor=True discards the learned noise floor, which is mandatory
    rather than tidy when the gain genuinely MOVES: the floor is an absolute
    level in the OLD units, so leaving it in place would have the gate
    comparing new levels against an old yardstick. Callers that have a
    freshly measured floor (calibrate() does) should seed it immediately
    after; the reset only guarantees that a caller which forgets cannot leave
    a stale one behind.

    reset_floor=False is for RESTORING a previously measured gain at boot.
    There the floor on disk was measured at this very divisor, so the two
    already agree and throwing it away would discard the good value to
    protect against a mismatch that does not exist.
    """
    if not (divisor > 0.0) or not math.isfinite(divisor):
        return
    was = _mic_scale[0]
    # DEADBAND. The chime peak this divisor is solved from is an acoustic
    # measurement, so it varies by a few tenths of a dB from boot to boot for
    # reasons that have nothing to do with the mic: amp warm-up, where the
    # case is resting, a door closing during the chime. Acting on that
    # variation is worse than ignoring it, because every accepted change
    # calls _reset_mic_floor() and throws away the floor that was measured at
    # startup — so a unit whose gain is already correct would relearn its
    # floor from scratch on every single boot, which is exactly the churn the
    # resume-on-boot design exists to prevent.
    #
    # MIC_CAL_GAIN_DEADBAND_DB is well below anything audible or anything the
    # ratio-based gate can notice (the gate's own open threshold sits ~1.9 dB
    # over the floor and both scale together), and comfortably above the
    # observed run-to-run spread.
    if abs(20.0 * math.log10(divisor / was)) < MIC_CAL_GAIN_DEADBAND_DB:
        return                                   # nothing worth announcing
    _mic_scale[0] = float(divisor)
    print(f"  🔊 Mic gain {20.0 * math.log10(was / divisor):+.1f} dB "
          f"(÷{was:.0f} → ÷{divisor:.0f}) — {why}")
    if reset_floor:
        _reset_mic_floor("mic gain changed")


def _restore_mic_scale() -> None:
    """Resume the last calibrated divisor at import, before anything reads audio.

    This has to happen HERE, beside the channel and floor resumes, rather than
    inside mic_calibrate.calibrate(). Three pieces of learned state have to
    agree for the gate to mean anything:

        channel  <- .mic_channel.json  resumed at import (_MicChannelLiveness)
        floor    <- .mic_floor.json    resumed at import (AdaptiveGate)
        gain     <- .mic_cal.json      *this*

    The floor is an ABSOLUTE level, so it is only meaningful at the divisor it
    was measured at. Resuming two of the three and leaving the gain at
    S32_SHIFT pairs a floor measured at ÷230005 with audio scaled at ÷65536 —
    11 dB of mismatch, in whichever direction is worse: the gate either sits
    far below the noise and latches open on room tone, or far above speech and
    the unit goes deaf. Nothing in the log looks wrong either way.

    Doing it at import means every entry point gets it: main.py, but also
    mic_check.py, song_lag_bench.py and mic_noise_probe.py, whose whole value
    is that their numbers describe the SHIPPED pipeline rather than a
    re-implementation of it. calibrate() still calls set_mic_scale() afterwards
    with a freshly measured value; this only ensures that the units are
    coherent before it does, and stay coherent if it never runs.

    reset_floor=False because the floor on disk was measured at this very
    divisor — see set_mic_scale's docstring.
    """
    try:
        with open(MIC_CAL_STATE_PATH, "r") as f:
            st = json.load(f)
        scale = float(st.get("mic_scale", 0.0))
        if scale > 0.0 and math.isfinite(scale):
            set_mic_scale(scale, "restored from the last calibration",
                          reset_floor=False)
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"  ⚠️  could not read {MIC_CAL_STATE_PATH}: {e}")


_restore_mic_scale()


def _mono16k_float(raw: bytes) -> np.ndarray:
    """The speech path, up to but NOT including the int16 clip.

    Split out so the calibration can measure the true peak. Measuring it on
    the clipped output would read back at most 32767 no matter how far over
    the signal actually went, so a badly-set gain would look perfectly
    scaled — the measurement has to survive the fault it exists to detect.
    """
    s32 = np.frombuffer(raw, dtype=np.int32)
    if s32.size < 2:
        return np.zeros(0, dtype=np.float32)
    # Combine the mics in FLOAT, before any scaling or clipping. The old
    # code shifted and cast each channel to int16 first, so a loud sample was
    # already clipped (previously: silently WRAPPED to a large opposite-sign
    # spike) before the two channels were even combined.
    _l = s32[0::2].astype(np.float32)
    _r = s32[1::2].astype(np.float32)
    _mic_live.observe(_l, _r)
    _m = _mic_ch_mode[0] = _mic_live.mode
    if   _m == "left":  mono48 = _l
    elif _m == "right": mono48 = _r
    else:               mono48 = (_l + _r) * 0.5
    return _mic_chain.process(mono48) / _mic_scale[0]


def s32_stereo_to_s16_mono_16k(raw: bytes) -> bytes:
    mono16 = _mono16k_float(raw)
    if mono16.size == 0:
        return b""
    pcm = np.clip(mono16, -32768, 32767).astype(np.int16)
    return pcm.tobytes()

def s32_stereo_to_s16_stereo_channels(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Same S32->S16 downshift as s32_stereo_to_s16_mono_16k, but returns
    the two channels SEPARATELY instead of averaging them together. Needed
    for direction-of-arrival estimation, which requires the phase/timing
    difference between the two physical mics — information that's
    destroyed the instant left+right get averaged into mono.

    Deliberately NOT band-limited/decimated: DOA needs the full 48kHz rate for
    sub-sample time resolution, and GCC-PHAT already whitens the spectrum, so
    the out-of-band energy that hurts the speech path doesn't hurt this one."""
    s32 = np.frombuffer(raw, dtype=np.int32)
    if s32.size < 2:
        return np.array([], dtype=np.int16), np.array([], dtype=np.int16)
    left  = np.clip(s32[0::2] / _mic_scale[0], -32768, 32767).astype(np.int16)
    right = np.clip(s32[1::2] / _mic_scale[0], -32768, 32767).astype(np.int16)
    return left, right


def s32_stereo_to_float_channels(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Both channels in int16 units as float, WITHOUT the clip.

    This is the startup calibration's input, and the missing clip is the
    whole point of it existing separately. Calibration has to measure a path
    whose gain may currently be wrong — that is the fault it is there to
    correct — and a clipped capture lies in exactly the direction that hides
    the problem: it flattens the peaks that prove the gain is too high, and
    it folds tone energy into harmonics that corrupt the per-tone SNR the
    channel verdict is drawn from. Measuring unclipped keeps the first boot
    after a gain mistake able to diagnose itself.
    """
    s32 = np.frombuffer(raw, dtype=np.int32)
    if s32.size < 2:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    return (s32[0::2] / _mic_scale[0], s32[1::2] / _mic_scale[0])

# ── Direction-of-arrival (DOA) via GCC-PHAT ─────────────────────────────
# INMP441 mic spacing on the v32 BODY board — matches the physical
# separation between the two I2S mics on the PCB. MIC_DISTANCE_M lives in
# config.py; adjust it there if your actual build differs — this value
# directly scales the angle estimate (wrong spacing = systematically wrong
# angle, not just noisy).

def estimate_doa_angle(left: np.ndarray, right: np.ndarray,
                       sample_rate: int = CAPTURE_RATE) -> float:
    """
    Generalized Cross-Correlation with Phase Transform (GCC-PHAT) — a
    standard, well-understood technique for estimating the direction a
    sound arrived from using two microphones. Returns an angle in degrees:
    negative = sound arrived from the left, positive = from the right,
    0 = directly ahead/center. Cheap enough to run per-chunk on a Pi Zero
    2W (a handful of FFTs on ~1600-sample windows).

    This does NOT replace Gemini's own audio understanding — it's a
    separate, local signal DGEN can use for physical reactions (turning
    the neck toward a speaker, or telling the model roughly where a voice
    came from) without waiting on a model round-trip.
    """
    try:
        if left.size == 0 or right.size == 0 or left.size != right.size:
            return 0.0
        n = 1 << (int(left.size) - 1).bit_length()  # next pow2 for speed
        L = np.fft.rfft(left.astype(np.float32), n=n)
        R = np.fft.rfft(right.astype(np.float32), n=n)
        cross = L * np.conj(R)
        denom = np.abs(cross)
        denom[denom < 1e-10] = 1e-10  # avoid div-by-zero on silence
        cc = np.fft.irfft(cross / denom, n=n)

        max_shift = int(sample_rate * MIC_DISTANCE_M / SOUND_SPEED_MPS) + 1
        cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
        shift = int(np.argmax(cc)) - max_shift

        val = (shift / sample_rate) * SOUND_SPEED_MPS / MIC_DISTANCE_M
        val = float(np.clip(val, -1.0, 1.0))
        return float(np.degrees(np.arcsin(val)))
    except Exception:
        return 0.0

# ═════════════════════════════════════════════════════════════════════════════
# SPEAKER CHAIN  (Gemini's 24kHz mono S16  ->  48kHz stereo S16 for aplay)
# ═════════════════════════════════════════════════════════════════════════════
# 24000 -> 48000 is exactly 2x. The obvious implementation — repeat each sample
# and average with its neighbour for the midpoint — is LINEAR INTERPOLATION, and
# it is a poor reconstruction filter. Its response is (1 + cos(2*pi*f/48000))/2:
#
#     1 kHz  -0.02 dB      6 kHz  -0.86 dB
#     8 kHz  -2.0  dB     10 kHz  -4.0  dB      12 kHz  -6.0 dB
#
# Gemini sends 24 kHz audio, so its band runs to 12 kHz and the entire top
# octave — where /s/, /sh/, /t/ and every other sibilant and stop burst lives —
# was being attenuated by 2-6 dB. That is heard exactly as speech that is muffled
# and "not clear", which is what remained after the SPEAKER_GAIN clipping fix.
#
# Replaced with a proper 2x polyphase interpolator: a 63-tap windowed-sinc
# low-pass at 11.4 kHz, split into its two phases. Zero-stuffing then filtering
# is mathematically what upsampling means; the polyphase form just skips the
# multiplications by the inserted zeros, so only 32 taps per output sample are
# ever evaluated. Measured on the Pi against the linear version it replaces:
#
#     freq      new       old linear
#     3 kHz    -0.00 dB    -0.33 dB
#     6 kHz    -0.01 dB    -1.25 dB
#     8 kHz    -0.01 dB    -2.04 dB
#    10 kHz    -0.10 dB    -2.73 dB
#
# 63 taps rather than 31: at 31 the Hamming transition band is ~3.7 kHz wide, so
# with the cutoff below the source's 12 kHz Nyquist the rolloff had already
# reached -1.9 dB by 10 kHz — better than linear but still not flat.
#
# It carries filter state across chunks for the same reason the mic chain does:
# Gemini streams its reply as many small chunks, and the previous implementation
# ran
#
#     np.interp(np.linspace(0, mono.size - 1, mono.size * 2), ...)
#
# independently on each one. Two further defects followed. First, linspace over
# [0, size-1] in size*2 steps has a spacing of (size-1)/(size*2-1), not 0.5, so
# the chunk was resampled at slightly the wrong rate. Second, every chunk was
# forced to start and end exactly ON an input sample, so each boundary duplicated
# a sample and broke the waveform's slope — a discontinuity many times per
# second. Carrying state makes the output one continuous stream.

_UP_TAPS = 63
_UP_FIR  = _design_lowpass(11400.0, PLAYBACK_RATE, _UP_TAPS) * 2.0   # 2x for the
                                                                    # inserted zeros
# Split into polyphase branches. With y[2n]=x[n], y[2n+1]=0, the output
# out[m] = sum_k h[k]*y[m-k] separates exactly into
#   out[2n]   = (h[0::2] * x)[n]      out[2n+1] = (h[1::2] * x)[n]
# so each output sample only ever touches the real input samples. Pad the
# shorter branch so both share one tail length.
_UP_PH0 = _UP_FIR[0::2]
_UP_PH1 = _UP_FIR[1::2]
_UP_PH_LEN = max(_UP_PH0.size, _UP_PH1.size)
_UP_PH0 = np.pad(_UP_PH0, (0, _UP_PH_LEN - _UP_PH0.size))[::-1].copy()
_UP_PH1 = np.pad(_UP_PH1, (0, _UP_PH_LEN - _UP_PH1.size))[::-1].copy()


class _SpkChain:
    def __init__(self) -> None:
        self._tail = np.zeros(_UP_PH_LEN - 1, dtype=np.float32)

    def upsample_2x(self, mono: np.ndarray) -> np.ndarray:
        buf = np.concatenate((self._tail, mono.astype(np.float32, copy=False)))
        self._tail = buf[-(_UP_PH_LEN - 1):].copy()
        win = np.lib.stride_tricks.sliding_window_view(buf, _UP_PH_LEN)
        out = np.empty(mono.size * 2, dtype=np.float32)
        out[0::2] = win @ _UP_PH0
        out[1::2] = win @ _UP_PH1
        return out


_spk_chain = _SpkChain()

# Clip accounting for the playback path. SPEAKER_GAIN multiplies Gemini's TTS,
# which already arrives near full scale, so too much gain saturates instead of
# getting louder — the distortion reads as "the speaker sounds broken". At the
# old gain of 2.5 a normal -3dBFS TTS peak (~23,000) lands at ~57,500, well past
# int16's 32,767, so loud syllables were flat-topped. Counting it makes the
# problem visible in the log rather than something to guess at; speaker() prints
# and resets these. For MORE VOLUME raise the ALSA/hardware level, not the gain.
spk_clip_samples = [0]
spk_total_samples = [0]


def _soft_limit(x: np.ndarray) -> np.ndarray:
    """Replace hard clipping with a smooth, bounded soft knee.

    THE BUG THIS FIXES, in the user's words: "adam's speaker sounds like its gain
    is increasing from low to mid where it was working perfectly then the gain
    goes high where it had lots of noise."

    That is the signature of hard clipping, not of a gain control. SPEAKER_GAIN is
    a fixed multiplier — nothing in ADAM ramps the output level (the volume tools
    in laptop_agent_client.py act on the LAPTOP, not this speaker) — so what
    varies is the CONTENT. Quiet and mid-level passages stay under int16 full
    scale and reproduce cleanly; loud syllables cross it and used to be
    flat-topped by np.clip. Flat-topping a waveform synthesises broadband
    harmonics, so the distortion appears and disappears with the loudness of what
    is being said, which is heard as the gain lurching up into noise. The log has
    been reporting the mechanism every turn: "Speaker clipped 0.1% of samples this
    turn at SPEAKER_GAIN=1.3".

    A soft knee bounds the signal without ever flat-topping it:
      • |x| below the knee is returned UNCHANGED — at the default knee that is
        every peak which was not going to clip anyway;
      • above the knee the excess is compressed through tanh, which is monotonic
        and asymptotic to full scale, so peaks are squashed rather than sheared.
    tanh'(0) == 1, so the curve's slope matches the linear region exactly at the
    knee — no discontinuity to hear at the transition.

    MEASURED HONESTLY, this is a backstop and not the cure. Sweeping the knee
    against hard clipping on the real chain (see the table in config.py under
    SPEAKER_GAIN) recovered at most ~0.2 THD points once the signal was over the
    ceiling: 3.9% vs 3.9% at 1.10 FS, 10.2% vs 10.3% at 1.30 FS. Removing energy
    that does not fit in int16 costs distortion no matter how gracefully it is
    done. What fixed the user's complaint was dropping SPEAKER_GAIN to 1.0 so the
    signal stops exceeding full scale at all. This function stays because it is
    cheap (1.4 ms per 20 ms chunk) and because it eliminates flat tops outright
    (7,000-11,000 sheared samples per half-second tone became 0), and flat tops
    radiate harmonics well above the 12th that a THD figure never counts — so it
    keeps the residual peaks, including the resampler's own overshoot at gain 1.0,
    from ever turning into hard edges.

    An earlier revision of this had the knee at 0.70, which made things WORSE for
    mid-loud content: it compressed everything above 0.70 FS, so a 0.91 FS peak
    that hard clipping left alone at 0.002% THD came out at 1.205%. The knee
    belongs just below full scale, where the limiter only acts where clipping
    would have.
    """
    limit = 32767.0
    knee  = SPEAKER_LIMITER_KNEE * limit
    if knee >= limit:
        return x
    span = limit - knee
    mag  = np.abs(x)
    over = mag > knee
    if not over.any():
        return x
    out = x.copy()
    out[over] = (np.sign(x[over])
                 * (knee + span * np.tanh((mag[over] - knee) / span)))
    return out


def s16_mono_24k_to_s16_stereo_48k(raw: bytes, gain: float = 1.0) -> bytes:
    mono = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if mono.size == 0:
        return b""
    if gain != 1.0:
        mono = mono * gain
    up = _spk_chain.upsample_2x(mono)
    # Counted BEFORE limiting, and deliberately after upsampling: the polyphase
    # interpolator can overshoot between two in-range samples, so this is the
    # count of samples that would actually have been flat-topped on their way to
    # the speaker, not an estimate taken earlier in the chain.
    n_clip = int(np.count_nonzero((up > 32767) | (up < -32768)))
    spk_clip_samples[0] += n_clip
    spk_total_samples[0] += up.size
    up = _soft_limit(up)
    # np.clip stays as a backstop for float rounding at the asymptote; after
    # _soft_limit it should have nothing left to do.
    up = np.clip(up, -32768, 32767).astype(np.int16)
    return np.repeat(up[:, None], 2, axis=1).reshape(-1).tobytes()

def rms_s32(raw: bytes) -> float:
    s = np.frombuffer(raw, dtype=np.int32).astype(np.float64)
    return float(np.sqrt(np.mean(s * s))) if s.size > 0 else 0.0

def rms_pcm16(pcm: bytes) -> float:
    """RMS of the FILTERED 16kHz mono audio, in int16 units (0..32767).

    This — not rms_s32 — is what the level gates in session.py compare against.
    rms_s32 measures the raw S32 capture, which on this hardware is ~85%
    out-of-band rumble and hiss (see the mic chain above): it read 68M-108M in a
    silent room, i.e. ~40x above the old MIC_SILENCE_FLOOR, so the silence gate
    and the adaptive noise-floor gate could never fire and pure room noise was
    streamed to Gemini continuously. Measuring AFTER the band-pass gives a
    number that actually tracks speech, in an intuitive unit.
    """
    s = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    return float(np.sqrt(np.mean(s * s))) if s.size > 0 else 0.0


# The units every persisted absolute level is measured in, stamped into
# .mic_floor.json so a floor measured under a different yardstick is discarded
# instead of silently trusted. Derived from the band rather than hardcoded, so
# retuning MIC_BAND_LO_HZ/HI_HZ invalidates old floors automatically — the
# person who retunes the band is exactly the person who will forget to.
_FLOOR_UNITS = f"{MIC_BAND_LO_HZ:.0f}-{MIC_BAND_HI_HZ:.0f}Hz"

# Cache: {(n_samples): (band_mask, parseval_weights)}. The chunk length is
# constant in practice (533 samples), so this is a one-entry dict after the
# first chunk — it exists so a short final chunk cannot force a rebuild loop.
_BAND_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def speech_band_rms(pcm: bytes) -> float:
    """RMS of the 16 kHz mono restricted to MIC_BAND_LO_HZ..MIC_BAND_HI_HZ.

    This is the gate's yardstick — the number observe()/is_speech() compare
    against the learned floor. rms_pcm16 measures the same audio FULL-BAND and
    is still what the barge-in and song paths use; only the gate moved.

    Why the gate needed its own measure: this mic's noise is broadband and
    impulsive while speech is neither, so a full-band RMS reports mostly
    energy that carries no speech. Measured on the shipped chain over 12 s of
    an empty room (2026-10-01), moving the measurement into 300-3400 Hz drops
    the noise 3.9 dB at p20, drops its kurtosis from 9.1 to 4.9, and drops the
    margin needed to clear the worst noise chunk from 13.5 dB to 9.1 dB —
    while leaving speech, which lives in that band, essentially untouched.
    That is what made an open ratio wide enough to actually close the gate
    affordable; see the MIC_OPEN_RATIO note in config.py.

    Computed from one rFFT via Parseval rather than by filtering, for two
    reasons. It is exact — a rectangular window makes Parseval an identity, so
    this returns a true band RMS in the same int16 units as rms_pcm16, not a
    filter's approximation of one. And it is fast: _BiquadHP.process is a
    per-sample Python loop, and the gate runs on every chunk on a Pi Zero 2 W,
    where a second such loop is real budget. The rFFT is vectorised in C.

    Spectral leakage from the rectangular window is negligible here because
    this integrates a 3.1 kHz-wide band, not a single bin.
    """
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    n = x.size
    if n < 32:
        return 0.0
    ent = _BAND_CACHE.get(n)
    if ent is None:
        freqs = np.fft.rfftfreq(n, 1.0 / GEMINI_SEND_RATE)
        mask = (freqs >= MIC_BAND_LO_HZ) & (freqs <= MIC_BAND_HI_HZ)
        # Parseval for a real signal: sum(x^2) = (1/n) * sum_k w_k |X_k|^2,
        # where every bin counts twice for its negative-frequency twin except
        # DC and, when n is even, Nyquist — which have no twin.
        w = np.full(freqs.size, 2.0)
        w[0] = 1.0
        if n % 2 == 0:
            w[-1] = 1.0
        ent = (mask, w)
        _BAND_CACHE[n] = ent
    mask, w = ent
    p = np.abs(np.fft.rfft(x)) ** 2
    ms = float(np.sum(p[mask] * w[mask])) / float(n * n)
    return math.sqrt(ms) if ms > 0.0 else 0.0

def is_valid_pcm16_chunk(mono16k: bytes) -> bool:
    """
    Sanity gate — structural validation instead of amplitude heuristics.

    Earlier revisions tried to detect corruption by how many samples were
    clipped (0.35, then loosened to 0.60 after legitimate loud speech kept
    getting dropped). That was the wrong signal: clipping/amplitude is a
    property of how loud someone is talking and how hot the mic gain is
    set, NOT a reliable indicator of whether the buffer is structurally
    corrupt. Tightening it caused real speech loss ("only hears the last
    part"); loosening it let a genuinely malformed buffer through to
    Gemini, which triggered:
        "1007 invalid frame payload data — Request contains an invalid
         argument" — a protocol-level close that kills the whole session.

    The reliable check is structural: PCM16 audio must be a whole number
    of 2-byte samples. The S32->S16 mono 16kHz conversion always produces
    a deterministic, even-length output for valid input. An odd byte
    count (or empty buffer) is a definitive corruption/truncation signal
    regardless of how loud or quiet the audio inside it is — and never
    penalizes legitimate loud speech, which is a completely separate,
    unrelated property that should not be used as a corruption proxy.
    """
    if not mono16k:
        return False
    if len(mono16k) % 2 != 0:
        return False
    arr = np.frombuffer(mono16k, dtype=np.int16)
    if arr.size == 0:
        return False
    return True

# ── ADAPTIVE SPEECH GATE ────────────────────────────────────────────────
# Everything above this line is signal processing. This is the decision:
# "is someone talking to ADAM right now?" — and it is the part that has to
# work in a room nobody measured beforehand.
#
# WHY THE OLD ABSOLUTE THRESHOLDS COULD NOT SHIP. The gate used to compare
# the filtered int16 RMS against constants: MIC_SILENCE_FLOOR = 1800, with
# an adaptive term clamped by MIC_AMBIENT_MAX = 1650. Both numbers were
# measured in ONE room on ONE unit, and the clamp's own comment records the
# trap: MIC_AMBIENT_MAX * MIC_SPEECH_MARGIN has to stay below the quietest
# speech (2357 on that day), so the ceiling can never rise above ~1746. A
# live log then showed exactly what that means in a different room — a
# phone call playing across the desk put the floor at p50 1872, ABOVE the
# 1800 open threshold, while the adaptive tracker sat pinned at its 1650
# ceiling and could not follow. The gate latched open for 45s at a time,
# fed room noise to Gemini, and ADAM printed advice to hand-edit .env with
# a computed MIC_SILENCE_FLOOR. Needing an engineer per room is not a
# product.
#
# WHAT REPLACES IT. Two independent votes, neither of which contains a
# number specific to this room, this unit, or this user's voice:
#
#   1. A LEARNED FLOOR. Per-chunk RMS goes into a ring buffer covering
#      MIC_FLOOR_WINDOW_S seconds; the floor is a low percentile
#      (MIC_FLOOR_PERCENTILE) of that window. In conversation speech is a
#      minority of wall-clock time, and even during continuous talking the
#      gaps between syllables land in the low percentiles — so a low
#      percentile IS the noise floor, by construction, at any absolute
#      level. That is the property the old exponential average lacked: an
#      EMA integrates speech into its own estimate, which is why it needed
#      a clamp and a cooldown to stop it poisoning itself, and the clamp is
#      what then broke in a louder room. Thresholds become ratios of that
#      floor, so a quiet bedroom and a noisy office get the same behaviour
#      at different absolute levels.
#
#   2. A SPEECH-SHAPE VOTE that ignores level entirely, computed from one
#      1024-point rFFT of the same 16 kHz mono the model gets:
#        • SPECTRAL FLATNESS over 120-6800 Hz (geometric mean / arithmetic
#          mean of the power spectrum). Noise is flat, speech is not: voiced
#          speech puts its energy into a harmonic comb under a few formant
#          peaks, so the geometric mean collapses. This is the strongest
#          single discriminator measured on this hardware.
#        • LO/HI BAND RATIO, 120-1000 Hz against 1000-6800 Hz. Voiced
#          speech is bottom-heavy; hiss and fan whine are not.
#      Both are scale-invariant, so they carry no number specific to this
#      room, unit or user — the property the level test cannot have.
#
# DO NOT PUT webrtcvad BACK HERE. It was the first design and it was
# refuted by measurement on this HAT, not by argument: on 25 s of ordinary
# room noise it called 100.0% of frames "speech" at aggressiveness 0, 1 and
# 2, and 98.6% at 3. A vote that says yes to everything is not a vote. Its
# knobs survive as MIC_VAD_BACKEND / MIC_VAD_AGGRESSIVENESS, defaulted off,
# only so a different microphone can be tried without a code change.
#
# The votes cover each other's blind spots: the floor ratio rejects distant
# or other-room speech (level), the shape test rejects loud steady noise
# (spectrum). Opening needs both, on MIC_VAD_ONSET_CHUNKS consecutive
# chunks, unless the level is overwhelming (MIC_OPEN_STRONG x the floor),
# which is its own evidence.
#
# HOLDING deliberately does NOT ask "did the shape pass recently". At the
# per-chunk false-positive rate this feature really has (~5-10% on noise), a
# 0.5 s "recently" window is true ~79% of the time on noise alone, so it can
# never help the gate CLOSE — and under Gemini's manual activity detection a
# gate that cannot close means no reply at all. Holding instead needs a
# FRACTION of the sustain window to pass (MIC_SHAPE_HOLD_FRAC), mirroring
# the rolling median used for level, with a little slack on the flatness
# threshold (MIC_SHAPE_FLAT_SLACK) so consonants and inter-syllable dips do
# not truncate a turn.
#
# The learned floor is also persisted to MIC_FLOOR_STATE_PATH, so a restart
# resumes with the room it already knows instead of a cold ring buffer.


# Spectral-shape constants. Precomputed once: the window and the band masks
# never change, and at 30 chunks/s the whole feature costs 1.02 ms against a
# 33.3 ms budget (measured on the Pi Zero 2 W), so this is free.
_SHP_NFFT = 1024
_SHP_WIN  = np.hanning(_SHP_NFFT).astype(np.float32)
_SHP_FREQ = np.fft.rfftfreq(_SHP_NFFT, 1.0 / GEMINI_SEND_RATE)
_SHP_B_LO = (_SHP_FREQ >= 120) & (_SHP_FREQ < 1000)
_SHP_B_HI = (_SHP_FREQ >= 1000) & (_SHP_FREQ < MIC_LP_HZ)
_SHP_B_SP = (_SHP_FREQ >= 120) & (_SHP_FREQ < MIC_LP_HZ)


class AdaptiveGate:
    """Learns a room's noise floor and scores each chunk for speech shape.

    Self-contained and unit-testable: feed it observe(rms) and
    shape_ok(pcm) and read .floor / .open_th / .hold_th / .shape_frac. No
    config beyond ratios, no per-room constants, no calibration step the
    user has to perform.
    """

    def __init__(self, chunks_per_s: float) -> None:
        self._n_win  = max(30, int(round(MIC_FLOOR_WINDOW_S * chunks_per_s)))
        self._ring   = collections.deque(maxlen=self._n_win)
        self._floor  = 0.0
        self._recalc_every = max(1, int(round(chunks_per_s / 6.0)))
        self._since_calc   = 0
        self._min_n   = max(8, int(round(MIC_FLOOR_MIN_S * chunks_per_s)))
        self._loaded  = False
        self._saved_t = 0.0
        # Shape history, same length as the level sustain window so the two
        # hold tests see the same span of time.
        self._n_sus   = max(3, int(round(MIC_VAD_SUSTAIN_S * chunks_per_s)))
        self._shp_win = collections.deque(maxlen=self._n_sus)
        self.flat     = 1.0     # last measured flatness   (1.0 = pure noise)
        self.lohi     = 0.0     # last measured lo/hi ratio
        self.backend  = "shape"
        # LEARNED FLATNESS THRESHOLD — the shape test's own version of the
        # learned level floor, and the reason this gate can be shipped to a
        # room nobody has measured.
        #
        # MIC_SHAPE_FLAT_MAX = 0.35 was measured in ONE room: there the noise
        # bed's flatness sat high enough that 0.35 passed only 4.3% of noise
        # chunks. In a room with a flatter, hissier bed (a fan, an air
        # conditioner, a PC next to the mic) 0.35 is far stricter than it
        # needs to be, and strictness here is not free — it is paid for in
        # rejected speech, because speech recorded at low SNR is itself
        # flatter than clean speech: the noise fills in the spectral valleys
        # between the harmonics that this statistic exists to see.
        #
        # So the bed's flatness is measured, at a low percentile, from chunks
        # the caller labels as known noise (gate shut, level under the open
        # threshold, amplifier off), and the threshold is placed just under
        # it. Deliberately ONE-SIDED: the learned value may only ever LOOSEN
        # the test, never tighten it past the measured 0.35. A learned
        # threshold that can tighten could, in a room whose noise is TONAL
        # (a whine, a hum — low flatness, lower than speech), walk itself
        # down until nothing passes and deafen ADAM completely. The floor at
        # MIC_SHAPE_FLAT_MAX means the worst case is exactly today's
        # behaviour and the best case is a room that finally works.
        self._flat_ring = collections.deque(maxlen=self._n_win)
        self._flat_max  = MIC_SHAPE_FLAT_MAX
        self._flat_calc = 0
        # webrtcvad wants exactly 10/20/30 ms frames; a 1600-frame @48k
        # capture chunk decimates to 533 samples (33.3 ms), which is not a
        # legal size, so frames are cut from a carry buffer instead. Off by
        # default — see the refutation above; kept only as an escape hatch
        # for a different microphone.
        self._vad         = None
        self._vad_frame   = int(GEMINI_SEND_RATE * MIC_VAD_FRAME_MS / 1000)
        self._vad_carry   = b""
        if MIC_VAD_BACKEND == "webrtc":
            try:
                import webrtcvad
                self._vad    = webrtcvad.Vad(MIC_VAD_AGGRESSIVENESS)
                self.backend = "shape+webrtc"
            except Exception as e:
                print(f"  ⚠️  MIC_VAD_BACKEND=webrtc but webrtcvad is "
                      f"unavailable ({e}) — using shape only")
        self._load()

    # ── learned noise floor ─────────────────────────────────────────
    def observe(self, rms: float) -> float:
        """Feed one chunk's RMS. Returns the current floor estimate."""
        self._ring.append(float(rms))
        self._since_calc += 1
        if self._since_calc >= self._recalc_every or self._floor <= 0.0:
            self._since_calc = 0
            n = len(self._ring)
            if n >= self._min_n:
                new = float(np.percentile(np.fromiter(self._ring, np.float64,
                                                      n),
                                          MIC_FLOOR_PERCENTILE))
            else:
                # Not enough history yet. Prefer a persisted floor from the
                # last run over a guess; otherwise use the running minimum,
                # which errs sensitive rather than deaf.
                new = (self._floor if self._loaded
                       else min(self._ring) if self._ring else 0.0)
            # Rise slowly, fall quickly. A room that gets NOISIER should not
            # deafen ADAM instantly on one door slam; a room that goes quiet
            # should regain sensitivity right away.
            if self._floor <= 0.0:
                self._floor = new
            elif new > self._floor:
                self._floor += (new - self._floor) * MIC_FLOOR_RISE
            else:
                self._floor += (new - self._floor) * MIC_FLOOR_FALL
            self._maybe_save()
        return self._floor

    @property
    def floor(self) -> float:
        return self._floor

    def reset_floor(self, why: str = "") -> None:
        """Throw away the learned floor and relearn from scratch.

        Called when the SIGNAL ITSELF changes scale — today only when the
        channel selector drops or readmits a mic, which moves the noise level
        by many dB in one step. Everything level-derived has to go with it:
        the ring, the persisted state, and the learned flatness threshold
        (measured on the old bed). MIC_FLOOR_MIN_S of audio rebuilds it.
        """
        self._ring.clear()
        self._flat_ring.clear()
        self._floor = 0.0
        self._loaded = False
        self._flat_max = MIC_SHAPE_FLAT_MAX
        self._since_calc = 0
        self._saved_t = time.time()      # don't persist a half-learned floor
        try:
            os.remove(MIC_FLOOR_STATE_PATH)
        except Exception:
            pass
        print(f"  🎚️  Mic floor reset ({why}) — relearning over "
              f"{MIC_FLOOR_MIN_S:.1f}s")

    @property
    def ready(self) -> bool:
        return len(self._ring) >= self._min_n or self._loaded

    def seed_floor(self, floor: float, why: str = "") -> None:
        """Install a MEASURED floor, from startup calibration.

        Different from both _load() and observe(): those resume or infer a
        floor from history, and history is exactly what is untrustworthy after
        the hardware changes. This value was measured seconds ago, in silence,
        on this body, through the real capture path — so it replaces the ring
        rather than joining it. Leaving the old observations in place would let
        a stale floor drag the fresh one for a whole MIC_FLOOR_WINDOW_S; the
        slow-rise/fast-fall smoothing in observe() is there to resist sudden
        jumps from door slams, and a deliberate recalibration is precisely the
        sudden jump it must NOT resist.

        _loaded is set so the floor is trusted immediately, before _min_n
        chunks have arrived: ADAM's first words after boot are heard through
        this number, and it is a better number than the running minimum
        observe() would otherwise fall back to.
        """
        if floor <= 0.0:
            return
        was = self._floor
        self._ring.clear()
        self._floor  = float(floor)
        self._loaded = True
        self._since_calc = 0
        delta = (20.0 * math.log10(max(floor, 1.0) / max(was, 1.0))
                 if was > 0.0 else 0.0)
        moved = f" ({delta:+.1f} dB vs the resumed {was:.0f})" if was > 0 else ""
        print(f"  🎚️  Mic floor measured {floor:.0f}{moved} — "
              f"{why} (open≥{self.open_th:.0f}, strong≥{self.strong_th:.0f})")
        self._saved_t = 0.0          # bypass the save rate-limit: persist now
        self._maybe_save()

    @property
    def open_th(self) -> float:
        return max(MIC_OPEN_MIN, self._floor * MIC_OPEN_RATIO)

    @property
    def strong_th(self) -> float:
        """Loud enough to open on level alone, without the shape vote — a
        shout must always work. Measured: at this room's floor of 1512 that
        puts the bar at 4838, above the loudest single noise chunk seen
        (3824), so it is not a back door for noise."""
        return max(MIC_OPEN_MIN, self._floor * MIC_OPEN_STRONG)

    @property
    def hold_th(self) -> float:
        return min(self.open_th * MIC_HOLD_MAX_RATIO,
                   max(MIC_OPEN_MIN * 0.75, self._floor * MIC_HOLD_RATIO))

    @property
    def cand_th(self) -> float:
        """Below open_th but above the floor: loud enough to be WORTH asking
        the spectrum about, too quiet for level alone to convict. Deliberately
        derived from the floor directly rather than as a fraction of open_th,
        so the two rails stay independent — lowering open_th to recover quiet
        speech would also have moved strong_th and hold_th, which is why that
        was not the fix (see MIC_CAND_RATIO in config.py)."""
        return max(MIC_OPEN_MIN, self._floor * MIC_CAND_RATIO)

    # ── speech-shape vote ───────────────────────────────────────────
    @property
    def flat_max(self) -> float:
        """The flatness threshold actually in force this chunk — learned from
        the room's own noise bed, floored at MIC_SHAPE_FLAT_MAX so it can only
        ever be looser than the measured default (see __init__)."""
        return self._flat_max

    def shape_ok(self, mono16k: bytes, learn_noise: bool = False) -> bool:
        """Score this chunk's SPECTRUM for speech and return the strict
        (opening) verdict. Also pushes the relaxed (holding) verdict into
        the sustain window, so callers make exactly one call per chunk.

        learn_noise: True when the CALLER knows this chunk is not speech —
        gate shut, level below the open threshold, amplifier off. Only those
        chunks teach the learned flatness threshold. The caller has to say so
        because only the caller knows the gate state; this class deliberately
        never consults level, and inferring "quiet" from the spectrum alone
        would be circular.

        Level is not consulted anywhere in here — that is the point.
        """
        try:
            x = np.frombuffer(mono16k, np.int16).astype(np.float32)
            if x.size < 64:
                self._shp_win.append(0.0)
                return False
            x   = x - x.mean()
            buf = np.zeros(_SHP_NFFT, np.float32)
            m   = min(x.size, _SHP_NFFT)
            buf[:m] = x[:m] * _SHP_WIN[:m]
            P  = np.abs(np.fft.rfft(buf)) ** 2 + 1e-9
            sp = P[_SHP_B_SP]
            # Flatness as geometric/arithmetic mean. exp(mean(log)) is the
            # geometric mean computed without overflowing on a long product.
            self.flat = float(math.exp(float(np.log(sp).mean())) / sp.mean())
            self.lohi = float(P[_SHP_B_LO].sum() / P[_SHP_B_HI].sum())
        except Exception as e:
            # A broken feature must not deafen ADAM: fail open (vote yes) so
            # the gate degrades to level-only, and say so once.
            if self.backend != "level":
                print(f"  ⚠️  mic shape feature failed ({e}) — level-only")
                self.backend = "level"
            self._shp_win.append(1.0)
            return True
        if self.backend == "level":
            self._shp_win.append(1.0)
            return True
        if learn_noise and MIC_SHAPE_ADAPT:
            self._flat_ring.append(self.flat)
            self._flat_calc += 1
            if (self._flat_calc >= self._recalc_every
                    and len(self._flat_ring) >= self._min_n):
                self._flat_calc = 0
                n  = len(self._flat_ring)
                p  = float(np.percentile(
                    np.fromiter(self._flat_ring, np.float64, n),
                    MIC_SHAPE_FLAT_PCTL))
                self._flat_max = min(MIC_SHAPE_FLAT_CEIL,
                                     max(MIC_SHAPE_FLAT_MAX,
                                         p * MIC_SHAPE_FLAT_MARGIN))
        strict = (self.flat <= self._flat_max
                  and self.lohi >= MIC_SHAPE_RATIO_MIN)
        # Holding gets slack on flatness only: an unvoiced consonant is
        # flatter than a vowel but still is not a fan.
        self._shp_win.append(
            1.0 if self.flat <= self._flat_max + MIC_SHAPE_FLAT_SLACK
            else 0.0)
        if strict and self._vad is not None:
            strict = self._webrtc_vote(mono16k)
        return strict

    @property
    def shape_frac(self) -> float:
        """Fraction of the sustain window whose shape passed. This, not a
        'heard speech recently' timer, is what lets the gate close."""
        if not self._shp_win:
            return 0.0
        return float(sum(self._shp_win) / len(self._shp_win))

    def shape_hold_ok(self) -> bool:
        return self.shape_frac >= MIC_SHAPE_HOLD_FRAC

    def _webrtc_vote(self, mono16k: bytes) -> bool:
        """Optional extra AND term. Off by default: measured 100% false
        positive on this HAT's room noise (see the note above)."""
        buf  = self._vad_carry + mono16k
        step = self._vad_frame * 2                        # bytes per frame
        hit  = False
        i    = 0
        while i + step <= len(buf):
            try:
                if self._vad.is_speech(buf[i:i + step], GEMINI_SEND_RATE):
                    hit = True
            except Exception:
                self._vad = None                          # never retry-storm
                self.backend = "shape"
                return True
            i += step
        self._vad_carry = buf[i:]
        return hit

    def is_speech(self, mono16k: bytes, rms: float) -> bool:
        """Dynamic speech presence decision combining learned room floor and
        spectral shape. Adapts to any room noise floor and any vocal
        pitch/frequency (male/female/child).

        Three tiers, in decreasing order of what the LEVEL alone establishes:

          >= strong_th   level alone convicts. A shout must always work, even
                         if the shape vote is confused by distortion.
          >= open_th     level has cleared the room's noise p95, so one strict
                         shape pass is enough.
          >= cand_th     level establishes nothing, so the spectrum decides
                         outright — a strict pass AND sustained shape evidence.

        That third tier is the 2026-10-01 fix. Level used to be a hard VETO
        here: below open_th this returned False without ever looking at the
        spectrum, which discarded the only measurement capable of telling a
        quiet syllable from a loud click. The shape machinery was present,
        learning, and explicitly level-independent, and it was never asked.
        Field log: speech arrived at 1169 against open_th 756 — a margin of
        +3.8 dB, so anything less than a projected vowel fell under the rail
        and was dropped unexamined. That was ADAM's "missing a few words".

        What keeps the new tier honest is NOT its level but the pair of shape
        tests, each of which this room's noise bed fails outright: the strict
        vote passes 0.0% of 900 measured noise chunks, and shape_frac must
        clear 0.60 where noise reaches at most 0.53. shape_frac is read AFTER
        shape_ok has pushed this chunk's verdict, so the fraction includes the
        chunk being judged.
        """
        if rms >= self.strong_th:
            return True
        if rms >= self.open_th:
            return self.shape_ok(mono16k)
        if rms >= self.cand_th:
            # Exactly one shape_ok per chunk is an invariant of this pair of
            # methods: observe_background() now learns only below cand_th, so
            # in this band it made no call and the features would otherwise be
            # a stale chunk's. The call moved here rather than being added.
            if not self.shape_ok(mono16k):
                return False
            return self.shape_frac >= MIC_CAND_SHAPE_FRAC
        return False

    def observe_background(self, rms: float, mono16k: bytes | None = None) -> float:
        """Update room noise floor during listening mode when speaker and
        servos are idle.

        The shape learner is fed only below cand_th, not below open_th. The
        band between them is now where quiet speech is CONVICTED (see
        is_speech), and teaching the noise-flatness reference from chunks that
        may be speech would raise _flat_max toward speech's own flatness and
        quietly undo the tier that depends on it. Same self-sustaining trap
        documented in development_log.md Part 26 §2, one tier down.
        """
        floor = self.observe(rms)
        if mono16k is not None and rms < self.cand_th:
            self.shape_ok(mono16k, learn_noise=True)
        return floor

    # ── persistence ─────────────────────────────────────────────────
    def _load(self) -> None:
        try:
            with open(MIC_FLOOR_STATE_PATH, "r") as f:
                st = json.load(f)
            if (time.time() - float(st.get("t", 0))) > MIC_FLOOR_STATE_MAX_AGE_S:
                return
            # UNITS CHECK. The floor is an absolute level, so it is only
            # meaningful in the units it was measured in — and the units are
            # not a constant of the software: they changed the day the gate's
            # yardstick moved from a full-band RMS to a 300-3400 Hz one, which
            # shifted every level on this hardware by 3.9 dB. A floor resumed
            # across that change is silently wrong in the deaf direction, and
            # nothing in the log looks wrong: the gate simply never opens.
            #
            # So the band is stamped into the file and a mismatch is discarded
            # rather than trusted. One warmup is the entire cost; the
            # alternative is a unit that boots unable to hear and gives no
            # reason. A file with no stamp predates the stamp, which means it
            # is a full-band floor, which means it does not match.
            if st.get("band") != _FLOOR_UNITS:
                print(f"  🎚️  Discarding the saved mic floor — it was measured "
                      f"in different units ({st.get('band', 'full-band')} ≠ "
                      f"{_FLOOR_UNITS}); relearning over {MIC_FLOOR_MIN_S:.1f}s")
                return
            f0 = float(st.get("floor", 0.0))
            if f0 > 0.0:
                self._floor  = f0
                self._loaded = True
                print(f"  🎚️  Resuming learned mic floor {f0:.0f} from the "
                      f"last run (open≥{self.open_th:.0f})")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"  ⚠️  could not read {MIC_FLOOR_STATE_PATH}: {e}")

    def _maybe_save(self) -> None:
        now = time.time()
        if now - self._saved_t < MIC_FLOOR_SAVE_EVERY_S or self._floor <= 0:
            return
        self._saved_t = now
        try:
            tmp = MIC_FLOOR_STATE_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"t": now, "floor": round(self._floor, 2),
                           "band": _FLOOR_UNITS}, f)
            os.replace(tmp, MIC_FLOOR_STATE_PATH)
        except Exception:
            pass            # a lost floor costs one warmup, never a crash


# ═════════════════════════════════════════════════════════════════════════════
# DYNAMIC SPEAKER ENVELOPE TRACKER (Self-tuning Barge-In)
# ═════════════════════════════════════════════════════════════════════════════

class SpeakerEnvelopeTracker:
    """Tracks the real-time acoustic/mechanical speaker vibration envelope to provide
    a dynamic, self-tuning barge-in threshold.

    Adapts automatically to:
      1. Any speaker volume level (SPEAKER_GAIN).
      2. Any room ambient noise floor (from AdaptiveGate).
      3. Word pauses & syllable boundaries (threshold drops during gaps, enabling easy barge-in).
      4. Loud vowel bursts (threshold rises above measured vibration, preventing self-interruption).
    """

    def __init__(self, sample_rate: int = 16000, coupling_factor: float = 0.70) -> None:
        self.sample_rate = sample_rate
        self.coupling_factor = float(coupling_factor)   # measured ~0.65 on 3D body shell
        self.envelope = 0.0
        self.last_feed_t = 0.0
        self.debounce_hits = 0
        self._lock = threading.Lock()

    def feed_speaker(self, data: bytes | float, gain: float = 1.0) -> None:
        """Feed speaker playback audio to update the instantaneous envelope."""
        if not data:
            return
        if isinstance(data, (int, float)):
            rms = float(data) * gain
        elif isinstance(data, (bytes, bytearray)):
            rms = rms_pcm16(data) * gain
        else:
            return

        now = time.time()
        with self._lock:
            dt = max(0.0, now - self.last_feed_t)
            self.last_feed_t = now
            # Decay envelope with ~100ms time constant (matching room reverberation & DAC lag)
            decay = math.exp(-dt / 0.10) if dt > 0 else 1.0
            decayed = self.envelope * decay
            self.envelope = max(rms, decayed)

    def get_barge_threshold(self, room_floor: float) -> float:
        """Calculate the dynamic barge-in threshold for the current millisecond."""
        now = time.time()
        with self._lock:
            dt = max(0.0, now - self.last_feed_t)
            decay = math.exp(-dt / 0.10) if dt > 0 else 1.0
            cur_env = self.envelope * decay

            # Floor based on room ambient: during word pauses, threshold drops to ~1.8x room floor
            quiet_th = max(1100.0, room_floor * 1.8)

            # Vibration ceiling: proportional to current speaker vibration output
            vibration_est = cur_env * self.coupling_factor
            vibration_th = vibration_est * 1.25   # 25% safety margin above vibration

            return max(quiet_th, vibration_th)

    def check_barge_in(self, mic_rms: float, room_floor: float) -> tuple[bool, float]:
        """Check if incoming mic RMS constitutes a genuine barge-in.
        Requires 2 consecutive frames (40ms) exceeding threshold to reject clicks.
        Returns (is_barge, current_threshold)."""
        thresh = self.get_barge_threshold(room_floor)
        with self._lock:
            if mic_rms > thresh:
                self.debounce_hits += 1
            else:
                self.debounce_hits = 0

            if self.debounce_hits >= 2:
                self.debounce_hits = 0
                return True, thresh
            return False, thresh

    def reset(self) -> None:
        """Reset tracker state on turn end or interruption."""
        with self._lock:
            self.envelope = 0.0
            self.debounce_hits = 0
            self.last_feed_t = 0.0


class SongBargeIn:
    """Interrupt a song with a voice. Self-calibrating: no room constant.

    WHY THIS IS REFERENCE-BASED (a level-only detector was built first and
    measured to be unfixable)
    ------------------------------------------------------------------------
    The obvious detector -- learn the music's mic level, fire when the mic
    rises above it and is speech-shaped -- cannot work here, and this was
    established empirically, not assumed:

      * A SINGING voice and a SPEAKING voice are the same instrument. Both are
        formant-peaked and low-frequency dominant, so both pass any spectral
        shape test. The shape test cannot reject a vocal track.
      * Therefore a verse -> loud-vocal-chorus transition and a genuine 400 ms
        human barge-in produce the IDENTICAL signature: a fast, speech-shaped
        level rise above a floor that has not caught up yet. In the bench both
        fired on the same chunk indices. No level ratio separates them, because
        there is nothing to separate -- reference-free, they are the same event.

    The one thing that does distinguish them: the chorus is present in the
    SONG'S OWN PCM, and the human's voice is not. So the detector compares the
    mic against the song's own level and watches the RATIO between them:

        ratio = mic_rms / expected_music_level

    A chorus raises mic and expected together, so the ratio is flat -> no
    trigger. A human voice adds energy the reference does not contain, so the
    ratio jumps -> trigger. This is what makes it immune to song dynamics.

    ALIGNING THE REFERENCE (measured, not guessed)
    ----------------------------------------------
    song_playback.py writes into aplay's stdin, and aplay buffers, so a chunk
    written at wall-time T is HEARD at T + B. B is NOT the ~5.9 ms acoustic
    flight time. song_lag_bench.py measured it on this unit:

        burst 0   541 ms      (ALSA buffer cold)
        burst 1   793 ms      (filling -- SONG_PACE_FRAC=0.9 overfeeds by 11%)
        burst 2+  929-956 ms  (saturated: 941 ms +/- 14 ms over 25 s)

    So B ramps for the first ~5 s of a song and then locks hard. _expected()
    therefore takes the MAX reference level over the window
    [T - LAG_MAX, T - LAG_MIN], which spans both the cold and saturated cases.
    MAX, not mean, is deliberate: it makes the expected level err HIGH, and a
    too-high expectation only makes the detector briefly deaf, whereas a
    too-low expectation stops the song by itself. Deafness is recoverable --
    the user can still say "adam stop", which Vosk hears offline. A spurious
    stop is not recoverable; the song just dies.

    HOW IT STAYS SELF-CALIBRATING (the production-ready requirement)
    ---------------------------------------------------------------
      * `coupling` -- how loud the mic reads the song per unit of the song's
        own digital level -- is LEARNED, never configured. It absorbs the
        speaker volume, the room, the mic gain, the shell coupling and the
        capture chain's own filtering in one number, so nothing here needs
        tuning per unit or per room.
      * It is a HIGH percentile (p90) of the observed music-only ratios, so
        the bar sits above music's normal variation rather than at its middle.
      * Only chunks that FAIL the shape test teach `coupling`, so a talking
        human can never raise the bar above their own voice. Excluding vocal
        chunks costs nothing now: during a chorus the ratio is flat anyway,
        because the reference rose too.
      * A barge-in needs the ratio to exceed coupling * SONG_BARGE_RATIO for
        SONG_BARGE_HOLDS consecutive chunks (~100 ms), which rejects transients.

    The caller must feed BOTH sides: feed_reference() from song_playback.py at
    write time, and feed() with mic chunks during the song. reset() at song
    start clears both.
    """

    def __init__(self, sample_rate: int = GEMINI_SEND_RATE) -> None:
        self.sample_rate = sample_rate
        self.coupling = 0.0                  # learned mic-per-song-level ratio
        self.debounce_hits = 0
        n = max(2, int(round(SONG_BARGE_WINDOW_S * sample_rate / 480.0)))
        self._coup = collections.deque(maxlen=n)   # music-only observed ratios
        self._ref = collections.deque()            # (write_wall_time, rms)
        self._lock = threading.Lock()
        # Stale-audio guard. When the capture path stalls (an arecord overrun,
        # which on the VoiceHAT surfaces as "overrun!!! (at least N ms long)"),
        # the next chunks arecmthe delivers were captured N ms in the past. A
        # chunk 2.8 s old has NO reference in [T-LAG_MAX, T-LAG_MIN], so
        # `_expected(T)` compares it against the wrong song passage and the
        # ratio spikes, which false-triggers a barge-in and kills the song.
        # Track the wall-time gap between mic feeds; if the capture gap ever
        # exceeds STALL_S we enter a recovery window during which barge-in is
        # refused, so the stale burst is dropped on the floor instead of being
        # judged as if it were live.
        self._last_feed_t = 0.0
        self._stalled_until = 0.0
        self._stall_s = float(SONG_BARGE_STALL_S)

    def stale(self, now: float) -> bool:
        """True while we are inside the post-overrun recovery window.

        Caller should treat the mic chunk as untrustworthy and NOT ask the
        detector to judge it. Returns False once the window passes so a real
        voice after recovery barges in normally.
        """
        return now < self._stalled_until

    # ── reference side (fed by song_playback.py at write time) ──────────
    def feed_reference(self, pcm: bytes) -> None:
        """Record the digital level of a song chunk at the moment it is written
        into aplay's stdin. Any PCM16 layout works -- the level is only ever
        used as a ratio, so channel count and rate cancel out in `coupling`."""
        if not pcm:
            return
        rms = rms_pcm16(pcm)
        now = time.time()
        with self._lock:
            self._ref.append((now, rms))
            # Keep only what _expected() could still ask for.
            cutoff = now - (SONG_BARGE_LAG_MAX_MS / 1000.0) - 0.5
            while self._ref and self._ref[0][0] < cutoff:
                self._ref.popleft()

    def _expected(self, t: float) -> float:
        """MAX reference level over [t-LAG_MAX, t-LAG_MIN]. Caller holds lock."""
        lo = t - SONG_BARGE_LAG_MAX_MS / 1000.0
        hi = t - SONG_BARGE_LAG_MIN_MS / 1000.0
        best = 0.0
        for ts, r in self._ref:
            if lo <= ts <= hi and r > best:
                best = r
        return best

    def _shape_is_speech(self, mono16k: bytes) -> bool:
        """Independent copy of the gate's spectral shape test, so it advances
        NO state. Fail-open (True) so a broken feature can never make a song
        uninterruptible -- the detector degrades to ratio-only.

        Reads the gate's ADAPTED flat_max (a read-only property, no state
        change) rather than the static MIC_SHAPE_FLAT_MAX, because that config
        value is only the FLOOR of the adaptive range -- see the block above
        its definition in config.py. Using the static floor here would make
        this test stricter than the gate's in a noisy room and silently cost
        the user barge-ins.
        """
        try:
            x = np.frombuffer(mono16k, np.int16).astype(np.float32)
            if x.size < 64:
                return False
            x = x - x.mean()
            buf = np.zeros(_SHP_NFFT, np.float32)
            m = min(x.size, _SHP_NFFT)
            buf[:m] = x[:m] * _SHP_WIN[:m]
            P = np.abs(np.fft.rfft(buf)) ** 2 + 1e-9
            sp = P[_SHP_B_SP]
            flat = float(math.exp(float(np.log(sp).mean())) / sp.mean())
            lohi = float(P[_SHP_B_LO].sum() / P[_SHP_B_HI].sum())
            try:
                flat_max = float(_adaptive_gate.flat_max)
            except Exception:
                flat_max = MIC_SHAPE_FLAT_MAX
            return (flat <= flat_max
                    and lohi >= MIC_SHAPE_RATIO_MIN)
        except Exception:
            return True

    def _recalc_coupling(self) -> float:
        """coupling = p95 of the observed music-only ratios.

        A HIGH percentile, not a median, because of the structural bound: since
        _expected() is a MAX over a window that contains the instant the mic is
        hearing, music alone can only ever produce ratio <= its true coupling.
        The bound sits at the TOP of the distribution, so that is what has to be
        estimated. Measured on synthetic instrumental, vocal, and 2.5x-chorus
        material, the music-only ratio was p50 1.23-1.30 / p95 1.50 / max 1.50
        in every case -- tight, and invariant to song dynamics.
        """
        if len(self._coup) >= SONG_BARGE_MIN_N:
            arr = np.fromiter(self._coup, np.float64, len(self._coup))
            self.coupling = float(np.percentile(arr, 95))
        return self.coupling

    def bar(self) -> float:
        """The ratio a chunk must exceed to count as a barge-in."""
        return self.coupling * SONG_BARGE_RATIO

    # ── mic side ────────────────────────────────────────────────────────
    def feed(self, mono16k: bytes) -> bool:
        """Feed one 16 kHz mono mic chunk. True iff this is a genuine barge-in.

        Only call this while a song is actually playing, and call reset() at
        song start so the previous track cannot set this song's bar.
        """
        if not mono16k:
            return False
        rms = rms_pcm16(mono16k)
        now = time.time()
        with self._lock:
            # Stale-audio guard: if the capture path just overran, these chunks
            # were captured seconds ago and have no matching reference. Drop
            # them and enter a recovery window so the burst degrades to silence
            # instead of a false bar.
            if self._last_feed_t > 0.0 and (now - self._last_feed_t) > self._stall_s:
                self._stalled_until = now + self._stall_s * 2.0
                self.debounce_hits = 0
                self._last_feed_t = now
                return False
            self._last_feed_t = now
            if self.stale(now):
                self.debounce_hits = 0
                return False
            exp = self._expected(now)
            if exp <= 0.0:
                # No reference covering this instant -- the song has only just
                # started, or feed_reference() is not wired. Judging a
                # barge-in without a reference is exactly the mistake this
                # class exists to avoid, so decline to guess.
                self.debounce_hits = 0
                return False

            ratio = rms / exp

            # ── learn the coupling ──
            # Every chunk teaches it, EXCEPT ones already over the bar. The
            # earlier design gated learning on the spectral shape test instead,
            # and that deadlocked: on an all-vocal song every chunk is
            # speech-shaped, so nothing ever taught coupling and the detector
            # stayed permanently disarmed (measured: coupling 0.00, 0 fires
            # even for a real voice). Excluding only over-the-bar chunks is
            # what actually prevents a sustained voice from lifting the bar
            # above itself, and it works on any material.
            if self.coupling <= 0.0 or ratio <= self.bar():
                self._coup.append(ratio)
            self._recalc_coupling()

            if len(self._coup) < SONG_BARGE_MIN_N:
                self.debounce_hits = 0      # not armed yet
                return False

            # The shape test is a CONFIRMATION here, never a learning gate: it
            # rejects a door slam or a dropped object, which raise the ratio
            # just as a voice does but are broadband. It cannot distinguish
            # singing from speaking -- that job belongs to the reference.
            if ratio > self.bar() and self._shape_is_speech(mono16k):
                self.debounce_hits += 1
            else:
                self.debounce_hits = 0
            return self.debounce_hits >= SONG_BARGE_HOLDS

    def reset(self) -> None:
        """Forget the learned coupling and the reference history.

        Called at song start: each track has its own level and the ALSA buffer
        is cold, so carrying state over would set the wrong bar. No barge-in
        can fire for the first few chunks after this, which is correct -- there
        is nothing to be relative to yet.
        """
        with self._lock:
            self._coup.clear()
            self._ref.clear()
            self.coupling = 0.0
            self.debounce_hits = 0
            self._last_feed_t = 0.0
            self._stalled_until = 0.0


# Module-level instances for zero-configuration dynamic operation
_adaptive_gate = AdaptiveGate(chunks_per_s=30.0)
_speaker_tracker = SpeakerEnvelopeTracker(coupling_factor=0.70)
_song_barge = SongBargeIn()


def beep_s16_stereo(freq=880.0, dur=0.2) -> bytes:
    n    = int(PLAYBACK_RATE * dur)
    t    = np.arange(n, dtype=np.float32) / PLAYBACK_RATE
    mono = np.clip(np.sin(2 * np.pi * freq * t) * 0.3 * 32767, -32768, 32767).astype(np.int16)
    return np.repeat(mono[:, None], 2, axis=1).reshape(-1).tobytes()

def read_exact(pipe, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = pipe.read(n - len(buf))
        if not chunk:
            raise EOFError("pipe closed")
        buf.extend(chunk)
    return bytes(buf)


def write_all(pipe, data: bytes, frame_bytes: int = 4) -> int:
    """Write every byte of `data` into `pipe`, looping over partial writes.

    THIS IS NOT PEDANTRY — a dropped byte is audible, and the failure mode is
    spectacular. aplay is spawned with bufsize=0, which makes proc.stdin a raw
    _io.FileIO. A raw write() on a pipe is allowed to be SHORT: it returns the
    number of bytes the kernel accepted, and when the 64 KiB pipe buffer is
    full (a loaded Pi Zero 2 W, a song and a reply competing for one aplay)
    that is less than len(data). `pipe.write(data)` on its own therefore
    silently loses the remainder.

    Playback is s16 stereo, so a frame is 4 bytes: [L_lo, L_hi, R_lo, R_hi].
    Lose a number of bytes that is not a multiple of 4 and every following
    sample is reassembled from the wrong pair of bytes — the low and high
    halves of each int16 swap. A quiet passage at amplitude 100 comes back as
    100 * 256 = 25,600, i.e. +48 dB, and the rest of the stream is full-scale
    buzz until the stream is restarted. That is exactly the "volume suddenly
    jumps and turns into distortion" symptom, and no amount of gain tuning
    can fix it because the samples are structurally wrong.

    Returns the number of bytes written (== len(data) unless it raised).
    Raises whatever the underlying write raises, having written a whole
    number of frames where possible so a retry stays aligned.
    """
    if not data:
        return 0
    if frame_bytes > 1 and (len(data) % frame_bytes):
        # Never hand ALSA a partial frame. The tail is dropped rather than
        # written, because writing it would shift every subsequent frame.
        data = data[:len(data) - (len(data) % frame_bytes)]
        if not data:
            return 0
    mv    = memoryview(data)
    total = 0
    while mv:
        n = pipe.write(mv)
        # A buffered stream returns None on success (it took everything); a
        # raw FileIO returns the count, and may return 0 on a full pipe.
        if n is None:
            total += len(mv)
            break
        if n <= 0:
            # Nothing accepted and no exception: the pipe is full. Give the
            # reader a moment rather than spinning on the CPU it needs to
            # drain us.
            time.sleep(0.002)
            continue
        total += n
        mv = mv[n:]
    return total


# One aplay process serves the WHOLE session, and five different code paths
# write into its stdin: the startup beep, end_of_turn()'s tail flush, the DAPM
# keep-alive, Gemini's TTS chunks, and _play_song_task(). Several of those run
# concurrently in separate asyncio.to_thread workers, so without this lock two
# writers can interleave INSIDE write_all()'s partial-write loop: writer A gets
# a short write, writer B's bytes land in the gap, and every frame after that
# point is reassembled from the wrong byte pair — the +48 dB full-scale buzz
# described in write_all()'s docstring, permanent for the rest of the session.
#
# The lock covers write AND flush together, because a flush between another
# writer's partial write and its continuation is the same hazard.
_pcm_write_lock = threading.Lock()


def write_pcm(pipe, data: bytes, frame_bytes: int = 4) -> int:
    """Atomically write + flush one PCM chunk into a pipe shared by many writers.

    Always use this instead of a bare write_all()/flush pair on the shared
    aplay stdin. write_all() alone is frame-safe only for a SINGLE writer; it
    is this lock that makes it frame-safe when the speaker task and the song
    task are both feeding the same process.
    """
    with _pcm_write_lock:
        n = write_all(pipe, data, frame_bytes)
        try:
            pipe.flush()
        except Exception:
            pass
        return n


def drain_stderr(proc: subprocess.Popen, label: str,
                 benign_underrun=None) -> None:
    """Forward a subprocess's stderr to the log, summarising ALSA underruns.

    Underruns used to be dropped outright (`if "underrun" not in txt.lower()`).
    That hid the single most useful clue about broken playback: an underrun means
    aplay ran out of audio and the sound card played whatever was left in the
    buffer, which is heard as a click, a gap, or a burst of crackle. Silently
    discarding them meant "the speaker sounds broken" had no corresponding
    evidence anywhere in the log, and left DSP bugs and CPU starvation
    indistinguishable. They are still not printed one-per-line — on a loaded Pi
    that would flood the journal — but they are counted and reported.

    benign_underrun: optional predicate, called when an underrun line arrives.
    If it returns True the underrun is EXPECTED and is not reported with the
    alarming message. ADAM holds the playback device open for
    SPEAKER_IDLE_CLOSE_S after a reply finishes, and a running ALSA stream with
    no data is an XRUN by definition — so exactly one underrun per turn is
    structural, happens after every sample has already been heard, and means
    nothing. Attributing it to "CPU starvation or too-small buffer" sent a real
    debugging session chasing a non-problem. Underruns that arrive WHILE audio
    is flowing are the ones that are audible, and those still get the full
    warning.
    """
    n_under = 0
    last_report = 0.0
    try:
        for line in proc.stderr:
            txt = line.decode(errors="replace").strip()
            if not txt:
                continue
            if "underrun" in txt.lower():
                now = time.time()
                if benign_underrun is not None:
                    try:
                        benign = benign_underrun()
                    except Exception:
                        benign = False
                else:
                    benign = False
                if benign:
                    # Expected ALSA XRUN while idle between turns — inaudible and normal.
                    continue
                n_under += 1
                if now - last_report > 5.0:
                    print(f"  ⚠️  [{label}] {n_under} buffer underrun(s) — audio "
                          f"dropouts/crackle. CPU starvation or too-small buffer.")
                    last_report = now
                    n_under = 0
                continue
            print(f"  [{label}] {txt}")
    except Exception:
        pass
