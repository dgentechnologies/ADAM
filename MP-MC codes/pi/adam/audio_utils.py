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
    MIC_NR, MIC_NR_FRAME, MIC_NR_OVERSUB, MIC_NR_FLOOR_DB, MIC_NR_NOISE_S,
    MIC_NR_SMOOTH,
    MIC_CHANNEL,
    MIC_CH_WATCH_S, MIC_CH_WATCH_MIN_CLIPS,
    MIC_CH_CLIP_FRAC,
    MIC_FLOOR_WINDOW_S,
    MIC_FLOOR_PERCENTILE,
    MIC_FLOOR_MIN_S,
    MIC_FLOOR_RISE,
    MIC_FLOOR_FALL,
    MIC_FLOOR_STATE_PATH,
    MIC_FLOOR_STATE_MAX_AGE_S,
    MIC_FLOOR_SAVE_EVERY_S,
    MIC_OPEN_RATIO,
    MIC_OPEN_STRONG,
    MIC_OPEN_MIN,
    MIC_HOLD_RATIO,
    MIC_HOLD_MAX_RATIO,
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
)

# ═════════════════════════════════════════════════════════════════════════════
# MIC BAND-LIMITING CHAIN  (48kHz stereo S32  ->  16kHz mono S16 for Gemini)
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

# High-pass built as (signal - moving average). A boxcar average of length M is
# a low-pass with its corner near 0.443*fs/M, so subtracting it high-passes at
# the same corner. Done with cumsum it is O(n) and fully vectorised — no scipy
# (not installed on the Pi) and no per-sample Python loop.
_MA_LEN = max(3, int(round(0.443 * GEMINI_SEND_RATE / MIC_HP_HZ)) | 1)   # odd


class _MicChain:
    """Stateful 48k stereo -> 16k mono band-limited converter."""

    def __init__(self) -> None:
        self._lp_tail   = np.zeros(_LP_TAPS - 1, dtype=np.float32)
        self._ma_tail   = np.zeros(_MA_LEN - 1, dtype=np.float32)
        self._dec_phase = 0

    def process(self, mono48: np.ndarray) -> np.ndarray:
        # ── anti-alias low-pass + decimation, fused (polyphase) ─────────────
        # Only every DECIM'th low-pass output survives the decimation, so
        # convolving the full 48kHz stream and then throwing 2/3 of it away
        # does 3x the necessary multiplies. Computing just the kept phases via
        # a strided sliding window costs 533x63 MACs per chunk instead of
        # 1600x63, and hands the inner loop to BLAS. That headroom matters:
        # this runs 30x/second on a Pi Zero 2W that is simultaneously feeding
        # aplay, and starving the playback task is audible as crackle.
        buf = np.concatenate((self._lp_tail, mono48))
        self._lp_tail = buf[-(_LP_TAPS - 1):].copy()
        win = np.lib.stride_tricks.sliding_window_view(buf, _LP_TAPS)
        # win[i] == buf[i:i+_LP_TAPS], so win[i] @ _LP_FIR_R reproduces
        # np.convolve(buf, _LP_FIR, "valid")[i] exactly.
        lp = win[self._dec_phase::DECIM] @ _LP_FIR_R

        # 1600 frames/chunk is not a multiple of 3, so restarting at index 0
        # every chunk would repeat/skip a sample each time (~0.1% rate error
        # plus a boundary glitch). Track where the next chunk should start.
        self._dec_phase = (self._dec_phase - int(mono48.size)) % DECIM

        # ── de-rumble high-pass, at 16k (3x fewer samples than at 48k) ─────
        buf = np.concatenate((self._ma_tail, lp))
        self._ma_tail = buf[-(_MA_LEN - 1):].copy()
        c   = np.cumsum(np.concatenate(([0.0], buf.astype(np.float64))))
        ma  = (c[_MA_LEN:] - c[:-_MA_LEN]) / _MA_LEN        # len == lp.size
        half = (_MA_LEN - 1) // 2                           # symmetric => no
        mid  = buf[half:half + lp.size]                     # phase distortion
        return mid - ma.astype(np.float32)


_mic_chain = _MicChain()


# ── NOISE SUPPRESSION FOR THE RECOGNISER-BOUND STREAM ───────────────────────
# Why this exists, and why it is not in the gate's path:
#
# Measured live on this unit, post-filter int16 RMS: noise floor 1550-1591,
# speech p90 2041-4256, speech peaks 2783-6487. That is roughly +6 dB SNR. The
# adaptive gate copes fine with that — it is a ratio detector and it opened on
# every utterance — but a speech RECOGNISER does not. Words come back wrong
# because the acoustic evidence really is buried, not because anything upstream
# is broken.
#
# So the recogniser gets a cleaned copy and the gate keeps the original. That
# split is deliberate: the gate's floor percentile, learned flatness ceiling and
# lo/hi ratio were all tuned against the raw signal's statistics, and quietly
# changing what it sees would invalidate that tuning and risk trading a
# recognition problem for a "cannot hear me at all" problem.
#
# Method: WOLA spectral magnitude subtraction with a per-bin noise floor from
# minimum statistics. No VAD input at all — the noise estimate is the running
# minimum of the smoothed power in each frequency bin over MIC_NR_NOISE_S, and
# a minimum over 1.5 s cannot be contaminated by speech because no phoneme
# sustains one bin's energy that long. That independence matters here: the gate
# and the suppressor cannot cascade each other's mistakes.
#
# sqrt-Hann analysis AND synthesis windows with hop = frame/2 satisfy COLA
# (their squares sum to a periodic Hann, which sums to exactly 1.0), so with all
# gains at 1.0 this reconstructs the input bit-for-bit apart from a hop of
# delay. That property is what makes MIC_NR=0 a genuine A/B test rather than a
# different signal path.
class _NoiseSuppressor:
    """Streaming single-channel denoiser for int16 mono at GEMINI_SEND_RATE."""

    def __init__(self, frame: int, oversub: float, floor_db: float,
                 noise_s: float, rate: int, smooth: float = MIC_NR_SMOOTH) -> None:
        self._n   = max(64, int(frame) & ~1)          # even
        self._h   = self._n // 2                      # COLA hop for sqrt-Hann
        # Periodic (not symmetric) Hann is the one that satisfies COLA at N/2.
        hann      = np.hanning(self._n + 1)[:self._n]
        self._w   = np.sqrt(hann).astype(np.float32)
        self._in   = np.zeros(0, dtype=np.float32)
        self._acc  = np.zeros(self._n, dtype=np.float32)
        nb         = self._n // 2 + 1
        self._pwr  = np.zeros(nb, dtype=np.float32)   # smoothed power
        self._gain = np.ones(nb, dtype=np.float32)    # previous frame's gain
        self._floor   = float(10.0 ** (floor_db / 20.0))
        self._oversub = float(oversub)
        self._alpha   = min(0.99, max(0.0, float(smooth)))
        # Minimum statistics: a deque of per-sub-window minima plus the minimum
        # so far in the current sub-window. min(deque + current) is the estimate.
        # Four sub-windows is the usual compromise — enough that the estimate
        # updates several times a second, few enough that the window really is
        # noise_s long.
        frames_per_s   = float(rate) / float(self._h)
        self._sub_len  = max(1, int(round(noise_s * frames_per_s / 4.0)))
        self._subs     = collections.deque(maxlen=4)
        self._cur_min  = None                         # np.ndarray | None
        self._sub_n    = 0
        self._primed   = False
        # Mean applied gain over the speech band, in dB, for the stats line.
        # Without this the suppressor is invisible: you cannot tell "working"
        # from "primed but doing nothing" from "MIC_NR=0" by listening to a
        # 16 kHz stream you are not holding.
        lo_bin = max(1, int(round(300.0 * self._n / float(rate))))
        hi_bin = min(nb, int(round(3400.0 * self._n / float(rate))) + 1)
        self._band = slice(lo_bin, hi_bin)
        self._db   = 0.0

    def reset(self) -> None:
        """Drop the overlap state at a stream discontinuity (ADAM speaking mutes
        the mic, so the 16 kHz stream really does have gaps). The NOISE estimate
        is kept — the room is the same room on the other side of the gap, and
        re-learning it would leave the first second after every reply
        unprocessed, which is exactly when the user is most likely to talk."""
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
            # Smooth the power before the minimum tracker. The heavier this
            # smoothing, the less the minimum sits below the true mean — which
            # is the bias MIC_NR_OVERSUB has to make up for. See the note there.
            self._pwr = (self._alpha * self._pwr
                         + (1.0 - self._alpha) * pwr)
            noise = self._noise_est(self._pwr)
            if self._primed:
                # Magnitude subtraction, expressed in power to avoid two sqrts:
                # gain = sqrt(max(P - a*N, 0) / P).
                clean = np.maximum(pwr - self._oversub * noise, 0.0)
                g     = np.sqrt(clean / np.maximum(pwr, 1e-9))
                np.maximum(g, self._floor, out=g)
                # Smooth across frequency (3-bin) and time (1-pole). Both fight
                # musical noise: isolated surviving bins get pulled down by
                # their neighbours, and bins cannot flip between full pass and
                # full floor from one 16 ms hop to the next.
                g[1:-1] = (g[:-2] + g[1:-1] + g[2:]) / 3.0
                # Fast-attack on rising speech (preserves word onsets), smooth decay on noise
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
            return b""
        y = np.concatenate(out)
        return np.clip(y, -32768, 32767).astype(np.int16).tobytes()


_mic_nr = (_NoiseSuppressor(MIC_NR_FRAME, MIC_NR_OVERSUB, MIC_NR_FLOOR_DB,
                            MIC_NR_NOISE_S, GEMINI_SEND_RATE)
           if MIC_NR else None)


def denoise_16k(pcm: bytes) -> bytes:
    """Cleaned copy of one 16 kHz mono chunk for the recogniser. Returns the
    input unchanged when MIC_NR=0. May return fewer or more bytes than it was
    given (WOLA runs on a fixed frame/hop, not on chunk boundaries) and returns
    b"" while the first frame is still filling — callers must tolerate both."""
    return pcm if _mic_nr is None else _mic_nr.process(pcm)


def denoise_reset() -> None:
    """Call at a discontinuity in the 16 kHz stream, i.e. whenever the mic has
    been muted. Cheap; safe to call when MIC_NR=0."""
    if _mic_nr is not None:
        _mic_nr.reset()


def denoise_db() -> float | None:
    """Mean gain the suppressor is currently applying across 300-3400 Hz, in dB.
    None when MIC_NR=0. 0.0 means primed-but-passing (the frame looked like pure
    speech) or not yet primed; the floor is MIC_NR_FLOOR_DB."""
    return None if _mic_nr is None else _mic_nr._db


# ── Which physical mic feeds the speech path ────────────────────────────
# Measured on this hardware, 5.9s of AMBIENT room with ADAM stopped (so no
# servo PWM), raw S32 before any filtering:
#
#     L: dc +9.10e+05 (-67.5 dBFS)  peak 1.795e+09 (-1.6 dBFS)  rms -18.4 dBFS
#     R: dc -2.94e+03 (-117.3 dBFS) peak 9.98e+08 (-6.7 dBFS)  rms -25.5 dBFS
#     energy: 80.85% below 60 Hz, loudest component 26.4 Hz,
#             250-4000 Hz (the speech band) = 2.7% of total
#
# Two conclusions, and they are what "ADAM keeps mis-hearing everything"
# actually is:
#
# 1. The ADC has 1.6 dB of headroom left on SILENCE. Speech adds roughly
#    -13 dBFS in-band on top of a rumble that is already at -1.6, so the
#    CONVERTER clips whenever someone talks. That distortion is created
#    upstream of every filter in this file, so no amount of DSP can undo
#    it, and clipping a 26 Hz carrier splatters intermodulation products
#    straight into the speech band. There is also no capture-gain control
#    to turn down: `amixer -c sndrpigooglevoi controls` returns nothing.
#
# 2. The 5.1 dB peak / 7.1 dB rms gap between L and R cannot be acoustic.
#    A 26 Hz wave is 13 m long; two mics ~5 cm apart see the same pressure
#    to within a small fraction of a dB. A difference this large, together
#    with a DC offset 300x bigger on L, means the left channel is picking
#    up something electrical or structure-borne of its own — grounding,
#    lead dress, a cold joint, or simply a weak INMP441.
#
# Averaging L and R therefore imports L's clipping risk into the mono speech
# signal. BUT dropping L is NOT a free win, and the measurement says so: on
# the same 8s capture, R alone had a post-filter noise floor of p50 1498
# against 804 for the L+R mix — 5.4 dB WORSE in band. L's excess is subsonic
# and the 120 Hz high-pass already removes it, while averaging two mics
# cancels uncorrelated noise. Trading 5.4 dB of speech-band SNR for headroom
# that is only needed on the loudest syllables makes recognition worse, not
# better, so "auto" drops a channel only when it is genuinely SATURATING
# (samples at/over MIC_CH_CLIP_FRAC of full scale, where the damage is real
# and unfixable downstream) and mixes in every other case. "mix" forces the
# average, "left"/"right" force one channel. DOA is unaffected either way —
# it reads the channels separately via s32_stereo_to_s16_stereo_channels().
_mic_ch_mode  = [None if MIC_CHANNEL == "auto" else MIC_CHANNEL]
_mic_ch_peak  = [0.0, 0.0]
_mic_ch_clip  = [0, 0]
_mic_ch_n     = [0]
_MIC_CH_CAL_CHUNKS = 30                      # 30 x 1600 frames @48k = 1.0 s

# Continuous saturation watch — see the MIC_CH_WATCH_S block in config.py for
# why the one-shot calibration above could never fire. `_mic_ch_forced` is the
# session-long flag that stops the watch once a channel has been dropped (or
# when the operator named a channel explicitly).
_MIC_CH_WATCH_CHUNKS = max(1, int(round(MIC_CH_WATCH_S * 48000.0 / 1600.0)))
_mic_ch_wclip   = [0, 0]
_mic_ch_wn      = [0]
_mic_ch_wpeak   = [0.0, 0.0]
_mic_ch_forced  = [MIC_CHANNEL != "auto"]


def _mic_ch_calibrate(l: np.ndarray, r: np.ndarray) -> None:
    """Track each channel's peak/saturation for 1s, then latch a mode."""
    fs  = 2.0 ** 31
    thr = MIC_CH_CLIP_FRAC * fs
    for i, ch in ((0, l), (1, r)):
        _mic_ch_peak[i] = max(_mic_ch_peak[i], float(np.abs(ch).max()))
        _mic_ch_clip[i] += int(np.count_nonzero(np.abs(ch) >= thr))
    _mic_ch_n[0] += 1
    if _mic_ch_n[0] < _MIC_CH_CAL_CHUNKS:
        return
    dbl = 20.0 * math.log10(max(_mic_ch_peak[0], 1.0) / fs)
    dbr = 20.0 * math.log10(max(_mic_ch_peak[1], 1.0) / fs)
    cl, cr = _mic_ch_clip
    if   cl and not cr: _mic_ch_mode[0] = "right"
    elif cr and not cl: _mic_ch_mode[0] = "left"
    else:               _mic_ch_mode[0] = "mix"
    print(f"  🎚️  Mic headroom L {dbl:+.1f} dBFS / R {dbr:+.1f} dBFS, "
          f"saturated samples L {cl} / R {cr} (raw, pre-filter) → speech "
          f"path uses {_mic_ch_mode[0].upper()}")
    if _mic_ch_mode[0] == "mix" and max(dbl, dbr) > -3.0:
        print(f"     ⚠️  only {-max(dbl, dbr):.1f} dB of converter headroom "
              f"left on room noise alone, and this HAT exposes no capture "
              f"gain — loud speech will clip in the ADC, upstream of every "
              f"filter. That is hardware (see the note in audio_utils.py).")


def _mic_ch_watch(l: np.ndarray, r: np.ndarray) -> None:
    """Keep counting saturation for the whole session, not just the first
    second, and drop a channel that proves it hits the converter rails.

    This is what makes MIC_CHANNEL=auto actually mean something: the initial
    1s calibration runs on boot silence, where clipping is impossible by
    construction, so without this the decision was always "mix" regardless of
    the hardware. Clipping is created inside the ADC, upstream of every filter
    in this file, so it is the one defect no amount of DSP can repair — which
    is why it is allowed to override the 5.4 dB in-band SNR advantage that
    mixing otherwise has.
    """
    fs  = 2.0 ** 31
    thr = MIC_CH_CLIP_FRAC * fs
    for i, ch in ((0, l), (1, r)):
        _mic_ch_wclip[i] += int(np.count_nonzero(np.abs(ch) >= thr))
        _mic_ch_wpeak[i]  = max(_mic_ch_wpeak[i], float(np.abs(ch).max()))
    _mic_ch_wn[0] += 1
    if _mic_ch_wn[0] < _MIC_CH_WATCH_CHUNKS:
        return

    cl, cr = _mic_ch_wclip
    hot    = None
    if cl >= MIC_CH_WATCH_MIN_CLIPS and cr * 8 <= cl:
        hot = ("left", "right", cl, cr)
    elif cr >= MIC_CH_WATCH_MIN_CLIPS and cl * 8 <= cr:
        hot = ("right", "left", cr, cl)

    if hot:
        bad, good, nbad, ngood = hot
        _mic_ch_mode[0]   = good
        _mic_ch_forced[0] = True
        print(f"  🎚️  Mic channel → {good.upper()}: the {bad} channel "
              f"saturated {nbad} samples in the last "
              f"{MIC_CH_WATCH_S:.0f}s (the {good} channel: {ngood}). "
              f"Clipping happens inside the ADC, so no filter can undo it — "
              f"dropping {bad} even though mixing is ~5.4 dB better in band. "
              f"Set MIC_CHANNEL=mix in ~/adam/.env to override.")
    _mic_ch_wclip[0] = _mic_ch_wclip[1] = 0
    _mic_ch_wpeak[0] = _mic_ch_wpeak[1] = 0.0
    _mic_ch_wn[0] = 0


def s32_stereo_to_s16_mono_16k(raw: bytes) -> bytes:
    s32 = np.frombuffer(raw, dtype=np.int32)
    if s32.size < 2:
        return b""
    # Combine the mics in FLOAT, before any scaling or clipping. The old
    # code shifted and cast each channel to int16 first, so a loud sample was
    # already clipped (previously: silently WRAPPED to a large opposite-sign
    # spike) before the two channels were even combined.
    _l = s32[0::2].astype(np.float32)
    _r = s32[1::2].astype(np.float32)
    if _mic_ch_mode[0] is None:
        _mic_ch_calibrate(_l, _r)               # mixes until it latches
    elif not _mic_ch_forced[0]:
        _mic_ch_watch(_l, _r)                   # keeps watching for clipping
    _m = _mic_ch_mode[0]
    if   _m == "left":  mono48 = _l
    elif _m == "right": mono48 = _r
    else:               mono48 = (_l + _r) * 0.5
    mono16 = _mic_chain.process(mono48) / float(1 << S32_SHIFT)
    return np.clip(mono16, -32768, 32767).astype(np.int16).tobytes()

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
    left  = np.clip(s32[0::2] >> S32_SHIFT, -32768, 32767).astype(np.int16)
    right = np.clip(s32[1::2] >> S32_SHIFT, -32768, 32767).astype(np.int16)
    return left, right

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

    @property
    def ready(self) -> bool:
        return len(self._ring) >= self._min_n or self._loaded

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

    # ── persistence ─────────────────────────────────────────────────
    def _load(self) -> None:
        try:
            with open(MIC_FLOOR_STATE_PATH, "r") as f:
                st = json.load(f)
            if (time.time() - float(st.get("t", 0))) > MIC_FLOOR_STATE_MAX_AGE_S:
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
                json.dump({"t": now, "floor": round(self._floor, 2)}, f)
            os.replace(tmp, MIC_FLOOR_STATE_PATH)
        except Exception:
            pass            # a lost floor costs one warmup, never a crash

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
    n_benign = 0
    last_report = 0.0
    last_benign = 0.0
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
                    n_benign += 1
                    if now - last_benign > 60.0:
                        print(f"  ℹ️  [{label}] {n_benign} underrun(s) while the "
                              f"device sat idle between replies — expected, "
                              f"nothing was playing.")
                        last_benign = now
                        n_benign = 0
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
