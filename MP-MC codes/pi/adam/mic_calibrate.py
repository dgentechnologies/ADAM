"""
mic_calibrate.py — ADAM v40 startup acoustic self-calibration
==============================================================================
ADAM plays a tone into its own body at startup, listens to itself, and tunes
the microphone path from what comes back.

WHY (the failure this exists to end)
------------------------------------------------------------------------------
Every other adaptive part of the mic path learns PASSIVELY — from whatever the
room happens to be doing — and persists what it learns so a restart is not
paid for with a re-warmup. That is correct while the hardware is constant and
silently wrong the moment it is not, because learned state carries no evidence
about the machine it describes.

Fitting the components into a new plastic body is exactly that moment. The
noise floor in .mic_floor.json moved 239 -> 1634 -> 3699 across the swap, and
every boot afterwards printed

    Resuming learned mic floor 3700 from the last run (open>=4625)
    Resuming learned mic channel LEFT (L live +11.9 dB, R DEAF +1.4 dB)

and then ran a gate calibrated for a body that no longer existed, on a channel
verdict measured 16 days earlier in that old body. With the open threshold at
4625 the quietest parts of speech — onsets, unvoiced consonants, the parts that
carry word IDENTITY rather than word ENERGY — never clear the bar, so Gemini is
fed vowels with the edges shaved off. That is what mishearing sounds like from
the inside.

The passive learner cannot detect this on its own, and not because it is badly
written. The 20th percentile of the mic over 45 s IS the noise floor, by
definition; if the new case rings or couples amplifier hiss, the learner
faithfully measures the ring and lifts the gate above it. It answers its
question correctly. The question it cannot answer is which of these happened:

    the ROOM got noisier          -> raising the gate is right
    the MIC PATH itself changed   -> everything learned is now stale

Those two are indistinguishable from passive observation, because passive
observation has no reference to compare against. Only a KNOWN stimulus can
separate them, and ADAM is the only thing present that can produce one. So it
produces one.

THE STIMULUS IS SIMULTANEOUS, NOT SWEPT — the load-bearing design choice
------------------------------------------------------------------------------
A chunk written to aplay is heard 541-941 ms later (measured by
song_lag_bench.py) and the figure DRIFTS with buffer fill. So the obvious
design — play 200 Hz, then 400 Hz, then 800 Hz, and slice the recording into
matching time windows — needs that lag to be known in order to attribute
energy to the right tone, and it fails quietly rather than loudly when the lag
is wrong: the bands simply smear into each other and the result still looks
like a plausible frequency response.

Summing the tones and playing them ALL AT ONCE removes the problem instead of
solving it. The tones are separated by FREQUENCY, by the FFT, so it does not
matter when the burst arrives or how long the buffer sat on it. One recording,
one transform, every band at once, and no alignment step that can silently be
wrong. Finding the burst inside the recording needs no timing knowledge
either: the frames containing the stimulus are found by their own tone-bin
energy (see _tone_power), which is self-locating by construction.

WHAT IS MEASURED, AND WHAT EACH MEASUREMENT DECIDES
------------------------------------------------------------------------------
  silence, amp on, nothing playing
        -> the true noise floor of the new body, in the GATE'S OWN UNITS
           (see _gate_floor_from: the capture is pushed through the real
           s32_stereo_to_s16_mono_16k path, because a raw-transducer RMS is
           not on the same scale as the number the gate compares against)
  per-tone SNR, per channel
        -> which microphone is actually live, decided against a known
           stimulus instead of inferred from how lively the room happened to
           be during MIC_CH_MIN_S of ambient
  per-tone SNR, shape across the ladder
        -> the body's frequency response: how much low-end the sealed case
           adds and how much consonant band it absorbs
  all of the above, versus the stored fingerprint
        -> whether the PATH changed since last boot. This is the check that
           makes a body swap self-healing: exceed MIC_CAL_CHANGE_DB and every
           passively-learned value is thrown away rather than resumed.

FAIL-OPEN, ALWAYS
------------------------------------------------------------------------------
Calibration is an optimisation, never a precondition. Missing arecord, a busy
device, a muted amplifier, someone talking over the tone — every one of these
returns None and leaves ADAM in exactly the behaviour it would have had if this
module did not exist. A calibration step that can prevent startup is worse than
no calibration step, because it converts a degraded mic into a dead robot.
"""

import os
import json
import math
import time
import subprocess

import numpy as np

from config import (CAPTURE_DEVICE, CAPTURE_FORMAT, CAPTURE_RATE,
                    CAPTURE_CHANNELS, PLAYBACK_DEVICE, PLAYBACK_FORMAT,
                    PLAYBACK_RATE, PLAYBACK_CHANNELS, CHUNK_FRAMES,
                    MIC_CAL_ENABLE, MIC_CAL_TONES, MIC_CAL_TONE_S,
                    MIC_CAL_SILENCE_S, MIC_CAL_CAPTURE_S, MIC_CAL_LEVEL,
                    MIC_CAL_LIVE_SNR_DB, MIC_CAL_DEAF_MARGIN_DB,
                    MIC_CAL_NOISE_PCTL,
                    MIC_CAL_CHANGE_DB, MIC_CAL_STATE_PATH,
                    MIC_CAL_GAIN_ENABLE, MIC_CAL_HEADROOM_DB,
                    MIC_CAL_SHIFT_MIN, MIC_CAL_SHIFT_MAX,
                    MIC_FLOOR_PERCENTILE, MIC_CHANNEL,
                    MIC_FS_SPL, MIC_SELF_NOISE_SPL, MIC_NOISE_WARN_DB,
                    MIC_HP_HZ, GEMINI_SEND_RATE)
from audio_utils import (s32_stereo_to_float_channels,
                         s32_stereo_to_s16_mono_16k, rms_pcm16,
                         speech_band_rms,
                         set_mic_scale, _mic_scale, _mono16k_float,
                         _adaptive_gate, _mic_live, read_exact)

# 8192 samples at 48 kHz = 171 ms per frame, 5.86 Hz per bin. The octave-spaced
# ladder (200 Hz upward) is separated by tens of bins at every rung, so no two
# tones can leak into each other's measurement window even with a badly offset
# sound-card clock.
CAL_FFT = 8192
# Tone energy is summed over the centre bin +/- this many, so a clock offset or
# the Hann window's own spreading cannot cost the tone its own measurement.
CAL_BIN_HALFWIDTH = 2
# Frames within this many dB of the loudest frame count as "the stimulus is
# present here". 6 dB = half amplitude: wide enough to survive the burst's
# attack and decay, tight enough to exclude frames that are purely room.
CAL_PRESENT_DB = 6.0


def _tone_freqs() -> list[float]:
    out = []
    for part in str(MIC_CAL_TONES).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            f = float(part)
        except ValueError:
            continue
        # Nyquist is the hard bound; anything at or above it aliases and would
        # be measured in the wrong bin entirely.
        if 20.0 < f < CAPTURE_RATE * 0.5:
            out.append(f)
    return out


def _stimulus(freqs: list[float], secs: float) -> bytes:
    """Sum of sines at `freqs`, as s16 stereo at PLAYBACK_RATE.

    Phases are staggered rather than aligned. Summing N sines in phase puts an
    N-times peak at t=0, which would force the whole stimulus down by 1/N to
    stay inside full scale and throw away most of the SNR the tones exist to
    provide; staggering keeps the crest factor near sqrt(N) instead. The result
    is then normalised by its ACTUAL peak, so MIC_CAL_LEVEL means the same
    thing — fraction of full scale — whatever ladder is configured.

    The raised-cosine envelope is not cosmetic: a rectangular burst starts and
    ends with a step, a step is broadband, and broadband energy lands in every
    tone bin at once and corrupts the very measurement being made. It also
    clicks, which is a poor thing for a robot to do on waking.
    """
    n = max(1, int(PLAYBACK_RATE * secs))
    t = np.arange(n, dtype=np.float64) / PLAYBACK_RATE
    x = np.zeros(n, dtype=np.float64)
    for i, f in enumerate(freqs):
        # Schroeder-style quadratic phase stagger: cheap, deterministic, and
        # keeps the crest factor low without a search.
        phase = math.pi * (i * i) / max(1, len(freqs))
        x += np.sin(2.0 * np.pi * f * t + phase)
    peak = float(np.max(np.abs(x))) or 1.0
    x *= (MIC_CAL_LEVEL * 32767.0) / peak

    fade = max(1, int(PLAYBACK_RATE * 0.02))          # 20 ms each end
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fade)))
    x[:fade] *= ramp
    x[-fade:] *= ramp[::-1]

    mono = np.clip(x, -32768, 32767).astype(np.int16)
    if PLAYBACK_CHANNELS == 1:
        return mono.tobytes()
    return np.repeat(mono, PLAYBACK_CHANNELS).tobytes()


def _tone_power(x: np.ndarray, freqs: list[float],
                select: bool) -> tuple[np.ndarray, int] | None:
    """Per-tone power for one channel.

    select=True  — keep only the frames where the stimulus is actually
                   present, located by the tone bins' OWN energy. This is what
                   makes the measurement independent of the 541-941 ms
                   write-to-hear lag: the burst is found by what it IS, not by
                   when it was expected. Reduced by the MEAN over those frames,
                   which is right because every kept frame is the same steady
                   tone and averaging suppresses noise.
    select=False — the silence capture, where there is no burst to find.
                   Reduced by MIC_CAL_NOISE_PCTL across frames, per bin, NOT by
                   the mean. A mean of power is dominated by its loudest term,
                   so a single hot frame — a servo step, a UART burst, a mains
                   tick — becomes the reference against which the tone is then
                   judged, and the chime appears not to have been heard at all.
                   See the measurement in config.py above MIC_CAL_NOISE_PCTL.
    """
    n = int(x.size)
    if n < CAL_FFT:
        return None
    w = np.hanning(CAL_FFT)
    bins = [int(round(f * CAL_FFT / CAPTURE_RATE)) for f in freqs]
    hop = CAL_FFT // 2

    spectra = []
    energies = []
    for i in range(0, n - CAL_FFT + 1, hop):
        seg = x[i:i + CAL_FFT].astype(np.float64)
        seg = (seg - seg.mean()) * w
        P = np.abs(np.fft.rfft(seg)) ** 2
        spectra.append(P)
        energies.append(sum(
            float(P[max(b - CAL_BIN_HALFWIDTH, 0):b + CAL_BIN_HALFWIDTH + 1]
                  .sum()) for b in bins))
    if not spectra:
        return None

    if select:
        e = np.asarray(energies)
        if float(e.max()) <= 0.0:
            return None
        keep = e >= e.max() * (10.0 ** (-CAL_PRESENT_DB / 10.0))
        chosen = [P for P, k in zip(spectra, keep) if k]
    else:
        chosen = spectra
    if not chosen:
        return None

    arr = np.asarray(chosen)
    Pavg = (np.mean(arr, axis=0) if select
            else np.percentile(arr, MIC_CAL_NOISE_PCTL, axis=0))
    out = np.array([
        float(Pavg[max(b - CAL_BIN_HALFWIDTH, 0):b + CAL_BIN_HALFWIDTH + 1]
              .sum()) for b in bins])
    return out, len(chosen)


def _snr_db(tone: np.ndarray, noise: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(tone, 1e-9) / np.maximum(noise, 1e-9))


def _noise_health(raw: bytes) -> dict:
    """Absolute noise-floor health of the microphone HARDWARE, in dB SPL.

    Every other number this module produces is RELATIVE — a tone SNR, a gate
    floor in gate units, a divisor. All of them can look perfectly healthy on
    a mic that is far too noisy to hear a person, because they all scale
    together: raising the digital gain lifts the floor and the signal by the
    same amount, and a loud near-field chime still beats a bad floor by a
    comfortable margin. That is exactly how a unit can report "+26 dB tone
    SNR, mic is fine" while being unable to hear anyone who is not shouting.

    This is the one measurement that is not relative. The INMP441 datasheet
    fixes the mapping from digital level to sound pressure: sensitivity is
    -26 dBFS at 94 dB SPL, so digital full scale is 120 dB SPL, and the part's
    own self-noise is 33 dB(A) SPL. Measuring the raw capture against ITS OWN
    full scale therefore converts the noise floor into an absolute dB SPL
    figure, which can be compared against how loud a person actually is:

        quiet room        30 dB SPL      normal speech @1m   60 dB SPL
        normal speech     55-65          raised voice @0.3m  75-80
        shout @0.1m       90-95          ADAM's own chime   105-115

    If the reported floor lands at or above normal speech, the microphone
    cannot hear a conversation no matter what the DSP does, and the only fix
    is on the board. Reporting that at boot is the difference between a unit
    that says it is fine and a unit that says what is wrong with it.

    Returns {} if the capture is unusable, so callers can skip reporting.
    """
    if len(raw) < CAPTURE_CHANNELS * 4 * 1024:
        return {}
    a = np.frombuffer(raw[:len(raw) - (len(raw) % (CAPTURE_CHANNELS * 4))],
                      dtype="<i4").reshape(-1, CAPTURE_CHANNELS)
    out: dict = {}
    for ch in range(min(2, CAPTURE_CHANNELS)):
        # The INMP441 left-justifies its 24-bit sample in a 32-bit slot, so
        # the microphone's own full scale is 2**31, NOT 2**15 and not 2**23.
        x = a[:, ch].astype(np.float64) / (2.0 ** 31)
        x = x - x.mean()                    # DC offset is not noise
        pk  = float(np.max(np.abs(x))) + 1e-20

        # Measure the noise IN THE BAND THE SPEECH PATH KEEPS, not across the
        # whole 24 kHz capture. This matters a lot on this hardware: the noise
        # is white, so three quarters of its total power sits above 8 kHz —
        # which the chain's decimation to 16 kHz throws away before the gate
        # or Gemini ever see it. Quoting the full-band figure would overstate
        # the floor by ~6 dB and contradict the unit's own observed behaviour
        # (it does occasionally hear a shout, which a full-band floor says is
        # impossible). Parseval via rFFT, so this needs no filter design.
        n    = len(x)
        spec = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(n, 1.0 / CAPTURE_RATE)
        band = (freqs >= MIC_HP_HZ) & (freqs <= GEMINI_SEND_RATE / 2)
        # rFFT power -> mean square over the original samples.
        ms = float(2.0 * spec[band].sum() / (n * n))
        rms = math.sqrt(max(ms, 1e-40)) + 1e-20
        dbfs = 20.0 * math.log10(rms)
        # CAUTION on the two SPL keys. Both are referred to the transducer's
        # own full scale (the division above is by 2**31, not by _mic_scale),
        # so "over_spec" — a difference of two SPL-mapped levels, in which the
        # mapping cancels — is a sound RELATIVE statement: how many dB noisier
        # this channel is than a clean INMP441 at the same reference.
        # "spl_equiv" is NOT sound as an absolute acoustic level on this
        # hardware, because most of the noise does not arrive through the
        # diaphragm at all. It is kept only because it is already persisted in
        # .mic_cal.json and documented in mic_noise_probe.py. Do not derive an
        # acoustic requirement from it; see _report_noise_health's docstring
        # for the measurements that rule the acoustic reading out.
        out[("left", "right")[ch]] = {
            "rms_dbfs":  round(dbfs, 1),
            "peak_dbfs": round(20.0 * math.log10(pk), 1),
            "spl_equiv": round(MIC_FS_SPL + dbfs, 1),
            "over_spec": round(MIC_FS_SPL + dbfs - MIC_SELF_NOISE_SPL, 1),
        }
    return out


def _report_noise_health(health: dict, channel: str,
                         median_snr: list | None = None,
                         dead: list | None = None) -> None:
    """Report what the noise floor measurably IS, and only that.

    This used to convert the floor into an absolute "dB SPL equivalent" via
    MIC_FS_SPL and declare a hardware fault whenever it exceeded the
    INMP441's 33 dB SPL self-noise spec. On this unit that printed "93 dB SPL
    equivalent, +60 dB vs spec, speech must reach ~99 dB SPL to open the gate,
    this is NOT fixable in software" — the verdict docs/development_log.md
    Part 25 recorded as final, which stopped work on the software path. It is
    wrong, and the refutation is in this same function's own inputs:

      * The RIGHT channel is acoustically DEAF — median tone SNR -0.1 dB, it
        does not hear the calibration chime at all — yet its noise floor sits
        within 3 dB of the LEFT channel's. Noise present in equal measure on
        a channel with no working acoustic path did not arrive acoustically.
      * The LEFT channel resolves that same chime at +27 dB SNR, from a small
        speaker a few centimetres away. A microphone that needed ~99 dB SPL
        before registering anything could not do that.
      * The noise is impulsive and channel-uncorrelated (kurtosis 9-44,
        corr(L,R) = -0.19). Acoustic room noise is neither.

    So the dominant noise is ELECTRICAL and injected AFTER transduction. An
    SPL figure is only meaningful for noise that came through the diaphragm;
    applied to noise added downstream it is a category error, and everything
    derived from it — the detect threshold, the transcribe threshold, the
    "unfixable" conclusion — inherits the error.

    The floor is still genuinely too high, and every dB removed from it is a
    dB of detection margin gained, so the wiring checklist stays. What goes is
    the claim that software cannot help, which 0.0% false opens on this very
    hardware (mic_check.py phase 1, after moving the gate to a 300-3400 Hz
    yardstick) disproves. The hardware fault that IS real and IS reported
    below is the dead channel.
    """
    if not health:
        return
    live = health.get(channel) or next(iter(health.values()))
    print(f"     noise floor: {live['rms_dbfs']:+.1f} dBFS in the "
          f"{MIC_HP_HZ:.0f}-{GEMINI_SEND_RATE / 2:.0f} Hz speech band, "
          f"peak {live['peak_dbfs']:+.1f} dBFS")
    if live["peak_dbfs"] > -0.5:
        print("     ⚠️  noise is RAILING to full scale — the capture is "
              "clipping on noise alone")

    # The only honest sufficiency test available at boot: a tone of known
    # level was just played and measured through this exact chain, so quote
    # that SNR rather than inferring a requirement from a datasheet.
    snr = None
    if median_snr:
        idx = {"left": 0, "right": 1}.get(channel)
        try:
            snr = (max(float(v) for v in median_snr) if idx is None
                   else float(median_snr[idx]))
        except (IndexError, TypeError, ValueError):
            snr = None
    if snr is not None:
        mark = "✅" if snr >= MIC_CAL_LIVE_SNR_DB else "⚠️ "
        print(f"     {mark} signal check: this path resolved the calibration "
              f"chime at {snr:+.1f} dB SNR")

    # The real, physical fault. Unlike the floor this is not a margin
    # question: no gain and no threshold recovers a mic that does not respond
    # to sound at all.
    if dead and any(dead):
        names = " and ".join(n for n, d in zip(("LEFT", "RIGHT"), dead) if d)
        print(f"  ❌ MIC HARDWARE FAULT — the {names} mic is DEAF, no response "
              f"to the chime.")
        print(f"     ADAM is running on one mic. Do NOT set MIC_CHANNEL=mix "
              f"until it is repaired:")
        print("     averaging a dead channel in halves the voice and keeps "
              "all of its noise (that cost")
        print("     21 dB of consonants once already — see "
              "docs/development_log.md Part 24).")

    # Relative noise excess. Reported as a comparison against a clean part at
    # the same reference, which is what the measurement supports — not as an
    # SPL, and not as a verdict on whether software can cope.
    over = live["over_spec"]
    if over <= MIC_NOISE_WARN_DB:
        print("     ✅ noise floor is within spec for the part")
        return
    print(f"     ⓘ  noise floor is {over:+.0f} dB above what a clean INMP441 "
          f"would give at this reference.")
    print("        It is not acoustic (see this function's docstring), so it "
          "is almost certainly pickup")
    print("        on the mic wiring or supply, and it costs detection margin "
          "directly. Check, in")
    print("        order: (1) mic wires short, twisted, away from the "
          "servo/amp/camera harness;")
    print("        (2) mic VDD separately decoupled, not sharing the servo "
          "rail; (3) SEL/LR pins tied")
    print("        correctly and both mics driving SD; (4) substitute a "
          "known-good mic module.")


def _gate_floor_from(raw: bytes) -> float:
    """Noise floor of a raw capture, expressed in the GATE'S units.

    The gate compares speech_band_rms(s32_stereo_to_s16_mono_16k(raw)) against
    its learned floor, and that path is not a plain scaling of the transducer:
    it picks a channel, high-passes, low-passes and decimates to 16 kHz, and
    then the level is taken across 300-3400 Hz only. An RMS taken off the raw
    int32 samples is therefore on a DIFFERENT SCALE, and seeding the gate with
    it would trade a stale floor for a wrong one. So the silence is pushed
    through the real function, in the real chunk size, and measured exactly
    where the gate measures.

    speech_band_rms rather than rms_pcm16 for exactly that reason: the gate's
    yardstick is band-limited (3.9 dB below full-band on this hardware), so a
    full-band seed would install a floor 3.9 dB too high and the unit would
    boot deaf. The two have to be the same function or the units diverge.

    The first chunks are dropped because the chain's biquad and FIR start from
    zero state: their output ramps for a few milliseconds, which reads as an
    artificially quiet floor and would leave the gate too sensitive.
    """
    step = CHUNK_FRAMES * CAPTURE_CHANNELS * 4       # S32_LE stereo
    vals = []
    for i in range(0, len(raw) - step + 1, step):
        mono = s32_stereo_to_s16_mono_16k(raw[i:i + step])
        if mono:
            vals.append(speech_band_rms(mono))
    if len(vals) <= 4:
        return 0.0
    vals = vals[3:]                                   # filter settling
    # Same statistic the gate itself uses, so the seeded value and the value
    # the gate would converge to on its own are the same kind of number.
    return float(np.percentile(np.asarray(vals, dtype=np.float64),
                               MIC_FLOOR_PERCENTILE))


def _post_chain_peak(raw: bytes) -> float:
    """Largest absolute sample the speech path would produce, before the clip.

    Measured through _mono16k_float rather than by scaling the raw int32
    peak, because the chain is not a scaling: the high-pass, the low-pass and
    the decimation all change the peak, and it is the value AT THE CLIP that
    decides whether a word survives. The 6.8 kHz low-pass alone takes several
    dB out of a white noise bed, so a raw-peak estimate would ask for more
    attenuation than the path actually needs.
    """
    step = CHUNK_FRAMES * CAPTURE_CHANNELS * 4       # S32_LE stereo
    peak = 0.0
    n = 0
    for i in range(0, len(raw) - step + 1, step):
        f = _mono16k_float(raw[i:i + step])
        n += 1
        if n <= 3 or f.size == 0:                    # filter settling
            continue
        peak = max(peak, float(np.abs(f).max()))
    return peak


def _solve_gain(tone_raw: bytes, silence_raw: bytes) -> tuple[float, str] | None:
    """Solve for the divisor that puts the measured peak at the target.

    The chime is emitted at a known fraction of full scale (MIC_CAL_LEVEL),
    so its recorded peak is a measurement of everything between the DAC and
    the int16 sample: amplifier gain, the distance and coupling from speaker
    to microphone inside this particular body, and the room. None of those
    are knowable in advance, which is exactly why the divisor is solved for
    here instead of being written down in config.py.

    The chain is linear and the divisor is a scalar, so the solve is exact:
    halving the peak takes precisely a doubling of the divisor. No search,
    no iteration, no convergence to worry about.

    Silence is measured too, but only as a veto. If the noise bed ALONE
    already exceeds the target peak there is no gain that can make speech fit
    underneath it, and quietly applying a huge attenuation would hide a
    hardware fault behind a very quiet microphone instead of reporting it.
    """
    target = 32767.0 * (10.0 ** (-MIC_CAL_HEADROOM_DB / 20.0))
    peak_t = _post_chain_peak(tone_raw)
    peak_s = _post_chain_peak(silence_raw)
    if peak_t <= 0.0:
        return None

    now = float(_mic_scale[0])
    want = now * (peak_t / target)

    lo, hi = 2.0 ** MIC_CAL_SHIFT_MIN, 2.0 ** MIC_CAL_SHIFT_MAX
    clamped = min(max(want, lo), hi)
    note = ""
    if abs(clamped - want) > 1e-6:
        note = (f", clamped from ÷{want:.0f} to stay inside "
                f"shift {MIC_CAL_SHIFT_MIN:.0f}-{MIC_CAL_SHIFT_MAX:.0f}")

    # Report both ends of the range the solve produced, because the pair is
    # what says whether the result is usable: a comfortable chime headroom
    # with the noise floor right behind it means there is no room left for a
    # voice, and that is a hardware problem, not a gain problem.
    scale = now / clamped
    print(f"     level: chime peak {peak_t * scale:.0f}, "
          f"noise peak {peak_s * scale:.0f} of 32767 "
          f"({20.0 * math.log10(32767.0 / max(peak_s * scale, 1.0)):.1f} dB "
          f"of headroom above the noise){note}")
    if peak_s * scale >= target:
        print("     ⚠️  the noise bed alone fills the target headroom — "
              "gain cannot fix this, the mic path itself is too noisy")

    shift_now = math.log2(now)
    shift_new = math.log2(clamped)
    return clamped, (f"chime peak set to {MIC_CAL_HEADROOM_DB:.0f} dB below "
                     f"full scale (shift {shift_now:.2f} → {shift_new:.2f})")


def _load_prev() -> dict | None:
    try:
        with open(MIC_CAL_STATE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _save(state: dict) -> None:
    try:
        tmp = MIC_CAL_STATE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=1)
        os.replace(tmp, MIC_CAL_STATE_PATH)
    except Exception as e:
        print(f"  ⚠️  could not write {MIC_CAL_STATE_PATH}: {e}")


def _capture(freqs: list[float]) -> tuple[bytes, bytes] | None:
    """Run the two captures. Returns (silence_raw, tone_raw).

    arecord is opened ONCE and held across both phases. Reopening it between
    them would re-pay the warm-up discard and, worse, would measure the silence
    and the tone through two different device states — the comparison between
    them is the entire measurement, so it has to happen down one barrel.
    """
    rec = subprocess.Popen(
        ["arecord", "-D", CAPTURE_DEVICE, "-f", CAPTURE_FORMAT,
         "-r", str(CAPTURE_RATE), "-c", str(CAPTURE_CHANNELS),
         "-t", "raw", "-q", "--buffer-size=48000"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    play = None
    try:
        if rec.stdout is None:
            return None
        bps = CAPTURE_RATE * CAPTURE_CHANNELS * 4     # bytes per second

        # The HAT delivers garbage for the first fraction of a second after
        # the stream opens — the same discard listen() does before trusting
        # anything, for the same reason.
        read_exact(rec.stdout, int(bps * 0.4))

        silence = read_exact(rec.stdout, int(bps * MIC_CAL_SILENCE_S))

        play = subprocess.Popen(
            ["aplay", "-D", PLAYBACK_DEVICE, "-f", PLAYBACK_FORMAT,
             "-r", str(PLAYBACK_RATE), "-c", str(PLAYBACK_CHANNELS),
             "-t", "raw", "-q"],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if play.stdin is None:
            return None
        play.stdin.write(_stimulus(freqs, MIC_CAL_TONE_S))
        play.stdin.flush()
        play.stdin.close()      # aplay drains what it has, then exits

        tone = read_exact(rec.stdout, int(bps * MIC_CAL_CAPTURE_S))
        return silence, tone
    except Exception as e:
        print(f"  ⚠️  calibration capture failed ({e})")
        return None
    finally:
        for p in (play, rec):
            if p is None:
                continue
            try:
                p.terminate()
                p.wait(timeout=2.0)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass


def calibrate() -> dict | None:
    """Measure the path, tune from it, persist the fingerprint.

    Returns the new state dict, or None if calibration could not be completed
    — in which case NOTHING has been changed and ADAM behaves exactly as it
    would without this module. The divisor was already restored to the last
    measured value when audio_utils was imported (see its _restore_mic_scale),
    so every early return below lands on the state the last good calibration
    left, not on a half-resumed one.
    """
    if not MIC_CAL_ENABLE:
        return None
    freqs = _tone_freqs()
    if not freqs:
        print("  ⚠️  MIC_CAL_TONES is empty — skipping startup calibration")
        return None

    print(f"  🎼 Calibrating mic path — playing a "
          f"{len(freqs)}-tone chime and listening to it "
          f"({MIC_CAL_SILENCE_S + MIC_CAL_CAPTURE_S + 0.4:.1f}s)")

    got = _capture(freqs)
    if got is None:
        return None
    silence_raw, tone_raw = got

    # Unclipped on purpose — see s32_stereo_to_float_channels. At the gain
    # this unit was actually running, the chime peaked at 71693 against an
    # int16 ceiling of 32767, so the clipped view of this very capture had
    # 18.5% of its samples flattened and every per-tone SNR below was being
    # read off harmonics of the clipping rather than off the chime.
    sl, sr_ = s32_stereo_to_float_channels(silence_raw)
    tl, tr  = s32_stereo_to_float_channels(tone_raw)

    noise = [_tone_power(sl, freqs, select=False),
             _tone_power(sr_, freqs, select=False)]
    tones = [_tone_power(tl, freqs, select=True),
             _tone_power(tr, freqs, select=True)]
    if any(v is None for v in noise + tones):
        print("  ⚠️  calibration capture too short to analyse — skipped")
        return None

    snr = [_snr_db(tones[i][0], noise[i][0]) for i in (0, 1)]
    med = [float(np.median(s)) for s in snr]

    # The tone level referred to the TRANSDUCER's own full scale, which is the
    # only stable fingerprint of the path available here.
    #
    # SNR is not one, and using it was a defect: SNR = tone - noise, and the
    # noise term is the room. Measured 2026-10-01 across three consecutive
    # calibrations as the room quietened — floor 675 -> 320 -> 255 — the chime
    # SNR rose +30.0 -> +37.5 -> +39.3 dB on hardware that had not been
    # touched, and the change detector duly announced "path CHANGED" all three
    # times, at 200 Hz then 1600 Hz. A detector that fires on the room is a
    # false alarm generator, and it fires in the one place it most misleads:
    # the boot log of a unit being diagnosed for a listening fault.
    #
    # Raw tone power is not a fingerprint either, because
    # s32_stereo_to_float_channels divides by _mic_scale[0] and the auto-gain
    # re-solves that divisor on every boot. Multiplying it back out removes
    # the divisor and leaves speaker level, acoustics and mic sensitivity —
    # i.e. the path. Read here, BEFORE _solve_gain runs below, so the divisor
    # used matches the one the capture was taken through.
    _g_db = 20.0 * math.log10(max(float(_mic_scale[0]), 1e-9))
    tone_ref = [[10.0 * math.log10(max(float(p), 1e-30)) + _g_db
                 for p in tones[i][0]] for i in (0, 1)]

    # Computed here rather than at the reporting site below because the
    # crest factor in it (peak - rms) is the evidence that distinguishes the
    # two reasons the chime check can fail, and the failure path needs it.
    health = _noise_health(silence_raw)

    # Did either channel hear the chime? If not, the channel verdict and the
    # gain solve have nothing to work from: both are measured AGAINST the tone,
    # and a verdict drawn from an unheard tone could mute a working mic.
    #
    # But the FLOOR is not measured against the tone — it is measured from the
    # silence capture alone, and that capture is just as valid either way. The
    # old code returned here, which threw the floor seed away as collateral,
    # and that is what put ADAM's "deaf for the first few minutes" into the
    # field log of 2026-10-01: the boot missed the live check by 0.7 dB
    # (L +11.3 against 12.0), skipped the seed, and ran on a STALE resumed
    # floor of 408 against a true room of 378. open_th was therefore 816
    # instead of 756, and every speech onset in that 60 dB-wide band was
    # invisible until the slow-rise/fast-fall estimator had walked the floor
    # down on its own — which is exactly the "then suddenly it started to
    # listen" that was reported. A 0.7 dB miss on one measurement must not
    # discard a different, independent, perfectly good measurement.
    heard = max(med) >= MIC_CAL_LIVE_SNR_DB
    if not heard:
        print(f"  ⚠️  calibration heard the chime weakly "
              f"(L {med[0]:+.1f} dB, R {med[1]:+.1f} dB SNR, "
              f"need {MIC_CAL_LIVE_SNR_DB:+.0f}) — keeping the existing "
              f"channel and gain, which the tone cannot re-decide.")
        crest = 0.0
        if health.get("left"):
            crest = (float(health["left"]["peak_dbfs"])
                     - float(health["left"]["rms_dbfs"]))
        if crest > 20.0:
            # The reference is the quiet part of the silence window, so a
            # large crest means something banged DURING it. That is a noise
            # burst, not a dead speaker — which matters, because the two
            # suggest opposite things to go and check.
            print(f"     the silence window had a {crest:.0f} dB crest, so "
                  f"something burst during it (servo, UART, supply) rather "
                  f"than the speaker being silent.")
        else:
            print(f"     the silence window was clean ({crest:.0f} dB crest), "
                  f"so suspect the speaker/amp or the mic wiring.")
        # Floor anyway — see above. This is the whole point of not returning.
        floor = _gate_floor_from(silence_raw)
        if floor > 0.0:
            _adaptive_gate.seed_floor(
                floor, "measured at startup (chime weak, floor still valid)")
        _report_noise_health(health, _mic_live.mode, med, [False, False])
        # No _save(): the fingerprint would be garbage, and the next boot
        # would compare a good measurement against it and declare a change.
        return None

    live = [m >= MIC_CAL_LIVE_SNR_DB for m in med]
    dead = [False, False]
    for i in (0, 1):
        j = 1 - i
        if med[j] - med[i] >= MIC_CAL_DEAF_MARGIN_DB and not live[i]:
            dead[i] = True

    # Body response, read off the live channel: how the case tilts the
    # spectrum. Consonants live in the hi band, so a strongly positive tilt is
    # the signature of a sealed box that is boosting boom and eating clarity.
    best = 0 if med[0] >= med[1] else 1
    lo = [s for f, s in zip(freqs, snr[best]) if f < 1000.0]
    hi = [s for f, s in zip(freqs, snr[best]) if f >= 1000.0]
    tilt = (float(np.mean(lo)) - float(np.mean(hi))) if lo and hi else 0.0

    # ── did the PATH change since last boot? ──────────────────────────
    prev = _load_prev()
    changed, why = False, ""
    if prev is None:
        changed, why = True, "no previous calibration on file"
    elif [round(f, 1) for f in freqs] != [round(float(f), 1)
                                          for f in prev.get("freqs", [])]:
        changed, why = True, "tone ladder changed"
    elif abs(float(prev.get("level", -1.0)) - MIC_CAL_LEVEL) > 1e-6:
        # A different stimulus level moves every SNR figure at once, so the
        # stored numbers are not comparable and a "change" would be spurious.
        changed, why = True, "stimulus level changed"
    else:
        p_ref = prev.get("tone_ref_db")
        if not p_ref:
            # Written by an older build, which stored only room-dependent
            # SNRs. There is nothing comparable on file, so relearn once.
            changed, why = True, "fingerprint predates the stable tone reference"
        else:
            try:
                d = np.abs(np.asarray(p_ref[best], dtype=np.float64)
                           - np.asarray(tone_ref[best], dtype=np.float64))
                worst = float(d.max())
            except Exception:
                worst, d = 999.0, None
            if worst > MIC_CAL_CHANGE_DB:
                k = int(np.argmax(d)) if d is not None else 0
                changed = True
                why = (f"tone level moved {worst:.1f} dB at {freqs[k]:.0f} Hz "
                       f"(limit {MIC_CAL_CHANGE_DB:.0f} dB)")
            elif prev.get("live") != live or prev.get("dead") != dead:
                changed, why = True, "microphone channel verdict changed"

    names = ("LEFT", "RIGHT")
    print(f"     tone SNR: L {med[0]:+.1f} dB {'(live)' if live[0] else '(DEAF)'}"
          f"  |  R {med[1]:+.1f} dB {'(live)' if live[1] else '(DEAF)'}")
    print("     body response ("
          + ", ".join(f"{f:.0f}Hz {s:+.0f}" for f, s in zip(freqs, snr[best]))
          + f" dB) — tilt {tilt:+.1f} dB lo/hi")

    # ── apply ─────────────────────────────────────────────────────────
    # Channel FIRST: the floor is a level, and which mics feed the path sets
    # that level. Measuring the floor before fixing the channel would measure
    # it for a path about to be replaced.
    if MIC_CHANNEL != "auto":
        print(f"     MIC_CHANNEL={MIC_CHANNEL} is pinned in .env — "
              f"calibration is reporting only, not overriding it")
    else:
        _mic_live.apply_calibration(live, dead, med)

    if changed:
        print(f"  🔄 Mic path CHANGED ({why}) — discarding everything learned "
              f"in the previous body and relearning from this one")
        _adaptive_gate.reset_floor("startup calibration: mic path changed")

    # Gain SECOND, for the same reason the channel came first: the divisor is
    # solved from a peak measured through the selected channel, so it has to
    # be solved after the selection and before anything reads a level.
    gain_why = ""
    if MIC_CAL_GAIN_ENABLE:
        got_gain = _solve_gain(tone_raw, silence_raw)
        if got_gain is not None:
            set_mic_scale(got_gain[0], got_gain[1])
            gain_why = got_gain[1]

    # Floor LAST. It is an absolute level, so it is only meaningful once the
    # channel and the divisor that produce that level are both settled — and
    # _gate_floor_from re-runs the same silence through the now-current path,
    # so what it returns is already in the new units.
    floor = _gate_floor_from(silence_raw)
    if floor > 0.0:
        _adaptive_gate.seed_floor(floor, "measured at startup in this body")

    # Hardware health, reported LAST so it is the final word the boot log
    # leaves on the microphone. The noise measurement is deliberately
    # independent of the channel and gain decisions above — it is referred to
    # the transducer's own full scale, so it stays true even when every
    # relative number looks good. The chime SNR and the dead-channel verdict
    # are passed in because the honest health question is "did this path
    # resolve a known signal", and that was just measured directly; inferring
    # it from the floor and a datasheet is what produced the wrong verdict
    # this function's docstring documents. (health was measured above, before
    # the chime verdict, because the weak-chime path needs it too.)
    _report_noise_health(health, _mic_live.mode, med, dead)

    state = {"t": time.time(), "freqs": [round(f, 1) for f in freqs],
             "level": MIC_CAL_LEVEL,
             "snr_db": [[round(float(v), 2) for v in s] for s in snr],
             "tone_ref_db": [[round(float(v), 2) for v in t] for t in tone_ref],
             "median_snr_db": [round(m, 2) for m in med],
             "live": live, "dead": dead, "tilt_db": round(tilt, 2),
             "gate_floor": round(floor, 2),
             "mic_scale": round(float(_mic_scale[0]), 1),
             "mic_shift": round(math.log2(float(_mic_scale[0])), 3),
             "gain_why": gain_why,
             "noise_health": health,
             "channel": _mic_live.mode}
    _save(state)
    return state
