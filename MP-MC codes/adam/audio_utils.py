"""
audio_utils.py — ADAM v40 audio DSP helpers
==============================================================================
Pure signal-processing helpers with no side effects and no hardware handles:

  • S32 stereo (ALSA capture) → S16 mono 16 kHz  (what Gemini wants to hear)
  • S32 stereo → separate L/R S16 channels        (needed for DOA — averaging
                                                    to mono destroys the phase
                                                    difference between mics)
  • estimate_doa_angle()  — direction-of-arrival via GCC-PHAT
  • S16 mono 24 kHz (Gemini output) → S16 stereo 48 kHz (what the speaker wants)
  • rms_s32 / is_valid_pcm16_chunk — sanity gates
  • beep_s16_stereo — local UI beep
  • read_exact / drain_stderr — subprocess pipe helpers

Tuning constants (S32_SHIFT, MIC_DISTANCE_M, sample rates) come from config so
there's a single source of truth for the wiring/audio parameters.
"""

import subprocess

import numpy as np

from config import (
    S32_SHIFT,
    CAPTURE_RATE,
    PLAYBACK_RATE,
    MIC_DISTANCE_M,
    SOUND_SPEED_MPS,
)


def s32_stereo_to_s16_mono_16k(raw: bytes) -> bytes:
    s32   = np.frombuffer(raw, dtype=np.int32)
    if s32.size < 2:
        return b""
    # CLIP before the int16 cast. `(s32 >> SHIFT).astype(np.int16)` on its own
    # WRAPS on loud speech: any shifted value above 32767 silently overflows to
    # a large opposite-sign spike (e.g. 36621 -> -28915), which is the harsh
    # distortion that made Gemini mis-transcribe. np.clip saturates cleanly
    # instead. (S32_SHIFT was also raised 14->15 in config for peak headroom.)
    left  = np.clip(s32[0::2] >> S32_SHIFT, -32768, 32767).astype(np.int16)
    right = np.clip(s32[1::2] >> S32_SHIFT, -32768, 32767).astype(np.int16)
    mono  = ((left.astype(np.int32) + right.astype(np.int32)) // 2).astype(np.int16)
    return mono[::3].tobytes()

def s32_stereo_to_s16_stereo_channels(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Same S32->S16 downshift as s32_stereo_to_s16_mono_16k, but returns
    the two channels SEPARATELY instead of averaging them together. Needed
    for direction-of-arrival estimation, which requires the phase/timing
    difference between the two physical mics — information that's
    destroyed the instant left+right get averaged into mono."""
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

def s16_mono_24k_to_s16_stereo_48k(raw: bytes, gain: float = 1.0) -> bytes:
    mono = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if mono.size == 0:
        return b""
    if gain != 1.0:
        mono = np.clip(mono * gain, -32768, 32767)
    out_len = mono.size * 2
    up = np.interp(
        np.linspace(0, mono.size - 1, out_len, dtype=np.float32),
        np.arange(mono.size, dtype=np.float32), mono
    ).astype(np.int16)
    return np.repeat(up[:, None], 2, axis=1).reshape(-1).tobytes()

def rms_s32(raw: bytes) -> float:
    s = np.frombuffer(raw, dtype=np.int32).astype(np.float64)
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

def drain_stderr(proc: subprocess.Popen, label: str) -> None:
    try:
        for line in proc.stderr:
            txt = line.decode(errors="replace").strip()
            if txt and "underrun" not in txt.lower():
                print(f"  [{label}] {txt}")
    except Exception:
        pass
