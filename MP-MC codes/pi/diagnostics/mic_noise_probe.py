"""
mic_noise_probe.py — ADAM v40 mic noise-floor characterisation
==============================================================================
Answers one question that decides whether "ADAM can't hear me unless I shout"
is fixable in software: IS THE NOISE FLOOR FLAT OR TONAL?

  • Flat / broadband  -> the noise is a wideband electrical fault (grounding,
                         supply, clock). Software can only claw back a few dB
                         with band-limiting and spectral subtraction; the real
                         fix is on the board.
  • Tonal / harmonic  -> mains hum, a switching regulator, or clock bleed.
                         A notch or comb filter removes it almost entirely and
                         the mic recovers most of its dynamic range in software.

It records RAW s32 straight from arecord — no high-pass, no gain, no gate — so
the numbers describe the HARDWARE, not the DSP chain layered on top of it.

Reported per channel:
  rms_dbfs      RMS level in dBFS of the INMP441's own 24-bit full scale
  spl_equiv     that level expressed as dB SPL, using the datasheet sensitivity
                (-26 dBFS @ 94 dB SPL, so 0 dBFS = 120 dB SPL)
  tonality      how much of the total noise power sits in discrete peaks vs
                the broadband bed. >0.5 means tonal and worth notching.
  top peaks     the loudest narrowband components, with the dB by which each
                stands above the local broadband median

Run on the Pi, in a quiet room, with nothing playing:

    cd ~/adam && source venv/bin/activate && python3 mic_noise_probe.py
"""

import subprocess
import sys

import numpy as np

from config import (CAPTURE_DEVICE, CAPTURE_RATE, CAPTURE_CHANNELS,
                    CAPTURE_FORMAT)

# INMP441 datasheet: sensitivity -26 dBFS at 94 dB SPL, so digital full scale
# corresponds to 120 dB SPL, and spec self-noise is 33 dB(A) SPL (61 dB SNR).
INMP441_FS_SPL   = 120.0
INMP441_SELF_SPL = 33.0

SECONDS = 8.0


def _record_raw(seconds: float) -> np.ndarray:
    """Capture raw interleaved s32 frames from arecord as (n, CAPTURE_CHANNELS)."""
    n_bytes = int(seconds * CAPTURE_RATE) * CAPTURE_CHANNELS * 4
    cmd = ["arecord", "-D", CAPTURE_DEVICE, "-f", CAPTURE_FORMAT,
           "-r", str(CAPTURE_RATE), "-c", str(CAPTURE_CHANNELS), "-t", "raw",
           "-q"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    buf = bytearray()
    try:
        while len(buf) < n_bytes:
            chunk = p.stdout.read(min(65536, n_bytes - len(buf)))
            if not chunk:
                break
            buf.extend(chunk)
    finally:
        p.kill()
        p.wait()
    n_frames = len(buf) // (CAPTURE_CHANNELS * 4)
    a = np.frombuffer(bytes(buf[:n_frames * CAPTURE_CHANNELS * 4]), dtype="<i4")
    return a.reshape(-1, CAPTURE_CHANNELS)


def _analyse(x_s32: np.ndarray, label: str) -> dict:
    """Characterise one channel's noise. x_s32 is raw int32 from the I2S bus."""
    # The INMP441 puts its 24-bit sample in the TOP 24 bits of the 32-bit slot,
    # so full scale for the microphone itself is 2**31, and the bottom 8 bits
    # are always zero. Normalise to that, not to 2**15.
    x = x_s32.astype(np.float64) / (2.0 ** 31)
    x = x - x.mean()                      # strip the DC offset before metering
    rms = float(np.sqrt(np.mean(x * x))) + 1e-20
    rms_dbfs = 20.0 * np.log10(rms)

    # Welch PSD with a long window: fine resolution is what separates a
    # discrete tone from a broadband bed.
    nfft = 8192
    win  = np.hanning(nfft)
    step = nfft // 2
    segs = []
    for i in range(0, len(x) - nfft, step):
        segs.append(np.abs(np.fft.rfft(x[i:i + nfft] * win)) ** 2)
    if not segs:
        return {"label": label, "rms_dbfs": rms_dbfs, "psd": None}
    psd   = np.mean(segs, axis=0)
    freqs = np.fft.rfftfreq(nfft, 1.0 / CAPTURE_RATE)

    # Broadband bed = running median of the PSD. A rolling median tracks the
    # noise shape while ignoring narrow peaks, so peak-over-median is a clean
    # "is this a tone?" test that does not care about overall tilt.
    k = 129
    pad  = np.pad(psd, k // 2, mode="edge")
    bed  = np.array([np.median(pad[i:i + k]) for i in range(len(psd))])
    excess = psd - bed
    excess[excess < 0] = 0.0
    # Fraction of total noise power that lives in peaks above the bed.
    tonality = float(excess.sum() / max(psd.sum(), 1e-30))

    # Rank the peaks by how far they stand above their local bed.
    ratio_db = 10.0 * np.log10((psd + 1e-30) / (bed + 1e-30))
    order = np.argsort(ratio_db)[::-1]
    peaks, seen = [], []
    for i in order:
        f = freqs[i]
        if f < 20.0 or f > CAPTURE_RATE / 2 - 100:
            continue
        if any(abs(f - g) < 40.0 for g in seen):   # one entry per peak
            continue
        seen.append(f)
        peaks.append((float(f), float(ratio_db[i])))
        if len(peaks) >= 8:
            break

    # Octave-band breakdown: shows tilt, and how much of the bed sits outside
    # the speech band where a tighter filter could simply discard it.
    bands, edges = [], [20, 60, 120, 250, 500, 1000, 2000, 4000, 8000]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (freqs >= lo) & (freqs < hi)
        p = psd[m].sum()
        bands.append((lo, hi, 10.0 * np.log10(p / max(psd.sum(), 1e-30) + 1e-30)))

    return {"label": label, "rms_dbfs": rms_dbfs, "tonality": tonality,
            "peaks": peaks, "bands": bands, "psd": psd, "freqs": freqs}


def main() -> int:
    print(f"Recording {SECONDS:.0f}s of RAW s32 from {CAPTURE_DEVICE} "
          f"({CAPTURE_RATE}Hz {CAPTURE_CHANNELS}ch) — keep the room quiet…")
    raw = _record_raw(SECONDS)
    if raw.shape[0] < CAPTURE_RATE:
        print("  ✗ capture failed or too short")
        return 1
    print(f"  captured {raw.shape[0]} frames "
          f"({raw.shape[0] / CAPTURE_RATE:.1f}s)\n")

    results = []
    for ch, name in enumerate(("LEFT", "RIGHT")[:CAPTURE_CHANNELS]):
        r = _analyse(raw[:, ch], name)
        results.append(r)
        spl = INMP441_FS_SPL + r["rms_dbfs"]
        print(f"=== {name} =========================================")
        print(f"  noise RMS      {r['rms_dbfs']:+7.1f} dBFS "
              f"(of the mic's own 24-bit full scale)")
        print(f"  equivalent SPL {spl:7.1f} dB SPL")
        print(f"  vs INMP441 spec self-noise ({INMP441_SELF_SPL:.0f} dB SPL): "
              f"{spl - INMP441_SELF_SPL:+.1f} dB")
        print(f"  tonality       {r['tonality']:.3f}   "
              f"({'TONAL — notchable' if r['tonality'] > 0.5 else 'broadband — not notchable'})")
        print("  loudest narrowband peaks (dB above local broadband bed):")
        for f, d in r["peaks"]:
            print(f"      {f:8.1f} Hz   {d:+6.1f} dB")
        print("  power by band (dB relative to total):")
        for lo, hi, d in r["bands"]:
            bar = "#" * max(0, int(40 + d))
            print(f"      {lo:5d}-{hi:5d} Hz  {d:+6.1f}  {bar}")
        print()

    # What a tighter speech-band filter would buy, measured rather than assumed.
    live = results[0]
    if live.get("psd") is not None:
        f, psd = live["freqs"], live["psd"]
        total = psd.sum()
        for lo, hi in ((80, 8000), (150, 5500), (200, 4500)):
            m = (f >= lo) & (f < hi)
            print(f"  band-limit {lo}-{hi} Hz keeps "
                  f"{10 * np.log10(psd[m].sum() / total):+.1f} dB of the noise")
    return 0


if __name__ == "__main__":
    sys.exit(main())
