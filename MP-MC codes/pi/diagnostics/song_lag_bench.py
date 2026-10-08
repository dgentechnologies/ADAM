#!/usr/bin/env python3
"""song_lag_bench.py — measure the WRITE-TO-HEAR lag of the song playback path.

WHY THIS EXISTS
===============
Reference-free song barge-in is impossible on this unit. Proven empirically:
a genuine 400 ms voice barge-in and an ordinary verse->vocal-chorus transition
produce the IDENTICAL detector signature (fast, speech-shaped level rise above
a stale floor), because a singing voice and a speaking voice are the same
instrument spectrally. No level ratio separates them.

The one thing that does separate them: the chorus is in the SONG'S OWN PCM,
and the human's voice is not. So the detector must compare the mic against the
song's own reference level. That needs the reference delayed to line up with
what the mic is hearing right now -- and that delay is NOT the ~5.9 ms acoustic
flight time. song_playback.py writes into aplay's stdin, and aplay buffers:
a chunk written at wall-time T is HEARD at T + B, where B is the ALSA buffer
depth (plus the pipe's own backlog from SONG_PACE_FRAC=0.9 pacing).

B is the single unknown blocking the reference-based detector. Guessing it is
not acceptable -- guess too small and the reference leads the mic, which is the
FALSE-TRIGGER direction (expected level too low => the song stops itself).
This bench measures it on the actual hardware instead.

WHAT IT DOES
============
Reproduces the real song path exactly: ONE long-lived aplay process (never a
second one -- see song_playback.py's docstring for why), fed 4096-frame chunks
paced at SONG_PACE_FRAC, while arecord captures concurrently. The written
signal is a sparse train of short chirp bursts separated by silence, so
cross-correlation has unambiguous peaks. Each burst's write wall-time is
recorded; the lag is (burst heard time) - (burst written time).

Reports per-burst lag so you can see whether B is stable or drifts as aplay's
buffer fills -- a drifting B would mean a fixed delay line is wrong and the
reference needs a max-over-window instead.

USAGE (on the Pi, in ~/adam)
    ./venv/bin/python song_lag_bench.py
    ./venv/bin/python song_lag_bench.py --secs 20 --burst-dbfs -12
"""
import argparse
import subprocess
import sys
import threading
import time

import numpy as np

from config import (CAPTURE_DEVICE, CAPTURE_RATE, CAPTURE_CHANNELS,
                    PLAYBACK_DEVICE, PLAYBACK_RATE, PLAYBACK_CHANNELS,
                    SONG_CHUNK_FRAMES, SONG_PACE_FRAC, S32_SHIFT)

BURST_MS   = 60      # chirp burst length
GAP_MS     = 940     # silence between bursts (overridable via --gap)
CHIRP_LO   = 300.0
CHIRP_HI   = 3800.0  # inside the mic's band (MIC_LP_HZ) so it survives capture


def build_pattern(secs: float, peak_dbfs: float, gap_ms: float = GAP_MS):
    """Return (stereo_s16_bytes, burst_start_frames) for a burst train."""
    amp = (10.0 ** (peak_dbfs / 20.0)) * 32767.0
    n_burst = int(PLAYBACK_RATE * BURST_MS / 1000.0)
    n_gap   = int(PLAYBACK_RATE * gap_ms / 1000.0)
    t = np.arange(n_burst) / float(PLAYBACK_RATE)
    # linear chirp, windowed so the onset is sharp but not a click
    ph = 2 * np.pi * (CHIRP_LO * t + 0.5 * (CHIRP_HI - CHIRP_LO) / t[-1] * t * t)
    burst = np.sin(ph) * np.hanning(n_burst)
    mono, starts = [], []
    total = 0
    while total < secs * PLAYBACK_RATE:
        starts.append(total)
        mono.append(burst)
        mono.append(np.zeros(n_gap))
        total += n_burst + n_gap
    x = np.concatenate(mono) * amp
    s16 = x.astype(np.int16)
    stereo = np.repeat(s16[:, None], PLAYBACK_CHANNELS, axis=1).ravel()
    return stereo.astype(np.int16).tobytes(), starts, burst


def capture_thread(proc, sink, stop):
    """Drain arecord into sink as (wall_time, bytes) so we can timestamp."""
    bytes_per_frame = CAPTURE_CHANNELS * 4      # S32_LE
    n = 4096 * bytes_per_frame
    while not stop.is_set():
        d = proc.stdout.read(n)
        if not d:
            break
        sink.append((time.time(), d))


def to_mono_f64(chunks):
    """Concatenate S32_LE stereo capture -> mono float64 at CAPTURE_RATE."""
    raw = b"".join(d for _, d in chunks)
    a = np.frombuffer(raw, np.int32)
    a = a[: (a.size // CAPTURE_CHANNELS) * CAPTURE_CHANNELS]
    a = a.reshape(-1, CAPTURE_CHANNELS).astype(np.float64)
    return a.mean(axis=1) / float(1 << S32_SHIFT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=12.0)
    ap.add_argument("--burst-dbfs", type=float, default=-12.0)
    ap.add_argument("--gap", type=float, default=GAP_MS,
                    help="ms of silence between bursts. MUST exceed the "
                         "largest lag being measured, or the peak search "
                         "window spans two bursts and argmax locks onto the "
                         "wrong one (that is what produced the bogus ~2ms "
                         "readings at gap=940 when the true lag reached 1s).")
    args = ap.parse_args()

    # Never let the search window span two bursts -- see --gap's help.
    search_s = (args.gap / 1000.0) * 0.9

    pattern, starts, burst = build_pattern(args.secs, args.burst_dbfs, args.gap)
    print(f"pattern: {len(pattern)/(PLAYBACK_CHANNELS*2)/PLAYBACK_RATE:.1f}s, "
          f"{len(starts)} bursts of {BURST_MS}ms @ {args.burst_dbfs:+.0f} dBFS, "
          f"gap {args.gap:.0f}ms, search window {1000*search_s:.0f}ms")

    rec = subprocess.Popen(
        ["arecord", "-D", CAPTURE_DEVICE, "-f", "S32_LE",
         "-r", str(CAPTURE_RATE), "-c", str(CAPTURE_CHANNELS), "-t", "raw"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    play = subprocess.Popen(
        ["aplay", "-D", PLAYBACK_DEVICE, "-f", "S16_LE",
         "-r", str(PLAYBACK_RATE), "-c", str(PLAYBACK_CHANNELS), "-t", "raw"],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    chunks, stop = [], threading.Event()
    th = threading.Thread(target=capture_thread, args=(rec, chunks, stop),
                          daemon=True)
    th.start()
    time.sleep(0.5)                      # let arecord settle
    t_cap0 = chunks[0][0] if chunks else time.time()

    # Feed aplay exactly like song_playback.py: 4096-frame chunks, paced 0.9x.
    step = SONG_CHUNK_FRAMES * PLAYBACK_CHANNELS * 2
    pace = (SONG_CHUNK_FRAMES / float(PLAYBACK_RATE)) * SONG_PACE_FRAC
    write_times = []                     # (frame_offset, wall_time)
    off = 0
    while off < len(pattern):
        play.stdin.write(pattern[off:off + step])
        play.stdin.flush()
        write_times.append((off // (PLAYBACK_CHANNELS * 2), time.time()))
        off += step
        time.sleep(pace)
    play.stdin.close()
    play.wait(timeout=10)
    time.sleep(0.7)                      # let the tail drain out of the buffer
    stop.set()
    rec.terminate()
    th.join(timeout=2)

    mic = to_mono_f64(chunks)
    if mic.size < CAPTURE_RATE:
        print("FAIL: captured almost nothing -- is the mic device right?")
        return 1

    # Envelope of the mic, and of the reference burst, both at CAPTURE_RATE.
    env = np.abs(mic)
    w = int(CAPTURE_RATE * 0.010)
    env = np.convolve(env, np.ones(w) / w, mode="same")

    # Find each burst's arrival by peak-picking within a generous search window
    # around its expected position (write time + 0..1.5 s of buffer).
    print(f"\n{'burst':>5} {'written@':>9} {'heard@':>9} {'lag':>9}")
    lags = []
    for i, s_frame in enumerate(starts):
        # wall time this burst's samples were WRITTEN into aplay's stdin
        w_t = None
        for f_off, t in write_times:
            if f_off <= s_frame:
                w_t = t
            else:
                break
        if w_t is None:
            continue
        rel = w_t - t_cap0                       # seconds into the capture
        lo = int(max(0, rel * CAPTURE_RATE))
        hi = int(min(env.size, (rel + search_s) * CAPTURE_RATE))
        if hi - lo < w * 2:
            continue
        seg = env[lo:hi]
        pk = int(np.argmax(seg))
        # reject a burst whose peak is not clearly above the segment's own median
        med = float(np.median(seg)) + 1e-12
        if seg[pk] / med < 4.0:
            print(f"{i:>5} {rel:>8.3f}s {'--':>9} {'(no clear peak)':>9}")
            continue
        heard = (lo + pk) / float(CAPTURE_RATE)
        lag = heard - rel
        lags.append(lag)
        print(f"{i:>5} {rel:>8.3f}s {heard:>8.3f}s {1000*lag:>7.1f}ms")

    if not lags:
        print("\nFAIL: no bursts located. Raise --burst-dbfs or check routing.")
        return 1

    a = np.array(lags)
    print(f"\nlag: n={a.size}  min={1000*a.min():.1f}ms  "
          f"median={1000*np.median(a):.1f}ms  max={1000*a.max():.1f}ms  "
          f"spread={1000*(a.max()-a.min()):.1f}ms")
    print(f"drift first->last: {1000*(a[-1]-a[0]):+.1f}ms")

    spread = 1000 * (a.max() - a.min())
    if spread <= 40:
        print(f"\nSTABLE: a fixed {1000*np.median(a):.0f}ms reference delay "
              f"line is valid (spread {spread:.0f}ms).")
    else:
        print(f"\nUNSTABLE (spread {spread:.0f}ms): a fixed delay line is NOT "
              f"safe. The reference must be compared as a MAX over a window "
              f"of [{1000*a.min():.0f}, {1000*a.max():.0f}]ms so the expected "
              f"level is never too LOW (the false-trigger direction).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
