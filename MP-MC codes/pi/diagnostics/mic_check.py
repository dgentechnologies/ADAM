#!/usr/bin/env python3
"""Does ADAM actually hear you? A two-phase pass/fail test of the REAL gate.

    ~/adam/venv/bin/python ~/adam/mic_check.py

Phase 1 asks for silence and checks the gate stays SHUT. Phase 2 asks you to
talk and checks it OPENS. Both numbers matter, and a fix for one that breaks
the other is the trap this project has fallen into repeatedly — a gate tuned
only against noise goes deaf, a gate tuned only against speech streams room
tone to Gemini and mishears continuously. So neither phase passes alone.

Everything here runs the SHIPPED path: the same arecord arguments session.py
uses, the same _mono16k_float, the same speech_band_rms, the same
_adaptive_gate singleton with the floor it resumed at import. Nothing is
re-implemented, because a re-implementation can pass while the product fails
(see docs/development_log.md Part 7 on self-confirming tests). The one thing
this does NOT share with the running system is the Gemini connection, so it
is safe to run with the service stopped and tells you nothing about it.
"""

import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, "/home/pi/adam")

from config import (CAPTURE_DEVICE, CAPTURE_FORMAT, CAPTURE_RATE,
                    CAPTURE_CHANNELS, CHUNK_FRAMES, GEMINI_SEND_RATE,
                    MIC_BAND_LO_HZ, MIC_BAND_HI_HZ)
from audio_utils import (_mono16k_float, speech_band_rms, rms_pcm16,
                         _adaptive_gate, _mic_live, _mic_scale, read_exact)

CHUNK_BYTES = CHUNK_FRAMES * CAPTURE_CHANNELS * 4       # S32_LE stereo
CHUNK_S     = CHUNK_FRAMES / CAPTURE_RATE

# Pass marks. Deliberately loose rather than flattering: the faults these
# exist to catch were 100%-open and 0%-open, so anything near the middle is a
# working gate and tightening these would only invite tuning to the test.
MAX_FALSE_OPEN_PCT = 2.0    # on silence
MIN_DETECT_PCT     = 20.0   # on speech — natural pauses make 100% impossible
MIN_SPEECH_SNR_DB  = 4.0    # peak speech over the learned floor


def db(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 20.0 * np.log10(a / b)


def capture(seconds: float, learn: bool, label: str) -> dict:
    """Run the real chain for `seconds` and return what the gate did.

    learn=True feeds observe_background(), exactly as session.py does while
    listening, so the floor converges on the room. learn=False leaves the
    floor alone — during the speech phase, feeding speech into the floor
    estimator would let the thing being measured move the ruler.
    """
    cmd = ["arecord", "-D", CAPTURE_DEVICE, "-f", CAPTURE_FORMAT,
           "-r", str(CAPTURE_RATE), "-c", str(CAPTURE_CHANNELS),
           "-t", "raw", "-q", "--buffer-size=48000"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, bufsize=0)
    time.sleep(0.4)
    if proc.poll() is not None:
        err = proc.stderr.read().decode(errors="replace").strip()
        print(f"  ❌ arecord failed: {err}")
        print("     Is ADAM still running? Stop it first: sudo systemctl stop adam")
        sys.exit(2)

    n_chunks = max(1, int(round(seconds / CHUNK_S)))
    levels, fulls, opened, in_cand = [], [], [], []
    bar_t = 0.0
    # Discard the first chunks of each capture. arecord has just restarted, so
    # the chain's biquad and FIR are stepping from a discontinuity and their
    # output is a settling transient, not room audio — measured at 2502
    # against a 598 room, i.e. 12 dB of pure artefact. Counting it would book
    # a false open against the silence phase and a phantom detection against
    # the speech phase, in both cases flattering whatever is being tested.
    # mic_calibrate._gate_floor_from drops 3 for exactly this reason.
    settle = 4
    try:
        for i in range(n_chunks + settle):
            raw = read_exact(proc.stdout, CHUNK_BYTES)
            if not raw or len(raw) < CHUNK_BYTES:
                break
            mono16 = _mono16k_float(raw)
            if mono16.size == 0:
                continue
            pcm = np.clip(mono16, -32768, 32767).astype(np.int16).tobytes()
            lvl = speech_band_rms(pcm)
            if i < settle:
                continue
            if learn:
                _adaptive_gate.observe_background(lvl, pcm)
            levels.append(lvl)
            fulls.append(rms_pcm16(pcm))
            opened.append(bool(_adaptive_gate.is_speech(pcm, lvl)))
            # Which tier decided this chunk. Without this the two phases
            # cannot be read against each other: a candidate-tier detection
            # on speech is the fix working, and a candidate-tier detection on
            # silence is the fix costing something, and they are the same
            # number in `opened`.
            in_cand.append(_adaptive_gate.cand_th <= lvl < _adaptive_gate.open_th)
            now = time.time()
            if now - bar_t > 0.25:
                bar_t = now
                oth = max(_adaptive_gate.open_th, 1.0)
                n = int(np.clip(20.0 + db(lvl, oth), 0, 46))
                mark = "SPEECH" if opened[-1] else "  --  "
                sys.stdout.write(f"\r   {label} [{'#'*n:<46}] "
                                 f"{lvl:7.0f}  {mark}")
                sys.stdout.flush()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except Exception:
            proc.kill()
    sys.stdout.write("\r" + " " * 78 + "\r")
    if not levels:
        print("  ❌ captured nothing — the I2S capture is wedged.")
        sys.exit(2)
    a = np.asarray(levels, dtype=np.float64)
    op = np.asarray(opened, dtype=bool)
    cd = np.asarray(in_cand, dtype=bool)
    return {"lvl": a,
            "cand_pct": 100.0 * float(np.mean(op & cd)),
            "full": np.asarray(fulls, dtype=np.float64),
            "open_pct": 100.0 * float(np.mean(opened)),
            "p20": float(np.percentile(a, 20)),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "max": float(a.max())}


def main() -> int:
    print()
    print("  ADAM mic check")
    print("  ─────────────────────────────────────────────────────────────")
    print(f"  channel     {_mic_live.mode.upper()}")
    print(f"  gain        ÷{_mic_scale[0]:.0f}")
    print(f"  gate band   {MIC_BAND_LO_HZ:.0f}-{MIC_BAND_HI_HZ:.0f} Hz "
          f"of {GEMINI_SEND_RATE // 2} Hz")
    print(f"  tiers       cand {_adaptive_gate.cand_th:.0f} "
          f"< open {_adaptive_gate.open_th:.0f} "
          f"< strong {_adaptive_gate.strong_th:.0f}")
    print(f"  floor now   {_adaptive_gate.floor:.0f} "
          f"({'resumed' if _adaptive_gate.ready else 'not learned yet'})")
    print()

    print("  PHASE 1 of 2 — please be QUIET for 10 seconds.")
    print("  Testing that room noise alone does NOT open the gate.")
    time.sleep(1.5)
    sil = capture(10.0, learn=True, label="quiet ")

    floor = _adaptive_gate.floor
    oth   = _adaptive_gate.open_th
    print(f"  floor learned      {floor:8.0f}")
    print(f"  open threshold     {oth:8.0f}   ({db(oth, floor):+.1f} dB over the floor)")
    print(f"  room noise         p20 {sil['p20']:.0f}   p50 {sil['p50']:.0f}   "
          f"p95 {sil['p95']:.0f}   max {sil['max']:.0f}")
    print(f"  out-of-band noise  {db(float(np.median(sil['full'])), sil['p50']):+.1f} dB "
          f"(how much the old full-band yardstick over-read by)")
    print(f"  FALSE OPENS        {sil['open_pct']:8.1f}%   "
          f"(must be under {MAX_FALSE_OPEN_PCT}%)")
    print(f"    of which cand tier {sil['cand_pct']:6.1f}%   "
          f"(what the shape-only tier costs on an empty room)")
    ok_silence = sil["open_pct"] <= MAX_FALSE_OPEN_PCT
    print(f"  → {'PASS' if ok_silence else 'FAIL'}")
    if not ok_silence:
        print("     The gate is opening on an empty room, so everything ADAM")
        print("     'hears' is noise and every transcript will be wrong.")
        print(f"     Needed open ratio ≥ {sil['p95'] / max(floor, 1.0):.2f}x "
              f"to clear this room's p95; MIC_OPEN_RATIO is "
              f"{oth / max(floor, 1.0):.2f}x.")
    print()

    print("  PHASE 2 of 2 — please TALK normally for 12 seconds,")
    print("  from where you actually sit when you talk to ADAM.")
    print("  (Count out loud, or read this paragraph aloud.)")
    time.sleep(2.0)
    spk = capture(12.0, learn=False, label="speak ")

    peak_snr = db(spk["p95"], floor)
    print(f"  speech level       p50 {spk['p50']:.0f}   p95 {spk['p95']:.0f}   "
          f"max {spk['max']:.0f}")
    print(f"  speech over floor  {peak_snr:+8.1f} dB at p95   "
          f"(must be over {MIN_SPEECH_SNR_DB:+.1f} dB)")
    print(f"  DETECTED           {spk['open_pct']:8.1f}%   "
          f"(must be over {MIN_DETECT_PCT}%)")
    print(f"    of which cand tier {spk['cand_pct']:6.1f}%   "
          f"(speech the old level-only veto would have dropped)")
    ok_speech = (spk["open_pct"] >= MIN_DETECT_PCT
                 and peak_snr >= MIN_SPEECH_SNR_DB)
    print(f"  → {'PASS' if ok_speech else 'FAIL'}")
    if not ok_speech:
        if peak_snr < MIN_SPEECH_SNR_DB:
            print("     Your voice is not arriving much above the room noise.")
            print("     That is a gain/placement problem, not a threshold one —")
            print("     lowering the threshold from here would just re-open the")
            print("     gate on noise. Try closer, or re-run the startup chime")
            print("     calibration, or check the mic is not obstructed.")
        else:
            print("     Loud enough but rejected, so the SHAPE vote is refusing")
            print("     it. See MIC_SHAPE_FLAT_MAX in config.py.")
    print()

    print("  ─────────────────────────────────────────────────────────────")
    if ok_silence and ok_speech:
        print("  RESULT: PASS — the gate is shut on the room and opens on you.")
        margin = db(spk["p95"], sil["p95"])
        print(f"  Your voice sits {margin:+.1f} dB above the room's p95 noise,")
        print(f"  with the threshold placed {db(oth, floor):+.1f} dB over the floor.")
        return 0
    print("  RESULT: FAIL — see the phase that failed above.")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n  interrupted")
        sys.exit(130)
