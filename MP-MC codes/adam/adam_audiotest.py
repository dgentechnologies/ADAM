#!/usr/bin/env python3
"""ADAM Pi audio round-trip test — mic capture + speaker playback.

Uses the EXACT arecord/aplay invocations from session.py and the real
audio_utils DSP helpers, so it validates the code path, not just the hardware.
    ~/adam/venv/bin/python ~/adam/adam_audiotest.py
"""
import os, sys, time, subprocess

os.environ.setdefault("GEMINI_API_KEY", "dummy")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import config
from audio_utils import rms_s32, beep_s16_stereo, s16_mono_24k_to_s16_stereo_48k
import numpy as np

REC_S = 2.0
print("=== ADAM audio round-trip test ===")
print(f"  capture : {config.CAPTURE_DEVICE} {config.CAPTURE_FORMAT} "
      f"{config.CAPTURE_RATE}Hz {config.CAPTURE_CHANNELS}ch")
print(f"  playback: {config.PLAYBACK_DEVICE} {config.PLAYBACK_FORMAT} "
      f"{config.PLAYBACK_RATE}Hz {config.PLAYBACK_CHANNELS}ch")

# ── [1] Capture (exact session.py arecord invocation) ────────────────────
print(f"\n[1] Recording {REC_S:.0f}s from the mics — make some noise...")
cmd = ["arecord", "-D", config.CAPTURE_DEVICE, "-f", config.CAPTURE_FORMAT,
       "-r", str(config.CAPTURE_RATE), "-c", str(config.CAPTURE_CHANNELS),
       "-t", "raw", "-q"]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
time.sleep(1.0)
if proc.poll() is not None:
    print(f"  ❌ arecord failed: {proc.stderr.read().decode(errors='replace').strip()}")
    sys.exit(1)
raw = proc.stdout.read(int(config.CAPTURE_RATE * config.CAPTURE_CHANNELS * 4 * REC_S))
proc.terminate()
try: proc.wait(timeout=2)
except Exception: proc.kill()

if not raw:
    print("  ❌ no audio captured"); sys.exit(1)
rms = rms_s32(raw)
peak = int(np.abs(np.frombuffer(raw, dtype=np.int32)).max())
print(f"  captured {len(raw)} bytes | RMS={rms:,.0f} peak={peak:,}")
print(f"  mic live: {'✅ YES' if rms > 1000 else '⚠️  very quiet (RMS<1000) — check mics'}")

# ── [2] Playback (exact session.py aplay invocation) ─────────────────────
print("\n[2] Playing test tone through the speaker...")
cmd = ["aplay", "-D", config.PLAYBACK_DEVICE, "-f", config.PLAYBACK_FORMAT,
       "-r", str(config.PLAYBACK_RATE), "-c", str(config.PLAYBACK_CHANNELS),
       "-t", "raw", "-q", "--buffer-size=96000"]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
if proc.poll() is not None:
    print(f"  ❌ aplay failed: {proc.stderr.read().decode(errors='replace').strip()}")
    sys.exit(1)
# 3 ascending beeps via the real beep helper
for f in (660.0, 880.0, 1100.0):
    proc.stdin.write(beep_s16_stereo(freq=f, dur=0.25))
proc.stdin.flush()
proc.stdin.close()
try: proc.wait(timeout=5)
except Exception: proc.kill()
err = proc.stderr.read().decode(errors="replace").strip()
print(f"  ✅ playback done{(' | aplay: ' + err) if err and 'underrun' not in err.lower() else ''}")
print("  (you should have heard 3 rising beeps)")

print("\n=== audio test done ===")
