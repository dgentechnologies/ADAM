# ADAM v40 — Startup Mic Calibration

> ADAM plays a chime through its own speaker at boot, listens to it through its
> own microphones, and solves its own mic gain from what it measured. Nothing in
> this path is hand-set, per-room, or per-unit.

**Written 2026-09-29**, after the components were refitted into a new plastic
body and ADAM began mishearing words. This document records what was actually
wrong, why the fix is a measurement rather than a constant, and what is still
open.

Companion documents:
- `mic_speaker_issues.md` — the fault guide. **A5** ("do not raise
  MIC_S32_SHIFT") and **A11** (the deaf RIGHT mic) are both directly relevant.
- `development_log.md` — the running history; line ~1019 is the clipping
  mechanism cited below.
- `full_duplex_and_song_bargein.md` — the other half-duplex work.

---

## Summary for someone in a hurry

The unit was running `MIC_S32_SHIFT=15`, which is 6 dB hotter than the
documented-safe 16. In the new plastic body the noise bed had risen enough that
**only 3.9 dB of headroom was left above silence** — so nearly every word hard-
clipped on the way in. Clipping "generates harsh odd harmonics, flattening
vowels and destroying consonant differentiation (/p/, /b/, /s/)"
(`development_log.md`). That is the mishearing, and it is a level fault, not a
recogniser fault.

The fix is not a better constant. `S32_SHIFT` has been hand-set to 13, 14, 15
and 16 at different points in this project's life — once per hardware change,
always by a person, always after the unit had already started failing. It is now
**solved from a measurement taken at every boot**, against ADAM's own chime,
through ADAM's own body.

Measured on one set of captures, so the three rows are directly comparable:

| Mic gain | Headroom over silence | Chime samples clipped |
|---|---|---|
| shift 15 — *what was running* | **+3.9 dB** | **18.316 %** |
| shift 16 — the documented default | +9.9 dB | 0.081 % |
| **calibrated from the tone** | **+20.2 dB** | **0.000 %** |

---

# Part 1 — The root cause

## 1.1 Symptom

ADAM mishears. Words come back wrong or not at all, worst during startup, after
the components were moved into a new plastic case. The recogniser, the gate and
the noise suppressor were all suspected first, because all three had been
touched recently.

## 1.2 What was actually happening

The I2S capture path delivers 32-bit words (an INMP441 is a 24-bit device in a
32-bit slot). `s32_stereo_to_s16_mono_16k` divides by `2**S32_SHIFT` to land in
int16. That divisor **is** the mic gain: a smaller divisor is a louder signal
and less headroom.

`~/adam/.env` on the Pi carried `MIC_S32_SHIFT=15`. The code default, and the
value `config.py` documents its thresholds against, is 16. So the shipped unit
was running one bit — 6 dB — hotter than everything else in the system assumed.

In the old body that was survivable. In the new one the noise bed had risen by
roughly 24 dB (measured floor moved from 1082 to ~7900 raw / 4013 post-chain),
and 6 dB of unearned gain on top of that consumed what was left:

```
shift 15:  noise peak alone is only 3.9 dB below full scale
```

With 3.9 dB of headroom above *silence*, a spoken word has nowhere to go. 18.3 %
of the calibration chime's samples were flattened against the ceiling. Real
speech, which is louder than the chime, was worse.

## 1.3 Why this hid so well

Three reasons this was not obvious from the logs:

1. **A clipped measurement cannot report that it clipped.** Every level readout
   in the system went through the int16 clip, so no matter how far over the
   signal actually was, it read back as at most 32767. The fault erased its own
   evidence.
2. **The floor estimator followed the noise up.** The gate is ratio-based
   against a learned floor, by design (see A1). It kept working — it just
   learned a floor of 4013 and opened at ~5016. Nothing in the gate's own
   telemetry looks wrong at that floor. What *is* wrong is that 4013 is inside
   the documented speech band and above the quietest real speech ever measured
   on this unit (2357). ADAM's own noise floor had risen above its owner's
   voice.
3. **Absolute constants were being crossed silently.** `MIC_LIVE_RMS_THRESHOLD`
   (1200) and `MIC_SILENCE_FLOOR` (1800) were both *below* the noise floor, so
   pure room noise crossed them continuously. ADAM never saw silence.

## 1.4 A hypothesis that was wrong, and how it was refuted

Before the level fault was found, the noise bed itself was characterised as
"HF hiss — 82 % of the power is above 3.2 kHz, exactly the consonant band". That
motivated looking for a high-pass or tilt correction.

It was an **artifact of unequal band widths**. The bands used to compute those
percentages were not equal in Hz, and 6.4–24 kHz alone is 73 % of the spectrum
at a 48 kHz sample rate — so *white* noise naturally puts ~73 % of its power
there. `_micnoise.py` measured the tilt directly and found **−1.5 to +0.9 dB**,
i.e. flat.

This mattered: no filter shape can help broadband noise. Recording it here so
the same wrong turn is not taken twice, and so the band-percentage style of
measurement is not trusted again without width normalisation.

---

# Part 2 — The fix: a measured gain, not a chosen one

## 2.1 The solve

The capture chain is linear and the gain is a single scalar divisor. That means
the correct divisor is not searched for — it is **solved in one step**:

```
new_divisor = old_divisor × (measured_peak / target_peak)
```

where

```
target_peak = 32767 × 10^(−MIC_CAL_HEADROOM_DB / 20)
```

Play a chime of known level, measure the peak it comes back at, and the divisor
that would have put that peak exactly at the target follows by proportion. No
iteration, no convergence criterion, no search. Re-running it is a *confirmation*
rather than a refinement, which is why repeat boots move by fractions of a dB:

| Boot | Gain change applied | Divisor | Equivalent shift | Floor learned |
|---|---|---|---|---|
| 1st (from shift 16) | **−11.2 dB** | ÷236 898 | 17.85 | 474 |
| 2nd (restored, re-measured) | +0.7 dB | ÷219 108 | 17.74 | 738 |
| 3rd (real boot, full startup) | −0.4 dB | ÷230 005 | 17.81 | 578 |

The first boot makes the whole correction. Later boots re-confirm it and absorb
whatever the room has done since.

## 2.2 Two measurement paths, and why they differ

The calibration needs two different peaks, and getting either from the wrong
place breaks it.

**For the gain solve — measure through the real chain.** `_post_chain_peak()`
runs the capture through `_mono16k_float()`, the same function the speech path
uses: channel select, high-pass, low-pass, decimate to 16 kHz. Scaling a raw
int32 peak instead would overstate the level, because the 6.8 kHz anti-alias
low-pass alone removes several dB from a white noise bed. The first three chunks
are discarded so the filters have settled.

**For the SNR analysis — measure unclipped.** Per-tone SNR is read from
`s32_stereo_to_float_channels()`, which applies the divisor but *not* the int16
clip. This was not a nicety: the chime peaked at **71 693** against a 32767
ceiling, so 18.5 % of its samples were flattened and the per-tone SNRs were
being computed off clipping harmonics rather than off the tones. Measuring
unclipped moved the LEFT channel's tone SNR from **+17.8 dB to +20.8 dB** — the
same signal, honestly measured.

This is the general trap: **a calibration cannot measure a path through the
distortion it exists to detect.**

## 2.3 Why 9 dB of headroom

`MIC_CAL_HEADROOM_DB = 9.0`. The target is set against the system's own
documented operating point, not picked for comfort:

- It predicts a post-calibration noise floor around **514 RMS**, which lands
  inside `config.py`'s documented band — *"ambient room noise sits at 300–600
  RMS; speech sits at 2000–8000 RMS"*. Measured floors came out at 474, 578 and
  738. That is the point: every absolute threshold in the system was written
  against that band, and landing the floor back inside it makes them meaningful
  again rather than requiring all of them to be re-tuned.
- It leaves noise **peaks** about 20 dB below full scale, so speech transients
  have room.
- Raising it further buys nothing. 16-bit quantisation noise is ~96 dB down,
  which is ~70 dB below this unit's measured noise floor. Trading real headroom
  for protection against a noise source 70 dB under the dominant one is not a
  trade.

The solved divisor is clamped to `2^13 … 2^20` (`MIC_CAL_SHIFT_MIN/MAX`). The
clamp exists so a pathological measurement — a dead speaker, a silent capture —
cannot drive the gain somewhere unrecoverable. When it engages, it says so.

## 2.4 The units-coherence bug this exposed

ADAM resumes three pieces of learned state at startup, and **all three have to
agree** for any of them to mean anything:

| State | File | Resumed by |
|---|---|---|
| channel (left / right / mix) | `.mic_ch.json` | `_MicChannelLiveness._load()` |
| noise floor | `.mic_floor.json` | `AdaptiveGate._load()` |
| gain divisor | `.mic_cal.json` | `_restore_mic_scale()` |

The first two were resumed at `audio_utils` import. The gain was **not** — it
restarted from `S32_SHIFT` every boot.

Those facts are incompatible. A learned floor is an absolute level, and it is
only meaningful at the divisor it was measured at. Resuming two of the three
pairs a floor measured at ÷230005 with audio scaled at ÷65536 — **11 dB of
mismatch, in whichever direction is worse**: either the gate sits far below the
noise and latches open on room tone, or it sits far above speech and the unit
goes deaf. Silently, with nothing in the log looking wrong.

**Where the fix had to go.** The first attempt put the restore inside
`mic_calibrate.calibrate()`, which fixes `main.py` and nothing else. Every
diagnostic — `mic_probe.py`, `mic_modes.py`, `mic_watch.py`, the whole table in
`mic_speaker_issues.md` — imports `audio_utils` and never calls `calibrate()`.
Those tools exist precisely because *"their numbers describe the shipped
pipeline rather than a re-implementation of it"*, and they would all have been
measuring at the wrong gain against a floor from the right one.

So the restore now happens at `audio_utils` import, beside the other two, and
`calibrate()` no longer does it at all. One place, and the three cannot come
back out of step by construction.

Verified with an entry point that never calls `calibrate()`:

```
🎙️  Resuming learned mic channel LEFT (L live +18.6 dB, R DEAF -0.2 dB)
🔊 Mic gain -10.9 dB (÷65536 → ÷230005) — restored from the last calibration
🎚️  Resuming learned mic floor 578 from the last run (open≥722)
```

Channel, then gain, then floor — in that order, so the floor is loaded into
units that already exist.

The second half of the fix is the `reset_floor` flag on `set_mic_scale()`:
*restoring* a gain the on-disk floor was already measured against must **not**
discard that floor, while *changing* the gain must. Never the other way round.

## 2.5 Why the chime is simultaneous, not swept

All six tones (200, 400, 800, 1600, 3200, 6000 Hz) sound **at once**, for 4.6 s.

A swept tone would need the calibrator to know *when* each tone arrived, and the
write-to-hear lag on this hardware is **541–941 ms and drifts** — it is a
function of aplay's buffer state, not a constant. Any fixed assumption about it
is wrong on some boots.

Sounding all six together removes the need to know the lag at all. Each tone is
found by its own bin energy, so the burst locates itself in the capture, and one
capture measures every frequency under identical conditions.

---

# Part 3 — What the calibration reports, and how to read it

A healthy boot, **first time** in a new body — two gain lines, because the
restored value and the measured one genuinely differ:

```
🔊 Mic gain -10.5 dB (÷65536 → ÷219108) — restored from the last calibration
🎼 Calibrating mic path — playing a 6-tone chime and listening to it (4.6s)
   tone SNR: L +18.6 dB (live)  |  R -0.2 dB (DEAF)
   body response (200Hz +41, 400Hz +28, 800Hz +16, 1600Hz +16, 3200Hz +21, 6000Hz +12 dB) — tilt +12.1 dB lo/hi
🎙️  Mic channel → LEFT (calibrated: L +18.6 dB, R -0.2 dB tone SNR)
   ⚠️  the RIGHT mic did not hear ADAM's own chime — it is contributing noise
       only and has been dropped from the speech path.
   level: chime peak 11626, noise peak 3214 of 32767 (20.2 dB of headroom above the noise)
🔊 Mic gain -0.4 dB (÷219108 → ÷230005) — chime peak set to 9 dB below full scale (shift 17.74 → 17.81)
🎚️  Mic floor reset (mic gain changed) — relearning over 1.5s
🎚️  Mic floor measured 578 — measured at startup in this body (open≥722, strong≥1849)
```

A **converged** boot — one gain line, because the solve agreed with the restored
value to within 0.1 dB and `set_mic_scale` declines to announce a non-change:

```
🔊 Mic gain -10.9 dB (÷65536 → ÷230005) — restored from the last calibration
🎼 Calibrating mic path — playing a 6-tone chime and listening to it (4.6s)
   tone SNR: L +21.1 dB (live)  |  R -0.3 dB (DEAF)
   level: chime peak 11626, noise peak 2984 of 32767 (20.8 dB of headroom above the noise)
🎚️  Mic floor measured 528 (-0.8 dB vs the resumed 578) — measured at startup in this body (open≥661, strong≥1691)
```

**This is the steady state, and it is what a healthy unit should print every
boot after the first.** The absence of a second gain line is the signal that
nothing has changed; the floor line reports its own drift so a slow change in
the room is visible without being acted on.

| Line | What to check |
|---|---|
| `tone SNR` | Each channel against a **known** stimulus. This is the only measurement in the system that can call a mic dead, because it is the only one with a reference. Below ~+3 dB is deaf. |
| `body response` | Per-tone SNR — the acoustic signature of this enclosure. Recorded, not corrected. |
| `level` | The gain solve's inputs. `noise peak` filling the target is the one unfixable case. |
| `Mic gain` | One line (restore only) on a settled unit. A **second** line means the solve disagreed by more than 0.1 dB — normal after a body or room change, worth investigating if it recurs every boot. |
| `Mic floor` | Should land **300–600**, may reach ~750. Well outside that band means something upstream is wrong. |

**The one case gain cannot fix** is called out explicitly:

```
⚠️  the noise bed alone fills the target headroom — gain cannot fix this,
    the mic path itself is too noisy
```

Gain is a scalar. It moves signal and noise together, so it can fix a level
fault and can never fix an SNR fault. If this warning appears, the answer is
hardware.

## 3.1 The new body's acoustic signature

Two independent runs, per-tone SNR in dB:

| | 200 Hz | 400 Hz | 800 Hz | 1600 Hz | 3200 Hz | 6000 Hz | tilt lo/hi |
|---|---|---|---|---|---|---|---|
| Run 1 | +43 | +30 | +12 | +19 | +22 | +11 | +10.9 dB |
| Run 3 (boot) | +41 | +28 | +16 | +16 | +21 | +12 | +12.1 dB |

Repeatable to a couple of dB. The plastic case is **low-frequency dominant** —
+41 dB at 200 Hz against +12 dB at 6 kHz. Some of that is the speaker's own
response and the short internal path, but the consequence is the same either
way: the enclosure couples ADAM's own bass into its mics efficiently, and
carries consonant-band energy poorly.

This is recorded, not corrected. Correcting it would mean an EQ fitted to one
body, and `it has to be dynamic as we have to production ready` rules that out.
What the number is *for* is the gain solve (it is part of the peak being
measured) and for recognising a body change across units.

---

# Part 4 — The noise bed: characterised, not cured

`_micnoise.py` was written to answer *where the new body's noise is coming
from*. It changes no settings; it only measures, so it is safe to run on a live
unit between sessions:

```bash
cd ~/adam && ./venv/bin/python _micnoise.py
```

It distinguishes four candidate causes by signature, from the **raw 32-bit
words** before any filtering:

| Measurement | Result | Verdict |
|---|---|---|
| Bit occupancy, bits 0–7 | always zero | **24-in-32 aligned correctly.** Not a frame/alignment fault, not a floating data line. |
| Spectrum tilt (hi − lo) | −1.5 to +0.9 dB | **Flat / white.** Not sigma-delta noise shaping, which rises with frequency. |
| Tonality (peak − median) | 4.7–6.8 dB | **Smooth.** Not clock or switching pickup, which appears as discrete tones or a comb. |
| Block-RMS spread p95/p05 | 0.6–2.7 dB | **Steady.** Not a duty-cycled aggressor (Wi-Fi, camera link, switching regulator under load). |
| corr(L, R) | −0.008 | **Uncorrelated.** Not a common-mode source — not a shared clock, shared supply rail, or radiated pickup driving both mics together. |
| Level | ~−20.8 dBFS raw | ~62 dB above INMP441 datasheet self-noise. |

Every mechanism that is cheap to fix is ruled out. What remains is a flat,
steady, per-channel electrical noise bed about 62 dB above spec on **both**
channels — including the deaf RIGHT one, which reads a steady 3665–3870, nearly
the same as LEFT. A mic that cannot hear a chime but still produces a full noise
bed is producing that noise **electronically, not acoustically**.

**Status: open hardware finding.** The calibration works around it by giving up
gain until the noise fits under the ceiling, which is why the fix held. But the
SNR that costs is gone for good — the ~20 dB of headroom bought here is headroom
over *noise*, and a quieter front end would be worth more than any further
software change in this area.

---

# Part 5 — Absolute-constant audit

The gain change is −10.9 dB relative to `S32_SHIFT=16` and −16.9 dB relative to
the 15 the unit was running. Every constant compared against an absolute RMS had
to be re-checked, because the ratio-based gate is immune to a gain change and
these are not.

| Constant | Value | Where used | Verdict at the new floor (474–738) |
|---|---|---|---|
| `MIC_SILENCE_FLOOR` | 1800 | — | **Legacy.** Superseded by the adaptive gate (A1); not in the live decision path. Was being crossed continuously by pure noise at the old floor of 4013. |
| `MIC_LIVE_RMS_THRESHOLD` | 1200 | `session.py:470` | Fallback only, used when the learned floor is 0. Correct. |
| `MIC_LIVE_RMS_THRESHOLD × 0.5` | 600 | `session.py:581` | DOA trigger for neck tracking. At the old floor of 4013 this fired on **every chunk** of pure noise, spending a thread-pool call each time. At 578 it behaves. Strict improvement. |
| `MIC_OPEN_MIN` | 90 | gate floor clamp | Floor of 578 is far above it; no effect. Correct. |

**No constant needed changing.** That is the intended outcome, and it is a
consequence of the design rather than luck: the calibration *targets* the
operating point those constants were written for, so landing inside the
documented 300–600 band restores all of them at once. Compare the alternative —
at shift 15 with a floor of 4013, two of these four were being crossed by silence
and would each have needed re-tuning by hand, per unit.

This is the argument for calibrating to a documented target instead of simply
"turning the gain down until it stops clipping".

---

# Part 6 — What changed in the code

| File | Change |
|---|---|
| `audio_utils.py` | Replaced the fixed `>> S32_SHIFT` with `_mic_scale`, a runtime float divisor. Added `set_mic_scale(divisor, why, reset_floor=True)` — logs the change in dB, ignores changes under 0.1 dB, and resets the learned floor unless told not to. Added `_restore_mic_scale()`, which runs **at import** beside the channel and floor resumes. Factored the chain into `_mono16k_float()` so the calibrator can measure the real path. Added `s32_stereo_to_float_channels()` — both channels in int16 units, **without** the clip. |
| `config.py` | New section *MEASURED MIC GAIN (replaces hand-picking S32_SHIFT)* carrying the solve formula and the measured table, plus `MIC_CAL_GAIN_ENABLE`, `MIC_CAL_HEADROOM_DB` (9.0), `MIC_CAL_SHIFT_MIN` (13), `MIC_CAL_SHIFT_MAX` (20). |
| `mic_calibrate.py` | Added `_post_chain_peak()` (measures through the real chain, skips 3 chunks for filter settling) and `_solve_gain()` (the proportional solve, the clamp, the level report, the noise-fills-headroom warning). SNR analysis switched to the unclipped path. Persists `mic_scale`, `mic_shift`, `gain_why` to `.mic_cal.json`. Does **not** restore the gain — `audio_utils` owns that, so the diagnostics get it too. |
| `_micnoise.py` | **New.** Read-only noise diagnostic; see Part 4. |
| `~/adam/.env` *(Pi only)* | `MIC_S32_SHIFT=15` → `16`. Backed up to `.env.bak.20260929` first. The pre-calibration fallback should be the documented-safe value, so that a unit whose calibration fails comes up conservative rather than clipping. |

Persisted state after a good boot, `~/adam/.mic_cal.json`:

```json
{
  "median_snr_db": [18.64, -0.18],
  "live": [true, false],
  "dead": [false, true],
  "tilt_db": 12.11,
  "gate_floor": 577.82,
  "mic_scale": 230004.6,
  "mic_shift": 17.811,
  "gain_why": "chime peak set to 9 dB below full scale (shift 17.74 → 17.81)",
  "channel": "left"
}
```

---

# Part 7 — Verification performed

| Check | Result |
|---|---|
| Song barge-in regression (`barge_stall_test.py`) — the scale refactor touched two functions on the song path | **PASS**, re-run after every edit. Coupling converged to 1.500 (expect ~1.50); stale burst 0 false barges; post-recovery voice 8/10; normal-path voice 8/10; music-only false opens 0/300. |
| Full startup (`main.py`) on real hardware | **Clean.** Restore → chime → channel select → gain solve → floor measure, then into the Gemini session with no ALSA collision against `listen()`/`speaker()`. |
| Gain resumes for entry points that never call `calibrate()` | Verified by importing `audio_utils` alone: channel, gain and floor all resume, in that order. |
| Mic live at the new gain | `🎤 Mic active (RMS: 876 \| Room floor: 528)`, floor stable at 509–551 across windows. |
| Pi ↔ laptop parity | **31 of 31 files byte-for-byte identical**, verified by `md5sum *.py *.txt` on both sides. |
| Calibration repeatability | Four runs: −11.2 dB, +0.7 dB, −0.4 dB, then **no change** (under the 0.1 dB announce threshold). Converged. |
| Tone SNR trend across those runs | +20.8 → +18.6 → +21.1 dB on LEFT; RIGHT stays at −0.3 to +0.1 dB throughout. |

## 7.1 What parity does and does not cover

The Pi ↔ laptop parity requirement covers **`*.py` and `*.txt` — code only.**

The three state files are deliberately excluded and are now in `.gitignore`.
They are *measurements of one physical unit*: its noise floor, its gain, which
of its mics is alive. A floor measured in one body at one gain is meaningless in
another, so syncing them is not merely unnecessary — it is the failure mode.
Pulling one unit's `.mic_floor.json` onto a second unit would silently override
a correct calibration with a foreign one, which is exactly the units mismatch
described in §2.4, arriving through version control instead of through a skipped
restore.

`.mic_floor.json` was tracked historically and needs untracking once:

```bash
git rm --cached "MP-MC codes/pi/adam/.mic_floor.json"
```

## 7.1 Still to verify

- **Real speech.** The gate opens and the floor is correct, but nobody has
  spoken to the calibrated unit yet and confirmed Gemini transcribes cleanly.
  This is the last unknown in the mishearing fix and it requires a person in the
  room. Everything measurable has been measured.
- **A full real song end-to-end** — no popping, no false auto-stop, a real voice
  still stops it, no `[arecord] overrun`. The two fixes are simulation-verified
  only; see `full_duplex_and_song_bargein.md` Part 5.

---

# Part 8 — Decision register

| # | Decision | Reason |
|---|---|---|
| 1 | Solve the gain, don't pick it | `S32_SHIFT` has been hand-set to 13, 14, 15 and 16 — once per hardware change, always after failure. The chain is linear and the divisor is a scalar, so the right value is one multiplication away from a measurement. |
| 2 | Measure at every boot, not once at manufacture | The body, the room and the noise bed all change. A factory calibration would be a constant again, just with a nicer provenance. |
| 3 | Target 9 dB below full scale | Lands the floor inside `config.py`'s documented 300–600 band, keeping every absolute threshold in the system valid. Noise peaks end ~20 dB down. More headroom buys nothing against a quantisation floor 70 dB below the real noise. |
| 4 | Measure the gain peak through the **real** chain | The 6.8 kHz anti-alias low-pass alone removes several dB from a white bed; scaling a raw int32 peak would overstate the level and under-gain the unit. |
| 5 | Measure SNR **unclipped** | The chime peaked at 71 693 against a 32767 ceiling. Per-tone SNRs were being read off clipping harmonics. A calibration must not measure through the distortion it exists to detect. |
| 6 | Sound all six tones simultaneously | The write-to-hear lag is 541–941 ms and drifts. Tone-bin detection locates the burst itself and needs no lag estimate. |
| 7 | Resume the divisor at `audio_utils` import, not in `calibrate()` | A learned floor is only meaningful at the divisor it was measured at. Putting the restore in `calibrate()` fixes `main.py` and leaves every diagnostic measuring at the wrong gain — and those tools exist specifically to describe the shipped pipeline. One place, beside the channel and floor resumes, so the three cannot drift apart. |
| 8 | Record the body response; do not equalise it | A fitted EQ is a per-body constant. *"it has to be dynamic as we have to production ready."* |
| 9 | Set the `.env` fallback to the documented 16 | A unit whose calibration fails should come up conservative, not clipping. |
| 10 | Don't re-tune the absolute constants | The calibration targets the operating point they were written for. Re-tuning them would couple them to this unit and undo that. |
| 11 | Leave `MIC_NR=0` | The fault was level, not noise shape. The suppressor addresses neither, and enabling it would have masked the clipping without fixing it. |

---

# Appendix — env overrides added

| Variable | Default | Effect |
|---|---|---|
| `MIC_CAL_GAIN_ENABLE` | `1` | Set `0` to calibrate the channel and floor but leave the gain alone. |
| `MIC_CAL_HEADROOM_DB` | `9.0` | Where the chime peak is placed below full scale. Larger = quieter and safer. |
| `MIC_CAL_SHIFT_MIN` | `13.0` | Loudest the solve may go (divisor `2^13`). |
| `MIC_CAL_SHIFT_MAX` | `20.0` | Quietest the solve may go (divisor `2^20`). |
| `MIC_S32_SHIFT` | `16` | Pre-calibration fallback only. Once calibration has run, the solved divisor supersedes it. **Do not hand-tune this** — see A5. |
