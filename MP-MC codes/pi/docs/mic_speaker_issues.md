# ADAM v40 — Mic & Speaker Fault Guide

What broke, why it broke, and either where it is fixed in code or what you
have to change yourself.

Written against the split package in `pi/adam/` with `main.py` as the
entrypoint — not the legacy monolith, and not the wiring in `setup.md` where
the two disagree. **The code is the source of truth.**

Nineteen faults are covered: thirteen that stop ADAM from HEARING or
UNDERSTANDING you (Part A) and six that make ADAM's VOICE sound wrong
(Part B). Twelve are fixed in code and need nothing from you. Four are hardware
and cannot be fixed in software at all, and one of those is a full stop: the
microphone noise floor is 61 dB above its own datasheet, which makes everything
else in Part A secondary. One is OS config. One needs no change because the
guard was already there. One is a misdiagnosis, documented here so nobody
"fixes" it later and makes things worse.

If you only read one section, read **A12**. It is the root cause of "I am
speaking and ADAM cannot listen — only a few times, and only if I shout." It is
the only fault in this document that a *relative* measurement can never find,
and the boot log looked healthy right up until the absolute one was taken.

If A12 comes back clean on your unit, read **A10** — the recogniser being
handed audio it cannot resolve is what "ADAM mis-hears everything" means once
the hardware is ruled out.

## Quick triage

| What you observe | Go to |
|---|---|
| **Speaks normally, ADAM hears nothing; has to shout at the mic** | **A12**, then A11 |
| **Sometimes hears you fine, sometimes mishears — no pattern** | **A12**, A8, A11 |
| **"Hello ADAM" → "Hello madam"; vowels fine, consonants wrong** | **A12**, A8, A11, then A10, A9 |
| **Direction sensing / neck tracking never works** | **A11** |
| **Boot log says the mic is fine but it demonstrably is not** | **A12** (relative metrics cannot see this) |
| ADAM talks (idle nudge fires) but never answers you | A1, A3 |
| ADAM stops hearing you a second or two after it finishes talking | A2 |
| ADAM answers only after you repeat yourself three times | A1 |
| ADAM hears you, then hangs forever without replying | A4 |
| Transcript comes back in a language nobody in the room speaks | **A8**, A6 |
| ADAM replies in Portuguese / Spanish / Korean and keeps doing it | A6 |
| ADAM answers half your sentence, then answers the other half | A7 |
| Speech is muffled or distorted only on the loudest syllables | A5 (not clipping — see A8) |
| Every word comes back as a similar-sounding wrong word | **A8**, A10, A9 |
| "ADAM" → "मैडम", "code" → "कोर्स", consonants swapped | **A8**, A10, A9 |
| Gate opens, `sent` is healthy, reply is confidently about nothing | A10 |
| **Mic gain line prints a slightly different value every boot** | **A13** |
| **Floor resets and relearns on every boot for no reason** | **A13** |
| **Hears you, but drops a few words per sentence — starts, ends, asides** | **A14** |
| **Deaf for the first minutes after boot, then suddenly fine** | **A15** |
| **Boot log says "calibration heard no tone" but the speaker clearly played** | **A15** |
| **"Mic path CHANGED" on every boot with nothing touched** | **A16**, then A13 |
| Voice level jumps around with no pattern | B1 |
| Voice is crackly or glitchy all the time | B2, B5 |
| Voice turns into loud buzz and stays buzzing | B3 |
| **Song is beautiful through `aplay` but distorted through ADAM** | **B6** |
| **Song has a click / dropout about twice a second** | **B6a** |
| Everything lags while a song plays | B4 |

**Start with A12 for any "it cannot hear me" complaint.** A microphone whose
electrical noise floor is equivalent to 94 dB SPL cannot be rescued by any gate,
gain or filter — speech at a normal 55–65 dB SPL is 30 dB below its own noise.
The boot log now measures and prints this on every start; read the last line of
the mic calibration block before investigating anything else.

**Then A11 for anything about direction, or about misheard words on the unit
that A12 has cleared.** The dead right mic costs 11.7 dB broadband and up to
21 dB in the consonant band — more than every other mic fix in this document
put together. Check the boot log for the `🎙️ Mic channel →` line.

**For song playback, go straight to B6.** It is the only fault in Part B that
is byte-verifiable, and the test listed there fails on the old code.

**Before trusting any mic measurement**, read the last bullet of "Things that
look like faults and are not": the room bed on this unit moves by 10 dB between
captures, so two runs minutes apart prove nothing. Use `mic_modes.py`, which
scores every candidate on one identical capture.

### The diagnostic scripts, and which question each answers

All live in `~/adam/` and use the real `config.py`/`audio_utils.py`, so their
numbers describe the shipped pipeline rather than a re-implementation of it.

| script | answers |
|---|---|
| `mic_probe.py [s]` | ADC saturation, energy distribution, in-band SNR, what the gate makes of it |
| `mic_modes.py [s]` | scores `left`/`right`/`mix` on **one identical capture**, plus what the selector decides |
| `mic_cause.py [s]` | is the bed acoustic, conducted from the CPU rail, or a dead channel? (idle / loaded / sound) |
| `mic_noise_probe.py [s]` | **is the noise floor itself within spec?** Absolute dB SPL against the INMP441 datasheet, tonality, narrowband peaks, octave-band power, and what three candidate band limits would keep |
| `mic_bits.py [s]` | is the I2S stream real 24-bit mic data or misread bits? |
| `mic_watch.py [s]` | per-channel level and spectral tilt over time — intermittent connection vs drifting room |
| `mic_geom.py` | inter-mic delay, L↔R coherence, per-channel response to sound (needs `mic_ab.py` captures first) |
| `_micnoise.py` | **what** the noise bed is: bit alignment, spectrum tilt, tonality, burstiness — distinguishes misalignment / sigma-delta shaping / clock pickup / duty-cycled aggressor |
| `nr_bench.py` | suppressor candidates against a known clean reference, with null controls |
| `song_stream_test.py` | **is ADAM's song playback byte-exact?** Runs the real song task against a bounded fake pipe drained at realtime, with a bursty competing writer, and diffs the result against the WAV. Two scenarios: song alone (B6a) and song + contention (B6b). Each fix revert-proven separately |

**Run `mic_noise_probe.py` first on any unit with a hearing complaint.** It is
the only script here that produces an *absolute* number rather than a relative
one, and A12 is invisible to every other row in this table.

**`song_stream_test.py` is the only test in this directory that can pass or
fail.** Everything else reports numbers for you to interpret; this one diffs
bytes and exits non-zero on a regression. It also refuses to report a hollow
PASS: if the pipe never hands out a short write, the contention scenario could
not have reproduced anything and the test fails itself with `SCENARIO VACUOUS`.

`mic_calibrate.py` is not a diagnostic — it runs automatically at every startup
and is what sets the channel, the gain and the floor. See
`startup_mic_calibration.md`.


## Reading the log

Every line quoted below is one ADAM actually prints. Follow it with
`journalctl -u adam -f` on the Pi, or run `main.py` in the venv by hand.

The mic stats line is the one to watch. It prints every `MIC_STATS_S`:

```
📊 Mic 20s: p50 1150 p90 2100 p99 2900 max 3400 | open≥2470 hold≥1730 | floor 1082 flat 0.31/0.35 lohi 1.90 shp 40% | opens 4 sent 118 | blocked 2 | nr -10.9dB | shut
```

- `p50/p90/p99/max` — post-filter int16 RMS distribution over the window
- `open≥` / `hold≥` — the two thresholds derived from the learned floor
- `floor` — learned noise floor; a trailing `?` means not yet converged
- `flat 0.31/0.35` — this chunk's spectral flatness / the live threshold
- `lohi` — low-band vs high-band energy ratio
- `shp` — fraction of the recent window that passed the shape test
- `opens` — gate openings in the window; `sent` — chunks sent to Gemini
- `blocked` — onset attempts that decayed without reaching quorum
- `nr` — mean noise-suppressor gain across 300–3400 Hz on the last frame
  (A10). Absent when `MIC_NR=0`. Near `-12.0dB` in silence (it is sitting on
  the floor), typically `-2` to `-6dB` mid-syllable
- mode — `shut` / `OPEN` / `IDLE` / `SONG`, plus `+AMP` when the floor
  estimate is frozen because the playback device is open

**Every number on that line except `nr` is measured on the RAW filtered
audio, before noise suppression.** The gate is deliberately fed the unprocessed
signal (A10), so `floor`, `p50`, `open≥` mean exactly what they meant before
suppression existed and old log lines stay comparable.

---

# Part A — ADAM cannot hear you

## A1. The VAD gate rejected real speech — FIXED IN CODE

**Symptom.** "I am constantly speaking but ADAM is not responding." The idle
nudge fires and ADAM talks, so the process is alive and the Gemini link is up.
On the stats line, `opens 0` while `blocked` climbs. Sometimes it takes three
attempts before one gets through.

**Root cause.** The gate opened only after `MIC_VAD_ONSET_CHUNKS` chunks in a
row passed BOTH the level test and the spectral-shape test. One chunk is
33.3 ms, so the old default of 5 demanded 167 ms of *uninterrupted* voiced,
tonal energy. Real speech does not contain that:

- Unvoiced consonants — `s`, `f`, `sh`, `t`, `k` — are broadband. Their
  spectral flatness is near 1.0, so they fail the shape test by design; that
  test exists precisely because broadband energy is what room noise looks
  like. At this SNR, fricatives and noise are genuinely inseparable.
- Stop closures — the silent beat inside `t`, `k`, `p`, `d` — run 50–80 ms,
  which is 2–3 chunks below the level threshold.

Under a *consecutive* rule either of those resets the counter to zero.
"Hey ADAM" at conversational volume has no 5 clean consecutive chunks in it,
so the counter never reached 5 and the gate never opened.

**Why the earlier "validation" missed it (recorded so it isn't repeated).**
An earlier parameter sweep on this codebase concluded that 5 was safe at
0.0 false opens/min. Its positive examples were `synth_vowel()` output —
synthetic *sustained vowels*, which have no fricatives and no stop closures.
It measured the one signal a consecutive rule handles perfectly. The 0.0
false-open figure was also at the resolution floor of a 25 s sample
(±2.4/min), so it was not evidence of much either way.

**Fix.** The run became a **quorum**: M passing chunks anywhere inside a
sliding window of N. Voiced segments now carry the decision and the
consonants between them cost nothing.

- `pi/adam/session.py` — `_onset_win` deque replaces the old `_onset_run`
  counter; a failing chunk appends `0` instead of clearing the count.
- `pi/adam/config.py` — `MIC_VAD_ONSET_CHUNKS` 5 → **3**, new
  `MIC_VAD_ONSET_WINDOW` = **6**. So: 3 passing chunks (100 ms of voiced
  energy) anywhere in 200 ms.

**Does the quorum let noise in?** Measured on this room: a single chunk
passes the level test 12.9% of the time and the shape test 4.3% of the time,
so P(both) ≈ 0.55%. P(≥3 passes in any 6-chunk window) ≈ 3×10⁻⁶, which at
30 chunks/s is on the order of **0.01 false opens per minute** — one every
couple of hours. The consecutive rule's measured 0.0/min was not meaningfully
better; it was the same number inside the noise.

**What was deliberately NOT done.** Loosening the shape test
(`MIC_SHAPE_FLAT_MAX=0.45`, `MIC_SHAPE_RATIO_MIN=0.40`) also makes the gate
open, but it re-admits the very room noise the test was built to reject —
webrtcvad called **100.0%** of this room's noise "speech" at aggressiveness
0/1/2 and 98.6% at 3, which is why it is off by default and why flatness is
the only working discriminator here. Loosening it is an emergency lever, not
the fix. The quorum addresses the actual defect: a *duration* rule was being
applied to a signal that is not continuous.

**Also added: the flatness threshold now learns the room.** A fixed 0.35 was
measured in one room. `AdaptiveGate.shape_ok(..., learn_noise=True)` collects
flatness during confirmed-quiet chunks and sets
`flat_max = clamp(p5(noise flatness) × 0.95, MIC_SHAPE_FLAT_MAX, CEIL)`.

The rule is **one-sided — it can only loosen** past the 0.35 baseline, never
tighten below it. A two-sided rule in a tonal room (a fan, a fridge, a hum)
would keep measuring low flatness and walk the threshold down until nothing
passed at all, which is the failure this whole section is about. Live value is
the second number in `flat 0.31/0.35`.

**Env overrides.**

| Variable | Default | Effect |
|---|---|---|
| `MIC_VAD_ONSET_CHUNKS` | 3 | passes needed inside the window |
| `MIC_VAD_ONSET_WINDOW` | 6 | window length in chunks (33.3 ms each) |
| `MIC_SHAPE_ADAPT` | 1 | set 0 to freeze `flat_max` at the baseline |
| `MIC_SHAPE_FLAT_CEIL` | 0.70 | how far learning may loosen |
| `MIC_SHAPE_FLAT_MARGIN` | 0.95 | safety factor under measured p5 |
| `MIC_SHAPE_FLAT_PCTL` | 5 | percentile of the noise-flatness window |
| `MIC_VAD_PREROLL_S` | 0.8 | audio prepended before the open |

`MIC_VAD_ONSET_CHUNKS=1` disables the duration test entirely. Try that only
to confirm the gate is the problem — then put it back.

Onset latency is **not** paid by the user: `MIC_VAD_PREROLL_S` (0.8 s = 24
chunks) prepends buffered audio ahead of the open, so the 100 ms the quorum
spends deciding is already in the stream Gemini receives.

**Verified live on the Pi, 2026-09-05.** The very first open after deploying
read:

```
🎙️  Speech detected (RMS 4051 ≥ 1811, 3/4 chunks)
```

`3/4` is the quorum doing exactly what it was built for — it opened on the
third passing chunk out of the first four, which a 5-consecutive rule could
not have done. Several full conversations followed (Hindi and English, both
transcribed correctly). Over ~2.5 minutes: `opens` matched the number of times
someone actually spoke, quiet windows showed `opens 0 blocked 0`, and no
window showed a false open.

One open read `RMS 1434 ≥ 1409` with `shp 27%` — a marginal-level chunk
admitted because the *shape* vote carried it. So both paths into the gate are
live, not just the loud-speech bypass.

---

## A2. I2S capture wedged, delivering digital silence — FIXED IN CODE

**Symptom.** ADAM stops hearing you a second or two after it finishes
speaking. `arecord` is still running and still delivering bytes at the right
rate, but every sample is exactly zero, so RMS sits at 0 and the gate can
never open. Nothing looks broken.

**Root cause.** One I2S device (`sndrpigooglevoi`) serves both capture and
playback, so they share a clock domain. Closing the playback side can leave
the capture DMA wedged — it keeps producing buffers, just all-zero ones.

**Fix.** A watchdog already existed (`MIC_DEAD_STREAM_S` = 3.0 s of exact
digital silence → restart `arecord`). It was made **two-fuse**, because 3 s is
far too long to wait in the one window where the cause is known:

| Situation | Silence tolerated before restart |
|---|---|
| Within `MIC_DEAD_AFTER_PLAY_WINDOW_S` (3.0 s) of a playback close | `MIC_DEAD_AFTER_PLAY_S` = **0.7 s** |
| Any other time | `MIC_DEAD_STREAM_S` = **3.0 s** |

The seconds right after ADAM stops talking are exactly when the user replies.
Donating them to a stream of zeros is what made ADAM feel deaf specifically
in conversation. Implemented in `pi/adam/session.py` (`_dead_limit_amp`,
gated on `amp_quiet_t`).

**Diagnostic.** The restart is loud about itself, and says whether a playback
close is implicated:

```
⚠️  Capture DEAD — 0.7s of exact digital silence from arecord (voiceHAT I2S capture wedged, playback closed 0.4s ago). Restarting arecord.
```

**The other lever — and it is now the default.** `SPEAKER_IDLE_CLOSE_S=0`
never closes the playback device, so the wedge cannot happen at all.

This section used to argue against it on a **measured +4.1 dB of amp hiss** on
the mic floor (floor p50 1082 → 1726 with `aplay` merely open on silence). That
measurement did not survive re-testing: a controlled amp-open vs amp-closed A/B
gives **−0.1 dB broadband, with every band inside ±0.2 dB**. The original
1082 → 1726 shift was the room and the dead right channel moving between two
captures taken minutes apart, not the amplifier — exactly the trap that made
the A8 diagnosis take so long. `mic_cause.py` also shows the floor is unmoved
by full CPU load (+0.2 dB), so there is no conducted-noise mechanism here at
all.

With no measured cost, holding the device open is strictly better: it removes
the wedge, and it removes one `aplay` spawn (~50-100 ms on a Pi Zero 2 W) from
the front of every single reply. `SPEAKER_IDLE_CLOSE_S=0` is the shipped
default in `config.py` and is set explicitly in `.env`.

---

## A3. Trapped in idle mode — FIXED IN CODE

**Symptom.** ADAM's idle nudge fires and ADAM talks, but ADAM never hears
anything you say. The stats line shows mode `IDLE`.

**Root cause — two separate bugs, both real.**

1. `ENABLE_IDLE` was a hardcoded literal `True` in `config.py`. Setting
   `ENABLE_IDLE=0` in `.env` had **no effect whatsoever** — the documented
   escape hatch did not exist.
2. Even where the flag was read, it was consulted at only **one** of the two
   places that enter idle: the inactivity-timeout path. Gemini's own
   `enter_idle_mode` tool call bypassed it entirely, so ADAM could still put
   itself to sleep with the flag off.

**Fix.**

- `pi/adam/config.py` — `ENABLE_IDLE` and `IDLE_TIMEOUT_S` are now read from
  the environment, so `.env` actually controls them.
- `pi/adam/session.py` — the Gemini tool path now checks the flag too.

**Diagnostic.** With `ENABLE_IDLE=0`, a blocked tool call announces itself:

```
🙉 enter_idle_mode ignored — ENABLE_IDLE=0
```

If you see mode `IDLE` on the stats line and no such message, you are on the
old build.

**Confirmed live, and it is not a rare corner.** On 2026-09-05 this was the
actual reason ADAM had gone deaf — nothing to do with the mic:

```
13:40:42  🔇 enter_idle_mode called — will go silent
13:40:42  🔇 Idle mode active (voice request) — servos centered; say "adam" to wake, Touch3, or wait 10 min
```

That is **one second into the session**, on the very first transcript. Gemini
decided a garbled opening line was a request to go to sleep. Every stats line
afterwards read `sent 0` while still showing `opens 1` — the gate was working
perfectly and the audio was being thrown away, which is exactly what makes this
fault look like a microphone problem. The previous service instance had been
sitting in the same state since 13:39:16.

Two things make it worse than a normal tool call: there is **no Touch3 on this
unit** (Part E — the ESP32-CAM link is dead), and the wake word only listens on
the Vosk offline path, so a mis-transcribed wake attempt cannot get you out
either. The only reliable exits were a 10-minute timeout or a restart.

**Set on this unit.** `ENABLE_IDLE=0` is now in `~/adam/.env` with a dated
comment. Verified over a full conversational run: 56 stats windows, **zero**
`IDLE` modes, zero `enter_idle_mode` calls, replies on every turn.

**Env overrides.** `ENABLE_IDLE=0` (never sleep), `IDLE_TIMEOUT_S` (seconds
of silence before the first nudge, default 90), `IDLE_MAX_S`.

---

## A4. Gate stuck open, so Gemini never gets ActivityEnd — ALREADY GUARDED

**Symptom.** ADAM clearly hears you — `🎙️ Speech detected` appears, chunks are
sent — and then nothing. No reply, indefinitely.

**Root cause.** The Live session runs in **manual activity detection** mode.
The gate's falling edge is the *only* thing that sends `activity_end`. If the
gate latches open, Gemini is still waiting for the end of your turn and will
never answer. Worse, the floor trackers deliberately freeze while the gate is
open — speech must not teach them what silence sounds like — so a latched gate
is self-sustaining: no update, no threshold movement, no escape.

**Status: no change needed.** Three bounds were already in place, and they are
time-based precisely because the level-based estimators are frozen:

| Guard | Default | Behaviour |
|---|---|---|
| `MIC_VAD_HANGOVER_S` | 1.0 s | normal close after speech stops (was 0.6 — see A7) |
| `MIC_VAD_MAX_OPEN_S` | 15 s | soft: force shut, resync both estimates *upward* to the level that fooled them, so the room's louder floor is learned in one step |
| `MIC_VAD_ABS_MAX_OPEN_S` | 45 s | hard ceiling regardless |

The HOLD condition also uses the shape *fraction* (`MIC_SHAPE_HOLD_FRAC`,
0.40) rather than a single chunk, which is what lets the gate close on a noise
bed instead of being held open by it.

**Cleanup done here.** `MIC_VAD_RELEASE_RATIO` was defined **twice** in
`config.py` with identical values, the first copy stranded in the middle of a
comment block that then referred to "the three constants below". The stranded
duplicate was deleted. No behaviour change — but a future edit to the wrong
copy would have been silently ignored.

**Verified live.** Every open in the 2026-09-05 run was followed by
`🤫 Speech ended`, and every turn produced a reply. No latch fired.

---

## A5. "Raise MIC_S32_SHIFT for more mic gain" — DO NOT DO THIS

This one is in the guide to stop it from being applied later. It is the one
suggested fix that would make things measurably worse.

**The proposal.** Raise `MIC_S32_SHIFT` (default 15, e.g. → 14) to add digital
gain so quiet speech clears the threshold. One shift bit is ×2, i.e. +6 dB.
(The env var is `MIC_S32_SHIFT`; the Python constant it sets is `S32_SHIFT`.)

**Why it does not help.** The gate is **ratio-based against a learned floor**.
Both the signal and the floor are scaled by the same digital gain, so the ratio
is unchanged and the gate behaves identically. Digital gain moves both numbers
on the stats line and changes nothing about the decision.

**Why it actively hurts.** `MIC_S32_SHIFT=14` is +6 dB. Measured headroom on
this hardware, on **room noise alone with nobody speaking**:

| Run | Left | Right |
|---|---|---|
| Earlier session | **−1.0 dBFS** | −7.1 dBFS |
| 2026-09-05 12:46 | **−3.7 dBFS** | −7.0 dBFS |
| 2026-09-05 13:40 | −7.0 dBFS | −11.2 dBFS |
| 2026-09-05 13:43 | −6.9 dBFS | −10.9 dBFS |

Note how much this moves **between boots of the same unit** — 6 dB on the left
channel across four measurements of the same room. That is consistent with the
electrical/structure-borne fault described in Part C rather than with an
acoustic level, and it is a second reason not to add fixed digital gain: you
would be sizing it against a number that is not stable.

Adding 6 dB to a channel already within 1–4 dB of full scale clips it. Clipping
is broadband, which drives spectral flatness toward 1.0 — i.e. it makes real
speech look *more* like noise to the shape test, tightening the very gate this
was meant to loosen.

**Status: no code change.** If a genuinely quieter unit ever needs adjusting,
check the headroom line first:

```
🎚️  Mic headroom L -3.7 dBFS / R -7.0 dBFS, saturated samples L 0 / R 0 (raw, pre-filter) → speech path uses MIX
```

Only consider a change if both channels show real headroom **and** `saturated
samples` is 0. On this unit the correct direction would be *down*, not up.

### Update 2026-09-29 — the gain is now measured, and this unit *was* clipping

Everything above is still true and this section still stands: **do not hand-pick
`MIC_S32_SHIFT`.** What changed is that there is now a principled way to change
the gain, so the answer is no longer "leave it alone" but "let it measure
itself". Full write-up in `startup_mic_calibration.md`.

Two things this section got right and one it got wrong:

- **Right:** *"on this unit the correct direction would be down, not up."* It is.
  The solved divisor is ÷230 005, equivalent to shift **17.8** — nearly two bits
  *quieter* than the 16 this guide assumed, and almost three quieter than the 15
  the unit was actually shipped running.
- **Right:** *"you would be sizing it against a number that is not stable."* The
  floor moved 1082 → ~7900 raw when the body changed. That is exactly why the
  divisor is now re-solved at every boot rather than written down anywhere.
- **Wrong, by omission:** this section argued digital gain cannot help *because
  the gate is ratio-based*, which is correct about the **gate** and silent about
  the **ceiling**. A wrong divisor does not fool the gate — it clips the audio
  before the gate ever sees it. The unit was running `MIC_S32_SHIFT=15` with
  **3.9 dB of headroom above silence** and 18.3 % of samples flattened, and the
  gate's telemetry looked healthy throughout, because a clipped measurement
  cannot report that it clipped.

So: a gain change cannot loosen the gate, and a gain *error* can absolutely
destroy intelligibility. Both are true at once.

**What to do now.** Nothing by hand. Startup calibration plays a chime, measures
the peak through the real mic chain, and solves the divisor that puts it 9 dB
below full scale. Check the boot log:

```
level: chime peak 11626, noise peak 3214 of 32767 (20.2 dB of headroom above the noise)
🔊 Mic gain -0.4 dB (÷219108 → ÷230005) — chime peak set to 9 dB below full scale
🎚️  Mic floor measured 578 — measured at startup in this body (open≥722, strong≥1849)
```

A floor of **300–600** (up to ~750) means the gain is right. A floor in the
thousands means it is not, and the fix is to let calibration run — not to edit
`.env`. `MIC_S32_SHIFT` is now only the pre-calibration fallback; leave it at the
documented **16** so a unit whose calibration fails comes up conservative rather
than clipping.

---

## A6. Speech transcribed as a language nobody in the room speaks — FIXED IN CODE

**Symptom.** You speak Hindi or Hinglish. The transcript comes back as
Portuguese, Spanish, Korean or Japanese, and because ADAM is instructed to
reply in the language it just heard, it *answers* in that language and the
conversation derails. Observed transcripts: `Tô com não, não` for "nahi nahi",
`peléan` for a Hindi fragment.

**Root cause — two layers, and both had to be fixed.**

1. **The Live config had no language hint at all.** `session.py` built
   `input_audio_transcription=types.AudioTranscriptionConfig()` — empty. The
   SDK is explicit about what that means: `language_codes` is
   `Optional[list[str]]`, and *"if omitted or empty, defaults to automatic
   language detection."* So every fragment was scored against 100+ languages
   with no prior. Given a 1-second clipped fragment, "nahi nahi" really is
   closer to Portuguese "não não" than to anything else in that space — the
   model was not malfunctioning, it was doing exactly what it was configured to
   do. (`language_auto`, `language_hints` and `adaptation_phrases` are
   deprecated in `google-genai`; `SpeechConfig.language_code` is output speech
   only and does not constrain recognition.)
2. **`SystemPrompt.txt` then locked the error in.** The LANGUAGE rule says to
   reply in the exact language of the user's most recent message. One
   mis-transcript therefore produced one foreign reply, which entered the
   conversation history as evidence that the user speaks that language.

**Fix.**

- `pi/adam/config.py` — new `STT_LANGUAGE_CODES`, default `hi-IN,en-IN`:

  ```python
  STT_LANGUAGE_CODES = [c.strip() for c in
                        os.getenv("STT_LANGUAGE_CODES", "hi-IN,en-IN").split(",")
                        if c.strip()]
  ```

- `pi/adam/session.py` — passed into the Live config, with `or None` so that
  clearing the variable restores auto-detection rather than sending an empty
  list:

  ```python
  input_audio_transcription=types.AudioTranscriptionConfig(
      language_codes=STT_LANGUAGE_CODES or None),
  ```

- `pi/adam/SystemPrompt.txt` — the LANGUAGE section gained an explicit
  exception: a turn in a language that has **not appeared earlier in this
  conversation**, and that is short, garbled or nonsensical in context, is to
  be treated as a mis-hearing of the language already being spoken, not as a
  language switch. A real switch from this user is fluent, in context, and
  sustained over more than one turn. ADAM must never answer in the
  mis-detected language and must never comment on the mis-detection.

**Verified live.** The exact phrase that used to come back as Portuguese now
comes back as Hindi:

```
🗣️  You: नहीं, नहीं, सर। अभी तो नहीं हो पाएगी। लेकिन
```

**What this does and does not buy you.** `language_codes` is a *bias*, not a
hard lock — a turn can still be transcribed outside the list. In the verified
run a Korean question was transcribed as Korean, and correctly so, because the
user was genuinely asking about a Korean word. That is the wanted behaviour;
the SystemPrompt rule, not the config, is what stops a *spurious* one from
hijacking the reply language. If you sell into a different market, set
`STT_LANGUAGE_CODES` to that market's languages — keep the list short, since
every extra language widens the space a garbled fragment can land in.

**Env override.** `STT_LANGUAGE_CODES=hi-IN,en-IN` (comma-separated BCP-47;
empty string restores full auto-detection).

---

## A7. Natural pauses chopped one sentence into several turns — FIXED IN CODE

**Symptom.** You say one sentence with a comma in it. ADAM answers the first
half, and while it is answering you are still finishing — so it answers the
second half separately, out of context. Or: you speak three times before you
get one combined answer, and it arrives late.

**Root cause.** `MIC_VAD_HANGOVER_S` was **0.6 s**. Natural clause pauses in
conversational speech are **0.5–0.8 s**, so an ordinary comma closed the gate.
Under manual activity detection the gate's falling edge *is* `activity_end`, so
closing early does not merely trim audio — it commits a turn. Each fragment
then arrives at the recogniser short and contextless, which is also what fed
A6: short fragments are exactly what language auto-detection gets wrong.

**Fix.** `MIC_VAD_HANGOVER_S` default **0.6 → 1.0** in `config.py`.

**The cost, stated plainly.** This adds **+0.4 s** to every reply, because the
gate must observe a full second of quiet before it tells Gemini the turn is
over. That is a real latency regression and it is deliberate: one coherent
answer 0.4 s later beats two wrong answers sooner. If a deployment values
snappiness over sentence integrity, `MIC_VAD_HANGOVER_S=0.7` is the shortest
value worth trying — below that you are back to cutting commas.

Do not compensate by lowering `MIC_VAD_MAX_OPEN_S`; that guard exists for
latched gates (A4) and has nothing to do with turn-taking.

**Env override.** `MIC_VAD_HANGOVER_S=1.0`.

---

## A8. `MIC_CHANNEL=auto` picked the channel on the wrong evidence — FIXED IN CODE

**Symptom.** "Sometimes ADAM listens properly, sometimes it mishears." Vowels
survive, consonants do not: "Hello ADAM" comes back "Hello madam", and the log
fills with phantom transcripts in languages nobody in the room speaks. The log
always says `speech path uses MIX`.

**Root cause — two layers.**

*Layer 1, the criterion.* `_mic_ch_calibrate()` selected on ADC **saturation**,
measured over `_MIC_CH_CAL_CHUNKS = 30` chunks (exactly 1.0 s of boot silence),
then latched for the session. Clipping is impossible by construction during
boot silence, so the counters were always `L 0 / R 0` and `auto` was a synonym
for `mix` on every unit ever shipped. Worse, saturation is the wrong question
entirely: measured on this unit there is **7.4 dB of headroom and 0 saturated
samples in 12 s**, so the criterion can never fire.

*Layer 2, the actual fault.* The **right microphone is deaf** (A11). `mix`
averages it into the live left one, which is what destroys the consonants.

**Measurement.** `mic_cause.py` — three phases in one process, so nothing can
drift between them:

| phase | L | R |
|---|---|---|
| idle, silence | −36.1 dBFS | −24.8 dBFS |
| silence, 4 cores spinning | −38.6 | −24.5 |
| broadband noise from ADAM's own speaker | **−18.6** | **−23.3** |

L gains **+19 to +27 dB in every band** when there is sound in the room. R
gains **+1.0 to +1.4 dB** — nothing. R is also unmoved by CPU load (+0.2 dB),
so this is not power-rail ripple, and `mic_bits.py` shows the low 8 bits of
every 32-bit word are exactly zero on both channels (1 distinct low byte out of
256), so the I2S link is correctly aligned and this is not a misread data line
either. R is simply a dead channel emitting a fixed white hiss, **11.4 dB
louder than the live mic**, uncorrelated with L at the estimator floor.

**Why `mix` is so expensive.** Averaging halves the signal and adds the dead
channel's noise, so SNR = s/(n_L+n_R) = s/(14.8·n_L):

| band | 100-300 | 300-1k | 1k-2k | 2k-3.4k | 3.4-5k | 5-8k |
|---|---|---|---|---|---|---|
| thrown away by `mix` | +4.6 | +7.7 | +18.3 | +21.1 | +21.2 | +21.4 dB |

**11.7 dB broadband, up to 21 dB in the consonant band** — worst exactly where
speech intelligibility lives, because R's noise is white while L's is
LF-weighted. For scale, the best suppressor benchmarked (dual-mic coherence
Wiener) buys 9 dB and the shipped WOLA one buys 3.4 dB. **The channel choice
dominates every other mic fix in this document.** And it explains the
intermittency: the result sits right at the decision threshold, so loud or
close speech clears it and normal speech does not.

**Why the old doc said right-only was better.** The previous text here argued
right-only is "5.4 dB worse in band than the mix (post-filter noise floor p50
1498 vs 804)". That number is real and it is exactly what a dead, noisy R
measures — **L alone was never measured.** The conclusion was drawn from a
comparison that omitted the only good channel.

**Fix.** `_MicChannelLiveness` in `audio_utils.py` selects on *acoustic
responsiveness* instead. Per channel it keeps the **first-difference RMS**
(`d = np.diff(ch); rms = sqrt(mean(d·d))` — one vector op, removes DC entirely
and emphasises 2-8 kHz) in a rolling `MIC_CH_WINDOW_S` (90 s) deque, and every
`MIC_CH_DECIDE_EVERY_S` (2 s) computes that channel's own dynamic range
`20·log10(p99/p20)`:

- **live** when the range clears `MIC_CH_LIVE_DR_DB` (8 dB). Sticky: a channel
  that has ever demonstrably responded to sound stays trusted.
- **deaf** on purely *relative* evidence — under `MIC_CH_DEAD_DR_DB` (3 dB) of
  its own movement while the other channel moves `MIC_CH_DEAD_MARGIN_DB` (5 dB)
  more. This needs no speech, so a cold start resolves from ordinary ambience.
- **demotion** likewise only relative: a channel that goes still *while the
  other one is responding* is dead, whereas both going still is just a quiet
  room and must change nothing.
- both live → `mix`; one live or one deaf → the good one; nothing proven →
  `mix` (conservative: never drop a channel on no evidence).
- persisted to `.mic_channel.json` (`MIC_CH_STATE_PATH`, 30-day max age) so the
  fault does not have to be rediscovered every boot.
- a mode change calls `_reset_mic_floor()`, because the learned gate floor is
  an int16 *level* and dropping a channel moves it by many dB at once.

**Why the window is 90 s and not 6.** The first attempt used p90/p10 over 6 s
and failed: it measured L at +7.2 dB in silence but only **+3.7 dB during
continuous speech**, because each short window is *homogeneous* — all speech or
all silence. The statistic needs a window long enough to contain both. At 90 s
the separation is decisive.

**Verified separation on this unit:**

| condition | L | R | verdict |
|---|---|---|---|
| 30 s ambient, nobody speaking | +9.4 dB | +1.2 dB | latched LEFT |
| with speech in the window | +27.9 dB | +1.7 dB | latched LEFT |
| dead-quiet, perfectly steady room | +1.1 dB | +1.4 dB | stays MIX (correct — no evidence) |

**Diagnostic.** When it decides it says so, with the evidence and the hardware
implication:

```
🎙️  Mic channel → LEFT: dynamic range L +9.4 dB / R +1.2 dB over the last 90s
    (≥8 dB = responds to sound, or ≥5 dB below the other channel = deaf)
   ⚠️  the RIGHT mic does not respond to sound — it is contributing noise only,
       so it has been dropped from the speech path. Check that mic's wiring on
       the Vero board; DOA/direction sensing cannot work until it does.
```

**Side effect worth knowing.** The stale floor is what caused the gate to
false-open on **44.4%** of ambient chunks in one run: 1640 had been learned
during the noisy `mix` era. With the channel fix and a fresh floor the same
measurement gives **1/900 chunks (0.1%)**. If you ever force `MIC_CHANNEL`
by hand, delete `.mic_floor.json` at the same time.

**Env overrides.** `MIC_CHANNEL` (`auto`/`mix`/`left`/`right`),
`MIC_CH_WINDOW_S=90`, `MIC_CH_MIN_S=15`, `MIC_CH_LIVE_DR_DB=8.0`,
`MIC_CH_DEAD_DR_DB=3.0`, `MIC_CH_DEAD_MARGIN_DB=5.0`,
`MIC_CH_DECIDE_EVERY_S=2.0`, `MIC_CH_STATE_PATH`, `MIC_CH_STATE_MAX_AGE_S`.
`MIC_CH_CLIP_FRAC`, `MIC_CH_WATCH_S` and `MIC_CH_WATCH_MIN_CLIPS` are **gone** —
they belonged to the saturation criterion, which was the wrong question.

---

## A11. The RIGHT microphone is deaf — HARDWARE, YOU MUST FIX

**Symptom.** Mishearing that comes and goes (A8), and direction sensing that
never works.

**Evidence.** Four independent measurements, all agreeing:

1. **Does not respond to sound.** +1.3 dB against L's +25 dB with broadband
   noise playing through ADAM's own speaker (`mic_cause.py` phase 3).
2. **Zero coherence with L.** Magnitude-squared coherence below 1 kHz is
   **0.005**, and the estimator's own noise floor is 1/n_frames = 0.004 — i.e.
   genuinely zero. Two mics centimetres apart in one head share a sound field;
   at 1 kHz the wavelength is 34 cm, so a real room bed *must* be strongly
   coherent between them. Zero coherence means at most one channel is hearing
   the room at all.
3. **Stationary.** Level spread of **0.5 dB over 90 s** (`mic_watch.py`), and
   +0.2 dB under full CPU load. A live mic in a real room always wobbles;
   L's spread over the same window was 3.9 dB.
4. **White.** 57.9% of its 0.1-8 kHz energy sits above 3.4 kHz, which is
   almost exactly the flat-spectrum share of that band by width — and
   `mean|Δ|/rms` is 1.39 against √2 = 1.41 for sample-to-sample-independent
   noise. It is broadband hiss, not sound.

**What it is not.** Not a misread I2S data line: the low 8 bits of every
32-bit word are exactly zero on both channels, so the receiver is latching
properly aligned 24-bit words (`mic_bits.py`). Not power-rail ripple: CPU load
moves it +0.2 dB. Not amplifier hiss: nothing was holding the card open, and an
amp-open/amp-closed A/B is −0.1 dB broadband, every band within ±0.2 dB. Not
the mic's own noise floor either — at −24.8 dBFS it is ~62 dB above an
INMP441's spec noise floor.

That last sentence is the important one, and it was written here as an
*exclusion* — "the bed is too big to be the part's own noise, so something is
injecting it." It is also, read the other way round, the whole of **A12**: a bed
62 dB over spec is 62 dB over spec on the *live* channel too, which means
switching to the left mic does not escape it. A11 and A12 are the same physical
fault seen from two sides. Fix A12's wiring and supply first; the right mic's
continuity is step 3 of that same list.

**Most likely cause.** One INMP441 actually wired (as L) with the SD line
floating during the R word slot, on the hand-soldered Vero board. Check
continuity of SD, WS/LRCL and ground for the right mic, and that its L/R select
pin is strapped to the opposite rail from the left one.

**What software does about it.** A8 detects it, drops the channel, warns in the
log, and persists the decision. That recovers 11.7-21 dB and is why ADAM hears
you at all today. It does **not** repair the hardware, and until the mic is
fixed:

- there is no stereo information, so **GCC-PHAT direction-of-arrival and neck
  tracking cannot work** — `estimate_doa_angle()` is reading one live mic
  against a noise generator;
- there is no second channel for a dual-mic coherence suppressor, which the
  benchmark says is worth 9 dB — the single best remaining SNR win, unavailable
  until the mic is repaired.

Repairing it promotes `auto` back to `mix` by itself, within one window, with no
config change.

---

## A9. Anti-alias filter too weak at the 8 kHz Nyquist — FIXED IN CODE

**Symptom.** Consonants come back as different consonants. Fricatives and
sibilants are the worst: `code` → `कोर्स` / `कोर्ट`, `ADAM` → `मैडम`. Vowels and
prosody are fine, so the transcript is fluent and confidently wrong rather than
garbled — which is why this reads as "mis-hearing" and not as "bad audio".

**Cause.** The mic runs at 48 kHz and the recogniser is fed 16 kHz, so
`audio_utils.py` low-passes then decimates by `DECIM=3`. The low-pass was a
fixed **63-tap** windowed-sinc. A Hamming-windowed sinc has a transition width
of roughly `3.3·fs/ntaps`, which at 63 taps and 48 kHz is **2514 Hz**. Centred
on `MIC_LP_HZ=6800` that puts the stopband edge at about **8057 Hz — above the
8000 Hz Nyquist of the 16 kHz output.** Attenuation *at* Nyquist was only
about **40 dB**, and the filter was still inside its transition band there.

Everything from 8 kHz to 9.3 kHz therefore folded back down into 6.7–8 kHz at
roughly −40 dB. That band is exactly where the energy that distinguishes
`s`/`ʃ`/`t`/`k` lives, so the aliased image landed on top of the cue the
recogniser needs and corrupted consonant identity while leaving the vowels
alone.

**Fix.** Derive the tap count from the transition band instead of hardcoding
it, and make the stopband edge land *at* Nyquist rather than past it:

```python
def _lp_taps_for(f_pass: float, f_stop: float, fs: float) -> int:
    width = max(1.0, float(f_stop) - float(f_pass))
    return max(31, int(math.ceil(3.3 * fs / width)) | 1)

_LP_TAPS = _lp_taps_for(MIC_LP_HZ, MIC_LP_STOP_HZ, CAPTURE_RATE)
_LP_FIR  = _design_lowpass((MIC_LP_HZ + MIC_LP_STOP_HZ) * 0.5,
                           CAPTURE_RATE, _LP_TAPS)
```

`MIC_LP_STOP_HZ` defaults to `GEMINI_SEND_RATE / 2` = 8000, which yields
**133 taps**. `fc` is set to the midpoint of pass and stop because a windowed
sinc's design frequency is its −6 dB point with the transition straddling it
symmetrically; using `MIC_LP_HZ` directly would push the whole transition into
the passband.

**Measured response of the new filter** (direct DFT of `_LP_FIR`):

| Frequency | 1 kHz | 4 kHz | 6.8 kHz | 8 kHz | 12 kHz | worst ≥ 8 kHz |
|---|---|---|---|---|---|---|
| Gain | −0.00 dB | 0.00 dB | −0.02 dB | **−50.8 dB** | −64.8 dB | **−52.5 dB** |

Passband is flat to `MIC_LP_HZ` to within 0.02 dB and the stopband is 50 dB
down before the fold-over point. Cost on the Pi: the decimation stage measures
**1.96 ms per 33.3 ms chunk** (~6 % of one core) at 133 taps.

**Why not just lower `MIC_LP_HZ` to 6200 instead.** That was the proposed fix
and it was deliberately rejected. Narrowing the passband would have moved the
transition band below Nyquist, yes — but by *discarding* the 6.2–8 kHz
fricative energy, which is the very cue the confused consonants depend on. It
trades an aliased `s` for an absent `s`. Raising the tap count fixes the
aliasing while keeping the band, and the only price is CPU the Pi has.

**Honest scope.** This was real but it is not the main cause of mis-hearing —
a −40 dB aliased image sits far below a noise floor that is only 6 dB under
the speech itself. A9 was worth fixing because it is cheap and exact; **A10 is
the one that moves the needle.**

**Env overrides.** `MIC_LP_HZ=6800`, `MIC_LP_STOP_HZ=8000` (lower
`MIC_LP_STOP_HZ` only if you also accept more taps; raising it above 8000
re-creates the bug).

---

## A10. In-band SNR too low for the recogniser — FIXED IN CODE

**This is the actual cause of "ADAM mis-hears everything".**

**Symptom.** The gate works — `opens` is non-zero, `sent` is 39–226 per
10 s window, every turn produces a reply — and the reply is about something
you did not say. Words are replaced by similar-sounding words rather than
dropped. Switching language does not help. Speaking louder helps a little.
Nothing in the log looks broken.

**Cause, measured.** In-band signal-to-noise ratio on this unit is only
**+2 to +12 dB, typically about +6 dB**:

| Quantity | Measured |
|---|---|
| Learned noise floor, post-filter RMS | 1550 – 1591 (and 1608–1790 in a warmer room) |
| Speech p90, post-filter RMS | 2041 – 4256 |
| Resulting in-band SNR | **+2.4 dB to +12.3 dB** |

The VAD gate copes with that easily because it is a *ratio* detector: it
compares the current chunk against a learned floor, and a 2:1 ratio is plenty.
A neural recogniser is not a ratio detector — it matches spectral detail, and
its accuracy falls off a cliff below roughly **+10 dB**. So the gate opening
correctly and the transcript being wrong are completely consistent, and no
amount of gate tuning can fix it. Neither can gain: `MIC_S32_SHIFT` scales
noise and speech together (see A5). The only lever with real headroom is
removing the noise.

**Fix — a WOLA spectral-subtraction suppressor in front of the recogniser
only.** `_NoiseSuppressor` in `audio_utils.py`, exposed as `denoise_16k()`,
`denoise_reset()`, `denoise_db()`.

How it works, in order:

1. **Weighted overlap-add framing.** `MIC_NR_FRAME=512` samples (32 ms at
   16 kHz), hop = 256, **sqrt-Hann on both analysis and synthesis.** Because
   `w² = periodic Hann` and periodic Hann at hop `N/2` sums to exactly 1.0,
   the transform reconstructs bit-for-bit when every gain is 1.0. Verified:
   max error **1 LSB** (int16 rounding) over 16 000 samples.
2. **No added latency.** Output sample `j` is emitted at index `j`; the only
   transient is a fade-in over the first `hop` samples after a reset. This
   matters because A7 already spends +0.4 s on hangover and the user's
   standing complaint was delay.
3. **Minimum-statistics noise estimate.** Per-bin power is smoothed with
   `MIC_NR_SMOOTH=0.90`, and the noise floor is the running minimum over four
   sub-windows spanning `MIC_NR_NOISE_S=1.5` s. No speech/silence decision is
   involved, so it cannot be fooled by a wrong VAD verdict, and it adapts to
   whatever room the unit is sold into.
4. **Subtraction with a conservative floor.** `clean = max(pwr − oversub·noise, 0)`,
   gain `= sqrt(clean/pwr)` clamped at `MIC_NR_FLOOR_DB=-12`, then smoothed
   across 3 neighbouring bins and 60/40 with the previous frame's gain.

**The gate never sees the suppressed audio.** `_read_and_convert` in
`session.py` returns both signals; every VAD, floor-learning and stats
computation uses the raw one, and only the recogniser paths — the Gemini
queue, the pre-roll and the Vosk wake-word queue — get the denoised one. The
standing instruction was "mic working perfectly so dont chnage it", and this
is how that is honoured: `MIC_NR=0` restores the previous behaviour exactly,
byte for byte, because the transform is unity-gain.

**Why `MIC_NR_OVERSUB=3.5` and not 2.0.** The first attempt used 2.0 and
*lost*: 2.8 dB of noise removed, 3.1 dB of speech removed, net **−0.3 dB**.
Minimum statistics is biased low by construction — the running minimum of a
fluctuating quantity is below its mean — and the bias depends entirely on the
smoothing constant. Measured on this unit's own noise:

| `MIC_NR_SMOOTH` (α) | true mean / min estimate | Power bias |
|---|---|---|
| 0.70 | 3.08× | +4.9 dB |
| 0.85 | 2.00× | +3.0 dB |
| 0.90 | **1.68×** | **+2.3 dB** |
| 0.95 | 1.38× | +1.4 dB |

At α=0.70, `MIC_NR_OVERSUB=2.0` (+3.0 dB) did not even cover the 4.9 dB bias,
so the subtraction sat *below* the true mean noise and did essentially
nothing while still costing speech. `MIC_NR_OVERSUB` has to carry two jobs at
once: cancel the bias **and** provide genuine over-subtraction. At α=0.90 the
bias is 1.68×, so 3.5 = 1.68 × ~2.1 of real over-subtraction. **If you change
`MIC_NR_SMOOTH`, that table no longer applies and `MIC_NR_OVERSUB` must be
re-derived.**

**Measured result** (synthetic speech in this room's own noise spectrum at a
realistic +7 dB input SNR, plus a clean-speech control):

| Metric | Before | After |
|---|---|---|
| Noise RMS in gaps | 1571 | 471 (**−10.5 dB**) |
| Speech RMS in bursts | 3504 | 2519 (−2.9 dB) |
| **In-band SNR** | **+7.0 dB** | **+14.6 dB (+7.6 dB)** |
| Clean speech with no noise present | — | **−0.36 dB** (transparent) |

That moves the unit from well below the recogniser's cliff to comfortably
above it. Cost on the Pi Zero 2 W: **2.15 ms per 33.3 ms chunk, 6.5 % of one
core.**

**Live confirmation.** After deployment the stats line carries the new field
and reports −10.7 to −11.4 dB of in-band attenuation in a quiet room, while
`floor`, `p50` and `open≥` are unchanged because they are still measured on
the raw path:

```
📊 Mic 10s: p50 1855 p90 1947 p99 2033 max 2096 | open≥2232 hold≥1893 | floor 1786 flat 0.56/0.48 lohi 0.13 shp 13% | opens 0 sent 0 | blocked 0 | nr -10.9dB | shut
```

**Deliberate design limits.**

- **`MIC_NR_FLOOR_DB=-12` is intentionally shallow.** A deeper floor buys more
  apparent quiet and strips consonants and creates musical noise; the standing
  instruction was to keep this "loose and little simple". One stage, one
  subtraction, no cascades.
- **The noise estimate survives `denoise_reset()`.** Only the frame buffers and
  gain history are cleared when ADAM starts speaking. It is the same room a
  moment later; re-learning would leave the first 1.5 s after every reply
  unprocessed, which is precisely when you start talking again.
- **Songs bypass it entirely.** The song-stop Vosk path keeps the raw signal:
  music is non-stationary, so a minimum-statistics estimate of it is
  meaningless, and a wrong estimate would suppress the stop phrase.
- **Not primed = pass-through.** For the first ~1.5 s of audio the gains are
  exactly 1.0, so a cold start degrades to the old behaviour instead of
  distorting.

**If it over-suppresses in your room** (voice sounds hollow, quiet speech gets
eaten), loosen it with `MIC_NR_OVERSUB=2.5` or `MIC_NR_FLOOR_DB=-9`, or set
`MIC_NR=0` to disable it outright. Do not add a second stage.

**Env overrides.** `MIC_NR=1`, `MIC_NR_FRAME=512`, `MIC_NR_OVERSUB=3.5`,
`MIC_NR_SMOOTH=0.90`, `MIC_NR_FLOOR_DB=-12`, `MIC_NR_NOISE_S=1.5`.

---

## A12. The microphone noise floor is 61 dB above spec — HARDWARE, WORTH FIXING

> **Read the Update at the end of this entry first (2026-10-01).** The relative
> measurement here is sound, but the absolute dB SPL figures are invalid and the
> "root cause" claim below is wrong — the listening fault was a software defect,
> fixed in `development_log.md` Part 26. The rest of this entry is kept because
> the noise excess is real and the repair checklist still applies.

**This was believed to be the root cause of "I am speaking, ADAM can't listen —
only a few times it can listen to me, rest of the time it can't; if I shout near
the mic maybe then it can listen."** It is not; see the Update. The instruction
this entry used to carry — "fix this before touching a single gate constant" —
is withdrawn: the gate constants were exactly what was wrong.

**Symptom.** ADAM occasionally transcribes something, usually wrongly, and
only when you are loud and close. Normal conversation produces nothing at all.
The boot log looks *healthy* — a good tone SNR, a sensible floor, a converged
gain — which is exactly what makes this fault so hard to find.

**Why every other number in the log missed it.** Every measurement the
calibration used to print is *relative*:

| Number | What it compares | Why a broken mic still passes |
|---|---|---|
| tone SNR `+23.4 dB` | chime vs noise | The chime is ADAM's own speaker, a few cm away in a sealed case. It hits the mic at ~120 dB SPL. It beats even a terrible floor. |
| `headroom above the noise 27 dB` | chime peak vs noise peak | Same stimulus, same problem. |
| gate floor `457` | a level in gate units | Gate units are `raw / divisor`. Change the divisor and this number changes with no physical meaning attached. |
| `Mic gain -10.7 dB` | a divisor | Pure arithmetic on digits. |

Raising the digital gain moves the floor and the voice by the *same* amount, so
none of these can distinguish "quiet mic" from "deafeningly noisy mic". A unit
can print `+26.6 dB tone SNR — mic is fine` and be unable to hear a person in
the same room. **That is what this unit was doing.**

**The measurement that found it.** One number is not relative: the mic's own
datasheet fixes the mapping from digital level to sound pressure. For the
INMP441, sensitivity is −26 dBFS at 94 dB SPL, so digital full scale is
`94 + 26 = 120 dB SPL`, and its 61 dB(A) SNR puts the part's own self-noise at
33 dB(A) SPL. Measure the raw capture against **its own** full scale — 2³¹,
because the part left-justifies its 24-bit sample in a 32-bit slot — and the
noise floor becomes an absolute dB SPL figure you can compare against a human.

Measured on this unit with `mic_noise_probe.py`, room quiet, nothing playing,
three consecutive runs:

```
                       LEFT (live)      RIGHT (deaf)
noise RMS, full band   -17.9 dBFS       -24.4 dBFS
noise RMS, 80-8000 Hz  -25.9 .. -26.2   -
SPL equivalent              94 dB SPL        96 dB SPL
vs INMP441 spec (33)        +61 dB           +63 dB
peak                     0.0 dBFS         -1.0 dBFS   <- railing
tonality                    0.037            0.044    <- nothing to notch
```

**Read that table again.** The microphone's electrical noise is equivalent to
someone running a lawnmower next to it, permanently. Nobody *hears* it, because
it is electrical, not acoustic — but the mic cannot tell the difference, and
neither can Gemini.

Against a 94 dB SPL floor:

* the gate first opens at ~96 dB SPL (`MIC_OPEN_RATIO` = 1.25 = +1.9 dB),
* reliable transcription needs ~104 dB SPL (the conventional 10 dB
  intelligibility margin),
* normal conversation is **55–65 dB SPL**,
* a shout at 10 cm is **90–95 dB SPL**.

So a shout lands just *below* the gate and just *sometimes* crosses it, which
is precisely the reported behaviour — and when it crosses by 1–2 dB the
consonants are still buried, which is why what comes back is `मैडम` rather than
`ADAM`. The gap between "gate opens" and "transcribes correctly" is the whole
reason this unit mishears rather than simply staying silent.

**It is broadband and it is not filterable.** Tonality 0.037 means 96 % of the
noise power is in the broadband bed, not in discrete peaks — there is no hum,
no regulator whistle, nothing a notch or comb could remove. The PSD is white
(power rises +3.0 dB per octave across every octave measured, which is flat
power *per hertz*), so the only thing band-limiting buys is the −6.3 dB the
chain already gets for free by decimating to 16 kHz. The two peaks that do
exist, +12.0 dB at 12 000 Hz and +6.7 dB at 6 000 Hz, are exact subharmonics of
the 48 kHz frame clock and both sit above the speech band.

**Bit-level forensics.** Raw `arecord` capture, `S32_LE`, no DSP:

```
bit toggle rate, consecutive samples (0.500 = fair coin flip)
 b31 0.364   b30 0.367   b29 0.383   b28 0.345
 b24 0.501   b20 0.502   b16 0.500   b12 0.502   b8 0.500
low 8 bits all zero: True          <- 24-in-32 left-justified, as expected
L/R cross-correlation: -0.20       <- NOT common-mode
kurtosis: L 7.3, R 32.1            <- impulsive, not Gaussian
peak: L 0.0 dBFS, R -1.0 dBFS      <- railing on noise alone
```

Bits 8 through 24 are fair coin flips, which simply means the noise is ~20 bits
tall — it is a genuinely enormous analog-domain signal, not a stuck line or a
bit-slip. The low 8 bits being exactly zero rules out an I2S format or
alignment error. The L/R cross-correlation of −0.20 rules out a shared clock or
a shared supply rail as the *dominant* path, because common-mode injection
would correlate the two channels strongly. Kurtosis of 7–32 against 3.0 for
Gaussian noise, plus rail-to-rail peaks, says the mics are being hit by
impulsive interference and their internal ADCs are clipping on it.

**What to fix, in this order.** This appeared with the new plastic body, which
is the strongest clue in the whole investigation — the electronics did not
change, the harness did.

1. **Mic wiring.** Short, twisted, and routed away from the servo, amp and
   camera harness. Long unshielded I2S runs next to a servo are the single most
   likely cause of impulsive noise that rails.
2. **Mic supply.** The INMP441 has poor PSRR. Give it its own decoupling
   (100 nF + 10 µF at each module) and do not share the servo rail. Servos
   brown out a shared 3V3 on every step, which matches the impulsive signature.
3. **SEL / L-R pins.** Confirm both are tied correctly and both mics actually
   drive SD. The right mic is deaf (A11) and still contributes noise, so it is
   already wired wrong in some way.
4. **Swap in a known-good mic module** to separate a bad part from a bad
   harness.

**What the code now does about it.** Nothing can recover 61 dB in DSP, and the
honest thing is to say so at boot instead of printing a cheerful tone SNR.
`mic_calibrate.py` now ends every calibration with an absolute health verdict
(`_noise_health` / `_report_noise_health`), computed inside the
`MIC_HP_HZ`–8000 Hz band the speech path actually keeps:

```
     noise floor: -26.2 dBFS in the 80-8000 Hz speech band = 94 dB SPL
       equivalent (+61 dB vs the INMP441's 33 dB SPL spec)
  ❌ MIC HARDWARE FAULT — noise floor is 61 dB above spec.
     Speech must reach ~96 dB SPL at the mic for the gate to open at all,
     and ~104 dB SPL to be transcribed reliably. ...
```

It is band-limited deliberately: the full-band figure is ~6 dB worse (102 dB
SPL) because the noise is white and three quarters of its power sits above
8 kHz, where the chain discards it before the gate or Gemini ever see it.
Quoting the full-band number would overstate the fault and contradict the
unit's own observed ability to occasionally hear a shout.

A healthy unit prints `✅ mic hardware healthy` with its margin against normal
speech, so this line is worth reading on every build.

**Do not "fix" this by raising the gain.** See A5. It cannot work, and the new
boot message says so in as many words.

**Env overrides.** `MIC_FS_SPL=120.0`, `MIC_SELF_NOISE_SPL=33.0`,
`MIC_NOISE_WARN_DB=25.0`. The first two are datasheet constants for the
INMP441; change them only if the microphone part changes.

### Update 2026-10-01 — the SPL figure was invalid, and this is not the root cause

The relative half of this entry stands: the floor **is** ~60 dB noisier than a
clean INMP441 at the same reference, that is real, and it is still worth fixing
in hardware — the wiring/supply checklist below is unchanged and every dB
removed there is a dB of detection margin gained.

**What is withdrawn is the absolute SPL claim and everything derived from it**
("93 dB SPL equivalent", "speech must reach ~96 dB SPL to open the gate",
"~104 dB SPL to be transcribed", "NOT fixable in software"). Three measurements
in the same boot log contradict it:

- The RIGHT mic is acoustically **deaf** (tone SNR −0.8 dB — see A11) yet its
  noise floor is within 3 dB of the LEFT mic's. Noise equal on a channel with
  no acoustic path did not arrive acoustically.
- The LEFT mic resolves the calibration chime at **+27.5 dB SNR** from a small
  speaker centimetres away. A mic needing ~96 dB SPL to register could not.
- The noise is impulsive and channel-uncorrelated (kurtosis 9–44,
  corr(L,R) = −0.19). Acoustic room noise is neither.

So the dominant noise is **electrical, injected after transduction**. dB SPL
describes noise that came through the diaphragm; applied to noise added
downstream it describes nothing. `MIC_FS_SPL` and `MIC_SELF_NOISE_SPL` are
correct datasheet values — the defect was applying an acoustic mapping to
non-acoustic noise, and the thresholds printed above inherited the error.

**The actual root cause of "ADAM can't listen" was a software defect**: the
gate's open threshold (floor 608 × `MIC_OPEN_RATIO` 1.25 = 760) sat *below* the
room's own noise p20 of 1202, so the gate was latched open on an empty room and
streamed noise to Gemini continuously. That is both reported symptoms at once.
Fixed by moving the gate's decision into 300–3400 Hz and widening the ratios:
**0.0% false opens on an empty room, measured twice on this hardware.** Full
account in `development_log.md` Part 26.

`_report_noise_health()` now prints the dBFS floor, the measured chime SNR, the
dead channel, and the noise excess as a relative figure. It no longer prints an
SPL requirement or a verdict on software. **Reclassified: HARDWARE, WORTH
FIXING — but not blocking, and not the root cause of the listening fault.**

---

## A13. The solved gain drifted every boot and reset the learned floor — FIXED IN CODE

**Symptom.** Every boot logged `🔊 Mic gain +0.2 dB` (or +0.5, or −0.3)
followed by `🎚️ Mic floor reset (mic gain changed) — relearning over 1.5s`,
even though nothing about the unit had changed. The floor that
`startup_mic_calibration.md` carefully arranged to resume from disk was thrown
away on every single start.

**Root cause.** The divisor is solved from the *measured* chime peak, which is
an acoustic measurement: amp warm-up, how the case is resting on the desk, a
door closing during the 4.6 s chime all move it by a few tenths of a dB.
`set_mic_scale()` ignored changes below 0.1 dB, but the observed run-to-run
spread was 0.2–0.5 dB — above that deadband. Every boot therefore "changed" the
gain, and every accepted change calls `_reset_mic_floor()`.

Three consecutive calibrations on an untouched unit walked the divisor
224392 → 211435 → 206611, a cumulative +0.7 dB of pure measurement noise.

**Why it mattered.** The three learned values — channel, gain, floor — are only
meaningful together, because the floor is an absolute level *in the units the
divisor defines* (see `startup_mic_calibration.md`). Resetting the floor on
every boot defeated the resume path entirely and left the gate running on a
1.5 s sample of whatever the room happened to be doing at power-on.

**The fix.** `set_mic_scale()` now uses `MIC_CAL_GAIN_DEADBAND_DB`, default
**1.5 dB** — comfortably above the measurement spread and far below anything
audible or anything the ratio-based gate can notice, since the gate's open
threshold is a *ratio* over the floor and both scale together.

Verified: a second calibration on an untouched unit now prints no gain line at
all and resumes the floor — `🎚️ Mic floor measured 457 (-0.2 dB vs the
resumed 466)` instead of a reset.

**Env override.** `MIC_CAL_GAIN_DEADBAND_DB=1.5`.

---

## A14. Level was a hard veto, so quiet speech was discarded unexamined — FIXED IN CODE

**Symptom.** *"It can listen to what I am telling, missing a few words."*
Words lost at the start and end of sentences, or when not speaking straight at
the unit, while clearly-projected speech worked.

**Root cause.** `AdaptiveGate.is_speech()` used level as a **veto**:

```python
if rms >= self.strong_th:   return True
if rms >= self.open_th:     return self.shape_ok(mono16k)
return False                 # spectrum never examined
```

Below `open_th` the spectral-shape machinery — present, learning every chunk,
and explicitly documented as level-independent — was never asked. The margin
this left was almost nothing: in the field log of 2026-10-01 the floor settled
at 377–380, putting `open_th` at 756, and real speech measured **1169**. That
is +9.8 dB over the floor but only **+3.8 dB over the threshold**, so any
syllable quieter than a projected vowel fell under the rail.

**The fix.** A third *candidate* tier between `cand_th` and `open_th` where the
spectrum decides outright — a strict shape pass **and** `shape_frac >= 0.60`.
Both constants were placed from a 900-chunk measurement of the room on a
settled floor: noise reaches at most **1.27x** the floor and `shape_frac` at
most **0.53**, so `MIC_CAND_RATIO=1.45` and `MIC_CAND_SHAPE_FRAC=0.60` each sit
above what the noise itself can reach. The strict shape vote passes **0.0%** of
pure noise chunks, so noise must fail two independent tests.

`observe_background()` now learns the noise-flatness reference only below
`cand_th`, not below `open_th`. This is not optional: the candidate band is
where quiet speech is now convicted, and letting it teach `_flat_max` would
raise the reference toward speech's own flatness and dismantle the tier that
depends on it. It also keeps the one-`shape_ok`-per-chunk invariant — the call
*moved* into `is_speech` rather than being added.

Verified: **0.0% false opens on silence across four 12 s runs**, including one
where 73.3% of chunks landed in the candidate band and the shape vote rejected
every one. Full reasoning and the rejected alternative (lowering
`MIC_OPEN_RATIO`, which would have moved `strong_th` and `hold_th` too) in
`development_log.md` Part 27 §3.

**Not yet verified:** the *benefit*. `mic_check.py` phase 2 needs a human
speaking and has still never been run. Its new `of which cand tier` line
reports exactly how much speech the old veto would have dropped.

**Env overrides.** `MIC_CAND_RATIO=1.45`, `MIC_CAND_SHAPE_FRAC=0.60`.

---

## A15. A 0.7 dB miss on the chime discarded the floor measurement too — FIXED IN CODE

**Symptom.** *"During the starting few minutes ADAM can't listen to anything,
then after some time it suddenly starts to listen."* Reported as two separate
behaviours; it is one.

**Root cause — two faults stacked.**

*First, the measurement.* The chime SNR is `tone − noise`, and the noise
reference was `np.mean` over the 13 frames of a 1.2 s silence window. A mean of
power is dominated by its loudest term, so three hot frames out of thirteen at
+25 dB inflate the reference by **+18.7 dB**. One boot measured **+11.3 dB**
against a 12.0 dB threshold where standalone runs of the same hardware gave
+25.6 to +30.0 dB. Measured: the tone half is steady to ~0.2 dB across runs
while the noise half moved 7 dB between an idle Pi and a loaded one.

*Second, and worse, what the failure did.* `calibrate()` makes three
independent decisions, and only two of them need the tone:

| decision | measured from | needs the tone? |
|---|---|---|
| which channel is live | tone, per channel | yes |
| the gain divisor | tone peak | yes |
| the gate's floor seed | **the silence capture** | **no** |

The old code did `return None` on a sub-threshold chime, which threw the floor
seed away as collateral damage. The gate then resumed a **stale floor of 408**
against a true room of **378** — `open_th` 816 instead of 756. Speech onsets in
that 0.7 dB-wide band were invisible until the floor estimator walked down on
its own (`MIC_FLOOR_RISE 0.02` over a 45 s ring), which is the "suddenly it
started to listen" exactly: a slow ramp that looks abrupt from outside.

The printed message — *"speaker muted, amp off, or both mics dead"* — was
wrong in every clause. The speaker was fine, the amp was on, and the LEFT mic
was resolving the chime at +25 dB.

**The fix.**
1. `_tone_power(select=False)` reduces the silence window by
   `MIC_CAL_NOISE_PCTL = 25` across frames, per bin, and `MIC_CAL_SILENCE_S`
   goes 1.2 s → 2.0 s so the percentile has ~22 frames. Same idiom the floor
   (p20) and flatness (p5) estimators already use, for the same reason: this
   room's noise is impulsive, so its average and its typical level are
   different numbers and only the typical one is a reference. Measured effect
   on identical hardware: **+25.6 → +30.0 dB**.
2. `calibrate()` is fail-soft. A weak chime skips only the channel and gain
   decisions and still measures and seeds the floor. It does **not** save a
   fingerprint, since one taken from an unheard tone would make the next boot
   declare a spurious change.
3. The message now reports the measured SNR, the threshold it missed, and the
   silence window's **crest factor** — which distinguishes a burst during the
   reference window from a genuinely silent speaker, and those send you to
   opposite places to look.

Verified by forcing the branch with `MIC_CAL_LIVE_SNR_DB=60`: channel and gain
preserved, `🎚️ Mic floor measured 255 (-2.0 dB vs the resumed 320) — measured
at startup (chime weak, floor still valid)`.

**Env override.** `MIC_CAL_NOISE_PCTL=25`, `MIC_CAL_SILENCE_S=2.0`.

---

## A16. "Mic path CHANGED" fired on the room going quiet — FIXED IN CODE

**Symptom.** `🔄 Mic path CHANGED (response moved 13.1 dB at 200 Hz) —
discarding everything learned in the previous body` on consecutive boots of
hardware nobody had touched. Each one also called `reset_floor()`.

**Root cause.** The stored fingerprint was the per-tone **SNR**, and SNR
contains the room. Three consecutive calibrations as the room quietened:

```
run 1   floor 675   chime +30.0 dB   🔄 CHANGED (13.1 dB at 200 Hz)
run 2   floor 320   chime +37.5 dB   🔄 CHANGED ( 9.9 dB at 1600 Hz)
run 3   floor 255   chime +39.3 dB
```

The tone was constant; the room fell 8.5 dB; the "response" duly appeared to
move. A detector that fires on the room is a false-alarm generator, and it
fires in the one place it most misleads — the boot log of a unit being
diagnosed for a listening fault.

Raw tone power is not a fingerprint either: `s32_stereo_to_float_channels`
divides by `_mic_scale[0]` and the auto-gain re-solves that divisor every boot.

**The fix.** Store `tone_ref_db` — the tone level with the gain divisor
multiplied back out, so it is referred to the transducer's own full scale.
What remains is speaker level, acoustics and mic sensitivity: the path, and
nothing else. A state file without the key relearns once.

Verified across three runs while the room moved **+4.1 dB** the other way
(floor 328 → 438 → 527, chime 36.0 → 34.7 → 32.5 dB): fired once for the
schema migration, then silent.

---

# Part B — ADAM's voice sounds wrong

## B1. MAX98357A GAIN pin left floating — HARDWARE, YOU MUST FIX

**Symptom.** Output level jumps between loud and quiet with no pattern and no
correlation to anything in software. Was crystal clear minutes ago, now is not.

**Root cause.** The amplifier's `GAIN` pin is left unconnected. A floating
CMOS input is high-impedance: it picks up coupled noise from the neighbouring
I2S clock lines and the 5V rail, so the internally-selected gain step is not
stable. Nothing in software can compensate, because the gain is being chosen
downstream of every byte ADAM writes.

**Fix — solder the pin to a definite level.** Pick one:

| GAIN pin wiring | Resulting gain |
|---|---|
| Direct to VDD | 6 dB |
| 100 kΩ resistor to VDD | 3 dB |
| Direct to GND | 12 dB |

Start at **6 dB (GAIN → VDD)**. If the result is too quiet, prefer the 12 dB
option over raising `SPEAKER_GAIN` in software — analog headroom is free,
digital headroom is not.

**How to tell this is your problem:** the log shows no underruns and no
warnings while the level is misbehaving. Software-side gain problems always
leave a trace; this one does not.

---

## B2. Kernel audio driver clock conflict — OS CONFIG, YOU MUST FIX

**Symptom.** Persistent crackle or glitching in the output that no software
change affects.

**Root cause.** `dtparam=audio=on` loads the Pi's onboard PWM/headphone audio
driver *alongside* the I2S codec. Both want the audio clocks; the contention
shows up as periodic dropouts.

**Fix.** Edit `/boot/firmware/config.txt`, comment the line out, reboot:

```bash
sudo sed -i 's/^dtparam=audio=on/#dtparam=audio=on/' /boot/firmware/config.txt && sudo reboot
```

Verify afterwards that only the voiceHAT device is present:

```bash
aplay -l
```

You should see `sndrpigooglevoi` and **not** `bcm2835 Headphones`.

---

## B3. Unbuffered pipe dropping bytes → byte-swap buzz — FIXED IN CODE

**Symptom.** The voice degenerates into loud buzz and **stays** buzzing for the
rest of the session. Restarting fixes it until it happens again.

**Root cause — the important one to understand.** `aplay` is spawned with
`bufsize=0`, which makes `proc.stdin` a raw `_io.FileIO`. A raw pipe write may
be **short**: it accepts fewer bytes than offered and *returns how many*. The
old code called `proc.stdin.write(data)` and discarded that return value, so
the unwritten tail was silently dropped.

Dropping bytes would only be a click — except when the dropped count is not a
multiple of 4 (one 48 kHz stereo s16 frame). Then the stream de-aligns, and
every following int16 has its low and high bytes swapped. A byte swap is a
**×256 error**, roughly **+48 dB**, i.e. full-scale buzz. Nothing ever re-syncs
a raw PCM pipe, so it persists until the process is replaced.

**Fix.** `write_all()` in `pi/adam/audio_utils.py` — a loop that honours the
returned count, retries on a full pipe, tolerates a buffered stream's `None`
return, and truncates any sub-frame remainder so the stream **cannot** go out
of alignment. Wired into all four write sites:

| Site | What it carries |
|---|---|
| `session.py` — main `out_q` path | every byte of ADAM's voice |
| `session.py` — teardown drain | the tail of the last reply |
| `session.py` — startup beep | the boot chime |
| `song_playback.py` | song audio |

**Verified on the Pi.** Against a pipe that deliberately accepts only 3000
bytes per call, 655 360 bytes of each of the three song files were pushed
through: **0 bytes lost, 0 misaligned**, across 240 write calls per file. A
misaligned payload is truncated to the frame boundary rather than de-aligning
the stream, and empty/sub-frame payloads are no-ops.

> Note on an alternative: switching to `bufsize=65536` would also fix the short
> write, since Python's buffered writer loops internally. `bufsize=0` was kept
> and the explicit loop added instead, because the frame-alignment guarantee is
> then visible in one place and does not depend on which stream type the process
> happened to get. `write_all()` handles both, so `bufsize` can be changed later
> without reintroducing the bug.

---

## B4. Unpaced song loop starving the CPU — FIXED IN CODE

**Symptom.** While a song plays, everything else lags: the song itself
stutters, the camera and servos become sluggish, replies are slow.

**Root cause.** The loop read the WAV and wrote it into the pipe with only
`await asyncio.sleep(0)` between chunks. Reading a file and writing to a pipe
run at memcpy speed, not at 48 kHz, so the loop tried to push a 3-minute song
through as fast as the pipe would take it.

**Measured on this Pi:** the read+write cost for 3.41 s of audio was 35–103 ms,
so the unpaced loop ran at **33× to 98× realtime**. On a Pi Zero 2 W that is
one asyncio task making hundreds of `to_thread` hops per second, starving the
camera, servo, Gemini and ALSA writer threads.

**Fix.** Pace to just under realtime in `pi/adam/song_playback.py`:

```python
pace_s = (SONG_CHUNK_FRAMES / float(PLAYBACK_RATE)) * SONG_PACE_FRAC
...
await asyncio.sleep(pace_s)
```

With the defaults (`SONG_CHUNK_FRAMES=4096`, `SONG_PACE_FRAC=0.9`): each chunk
is **85.3 ms** of audio and the loop sleeps **76.8 ms**, so it runs at
**111.1% of realtime** — always slightly ahead of ALSA, never 90× ahead. The
11% surplus is absorbed by the pipe's own backpressure, so this self-corrects
rather than drifting.

A bonus: the stop request is checked once per iteration, so `pace_s` also
bounds how long Touch3 or a spoken stop phrase waits — **76.8 ms**, still
instant to a listener.

**Env overrides.** `SONG_CHUNK_FRAMES` (larger = fewer wakeups, coarser stop
latency), `SONG_PACE_FRAC` (must stay **< 1.0**; at ≥ 1.0 the loop falls behind
realtime and ALSA underruns).

---

## B5. 5V rail sag and missing decoupling — HARDWARE, YOU MUST FIX

**Symptom.** Crackle and distortion that get worse on loud passages, and may
coincide with servo movement. In the worst case the Pi browns out and reboots.

**Root cause.** The MAX98357A draws current in bursts that track the audio
waveform. Powered from a marginal supply, or without local bulk capacitance,
the 5V rail sags on those peaks — and the same rail feeds the Pi and the
servos, so the amplifier, the CPU and the motors all fight each other.

**Fix.**

1. Use a dedicated **5V, 2.5–3.0 A** supply. A phone charger or a USB hub port
   is not sufficient, and neither is powering servos from the same regulator.
2. Add decoupling **right at the amplifier's VDD/GND pins**, as short as you
   can make the leads: **220–470 µF electrolytic in parallel with 0.1 µF
   ceramic**. The electrolytic covers the audio-rate sag, the ceramic covers
   the high-frequency switching edges — you need both.
3. Keep the amplifier's ground return separate from the servo ground return
   back to the supply, so servo current does not modulate the amp's reference.

---

## B6. Keep-alive silence injected *into the music*, plus an unserialised shared pipe — FIXED IN CODE

**Symptom.** `aplay -D plughw:1,0 song1.wav` sounds beautiful. The same file
through ADAM sounds distorted — not clipping, not muffled, but *broken up*:
a gritty, hollow quality that is worse on sustained notes and never appears on
speech. It is present from the first second of the song and it is not affected
by volume.

**This is the "song quality" issue, and it was two independent faults that had
to be fixed together.** Both live in the fact that **one `aplay` process serves
the whole session** and five separate code paths write into its stdin.

### B6a. The DAPM keep-alive fired while a song was playing

`speaker()` keeps the amplifier powered by writing 20 ms of digital silence
whenever nothing has come out of `out_q` for 0.5 s — see B3 and A-something
about the voiceHAT's DAPM powering the amp down into a Broken Pipe. The guard
on that branch was:

```python
elif not adam_speaking.is_set() and proc and proc.poll() is None:
```

It tests `adam_speaking`. But **`_play_song_task()` never sets
`adam_speaking`** — it sets `song_playing`, and it writes to `proc.stdin`
directly rather than through `out_q`. So for the entire duration of a song:

* `adam_speaking` is clear → the guard passes,
* `out_q` is empty → the 0.5 s `wait_for` timeout always fires,
* and 3840 bytes of digital silence get punched into the middle of the music
  **twice per second**.

3840 bytes is 960 frames, exactly 20 ms at 48 kHz stereo s16. So the song was
played with a **4 % duty cycle of dropouts**, each with a discontinuity at both
edges — a click, and a fractional-second gap, every 500 ms. That is exactly
what "distorted, doesn't sound properly" describes, and it is inaudible on
speech because speech has gaps in it anyway.

The keep-alive is also *pointless* during a song: the music is continuous
audio, which is all DAPM needs to stay powered up.

**Fix** — one clause, in `session.py`:

```python
elif (not adam_speaking.is_set()
      and not song_playing.is_set()
      and proc and proc.poll() is None):
```

### B6b. Nothing serialised the five writers

Even without the keep-alive, the shared pipe was not safe. `aplay` is spawned
with `bufsize=0`, so `proc.stdin` is a raw `_io.FileIO` whose `write()` is
allowed to accept **fewer bytes than offered**. `write_all()` loops over those
partial writes — but it is frame-safe only for a *single* writer.

`speaker()` (TTS chunks, end-of-turn flush, startup beep, keep-alive) and
`_play_song_task()` run as separate `asyncio.to_thread` workers. Without a
lock, writer A can be sitting between the two halves of a short write when
writer B's bytes land in the gap. Playback is s16 stereo = 4 bytes per frame,
so inserting or losing a byte count that is not a multiple of 4 swaps the low
and high halves of every following int16 — a ×256 error, **+48 dB of full-scale
buzz, permanent for the rest of the session**, because nothing ever re-syncs a
PCM pipe. This is the same failure mode `write_all()`'s own docstring predicted
in B3; the fix there was necessary but not sufficient once a second writer
existed.

**Fix** — `_pcm_write_lock` and `write_pcm()` in `audio_utils.py`:

```python
# The lock covers write AND flush together, because a flush between another
# writer's partial write and its continuation is the same hazard.
_pcm_write_lock = threading.Lock()

def write_pcm(pipe, data: bytes, frame_bytes: int = 4) -> int:
    with _pcm_write_lock:
        n = write_all(pipe, data, frame_bytes)
        try:
            pipe.flush()
        except Exception:
            pass
        return n
```

All five shared-pipe call sites now go through it: the startup beep,
`end_of_turn()`'s tail flush, the keep-alive, Gemini's TTS chunks, and
`_play_song_task()`. **Use `write_pcm()`, never a bare `write_all()` +
`flush()` pair, on the shared aplay stdin.**

### Regression test — `song_stream_test.py`

Both faults are byte-level, so the test is byte-level: it runs the real
`_play_song_task()` against a recording fake pipe instead of aplay, with a
competing writer hammering the same pipe, and compares the result to the source
WAV. It runs two scenarios — the song alone (covers B6a) and the song plus a
competing writer (covers B6b).

```
cd ~/adam && source venv/bin/activate && python3 song_stream_test.py
```

**The fake pipe is a bounded buffer drained at realtime, and that is the whole
test.** A real `write(2)` into aplay's stdin blocks when the 64 KiB pipe is full
and then returns a *short* count — which is why `write_all()` has a partial-write
loop at all, and the only window in which two unsynchronised writers can tear
each other's chunks. The contender writes in **bursts**, because `speaker()` is
unpaced: Gemini delivers a turn's audio faster than realtime and `speaker()`
pushes each chunk in as it arrives. Song (192 kB/s) + bursts (~48 kB/s) into a
192 kB/s sink keeps the pipe pinned full, so the loop is entered ~800 000 times
per run. The test **fails itself** if `pipe.short == 0` — a scenario that never
saw a partial write proved nothing, and says so rather than reporting a PASS.

On the fixed code, three consecutive runs:

```
  ── scenario 1: song alone (proves the keep-alive guard) ──
  wrote  : 2375680 bytes in 12.0s
  keep-alive injections while song was playing: 0
  short writes handed out by the pipe: 812245
  ✓ frame aligned (593920 frames)
  ✓ byte-identical to the WAV for all 2304000 compared bytes
  ✓ no 20 ms keep-alive silence injected into the song
  ✓ paced at 1.00x realtime
  RESULT: PASS

  ── scenario 2: song + a competing writer (proves write_pcm's lock) ──
  short writes handed out by the pipe: 818958
  competing TTS-style writes during the song: 212
  ✓ every competing write landed on a frame boundary; the song was never torn mid-frame
  ✓ all 212 competing writes landed whole; song bytes recovered cleanly
  ✓ byte-identical to the WAV for all 2179072 compared bytes
  RESULT: PASS
```

Each fix was then shown to be **necessary** by reverting it on its own. A test
that has not been shown to fail is not evidence:

| | scenario 1 | scenario 2 |
|---|---|---|
| fixed code | PASS (3/3 runs) | PASS (3/3 runs) |
| **revert A** — `song_playing` dropped from the keep-alive guard | **FAIL** | **FAIL** |
| **revert B** — the lock removed from the song write site | PASS — correctly, there is no contention to serialise | **FAIL** |

**Revert A** — 115 injections, `STREAM CORRUPTED: first difference at byte 81920
(0.427s into the song)`, `expected 96 08 42 0d …` / `got 00 00 00 00 …`,
`KEEP-ALIVE SILENCE FOUND inside the song at byte 81920 (0.427s)`. Literal
zeros, at the predicted ~0.5 s cadence.

**Revert B** — `✗ LOCK BROKEN (song torn): 125 competing write(s) landed off the
4-byte frame boundary, first at byte 105961 (offset 1) — every sample after that
point is byte-swapped`. Offset 1: a single-byte shift, which is exactly the
+48 dB buzz in `write_all()`'s docstring. Revert B passes scenario 1, as it
should — with nothing else writing, there is nothing for the lock to do.

#### The detector was blind to the fault it was written for

Worth recording, because the first version of this test passed with the lock
removed and that looked like evidence the lock was unnecessary. Two separate
harness defects, both of which flatter the code:

1. **The fake pipe was an infinite sink.** It appended to a `bytearray` and
   returned immediately, so each writer was inside `write()` for microseconds
   out of every 20–85 ms and the two never overlapped. Zero short writes, zero
   contention, guaranteed PASS. Hence the bounded-pipe model above and the
   `pipe.short == 0` self-check.
2. **`got.replace(MARKER_CHUNK, b"")` healed the corruption before measuring
   it.** The check counted marker *fragments*, i.e. it only caught the intruder
   being torn by the song. The dominant fault is the mirror image — a *whole*
   intruder chunk landing inside a partial song write — and removing that whole
   chunk splices the song back together perfectly, so the byte-compare passed.
   Note that revert B still reports `✓ all 310 competing writes landed whole`:
   the intruder was never torn. The fault is found only by checking **where**
   each marker sits, on the raw stream, before any removal — a marker at an
   offset that is not a multiple of 4 means the song's frame boundary moved.

The lesson generalises past this file: a test that reconstructs the expected
result from the actual one can only ever confirm itself. Check the raw artifact
first, then normalise.

`song1.wav` was separately confirmed to be 2 ch / 16 bit / 48000 Hz / 204.96 s,
ruling out a format mismatch as a contributing cause.

`_play_song_task()` selects with `random.choice(SONG_FILE_PATHS)`, so the test
pins the choice via a monkeypatch and fails loudly (`TEST HARNESS STALE`) if that
call ever moves. Without the pin the test compared whichever song the RNG picked
against whichever the test picked and passed only on coincidence — a 1-in-3 flake
that reports itself as `STREAM CORRUPTED at byte 0`.

Scenario 2's pacing figure is **informational, not asserted**: two writers
pushing more than realtime into a sink that drains at exactly realtime must make
the song take longer than its own duration. That is the hardware's arithmetic,
and it is also why barge-in during a song is audibly rough. Pacing is a real
requirement only in scenario 1.

### Why this was so hard to see

Neither fault produces anything in the boot log, `dmesg`, or `amixer`. Both are
silent in every software-visible sense: the process is healthy, the buffer
level is healthy, the pacing is correct. The only place either one shows up is
**in the bytes**, which is why the deliverable is a byte-comparison test.

**Env overrides.** None — the fix is unconditional. `song_stream_test.py` reads
`SONG_FILE_PATHS` from config.

---

# Part C — Measured hardware findings

Numbers from instrumented runs on this specific unit. They explain why several
of the "obvious" software fixes are the wrong lever.

**2026-09-29, new plastic body — the noise bed rose ~24 dB and it is white.**
Measured floor moved 1082 → ~7900 raw (4013 post-chain) when the components were
refitted. `_micnoise.py` characterised it from the raw 32-bit words:

| Measurement | Result | Rules out |
|---|---|---|
| Bit occupancy, bits 0–7 | always zero | misalignment, floating data line |
| Spectrum tilt (hi − lo) | −1.5 to +0.9 dB — **flat** | sigma-delta noise shaping |
| Tonality (peak − median) | 4.7–6.8 dB — smooth | clock / switching pickup |
| Block-RMS spread p95/p05 | 0.6–2.7 dB — steady | duty-cycled aggressor (Wi-Fi, camera, regulator) |
| corr(L, R) | −0.008 | common-mode: shared clock, shared rail, radiated pickup |

What is left is a flat, steady, **per-channel electrical** bed ~61 dB above
INMP441 datasheet self-noise in the speech band, on both channels — including
the deaf right one, which reads 3665–3870, nearly the same as left. A mic that
cannot hear a chime but still produces a full noise bed is producing it
electronically.

**2026-09-30 correction — the magnitude was understated, and the absolute
number is the one that matters.** Everything above is a *relative* measurement,
so it establishes the bed's *character* but not its *size*. `mic_noise_probe.py`
measures the raw capture against the microphone's own full scale and against the
datasheet, and the bed is worse than the table above implies:

* **+61 dB over spec in the 80–8000 Hz speech band, +69 dB full band on the
  left channel.** The `~62 dB` figure above was computed full-band from a
  quieter capture; the honest in-band number is +61 and the full-band left
  figure is +69.
* **The left stream rails to 0.0 dBFS on noise alone**, with the right at
  −1.0 dBFS. Nothing in the relative measurements above can show this, because
  the chain's gain and ceiling move with it.
* **94–96 dB SPL equivalent.** This is the number that explains the reported
  behaviour, and it is the one thing here a person can act on: see **A12**.
* **corr(L, R) = −0.20** on the new capture against −0.008 above. Both are far
  from +1 and both rule out the *dominant* path being common-mode, which is the
  conclusion that matters. The difference between them is capture-to-capture
  spread, not a change in the hardware.
* **Burstiness is unresolved, and the two probes disagree.** `_micnoise.py`
  reported block-RMS spread of 0.6–2.7 dB ("steady, rules out a duty-cycled
  aggressor"); the `mic_noise_probe.py` capture shows kurtosis of 7.3 (L) and
  32.1 (R) against 3.0 for Gaussian noise, i.e. clearly **impulsive**. Steady
  average power with high kurtosis is consistent with a *duty-cycled* aggressor
  — a servo, or a switching regulator under load — which is exactly what
  `_micnoise.py` ruled out. Do not treat that row as settled. It does not change
  the verdict (61 dB is 61 dB either way) but it does change the repair: if the
  noise is impulsive, the fix in A12 step 1–2 is the wiring and the supply, not
  the microphone part.

**Status: open, hardware — and it is the primary fault on this unit.** Startup
calibration works around it by giving up gain until the noise fits under the
ceiling, and that held. But the SNR is gone for good: a quieter front end is
worth more than any remaining software change in this area. See **A12** for the
repair order.

Note that an earlier characterisation of this bed as *"HF hiss, 82 % above
3.2 kHz"* was **an artifact of unequal band widths** — 6.4–24 kHz is 73 % of the
spectrum, so white noise naturally lands most of its power there. Band
percentages computed over unequal widths are not evidence of tilt; measure tilt
directly.

**2026-09-29 — the unit was shipped clipping.** `~/adam/.env` carried
`MIC_S32_SHIFT=15`, 6 dB hotter than the documented 16. Combined with the raised
bed above, that left **3.9 dB of headroom over silence** and flattened 18.3 % of
the calibration chime's samples. Measured on one set of captures:

| Mic gain | Headroom over silence | Chime samples clipped |
|---|---|---|
| shift 15 — as shipped | **+3.9 dB** | **18.316 %** |
| shift 16 — documented default | +9.9 dB | 0.081 % |
| calibrated from the tone | **+20.2 dB** | **0.000 %** |

This is the mishearing. Clipping generates odd harmonics that flatten vowels and
destroy /p/ /b/ /s/ differentiation. It hid because every level readout in the
system passed through the int16 clip, so a clipped measurement could not report
that it clipped. Fixed by solving the divisor from the startup chime rather than
hand-picking it — see `startup_mic_calibration.md`.

**The binding constraint is in-band SNR, not level, not clipping, not
filtering.** Post-filter floor 1550–1591 against speech p90 2041–4256 is
**+2 to +12 dB**, typically ~+6 dB. Every other mic finding below is real but
small next to that number: the pre-A9 aliased image sat ~40 dB down, and the
high-pass comb ripple is ±1.7 dB at 400 Hz decaying to ±0.15 dB by 800 Hz.
A ±1.7 dB wobble and a −40 dB image do not corrupt a recogniser that is
already working 6 dB above its noise. A10 is the fix that addresses the actual
constraint; A9 was fixed because it was cheap and exact, not because it was
the cause.

**The left mic channel has a fault that is not acoustic.** — ⚠️ **REFUTED, and
it was the wrong channel.** Superseded by A8/A11: the fault is on the **right**
channel, and the left one is the only good mic on this unit. The original
observations are kept below with what re-measurement actually showed.

- ~~Left peaks at −1.0 to −3.7 dBFS on room noise alone; right at −7.0 dBFS.~~
- ~~Left carries **7.1 dB more RMS** and a DC offset **~300× larger** than
  right.~~ Re-measured with `mic_cause.py`: at idle **L is −36.1 dBFS and R is
  −24.8 dBFS**, i.e. the *right* channel is 11.4 dB louder. The DC offset is
  irrelevant either way — the mono path removes DC before any scaling.
- ~~**80.85%** of raw ambient energy sits below 60 Hz, loudest component
  **26.4 Hz**.~~ True of the *raw* S32 stream, and already false by the time
  anything downstream sees it: measured on the real 16 kHz path, **0-60 Hz is
  0.07-0.95%** of total energy and the loudest component is 343.8 Hz. The
  high-pass at `MIC_HP_HZ` was already doing its job, so subsonic rumble was
  never eating the speech band and the "26 Hz rail hum" reasoning below does
  not lead anywhere.
- ~~Right-only measured 5.4 dB worse in band, so `MIC_CHANNEL` stays `mix`.~~
  The number is real; the conclusion inverted the fix. It is exactly what a
  dead, noisy right channel measures, and **left-only was never measured**.
  Left-only is worth **11.7 dB broadband and up to 21 dB in the consonant
  band** over the mix (A8).
- ~~A continuous watch will drop the left channel if it is ever caught
  saturating.~~ Saturation is the wrong criterion: this unit has **7.4 dB of
  headroom and 0 saturated samples in 12 s**, so no clipping-based test can
  ever fire. Replaced by the responsiveness test in A8.

Still true: there is **no capture gain control in this hardware**, so there is
nothing to turn down.

**Playback path.**

- Tone loopback rate and pitch are **exact** — every measured ratio 1.0000.
  Sample-rate handling is correct end to end.
- 1 kHz came back **21.6 dB below** 300 Hz. That is the speaker and enclosure,
  not the microphone and not the code. A physically larger driver or a sealed
  enclosure is the only fix.
- `SPEAKER_GAIN=1.0` costs about **2.3 dB** of loudness versus the old
  clipping-prone setting. That is the correct trade; `SPEAKER_GAIN=1.15` exists
  as an escape hatch if a unit is genuinely too quiet, but fix B1 first.
- The `aplay` prebuffer was reduced **1.0 s → 0.4 s**, which is most of the
  round-trip latency improvement.
- Underruns while the device sits idle between replies are **benign** and now
  say so in the log. Over 90 minutes: 58 benign underruns, 2 overruns.

**Refuted approaches, recorded so they are not retried.**

- **webrtcvad is useless in this room.** It labelled **100.0%** of the noise
  "speech" at aggressiveness 0, 1 and 2, and 98.6% at 3. It is off by default.
  Spectral flatness plus the low/high band ratio is what actually separates
  speech from this noise bed.
- **An EMA noise floor cannot work here.** Speech contaminates the average that
  is supposed to represent silence. The floor is a **low percentile of a long
  window**, which is immune to the speech it measures.
- **A 5-consecutive-chunk onset rule tested clean and was still wrong.** It
  passed its own validation because the validation used synthetic *sustained*
  vowels, which are exactly the signal a consecutive-run rule handles well.
  Real speech starts with plosives and unvoiced consonants that dip below the
  threshold mid-word. Replayed against the recorded pattern of a real
  "Hey ADAM" — `[0,0,1,1,0,1,1,1,0,0,1,1,1,0,1,1]` — the 5-consecutive rule
  never opens the gate and the 3-of-6 quorum does. Validate onset logic against
  captured speech patterns, never against generated tones.
- **`MIC_CHANNEL=right` as a fix for clipping.** See A8: clipping has never
  been measured on this unit (7.4 dB headroom, 0 saturated samples in 12 s), and
  `right` is the **deaf** channel. The old "costs 5.4 dB of in-band SNR" figure
  was measured against the mix, not against `left`, which is the only good
  channel here.
- **Comparing two mic measurements taken minutes apart.** The single biggest
  time-waster in this investigation. The room bed on this unit moves by 10 dB
  or more between captures, so `left` measured p50 380 in one run and 1842 in
  another and looked like a hardware intermittency. It was not. Force all
  candidates through the **same** captured bytes in **one** process
  (`mic_modes.py`) before believing any difference between channels or modes.
- **Narrowing `MIC_LP_HZ` to 6200 to escape the aliasing.** See A9. It does
  move the transition band below Nyquist, but by deleting the 6.2–8 kHz
  fricative energy that the confused consonants are made of. More taps fixes
  the same problem without paying for it.
- **More over-subtraction is not automatically better.** `MIC_NR_OVERSUB=2.0`
  at α=0.70 removed 2.8 dB of noise and 3.1 dB of speech — a net loss of
  0.3 dB. The constant is only meaningful relative to the measured
  minimum-statistics bias at the chosen `MIC_NR_SMOOTH`; see the bias table in
  A10 before touching either.
- **Deeper suppression floors.** `MIC_NR_FLOOR_DB` below about −15 dB produces
  audible musical noise and eats unvoiced consonants — the same class of damage
  A9 was fixing, reintroduced by the tool meant to help.

---

# Part D — Known-good reference state

Captured 2026-09-05 with the mic confirmed working by the user. If a future
change breaks hearing, compare against these numbers before changing anything.

```
📊 Mic 10s: p50 1181 p90 1266 p99 1401 max 1515 | open≥1416 hold≥1201 | floor 1133 flat 0.50/0.47 lohi 0.21 shp 40% | opens 0 sent 0 | blocked 0 | shut
```

- learned floor settles **1126–1144**, open threshold **1408–1430**
- `flat_max` learned up from the 0.35 baseline and settled at **0.46–0.47**
  (ceiling 0.70 never reached) — this room's noise flatness is 0.49–0.58, so
  the fixed baseline really was too tight here
- quiet windows: `opens 0`, `blocked 0` — no false opens observed
- `.mic_floor.json` is rewritten live, so the converged floor survives a
  restart: `{"t": 1788592745.1033168, "floor": 1135.05}`

**Second reference, conversation under way** (2026-09-05 13:43 run, with the
language lock, the 1.0 s hangover and `ENABLE_IDLE=0` all active):

```
📊 Mic 10s: p50 1625 p90 2041 p99 2508 max 2783 | open≥1963 hold≥1665 | floor 1570 flat 0.46/0.42 lohi 0.53 shp 80% | opens 1 sent 39 | blocked 0 | OPEN
```

- floor sits **1550–1591** here, ~400 higher than the quiet reference above,
  because the playback device is open for much of a real conversation
  (`+AMP` on the mode field) and the amp raises the room floor. Both thresholds
  track it, so this is the estimator working, not drifting.
- `flat_max` learned to **0.42** in this session against 0.46–0.47 in the
  quieter one — the learned value is per-session and per-room by design.
- `blocked` ran **0–3** per window and every open was followed by a reply.
- `sent` was non-zero on all 27 windows where anyone spoke, and `sent 0` on the
  29 windows where nobody did. `sent 0` during silence is correct; `sent 0`
  while `opens` is non-zero is the A3 signature.

**Third reference, everything in this document applied** (2026-09-05, quiet
room, A9 + A10 live — note the new `nr` field and that `floor`/`p50`/`open≥`
are still raw-path numbers so they remain comparable with the two blocks
above):

```
📊 Mic 10s: p50 1855 p90 1947 p99 2033 max 2096 | open≥2232 hold≥1893 | floor 1786 flat 0.56/0.48 lohi 0.13 shp 13% | opens 0 sent 0 | blocked 0 | nr -10.9dB | shut
```

- `nr` sits at **−10.7 to −11.4 dB** in silence, i.e. pinned near the −12 dB
  floor, which is what "no speech present, suppress hard" looks like. During
  speech it should rise toward −2 to −6 dB. `nr +0.0dB` on the very first
  window after a restart is normal: the estimator needs ~1.5 s to prime and
  passes audio through untouched until then.
- This room's floor was **1786** on this run against 1133 in the first
  reference and 1570 in the second — a 4 dB spread across three runs of the
  same unit. That spread is exactly why the gate learns its floor instead of
  using a constant, and why A10 estimates noise per-bin at runtime rather than
  shipping a fixed profile.
- Measured CPU on the Pi Zero 2 W: decimation **1.96 ms** and suppression
  **2.15 ms** per 33.3 ms chunk — about **12 %** of one core for the whole mic
  chain.

**Do not "tune" the mic further while it is behaving.** The gate has three
interacting adaptive loops (floor percentile, learned flatness, quorum window);
changing one constant to fix a symptom usually moves the working point of the
other two.

**Rollback.** The previous build of the four changed modules is on the Pi at
`~/adam_backup_20260905/`:

```bash
cp ~/adam_backup_20260905/*.py ~/adam/ && sudo systemctl restart adam
```

That snapshot predates **A6 through A10**, so rolling back also removes the
language lock, the 1.0 s hangover, the channel watch, the anti-alias fix and
the noise suppressor. To back out only the suppressor, set `MIC_NR=0` in
`~/adam/.env` and restart — the transform is unity-gain, so that restores the
previous audio bit for bit without touching anything else.

---

# Part E — Still open, not caused by any of the above

These were observed in the same runs but are separate problems.

- **The microphone noise floor (A12).** Listed here too because it is the one
  open item that blocks normal use: until the harness and supply are fixed, ADAM
  needs to be shouted at. Everything else in this list is an inconvenience.
  Software has done what software can do — it now measures the fault and says so
  at boot.
- **Whether the bed is impulsive or steady.** `_micnoise.py` says steady
  (block-RMS spread 0.6–2.7 dB); `mic_noise_probe.py` says impulsive (kurtosis
  7.3 / 32.1 vs 3.0 Gaussian, peaks railing to 0.0 dBFS). Both were run on this
  unit. A duty-cycled aggressor with a high repetition rate would produce
  exactly that combination — steady average power, spiky sample distribution —
  so the most likely reconciliation is a servo or switching regulator, not a
  measurement error. Resolving it would narrow A12's repair list from four steps
  to one. The way to settle it: capture with the servos unpowered and compare
  kurtosis.
- **ESP32-CAM link dead.** `⚠️ UART port is open but no data received from
  ESP32-CAM in 10s — running WITHOUT vision/touch (audio-only mode).` The port
  opens, so the Pi side is fine; check the ESP32-CAM is powered, that TX/RX are
  not swapped, and that both ends are at 921600. Until this is fixed there is
  **no Touch3**, which is why the spoken stop phrase for songs matters.
- **Laptop agent not discoverable.** `⚠️ mDNS discovery found no
  '_adam-laptop._tcp.local.' service within 3.0s`, repeatedly. The laptop-side
  advertiser is not running or is on a different subnet.
- **One large capture overrun** seen historically:
  `[arecord] overrun!!! (at least 1687.320 ms long)`. Not reproduced in the
  2026-09-05 run.
- **Reply latency 1.0–2.0 s** end to end, dominated by the model, not the audio
  path. In the verified run: speech detected → transcript ≈ 2 s, transcript →
  first audio out ≈ 2 s. A7 deliberately adds **+0.4 s** on top of this; if
  latency has to come down, the model round-trip is where the seconds are, not
  the gate.
- **Occasional empty reply turns.**
  `🤖 ADAM: [spoke but no output_transcription text captured — audio-only reply
  or empty turn]` appears a few times per conversation. Audio still plays, so
  this is a logging gap in `output_audio_transcription`, not a lost reply. Only
  worth chasing if you rely on the transcript for anything downstream.

## Things that look like faults in the log and are not

- **`⚠️ receive error: 1008 None. The operation was aborted.`** followed by
  `🔄 Session limit — reconnecting...` is Gemini Live's own session duration
  cap, roughly every 10 minutes. ADAM tears down all tasks, reopens `arecord`
  and `aplay`, and reconnects in about **5 s** — measured 14:49:44 → 14:49:49.
  `[arecord] Aborted by signal Terminated...` on the same second is that
  teardown, not a capture failure. Nothing to fix.
- **The learned floor jumping after a reconnect.** A run at 14:49 logged
  `🎚️ Resuming learned mic floor 2121 (open≥2651)` when the live floor a few
  seconds earlier had read 1874. That is correct: the room was genuinely loud
  (`p50` ~2370 sustained), the floor was still climbing toward it at
  `MIC_FLOOR_RISE`, and 2121 is what it had reached when `_maybe_save` last
  wrote. When the room went quiet the floor fell back to ~1600 within two
  windows, because `MIC_FLOOR_FALL=0.25` is 12× faster than the rise. The
  asymmetry is deliberate — see A1.
- **`flat 0.54/0.35` on the first stats line after a reconnect.** The learned
  flatness threshold is per-session and starts from the `MIC_SHAPE_FLAT_MAX`
  baseline. It re-converges after `MIC_FLOOR_MIN_S` = **1.5 s** of audio
  (45 chunks), not after a full 45 s window, so the tighter baseline is in
  force for about one and a half seconds. Visible in the log, not audible.
- **`nr +0.0dB` on the first stats line after a start or reconnect.** The
  minimum-statistics estimator needs ~1.5 s to prime and passes audio through
  at unity gain until it has. See A10.
- **`sent 0` during silence.** Correct. `sent 0` while `opens` is non-zero is
  the A3 signature and is the one to worry about.

---

# Appendix — every env override in one place

Set these in `~/adam/.env` on the Pi. **`.env` also holds `GEMINI_API_KEY` —
never paste its contents into a chat, an issue, or a log.** If that key has
ever been displayed in a shared terminal or session, rotate it.

Defaults below are read straight from `config.py`. Where the env var name and
the Python constant differ, the env var is what you set.

| Variable | Default | Issue |
|---|---|---|
| `MIC_VAD_ONSET_CHUNKS` | 3 | A1 |
| `MIC_VAD_ONSET_WINDOW` | 6 | A1 |
| `MIC_SHAPE_ADAPT` | 1 | A1 |
| `MIC_SHAPE_FLAT_CEIL` | 0.70 | A1 |
| `MIC_SHAPE_FLAT_MARGIN` | 0.95 | A1 |
| `MIC_SHAPE_FLAT_PCTL` | 5 | A1 |
| `MIC_SHAPE_FLAT_MAX` | 0.35 | A1 (baseline / floor of the learned value) |
| `MIC_SHAPE_RATIO_MIN` | 0.60 | A1 |
| `MIC_VAD_PREROLL_S` | 0.8 | A1 |
| `MIC_HP_HZ` | 100 | A1, C (rumble cut; **80** set in `.env` on this unit) |
| `MIC_LP_HZ` | 6800 | A1, A9 (anti-alias before 48k→16k) |
| `MIC_LP_STOP_HZ` | 8000 | A9 (stopband edge; sets the tap count) |
| `MIC_STATS_S` | 10.0 | reading the log |
| `MIC_DEAD_STREAM_S` | 3.0 | A2 |
| `MIC_DEAD_AFTER_PLAY_S` | 0.7 | A2 |
| `MIC_DEAD_AFTER_PLAY_WINDOW_S` | 3.0 | A2 |
| `SPEAKER_IDLE_CLOSE_S` | 0 | A2 (`0` = never close — the shipped default) |
| `ENABLE_IDLE` | 1 | A3 — **set to `0` on this unit** |
| `IDLE_TIMEOUT_S` | 90 | A3 |
| `MIC_VAD_HANGOVER_S` | 1.0 | A4, A7 (was 0.6) |
| `MIC_VAD_MAX_OPEN_S` | 15 | A4 |
| `MIC_VAD_ABS_MAX_OPEN_S` | 45 | A4 |
| `MIC_SHAPE_HOLD_FRAC` | 0.40 | A4 |
| `MIC_S32_SHIFT` | 15 | A5 — **leave alone** (sets `S32_SHIFT`) |
| `STT_LANGUAGE_CODES` | `hi-IN,en-IN` | A6 (empty = auto-detect all) |
| `MIC_CHANNEL` | `auto` | A8 (`auto`/`mix`/`left`/`right`) |
| `MIC_CH_WINDOW_S` | 90 | A8 (must span both speech and silence — 6 s failed) |
| `MIC_CH_MIN_S` | 15 | A8 (minimum window before deciding anything) |
| `MIC_CH_LIVE_DR_DB` | 8.0 | A8 (absolute: this channel responds to sound) |
| `MIC_CH_DEAD_DR_DB` | 3.0 | A8 (relative: still enough to be condemned) |
| `MIC_CH_DEAD_MARGIN_DB` | 5.0 | A8 (relative: how far the other must move) |
| `MIC_CH_DECIDE_EVERY_S` | 2.0 | A8 |
| `MIC_CH_STATE_PATH` | `adam/.mic_channel.json` | A8 (persisted decision) |
| `MIC_CH_STATE_MAX_AGE_S` | 30 days | A8 |
| `MIC_FS_SPL` | 120.0 | **A12** (INMP441 datasheet: −26 dBFS @ 94 dB SPL) |
| `MIC_SELF_NOISE_SPL` | 33.0 | **A12** (INMP441 datasheet: 61 dB(A) SNR) |
| `MIC_NOISE_WARN_DB` | 25.0 | **A12** (how far over spec before the fault block prints) |
| `MIC_CAL_GAIN_DEADBAND_DB` | 1.5 | **A13** (smallest gain change worth resetting the floor for) |
| `MIC_CAND_RATIO` | 1.45 | **A14** (candidate rail; room noise maxes at 1.27x) |
| `MIC_CAND_SHAPE_FRAC` | 0.60 | **A14** (sustain; noise maxes at 0.53 — must stay above it) |
| `MIC_CAL_NOISE_PCTL` | 25.0 | **A15** (percentile, not mean — a mean is one burst from failing) |
| `MIC_CAL_SILENCE_S` | 2.0 | **A15** (~22 frames for the percentile; was 1.2 s = 13) |
| `MIC_NR` | 0 | A10 — `1` enables the WOLA suppressor (+3.4 dB, 2.05 ms/chunk) |
| `MIC_NR_FRAME` | 512 | A10 (32 ms at 16 kHz; hop is half of it) |
| `MIC_NR_OVERSUB` | 3.5 | A10 — **re-derive if you change `MIC_NR_SMOOTH`** |
| `MIC_NR_SMOOTH` | 0.90 | A10 (per-bin power smoothing α) |
| `MIC_NR_FLOOR_DB` | −12 | A10 (raise toward −9 if voices sound hollow) |
| `MIC_NR_NOISE_S` | 1.5 | A10 (minimum-statistics window, 4 sub-windows) |
| `MIC_LIVE_RMS_THRESHOLD` | 1200 | attention/idle tracking, DOA trigger |
| `ENABLE_EXPANDER` | 1 | downward expander with pre-roll + hangover |
| `EXPANDER_FLOOR_DB` | −14.0 | how far non-speech is pushed down |
| `ENABLE_AEC` | 0 | full duplex — **deferred until half duplex is proven** |
| `POST_MUTE_S` | 0.35 (0.25 with AEC) | mic re-open delay after a reply |
| `SPEAKER_GAIN` | 1.0 | B1, C |
| `SONG_CHUNK_FRAMES` | 4096 | B4 |
| `SONG_PACE_FRAC` | 0.9 | B4 (must be < 1.0) |

**Actually set in `~/adam/.env` on this unit** — everything else is running on
its code default:

```
ENABLE_IDLE=0
SPEAKER_IDLE_CLOSE_S=0
ENABLE_SERVOS=0
MIC_S32_SHIFT=15
MIC_HP_HZ=80
MIC_LIVE_RMS_THRESHOLD=1200
STT_LANGUAGE_CODES=en-IN,en-US,hi-IN
MIC_NR=0
ENABLE_EXPANDER=1
EXPANDER_FLOOR_DB=-14.0
POST_MUTE_S=0.25
ENABLE_AEC=0
AEC_DELAY_MS=60
AEC_FILTER_LEN_MS=200
AEC_BARGE_RMS_MULT=14.5
```

`MIC_CHANNEL` is deliberately **not** set. `auto` now measures which mic
actually responds to sound (A8) and drops a deaf one; pinning it to `mix` is
what buried the consonant band under the dead right mic. The `AEC_*` values are
staged for full duplex and inert while `ENABLE_AEC=0`.

**The three `MIC_*_SPL` / `MIC_NOISE_WARN_DB` values in A12 are datasheet
constants, not tuning knobs.** They describe the INMP441 part, so the only
reason to change them is fitting a different microphone. Overriding them to
silence the boot warning does not make the unit hear you — it makes the one
measurement that found the fault stop reporting it.

## Per-unit state files — never sync these

Three files in `~/adam/` are measurements of **one physical unit**, not code:

| File | Written by | Holds |
|---|---|---|
| `.mic_channel.json` | `_MicChannelLiveness` | which mic is live, and each channel's dynamic range |
| `.mic_cal.json` | `mic_calibrate.py` | the solved divisor, and the noise-health report |
| `.mic_floor.json` | `AdaptiveGate` | the learned gate floor |

They are in `.gitignore` and they are **excluded from the Pi ↔ laptop parity
check**, which covers `*.py` and `*.txt` only. Copying one unit's state onto
another pairs a floor measured on one microphone with a divisor solved on a
different one — see `_restore_mic_scale()` in `audio_utils.py` for why that is
worse than having no state at all.

A legacy `.mic_ch.json` name appears in older notes and in `.gitignore`. It is
not a path any current code writes; the real name has always been
`.mic_channel.json` (`config.py:291`). If you find a stale zero-byte
`.mic_ch.json` on a unit, delete it.
