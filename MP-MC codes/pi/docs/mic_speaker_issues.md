# ADAM v40 — Mic & Speaker Fault Guide

What broke, why it broke, and either where it is fixed in code or what you
have to change yourself.

Written against the split package in `pi/adam/` with `main.py` as the
entrypoint — not the legacy monolith, and not the wiring in `setup.md` where
the two disagree. **The code is the source of truth.**

Fifteen faults are covered: ten that stop ADAM from HEARING or
UNDERSTANDING you (Part A) and five that make ADAM's VOICE sound wrong
(Part B). Eleven are fixed in code and need nothing from you. Three are hardware
or OS config and cannot be fixed in software at all. One is a misdiagnosis,
documented here so nobody "fixes" it later and makes things worse.

If you only read one section, read **A10**. Everything else in Part A is about
the gate opening or the turn ending; A10 is about the recogniser being handed
audio it cannot resolve, which is what "ADAM mis-hears everything" actually
means.

## Quick triage

| What you observe | Go to |
|---|---|
| ADAM talks (idle nudge fires) but never answers you | A1, A3 |
| ADAM stops hearing you a second or two after it finishes talking | A2 |
| ADAM answers only after you repeat yourself three times | A1 |
| ADAM hears you, then hangs forever without replying | A4 |
| Transcript comes back in a language nobody in the room speaks | A6 |
| ADAM replies in Portuguese / Spanish / Korean and keeps doing it | A6 |
| ADAM answers half your sentence, then answers the other half | A7 |
| Speech is muffled or distorted only on the loudest syllables | A8 |
| Every word comes back as a similar-sounding wrong word | A10, A9 |
| "ADAM" → "मैडम", "code" → "कोर्स", consonants swapped | A10, A9 |
| Gate opens, `sent` is healthy, reply is confidently about nothing | A10 |
| Voice level jumps around with no pattern | B1 |
| Voice is crackly or glitchy all the time | B2, B5 |
| Voice turns into loud buzz and stays buzzing | B3 |
| Everything lags while a song plays | B4 |

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

**The other lever, and why it is not the default.** `SPEAKER_IDLE_CLOSE_S=0`
never closes the playback device, so the wedge cannot happen at all. It costs
a **measured +4.1 dB of amp hiss** on the mic floor (floor p50 1082 → 1726
with `aplay` merely open on silence). At that floor the open threshold lands
around 3088, above this user's quietest measured speech of 2357 — the gate
would stop opening at all. Trading a 0.7 s recoverable stall for a permanently
deaf mic is the wrong trade, so the device still closes; use `0` only if the
wedge ever proves unrecoverable on some other unit.

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

## A8. `MIC_CHANNEL=auto` could never actually pick a channel — FIXED IN CODE

**Symptom.** None, directly — which is the problem. The log always says
`speech path uses MIX`, even on a unit where one channel is clipping, so a real
hardware fault stays invisible and untreated.

**Root cause.** `_mic_ch_calibrate()` in `audio_utils.py` tracks per-channel
peak and saturation for `_MIC_CH_CAL_CHUNKS = 30` chunks — 30 × 1600 frames at
48 kHz = **exactly 1.0 second** — and then latches a mode for the whole
session. Those 30 chunks are the first second after `arecord` starts, i.e.
**boot silence**, where clipping is impossible by construction. So the
saturation counters were always `L 0 / R 0`, the decision was always `mix`, and
`auto` was in practice a synonym for `mix` on every unit ever shipped.

That is why the headroom line prints exactly **once** per run (`journalctl |
grep -c` returns 1) and why the clipping hypothesis was untestable: the one
measurement that could have confirmed it was taken while nobody was speaking.

**Fix.** A continuous one-way watch alongside the existing calibration —
`_mic_ch_watch()`:

- accumulates saturation over rolling `MIC_CH_WATCH_S` (10 s) windows for the
  whole session, not just the first second;
- drops a channel only when it saturated ≥ `MIC_CH_WATCH_MIN_CLIPS` (20)
  samples in a window **and** the other channel saturated ≤ ⅛ as many — a
  ratio test, so a genuinely loud room does not cause a switch;
- is **one-way**: once a channel is dropped, `_mic_ch_forced` latches and the
  watch stops. No oscillation mid-conversation;
- is skipped entirely when the operator set `MIC_CHANNEL` explicitly.

Hooked in at the mono conversion:

```python
if _mic_ch_mode[0] is None:
    _mic_ch_calibrate(_l, _r)               # mixes until it latches
elif not _mic_ch_forced[0]:
    _mic_ch_watch(_l, _r)                   # keeps watching for clipping
```

**Why a watch and not just `MIC_CHANNEL=right`.** Forcing right-only was the
obvious fix and it was deliberately **not** applied. The measurement in Part C
says right-only is **5.4 dB worse in band** than the mix (post-filter noise
floor p50 1498 vs 804) — L's excess energy is subsonic and the 120 Hz
high-pass already removes it, while averaging two mics cancels uncorrelated
noise. Paying 5.4 dB of speech-band SNR *unconditionally*, to buy headroom that
is only needed on the loudest syllables, makes recognition worse on average.
Clipping happens inside the ADC, upstream of every filter in `audio_utils.py`,
so it is the one defect DSP cannot repair — which is the only reason it is
allowed to override that 5.4 dB, and only on evidence.

**Diagnostic.** When the watch fires it says so, with counts:

```
🎚️  Mic channel → RIGHT: the left channel saturated 34 samples in the last 10s (the right channel: 2)
```

**Current status on this unit: not fired.** Across the verified conversational
runs no switch line appeared, and both boots reported `saturated samples
L 0 / R 0`. So on *this* hardware the left-mic clipping hypothesis is **not
confirmed** — the headroom is uncomfortably small (A5) but actual ADC
saturation during speech has never been observed. If it ever happens the log
will now say so, with numbers, instead of leaving it to guesswork.

**Env overrides.** `MIC_CHANNEL` (`auto`/`mix`/`left`/`right`),
`MIC_CH_WATCH_S=10.0`, `MIC_CH_WATCH_MIN_CLIPS=20`, `MIC_CH_CLIP_FRAC=0.995`.

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

# Part C — Measured hardware findings

Numbers from instrumented runs on this specific unit. They explain why several
of the "obvious" software fixes are the wrong lever.

**The binding constraint is in-band SNR, not level, not clipping, not
filtering.** Post-filter floor 1550–1591 against speech p90 2041–4256 is
**+2 to +12 dB**, typically ~+6 dB. Every other mic finding below is real but
small next to that number: the pre-A9 aliased image sat ~40 dB down, and the
high-pass comb ripple is ±1.7 dB at 400 Hz decaying to ±0.15 dB by 800 Hz.
A ±1.7 dB wobble and a −40 dB image do not corrupt a recogniser that is
already working 6 dB above its noise. A10 is the fix that addresses the actual
constraint; A9 was fixed because it was cheap and exact, not because it was
the cause.

**The left mic channel has a fault that is not acoustic.**

- Left peaks at −1.0 to −3.7 dBFS on room noise alone; right at −7.0 dBFS.
- Left carries **7.1 dB more RMS** and a DC offset **~300× larger** than right.
- **80.85%** of raw ambient energy sits below 60 Hz, loudest component
  **26.4 Hz**.

At 26.4 Hz the wavelength is ~13 m. Two microphones 5 cm apart cannot differ by
5 dB at that wavelength *acoustically* — the path difference is negligible. So
this is electrical or structure-borne coupling into the left channel only.
Likely candidates: routing near a switching node, a ground loop, or mechanical
contact with the servo/chassis.

Consequences already handled in software: the FIR high-pass at `MIC_HP_HZ`
(120 Hz) removes it from the speech path, and `MIC_CHANNEL` stays **`mix`**
because right-only measured **5.4 dB worse in band** despite being the cleaner
channel out of band. Since A8 that decision is no longer permanent — a
continuous watch will drop the left channel if it is ever caught actually
saturating during speech, which so far it has not been. There is **no capture
gain control in this hardware**, so there is nothing to turn down.

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
- **`MIC_CHANNEL=right` as a fix for clipping.** See A8: it costs 5.4 dB of
  in-band SNR unconditionally to solve a problem that has never been measured
  on this unit.
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
| `MIC_HP_HZ` | 120 | A1, C (rumble cut — the left-channel fix) |
| `MIC_LP_HZ` | 6800 | A1, A9 (anti-alias before 48k→16k) |
| `MIC_LP_STOP_HZ` | 8000 | A9 (stopband edge; sets the tap count) |
| `MIC_STATS_S` | 10.0 | reading the log |
| `MIC_DEAD_STREAM_S` | 3.0 | A2 |
| `MIC_DEAD_AFTER_PLAY_S` | 0.7 | A2 |
| `MIC_DEAD_AFTER_PLAY_WINDOW_S` | 3.0 | A2 |
| `SPEAKER_IDLE_CLOSE_S` | 2.5 | A2 (`0` = never close) |
| `ENABLE_IDLE` | 1 | A3 — **set to `0` on this unit** |
| `IDLE_TIMEOUT_S` | 90 | A3 |
| `MIC_VAD_HANGOVER_S` | 1.0 | A4, A7 (was 0.6) |
| `MIC_VAD_MAX_OPEN_S` | 15 | A4 |
| `MIC_VAD_ABS_MAX_OPEN_S` | 45 | A4 |
| `MIC_SHAPE_HOLD_FRAC` | 0.40 | A4 |
| `MIC_S32_SHIFT` | 15 | A5 — **leave alone** (sets `S32_SHIFT`) |
| `STT_LANGUAGE_CODES` | `hi-IN,en-IN` | A6 (empty = auto-detect all) |
| `MIC_CHANNEL` | `auto` | A8 (`auto`/`mix`/`left`/`right`) |
| `MIC_CH_CLIP_FRAC` | 0.995 | A8 (fraction of full scale counted as saturation) |
| `MIC_CH_WATCH_S` | 10.0 | A8 |
| `MIC_CH_WATCH_MIN_CLIPS` | 20 | A8 |
| `MIC_NR` | 1 | A10 — `0` restores the pre-suppressor audio exactly |
| `MIC_NR_FRAME` | 512 | A10 (32 ms at 16 kHz; hop is half of it) |
| `MIC_NR_OVERSUB` | 3.5 | A10 — **re-derive if you change `MIC_NR_SMOOTH`** |
| `MIC_NR_SMOOTH` | 0.90 | A10 (per-bin power smoothing α) |
| `MIC_NR_FLOOR_DB` | −12 | A10 (raise toward −9 if voices sound hollow) |
| `MIC_NR_NOISE_S` | 1.5 | A10 (minimum-statistics window, 4 sub-windows) |
| `SPEAKER_GAIN` | 1.0 | B1, C |
| `SONG_CHUNK_FRAMES` | 4096 | B4 |
| `SONG_PACE_FRAC` | 0.9 | B4 (must be < 1.0) |

**Actually set in `~/adam/.env` on this unit** — everything else is running on
its code default:

```
ENABLE_IDLE=0
```
