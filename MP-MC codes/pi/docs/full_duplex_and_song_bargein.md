# ADAM v40 — Full Duplex & Song Barge-In

What was attempted, what the hardware actually permits, and what shipped.
Written 2026-09-08 against the code running on `adam-pi`.

This document exists because two things were asked for and only one of them is
physically possible on this unit. Full duplex — streaming the mic to Gemini
*while* ADAM's speaker is playing — was investigated, measured, and **refuted**.
Song barge-in — a human voice interrupting a playing song — was built, and the
first design was also refuted before a second one was measured and shipped.

Every number below was measured on this specific unit. The benches that
produced them are checked in next to the code so any of it can be re-run.

Companion documents: [`mic_speaker_issues.md`](mic_speaker_issues.md) is the
symptom→fix reference organised by fault; [`development_log.md`](development_log.md)
is the narrative record of the earlier mis-hearing work.

---

## Summary for someone in a hurry

| Goal | Verdict | Why |
|---|---|---|
| Full duplex (mic → Gemini during playback) | **Not possible on this hardware** | Linear AEC achieves 1–2 dB ERLE on the real echo path; 18 dB is needed |
| Barge-in during songs, level-only detector | **Refuted — do not revive** | A singing voice and a speaking voice are the same instrument |
| Barge-in during songs, reference-based | **Shipped** | Compares mic to the song's own PCM; immune to song dynamics |

The half-duplex design (mic muted while the speaker is active) stays. It is not
a workaround for missing effort — it is the correct response to a measured
property of the speaker→mic path.

---

# Part 1 — Full duplex: why it is not available

## 1.1 What was tried

`speexdsp`'s `EchoCanceller` was wired up against the live capture and playback
paths and swept across every axis that could plausibly matter:

- **Probe level:** −30, −24, −18, −12, −6 dBFS
- **Filter length:** 50, 100, 200, 400 ms
- **Alignment:** −150 ms to +50 ms around the correlation peak

The bench is [`aec_bench.py`](../adam/aec_bench.py). It reports ERLE (Echo
Return Loss Enhancement — how many dB of the speaker's own sound the canceller
removes from the mic) alongside clip %, so a good-looking number caused by a
saturated mic cannot pass unnoticed.

## 1.2 The result

**1–2 dB ERLE at every level, every filter length, every alignment.**

For context: usable full duplex needs roughly 18 dB. At 2 dB the canceller is
doing essentially nothing, and Gemini would hear ADAM's own voice as though a
second person were talking over it.

## 1.3 Ruling out the boring explanations

A 2 dB result is much more often a broken harness than a broken speaker. Each
alternative was eliminated:

**Was the mic clipping?** No. The level sweep is the direct test: at −30 dBFS
with **0.00 % clipped samples**, ERLE was still **+0.4 dB**. Clipping was the
first hypothesis and it was wrong — quieter probes did not help at all.

**Was the alignment wrong?** No. `find_lag` located the echo at 5.9 ms with a
peak-to-median ratio of ~210 — an unambiguous, sharp correlation peak. Sweeping
alignment by hand did not find a better position.

**Was the capture chain nonlinear?** No. `_MicChain` is a linear low-pass,
decimation and high-pass. Nothing in it can create the distortion being seen.

**Was some codec AGC fighting the canceller?** No. The Google voiceHAT codec
has no AGC or DSP block to blame.

**Was the bench itself broken?** This is the one that mattered, so it was tested
directly. [`aec_selftest.py`](../adam/aec_selftest.py) synthesises a *perfect
linear echo* — a delayed, scaled copy of the reference, no room, no hardware —
and runs it through the exact same `find_lag` and `erle` functions:

```
[synthetic] injected delay    = 5.88 ms
[synthetic] find_lag returned = 5.88 ms
[synthetic] ERLE (200ms)      = +20.4 dB   clip=0.00%
```

**The harness measures +20.4 dB on a linear echo and +1–2 dB on this speaker.**
The measurement is sound; the echo path is the problem.

## 1.4 What the numbers say the problem is

The filter-length curve is the tell:

| Filter length | ERLE |
|---|---|
| 50 ms | **+2.2 dB** |
| 100 ms | +1.6 dB |
| 200 ms | +1.1 dB |
| 400 ms | +0.8 dB |

A *longer* adaptive filter scoring *worse* is backwards for a linear echo — more
taps should capture more of the room's tail. This is the signature of a
**nonlinear echo path**: the extra taps have no consistent linear relationship
to fit, so they only add adaptation noise. The boxed speaker driving a small
plastic shell is distorting, and a linear canceller cannot model distortion by
construction.

## 1.5 Decision

**Do not stream the mic during playback.** No amount of tuning recovers 16 dB
from a nonlinearity. Getting full duplex on this unit would need either
nonlinear/neural echo cancellation or a mechanical fix to the speaker mounting —
both out of scope for a software change, and the mechanical route should be
tested first because it is cheaper and it also helps every other audio path.

This verdict is recorded at the top of `aec_bench.py` so the next person to open
it sees it before re-running the sweep.

---

# Part 2 — Song barge-in, attempt 1: refuted

The first design was reference-free: learn the music's own mic level, and fire
when the mic rises above it *and* is speech-shaped.

It was implemented in full, then tested, then deleted. The failure is worth
recording because the idea is the obvious one and will otherwise be reinvented.

## 2.1 The two bugs that were fixable

Both were found and fixed before the fatal one surfaced:

1. **Floor pumping.** The first version updated the level floor from every
   chunk. Sustained speech pumped the floor up above itself, so it could never
   trigger. Fixed by making the floor a rolling-window **p70 percentile** — a
   few chunks of speech are ~1.5 % of a 200-sample window, and a percentile
   ignores that.

2. **Wrong coupling constant.** An earlier attempt reused
   `SpeakerEnvelopeTracker`'s `coupling_factor = 0.70`, which was measured for
   ADAM's *speech* vibrating the shell. The mic reads full-range music at
   **≈ +3.6 dB (≈1.5×)** the digital level, so the song would have tripped its
   own barge-in continuously. Reverted.

## 2.2 The bug that killed the design

**A singing voice and a speaking voice are the same instrument.**

Both are formant-peaked and low-frequency dominant, so both pass any spectral
shape test. The shape test cannot reject a vocal track — and that is not a
tuning problem, it is what the two signals are.

The consequence, measured:

| Scenario | False stops |
|---|---|
| Instrumental, steady | 0 |
| Vocal song, steady | 0 |
| **Instrumental → loud vocal chorus** | **13** |
| **Vocal verse → chorus ×2.5** | **3** |

And the fatal comparison — a genuine 400 ms human barge-in was injected into
the instrumental track:

```
verse -> vocal chorus     fires at chunks [102, 105, 108, 111, 114, 117]
REAL 400ms human voice    fires at chunks [102, 105, 108, 111]
```

**Identical signature.** Reference-free, a chorus entering and a person speaking
are not merely hard to tell apart — they are the same event: a fast,
speech-shaped level rise above a floor that has not caught up. No level ratio
separates them, because there is nothing to separate.

A false stop is the worst possible failure here: the song dies unpredictably
mid-playback for no reason the user can see. This design could not ship.

---

# Part 3 — Song barge-in, attempt 2: shipped

## 3.1 The insight

The chorus is in **the song's own PCM**. The human's voice is not.

We are the ones playing the song, so the reference signal is free — the exact
bytes are in hand in `song_playback.py`. So instead of tracking the mic's
absolute level, track the **ratio**:

```
ratio = mic_rms / expected_music_level
```

- A chorus raises the mic **and** the reference together → **ratio flat** → no trigger.
- A voice adds energy the reference does not contain → **ratio jumps** → trigger.

This is what makes it immune to song dynamics, and it is why the shape test
alone could never have worked.

## 3.2 The obstacle: aplay buffers, so the reference is early

`song_playback.py` writes into the shared `aplay` process's stdin, and `aplay`
buffers. A chunk written at wall-time `T` is *heard* at `T + B`. `B` is **not**
the 5.9 ms acoustic flight time, and guessing it is unsafe in a specific
direction: guess too small and the reference leads the mic, the expected level
comes out too low, and the song stops itself. That is the false-trigger
direction.

So `B` was measured rather than assumed. [`song_lag_bench.py`](../adam/song_lag_bench.py)
reproduces the real path exactly — one long-lived `aplay`, 4096-frame chunks
paced at `SONG_PACE_FRAC`, `arecord` capturing concurrently — and writes a train
of chirp bursts whose write times are recorded, then locates each burst in the
capture.

**First run was contaminated** and it is worth saying how: with bursts 1 s apart
and a 1.5 s peak-search window, the window spanned two bursts and `argmax`
sometimes locked onto the neighbour, producing nonsense 2.3 ms readings among
sane ~950 ms ones. Re-run with 2.5 s spacing and a search window clamped below
the burst gap:

| burst | lag | |
|---|---|---|
| 0 | 541 ms | ALSA buffer cold |
| 1 | 793 ms | filling — `SONG_PACE_FRAC=0.9` overfeeds by 11 % |
| 2–11 | **929–956 ms** | **saturated: 941 ms ± 14 ms over 25 s** |

So `B` ramps for the first ~5 s of a track while the buffer fills, then locks
hard. The 415 ms "spread" is entirely the cold-start ramp, not jitter.

`SONG_BARGE_LAG_MIN_MS = 450` / `SONG_BARGE_LAG_MAX_MS = 1050` bracket both the
cold ramp and the saturated value with margin.

> **Re-run `song_lag_bench.py` if `SONG_CHUNK_FRAMES`, `SONG_PACE_FRAC` or the
> ALSA buffer settings ever change.** `B` is a property of that plumbing, and
> these two constants are the only place it is written down.

## 3.3 The structural guarantee that sets the threshold

`_expected()` takes the **maximum** reference level over
`[T − LAG_MAX, T − LAG_MIN]`. Max, not mean, and that choice does real work:

Because the window is guaranteed to contain the instant the mic is hearing,
**music alone can only ever produce `ratio ≤ its true coupling`.** The
distribution is bounded above. Anything above that bound must come from a source
not present in the reference — that is, from the room.

Measured music-only ratios, across three very different kinds of material:

| Material | p50 | p95 | max |
|---|---|---|---|
| Instrumental, steady | 1.30 | 1.50 | 1.50 |
| Instrumental → vocal chorus ×2.25 | 1.25 | 1.50 | 1.50 |
| Vocal song, steady | 1.24 | 1.50 | 1.50 |
| Vocal verse → chorus ×2.5 | 1.23 | 1.50 | 1.50 |

Tight, and **the bound is identical (1.50) in all four** despite wildly
different dynamics. That is the property the whole design rests on.

Measured voice ratios over instrumental music at the same level:

| Voice amplitude | min ratio observed |
|---|---|
| 0.30 | 1.88 |
| 0.45 | 2.48 |
| 0.60 | 3.14 |
| 0.80 | 4.07 |

The bar sits at `coupling × SONG_BARGE_RATIO = 1.50 × 1.25 = 1.87` — 25 % clear
of music's bound, and right at the quietest voice measured.

## 3.4 Why coupling is learned from a *high* percentile

Since the bound is at the top of the distribution, that is what has to be
estimated: `coupling = p95` of observed ratios, over a 20 s window.

Learning is gated on `ratio ≤ current bar` — every chunk teaches coupling
*except* ones already over the bar. This is what stops a sustained voice from
lifting the bar above itself.

**The earlier version gated learning on the shape test instead, and that
deadlocked.** On an all-vocal song every chunk is speech-shaped, so nothing ever
taught coupling and the detector stayed permanently disarmed — measured as
`coupling = 0.00` and zero fires even for a real voice. Gating on the bar
instead works on any material, and it is also the more honest rule: a voice only
ever *raises* the ratio, so excluding the top is the correct way to exclude it.

The shape test is kept, but only as a **confirmation** at fire time. It rejects
a door slam or a dropped object — broadband transients that raise the ratio just
as a voice does. It is never allowed to gate learning again.

It reads `_adaptive_gate.flat_max` (a read-only property; it advances no gate
state) rather than the static `MIC_SHAPE_FLAT_MAX`, because that config value is
only the *floor* of the adaptive range. Using the static floor would make this
test stricter than the gate's own in a noisy room and silently cost the user
barge-ins.

## 3.5 Test results

Ten scenarios against the real `feed()`, with the measured 941 ms lag simulated:

```
--- must NOT fire ---
PASS  A instrumental steady            coup=1.50 bar=1.87 fires= 0
PASS  B inst -> VOCAL CHORUS x2.25     coup=1.50 bar=1.87 fires= 0   <- killed v1 (13 false stops)
PASS  C vocal song steady              coup=1.50 bar=1.87 fires= 0
PASS  D vocal verse -> chorus x2.5     coup=1.50 bar=1.87 fires= 0
PASS  G sudden quiet passage           coup=1.50 bar=1.87 fires= 0
PASS  H door slam over music           coup=1.50 bar=1.87 fires= 0
--- MUST fire ---
PASS  E inst + voice amp=0.30          coup=1.50 bar=1.87 fires= 3  [202, 205, 208]
PASS  E inst + voice amp=0.45          coup=1.50 bar=1.87 fires= 4  [202, 205, 208, 211]
PASS  E inst + voice amp=0.60          coup=1.50 bar=1.87 fires= 4  [202, 205, 208, 211]
PASS  F vocal song + voice 0.60        coup=1.50 bar=1.87 fires= 3  [202, 206, 211]

10/10 passed
```

Coupling converged to exactly **1.50** — the true injected value — in every
scenario, including the all-vocal material that deadlocked the previous version.
Voice starts at chunk 200 and fires at 202 in every case: **~90 ms latency**,
which is the 3-chunk debounce.

## 3.6 Known limitation, and why it is acceptable

**Over a loud chorus, a voice must be comparably loud to move the ratio at all.**
Measured: against a chorus at amplitude 0.45, a voice at 0.45 produces ratio
1.10 and at 0.60 produces 1.35 — both below music's own 1.50 bound, therefore
undetectable.

This is physics, not tuning. The voice has to add meaningful energy on top of
what is already playing. Two things make it acceptable:

1. The failure direction is **deafness, which is recoverable** — the offline
   Vosk stop phrase ("adam stop" / "stop the song") still works during songs and
   needs no level headroom at all. A false stop would *not* be recoverable.
2. Every threshold errs toward deafness by construction (`max` for the expected
   level, high percentile for coupling, 3-chunk debounce, shape confirmation).
   That was deliberate.

---

# Part 4 — What changed in the code

| File | Change |
|---|---|
| [`audio_utils.py`](../adam/audio_utils.py) | New `SongBargeIn` class (reference-based) + `_song_barge` instance |
| [`song_playback.py`](../adam/song_playback.py) | `_song_barge.reset()` at track start; `feed_reference(data)` on every chunk written to `aplay` |
| [`session.py`](../adam/session.py) | Song branch feeds mic chunks to `_song_barge.feed()`; on a hit, sets `song_stop_requested`. Vosk stop-phrase path preserved underneath |
| [`config.py`](../adam/config.py) | Six `SONG_BARGE_*` keys, with the measurements that justify each in the comment block |
| [`aec_bench.py`](../adam/aec_bench.py) | Full-duplex sweep + the recorded verdict |
| [`aec_selftest.py`](../adam/aec_selftest.py) | Proves the AEC harness measures +20.4 dB on a linear echo |
| [`song_lag_bench.py`](../adam/song_lag_bench.py) | Measures the write-to-hear lag `B` |

## 4.1 Two further fixes, 2026-09-08 — popping and self-stopping

Reported after Part 3 shipped: *"in middle of song popping sound is comming and
after song playing few seconds adam stops it and starts speeking."* Those are
two symptoms of **one** root cause, plus the false barge-in it triggers.

### The cause: the song loop was writing faster than realtime

The old pacing was `pace_s = chunk_len × SONG_PACE_FRAC` with
`SONG_PACE_FRAC = 0.9`. That reads like "pace at 0.9× realtime" and is actually
the opposite: it wrote a 33 ms chunk every 30 ms, i.e. **110 % of realtime**.

It does not make the song play faster — `aplay` clocks that. It makes the
buffer grow, forever. Measured ramp: **541 → 941 ms**. Eventually the kernel
pipe filled, a blocking `write_all` stalled the entire song loop, and because
that loop shares the event loop with capture, the stall produced an
`[arecord] overrun`.

- **The popping** is that buffer pressure and the write stalls around it.
- **The self-stopping** is what happened next. After an overrun, the mic chunks
  that arrive were captured *seconds* ago. They have no matching reference
  inside the `[LAG_MIN, LAG_MAX]` window, so the barge-in detector judged a
  stale burst against a reference that never corresponded to it — and stopped
  the song, which from the outside looks exactly like ADAM interrupting itself
  to start talking.

### Fix 1 — pace to a fixed deadline

Each chunk is now scheduled against an absolute time computed from total audio
written:

```python
next_t = start_t + (frames_written / PLAYBACK_RATE) * lead   # lead == 1.00
```

This is **self-correcting**. The loop settles at exactly 1.00× realtime and
stays there: no unbounded growth, no pipe full, no feed stall. If the CPU runs
late the sleep clamps to zero and the chunk ships immediately, so the song
catches up rather than drifting further behind — the buffer shrinks to find the
correct rate instead of growing away from it.

A ~500 ms head-start (`frames_written` starts negative) lets the cold ALSA
buffer fill at track start, then the loop converges to realtime and holds.

`lead` is hardcoded to 1.00 and is **deliberately not** `SONG_PACE_FRAC`. Any
value below 1.0 reintroduces the bug exactly. `SONG_PACE_FRAC` survives in
`config.py` only because `song_lag_bench.py` still reproduces the old pacing in
order to measure the lag window.

### Fix 2 — stale-audio guard

Fix 1 removes the cause, but an overrun can still happen for reasons outside
this loop (CPU contention, a reconnect). So the detector now refuses to judge
audio it knows is stale:

> If the gap between consecutive mic chunks fed to the barge-in exceeds
> `SONG_BARGE_STALL_S` (0.60 s), an overrun has occurred. Drop those chunks and
> enter a recovery window of `2 × STALL_S` (1.2 s) during which barge-in is
> refused, then resume normally.

An overrun therefore degrades to a **brief deaf window** instead of a false stop.
That is the right direction: being briefly unable to hear a barge-in is
recoverable — the user repeats themselves, or uses the spoken stop phrase — while
the song stopping itself is not.

### Effect on the lag window

The lag bounds `450–1050 ms` were measured under the *old* overfeeding pacing,
where the buffer ramped 541 → 941 ms. Under deadline pacing it settles near the
541 ms end and stays there, so the window is still correct but its upper half is
now unused headroom. Left as-is deliberately — it is the conservative direction,
and a wider window costs nothing.

### Test

[`barge_stall_test.py`](../adam/barge_stall_test.py) drives the guard directly.
Last run, on the Pi, after the mic-gain recalibration:

```
coupling converged to 1.500 (expect ~1.50)
stale burst produced 0 false barge(s) (expect 0)
post-recovery voice fired 8 time(s) out of 10 (expect >=1)
normal-path voice fired 8/10 (expect >=1)
music-only false: 0/300 (expect 0)
PASS
```

The two rows that matter: a stale burst causes **zero** false barges, and the
detector still works normally **after** the recovery window closes — a guard that
latched deaf would pass the first test and fail the second.

## Configuration

Nothing below needs tuning per room or per unit — `coupling` is learned at
runtime and absorbs speaker volume, room, mic gain, shell coupling and the
capture chain's filtering in one number.

| Key | Default | Meaning |
|---|---|---|
| `SONG_BARGE_WINDOW_S` | 20 | Rolling window for learning coupling |
| `SONG_BARGE_RATIO` | 1.25 | Safety margin above music's measured bound |
| `SONG_BARGE_HOLDS` | 3 | Consecutive chunks required (~90 ms) |
| `SONG_BARGE_MIN_N` | 30 | Samples before the detector arms (~1 s) |
| `SONG_BARGE_LAG_MIN_MS` | 450 | Lower bound of the measured write-to-hear lag |
| `SONG_BARGE_LAG_MAX_MS` | 1050 | Upper bound of the same |
| `SONG_BARGE_STALL_S` | 0.60 | Gap between mic chunks that means an overrun happened. Larger = more conservative (longer deaf window after a stall) |

---

# Part 5 — Still to verify on hardware

Everything in Part 3 was validated against synthetic material with the measured
941 ms lag simulated, and Part 4.1 against a synthetic stall. Two things are
**not** yet confirmed on the unit with a real track playing:

1. **No false stops across a full real song.** Real music has transients,
   silences and production compression that synthetic material does not. The
   number to watch is whether `coupling` settles near a stable value and stays
   there.
2. **A real voice at conversational volume actually stops the song**, and how
   far the user has to be from the mic for that to hold.

Also unconfirmed on real material, from Part 4.1: **no popping**, and **no
`[arecord] overrun`** across a full track. Both should now be structurally
impossible rather than merely unlikely — the loop cannot exceed realtime — so an
overrun appearing anyway would mean the cause is somewhere other than this loop.

Both need someone in the room with the unit. If a false stop does appear, check
whether `SONG_CHUNK_FRAMES` has changed since the lag was measured — that would
move `B` out of the 450–1050 ms window. (`SONG_PACE_FRAC` no longer affects the
shipped loop; it only drives `song_lag_bench.py`.)

---

# Part 6 — Decision register

| Decision | Reason | Status |
|---|---|---|
| Keep half duplex; do not stream mic during playback | 1–2 dB ERLE measured vs 18 dB needed; nonlinear echo path | Final, hardware-limited |
| Prove the AEC harness before trusting its verdict | A 2 dB result is usually a broken bench; this one measured +20.4 dB on a linear echo | Done |
| Reject the level-only barge-in detector | Chorus entry and human speech are the same event without a reference | Refuted, do not revive |
| Compare mic against the song's own PCM | Only signal that distinguishes chorus from human | Shipped |
| Measure `B` instead of assuming it | Guessing low causes the song to stop itself | Measured: 941 ms saturated |
| `max` (not mean) for expected level | Errs toward deafness, which is recoverable; a false stop is not | Shipped |
| Learn coupling from p95, gated on the bar | The bound is at the top; shape-gating deadlocks on vocal songs | Shipped |
| Keep the shape test as confirmation only | Rejects broadband transients; cannot distinguish singing from speaking | Shipped |
| Keep the Vosk stop phrase underneath | Needs no level headroom; the fallback when a chorus is too loud | Unchanged |
| Pace the song loop to a fixed deadline, not a fraction of chunk length | `SONG_PACE_FRAC=0.9` wrote at 110 % of realtime and ramped the buffer 541 → 941 ms until the pipe blocked. A deadline is self-correcting and cannot drift | Shipped |
| Hardcode `lead = 1.00`; do not expose it | Any value below 1.0 reintroduces the bug exactly, and it is not a tuning knob — 1.00× realtime is the only correct rate | Shipped |
| Drop mic chunks after a capture stall instead of judging them | Post-overrun audio is seconds old and has no matching reference. A brief deaf window is recoverable; a self-stopping song is not | Shipped |
| Leave the 450–1050 ms lag window unchanged after the pacing fix | The lag now settles near 541 ms, so the upper half is unused headroom. Widening the margin in the conservative direction costs nothing | Shipped |
