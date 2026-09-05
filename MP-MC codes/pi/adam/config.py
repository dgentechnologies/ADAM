"""
config.py — ADAM v40 configuration
==============================================================================
All tunable constants, environment loading, and static config live here.
Nothing in this file should import from any other ADAM module — it sits at
the bottom of the dependency graph so everything else can import from it
safely without circular imports.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Environment ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not set in .env")

# ═════════════════════════════════════════════════════════════════════════════
# AI MODEL
# ═════════════════════════════════════════════════════════════════════════════

LIVE_MODEL = "gemini-3.1-flash-live-preview"
VOICE      = "Charon"

# ── Which languages the speech-to-text is allowed to consider ─────────────
# BCP-47 hints passed to AudioTranscriptionConfig(language_codes=...).
#
# WHY THIS EXISTS. Leaving it unset is not "neutral" — the SDK documents that
# an omitted/empty language_codes means "automatic language detection", i.e.
# the recogniser scores the audio against 100+ languages and returns whichever
# scores highest. On clean, long utterances that works. On a SHORT fragment of
# accented Hindi it does not: measured live, real Hindi came back as Portuguese
# ("Tô com não, não") and Spanish ("peléan"), because "nahi nahi" / "karo na"
# genuinely are close phonetic matches to "não não" once a clause is only a
# second long. The downstream damage is bigger than a bad transcript: the
# system prompt tells ADAM to reply in the language the user just spoke, so one
# mis-detected fragment makes ADAM abandon Hindi mid-conversation.
#
# Naming the two languages actually in use collapses that search space. Keep
# this list SHORT — every extra entry re-widens exactly the confusion it is
# meant to remove, so add a language only when someone really speaks it.
# Hinglish (Hindi/English code-switching) needs no third code; it is covered by
# listing both. Set to an empty string to fall back to full auto-detection.
STT_LANGUAGE_CODES = [c.strip() for c in
                      os.getenv("STT_LANGUAGE_CODES", "hi-IN,en-IN").split(",")
                      if c.strip()]

# ═════════════════════════════════════════════════════════════════════════════
# FILE PATHS
# ═════════════════════════════════════════════════════════════════════════════

MEMORY_FILE        = BASE_DIR / "adam_memory.json"
FACE_MEMORY_FILE   = BASE_DIR / "adam_faces.json"
SYSTEM_PROMPT_FILE = BASE_DIR / "SystemPrompt.txt"
CONV_MEMORY_FILE   = BASE_DIR / "adam_conversations.json"

# ═════════════════════════════════════════════════════════════════════════════
# AUDIO — CAPTURE / PLAYBACK
# ------------------------------------------------------------------------------
# Both capture and playback use the Google voiceHAT soundcard (dual INMP441
# I2S mics + MAX98357A I2S amp), addressed BY NAME rather than card index.
# ALSA numbers the voiceHAT AFTER the HDMI audio card (vc4hdmi), so its index
# is not fixed (typically 1) and can shift across boots/kernels — but the name
# "sndrpigooglevoi" is stable. The old "plughw:0,0" pointed at card 0 = HDMI,
# which has NO capture device (arecord fails) and would route playback to HDMI
# instead of the speaker. Verified on real hardware: S32_LE/48k/2ch capture
# (mics live) and S16_LE/48k/2ch playback both open on plughw:sndrpigooglevoi,0.
# Overridable via .env (CAPTURE_DEVICE / PLAYBACK_DEVICE) if the card name ever
# differs — check `arecord -l` / `aplay -l`.
# ═════════════════════════════════════════════════════════════════════════════

CAPTURE_DEVICE   = os.getenv("CAPTURE_DEVICE", "plughw:sndrpigooglevoi,0")
CAPTURE_FORMAT   = "S32_LE"
CAPTURE_RATE     = 48000
CAPTURE_CHANNELS = 2

PLAYBACK_DEVICE   = os.getenv("PLAYBACK_DEVICE", "plughw:sndrpigooglevoi,0")
PLAYBACK_FORMAT   = "S16_LE"
PLAYBACK_RATE     = 48000
PLAYBACK_CHANNELS = 2

GEMINI_SEND_RATE = 16000
GEMINI_RECV_RATE = 24000
CHUNK_FRAMES     = 1600      # 33ms at 48kHz

# ── MIC DIGITAL GAIN ────────────────────────────────────────────────────────
# The INMP441 sends 24-bit audio left-justified in each 32-bit I2S frame.
# S32_SHIFT is how far we right-shift the raw S32 sample before the int16 cast
# in audio_utils. It's effectively the mic's *digital gain*: smaller shift =
# louder (and closer to clipping); +1 to the shift HALVES the level.
#
# 14 was too hot for the observed mic levels (raw RMS 130M-310M in the logs).
# On loud speech the shifted value went past 32767 and the int16 cast WRAPPED
# it into loud opposite-sign spikes — that garbling is why Gemini mis-heard
# English as random languages. 15 halves the level to give clean headroom;
# the np.clip() now in audio_utils saturates any rare remaining peak instead of
# wrapping it. If speech still sounds hot/distorted, bump to 16 (quieter) with
# no code change:  echo 'MIC_S32_SHIFT=16' >> ~/adam/.env  &&  restart adam.
S32_SHIFT        = int(os.getenv("MIC_S32_SHIFT", "15"))

# ── SPEAKER SOFTWARE GAIN ───────────────────────────────────────────────────
# Multiplier applied to Gemini's TTS before playback. THIS is the fix for "the
# speaker gain is changing constantly and so much noise now, a few minutes back it
# was crystal clear quality — it sounds like the gain increases from low to mid
# where it was working perfectly, then goes high where it has lots of noise".
#
# Nothing in ADAM ramps the output level: this is a constant, and the volume tools
# in laptop_agent_client.py drive the LAPTOP, not this speaker. What varies is the
# CONTENT. Gemini's TTS already peaks around -3 to -1 dBFS, so a 1.3 multiplier put
# loud syllables 10-30% PAST int16 full scale while quiet and mid-level passages
# stayed inside it. Everything under the ceiling reproduced cleanly; everything
# over it was saturated. That is why the distortion came and went with how loud
# the sentence was, and why it sounded like a gain rising into noise.
#
# Measured on the Pi, THD of a 1 kHz tone through the real playback chain at
# gain 1.3, sweeping the soft-limiter knee (see SPEAKER_LIMITER_KNEE):
#     input FS   x1.3    hard clip   knee0.90   knee0.95   knee0.98
#       0.60     0.78       0.002%     0.002%     0.002%     0.002%
#       0.70     0.91       0.002%     0.002%     0.002%     0.002%
#       0.77     1.00       0.001%     0.863%     0.371%     0.100%
#       0.85     1.10       3.864%     3.979%     3.865%     3.861%
#       0.95     1.23       8.431%     8.271%     8.395%     8.431%
#       1.00     1.30      10.327%    10.187%    10.286%    10.324%
# The lesson is in the last three rows: once the signal is over the ceiling, EVERY
# way of getting rid of the excess costs 4-10% THD. Soft limiting is a wash — it
# only ever recovered ~0.2 points, well inside the noise. There is no clever way to
# reproduce a waveform that does not fit; the only cure is not to exceed the
# ceiling. At gain 1.0 Gemini's own samples cannot exceed full scale by
# construction, and THD drops to the ~0.002% floor of the resampler.
#
# The cost is real and worth stating plainly: 1.0 is 2.3 dB quieter than 1.3, and
# there is no hardware volume to make it up with. `amixer -c sndrpigooglevoi
# scontrols` AND `controls` both return nothing on this card — zero mixer
# controls, simple or raw — so this multiplier is the only loudness knob ADAM has,
# and loudness trades directly against distortion. If clean-but-quiet is the wrong
# trade for the room:
#   echo 'SPEAKER_GAIN=1.15' >> ~/adam/.env  &&  sudo systemctl restart adam
# 1.15 puts the ceiling at 0.87 FS, so only genuinely loud syllables saturate.
SPEAKER_GAIN     = float(os.getenv("SPEAKER_GAIN", "1.0"))
# Soft-limiter knee, as a fraction of int16 full scale. Samples below it pass
# through untouched; above it the excess is compressed through tanh, which is
# monotonic, bounded by full scale, and cannot flat-top. tanh'(0) == 1, so the
# slope matches the linear region exactly at the knee — nothing to hear at the
# transition.
#
# Per the sweep above this is a BACKSTOP, not the cure — it buys ~0.2 THD points
# where hard clipping was already destroying the waveform. It is kept because it
# is cheap (1.4 ms per 20 ms chunk, ~7% of realtime on a Pi Zero 2 W) and because
# it removes flat tops entirely (measured: 7,000-11,000 flat-topped samples per
# half-second tone became 0). Flat tops radiate harmonics far above the 2nd-12th
# that the THD figure counts, which is the part that reads as harshness.
#
# 0.95 rather than 0.70: at 0.70 the limiter distorted signals that were never
# going to clip at all (1.205% THD on a peak of 0.91 FS, against 0.002% for doing
# nothing), because it compressed everything above the knee instead of only what
# exceeded the ceiling. 0.95 is verified transparent for every peak below it, so
# the limiter now engages only where hard clipping would have.
SPEAKER_LIMITER_KNEE = float(os.getenv("SPEAKER_LIMITER_KNEE", "0.95"))

POST_MUTE_S      = 0.45
# ── ECHO GUARD ──────────────────────────────────────────────────────────────
# POST_MUTE_S ends the hard mute quickly on purpose, so a fast human reply isn't
# swallowed. But the room's reverb tail outlives it: measured live, the first
# unmuted chunk after a reply read RMS 2,693 against an open_th of 2,300 and
# opened the VAD gate on ADAM's own voice. Sending that back to Gemini is how a
# model starts answering itself.
#
# The fix is a raised OPEN threshold for a short window after the mic reopens —
# not a longer mute. The audio keeps flowing into the pre-roll buffer either way,
# so nothing is thrown away; it just takes a genuinely louder chunk to be called
# speech.
#
# The MULTIPLIER is tied to MIC_SILENCE_FLOOR and to the echo level, and BOTH
# have now been measured directly (adam/_floorcal.py plays a full-scale
# speech-band buffer through the real speaker chain and samples the decay):
#     33ms after playback ends   1,665
#     POST_MUTE_S (450ms)        1,114     <- where the mic actually reopens
#     worst chunk in the guard window  1,198
# That obsoletes the 2,693 this was originally sized against — the polyphase
# speaker interpolator and the SPEAKER_GAIN retune cut the tail by more than
# half. With the tail at ~1,200, the guard no longer has to fight real speech:
# 1.25 puts the bar at 2,250, which is 1.88x over the measured echo and still
# BELOW the 2,357 quietest measured speech peak. Both constraints hold at once
# for the first time; at the 2.0 this briefly was, the bar (3,600) sat above all
# measured speech and the guard silently ate barge-ins.
MIC_ECHO_GUARD_S      = float(os.getenv("MIC_ECHO_GUARD_S", "0.7"))
MIC_ECHO_GUARD_MARGIN = float(os.getenv("MIC_ECHO_GUARD_MARGIN", "1.25"))
MIC_Q_MAX        = 40
OUT_Q_MAX        = 200

# ── MIC BAND-PASS ───────────────────────────────────────────────────────────
# Corner frequencies for the mic chain in audio_utils. Measured on this build's
# INMP441 pair in a SILENT room: 72% of the left mic's energy sits below 300 Hz
# and 53% of the right mic's sits above 8 kHz, versus only ~14% in the speech
# band for either. The low-pass is also the anti-alias filter for the 48k->16k
# decimation, so MIC_LP_HZ must stay below GEMINI_SEND_RATE/2 = 8000.
MIC_HP_HZ = float(os.getenv("MIC_HP_HZ", "120"))    # kill rumble below this
MIC_LP_HZ = float(os.getenv("MIC_LP_HZ", "6800"))   # kill hiss above this

# Where the anti-alias low-pass must be FULLY down, not merely -6 dB.
#
# MIC_LP_HZ is the -6 dB point of a windowed-sinc, and a windowed-sinc's
# transition band straddles that point symmetrically. The old design used a
# fixed 63 taps, which at 48 kHz gives a Hamming transition width of about
# 3.3*48000/63 = 2514 Hz — so the stopband did not begin until roughly
# 6800 + 1257 = 8057 Hz, i.e. just ABOVE the 8 kHz Nyquist of the 16 kHz
# stream we decimate to. Two consequences, both measurable:
#   * everything from 8000 Hz up to ~8057 Hz folded back onto 7943-8000 Hz;
#   * attenuation AT 8 kHz was only ~40 dB instead of the window's -53 dB,
#     which matters on this hardware because the INMP441 puts 53% of its
#     energy above 8 kHz.
# Neither is large on its own, and neither is the reason speech is misheard
# (that is SNR — see MIC_NR below). But it is a genuine design error, and the
# fix is nearly free: specify the stopband edge instead of the tap count and
# let the filter designer solve for the taps it needs.
#
# Do NOT "fix" this by lowering MIC_LP_HZ instead. That throws away real
# fricative energy at 6-8 kHz, which is exactly the band that distinguishes
# the consonants being confused (/s/ vs /t/ vs /d/ — "code" heard as "course"
# or "court"). Keep the passband and pay for the taps.
MIC_LP_STOP_HZ = float(os.getenv("MIC_LP_STOP_HZ",
                                 str(GEMINI_SEND_RATE / 2)))   # 8000

# ── MIC NOISE SUPPRESSION (the fix that actually addresses mis-hearing) ─────
# Measured on this unit during a real conversation, post-filter int16 RMS:
#   noise floor      1550-1591
#   speech p90       2041-4256
#   speech max       2783-6487
# That is an in-band SNR of +2 to +12 dB, typically about +6 dB. No speech
# recogniser is reliable there; humans need ~+15 dB and neural STT degrades
# sharply below ~+10 dB. This — not clipping, not filter ripple — is why words
# come back wrong ("ADAM" as "madam", "code" as "course"/"court").
#
# Nothing about gain fixes it. Digital gain scales signal and noise together;
# the ratio is what is broken. The room noise here is STATIONARY (fan, amp
# hiss, the 26 Hz electrical fault on the left channel, switching noise), and
# stationary noise is the one case classical single-channel suppression handles
# well: estimate the noise magnitude spectrum, subtract it, keep the phase.
#
# Applied ONLY to the copy of the audio that goes to Gemini and to the Vosk
# wake-word detector. The adaptive gate keeps seeing the ORIGINAL signal, so
# every threshold, learned floor and flatness statistic behaves exactly as
# before and none of the gate tuning is invalidated. Set MIC_NR=0 to send the
# unprocessed stream and confirm the difference for yourself.
MIC_NR          = os.getenv("MIC_NR", "0").strip().lower() not in (
                      "0", "false", "no", "off")
# Frame/hop for the WOLA analysis. 512 @ 16 kHz = 32 ms with a 16 ms hop, which
# resolves pitch harmonics for adult voices while keeping the added latency to
# frame-hop = 16 ms. Must satisfy hop == frame/2 (sqrt-Hann is COLA there).
MIC_NR_FRAME    = int(os.getenv("MIC_NR_FRAME", "512"))
# How much of the estimated noise POWER to remove. This number has to carry two
# separate jobs, so it is bigger than the textbook 1.5-2.0:
#
#   1. Minimum-statistics estimation is biased LOW by construction — the
#     minimum of a fluctuating estimate is below its mean. Measured on this
#     unit's own noise with MIC_NR_SMOOTH=0.9: the tracker reads 2.3 dB (1.68x)
#     under the true mean noise power. Ignore that and the subtraction does
#     essentially nothing, which is exactly what the first version of this code
#     did (2.8 dB of noise removed, 3.1 dB of speech removed — a net loss).
#   2. Genuine over-subtraction on top, because removing exactly the mean
#     leaves half the noise bins above the estimate and those survivors are
#     what "musical noise" is made of.
#
# 3.5 = 1.68 bias x ~2.1 over-subtraction. If you change MIC_NR_SMOOTH you must
# re-measure the bias; the two are not independent.
MIC_NR_OVERSUB  = float(os.getenv("MIC_NR_OVERSUB", "1.8"))
# One-pole smoothing applied to the per-bin power before the minimum tracker
# sees it. Higher = steadier estimate = less downward bias to compensate for
# (0.70 -> 4.9 dB of bias, 0.90 -> 2.3 dB, 0.95 -> 1.4 dB), but the effective
# averaging window grows as 1/(1-a) frames and must stay well under the length
# of a sustained vowel or speech starts defining the noise floor. 0.90 is ~10
# frames = 160 ms, comfortably below the 400 ms of the longest vowel and
# comfortably below the 1.5 s minimum window.
MIC_NR_SMOOTH   = float(os.getenv("MIC_NR_SMOOTH", "0.90"))
# Maximum attenuation per bin. -12 dB is deliberately conservative: deeper
# floors (-25 dB and below) sound cleaner to a human but strip the low-energy
# consonants an STT model needs, and they produce "musical noise" — isolated
# surviving bins warbling in the residual — which recognisers dislike more than
# honest hiss. Going deeper here is the most likely way to make things worse.
MIC_NR_FLOOR_DB = float(os.getenv("MIC_NR_FLOOR_DB", "-6"))
# Length of the sliding window the per-bin noise minimum is tracked over. Must
# be comfortably longer than the longest continuous vowel (~0.4 s) so that
# speech cannot be mistaken for the noise floor, and short enough to follow a
# room that changes: 1.5 s satisfies both.
MIC_NR_NOISE_S  = float(os.getenv("MIC_NR_NOISE_S", "1.5"))

# Which physical mic feeds the SPEECH path: auto | mix | left | right.
# Measured ambient, ADAM stopped, raw S32 before filtering: left peaks at
# -1.6 dBFS and right at -6.7 dBFS, 80.85% of the energy below 60 Hz with the
# loudest component at 26.4 Hz, and the speech band (250-4000 Hz) holding just
# 2.7% of the total. At 26 Hz the wavelength is 13 m, so two mics 5 cm apart
# MUST see the same pressure — a 5.1 dB peak gap is not acoustic, it is
# something electrical or structure-borne on the left channel alone (its DC
# offset is also 300x larger). With 1.6 dB of headroom left on silence the
# CONVERTER clips on loud syllables, upstream of every filter, and there is no
# capture gain to lower (amixer exposes no controls on sndrpigooglevoi).
#
# Dropping the hot channel is still NOT the default, because measuring it
# settled the question: on the same capture, right-only had a post-filter
# noise floor of p50 1498 vs 804 for the mix — 5.4 dB worse IN BAND, since
# L's excess is subsonic (the 120 Hz high-pass already removes it) and
# averaging two mics cancels uncorrelated noise. So "auto" mixes unless a
# channel is genuinely saturating, which is the only case DSP cannot repair.
# Force a channel with:  echo 'MIC_CHANNEL=right' >> ~/adam/.env
MIC_CHANNEL = os.getenv("MIC_CHANNEL", "auto").strip().lower()
if MIC_CHANNEL not in ("auto", "mix", "left", "right"):
    MIC_CHANNEL = "auto"
MIC_CH_CLIP_FRAC = float(os.getenv("MIC_CH_CLIP_FRAC", "0.995"))

# THE HOLE IN THE ABOVE, AND WHAT CLOSES IT. "auto" decides by looking for
# saturated samples — but it only looked during the FIRST SECOND after arecord
# starts. At that moment ADAM has just booted and nobody is talking, so the one
# thing it is trying to detect is the one thing that cannot be present. It
# therefore latched "mix" every single time and never revisited the decision,
# including when a syllable later drove L into the rails. That is not a
# conservative default, it is a detector pointed at the wrong second.
#
# So the saturation count keeps running for the whole session. Every
# MIC_CH_WATCH_S of audio, if one channel has accumulated at least
# MIC_CH_WATCH_MIN_CLIPS saturated samples and the other is essentially clean
# (under an eighth as many), the hot channel is dropped and the switch is
# logged with the counts that caused it.
#
# The switch is ONE-WAY and only happens in "auto": clipping is unrepairable
# downstream, so evidence of it outweighs the 5.4 dB in-band penalty measured
# above, but a channel that has proven it can hit the rails should not be
# readmitted just because the room went quiet again — that would flap on every
# loud/quiet cycle. An explicit MIC_CHANNEL=mix/left/right disables the watch
# entirely, so a forced choice stays forced.
MIC_CH_WATCH_S         = float(os.getenv("MIC_CH_WATCH_S", "10.0"))
MIC_CH_WATCH_MIN_CLIPS = int(os.getenv("MIC_CH_WATCH_MIN_CLIPS", "20"))

# ── ADAPTIVE SPEECH GATE (the production path) ──────────────────────────────
# MIC_ADAPTIVE=1 is the default and means NO threshold below this line has to
# be tuned per room, per unit, or per user. The rationale, the failure it
# replaces and the two-vote design are documented in audio_utils.py's
# ADAPTIVE SPEECH GATE block; only the numbers live here.
#
# Set MIC_ADAPTIVE=0 to fall back to the old fixed-threshold path
# (MIC_SILENCE_FLOOR / MIC_AMBIENT_MAX below), which is kept as a rollback.
MIC_ADAPTIVE = os.getenv("MIC_ADAPTIVE", "1").strip().lower() not in (
    "0", "false", "no", "off")

# The floor is a low percentile of a long window, which is what makes it
# immune to the speech it is measuring. 45s at 30 chunks/s = 1350 samples;
# p20 of that is the room, not the talker, even during a monologue — the
# gaps between syllables are far more numerous than the syllables.
MIC_FLOOR_WINDOW_S     = float(os.getenv("MIC_FLOOR_WINDOW_S", "45"))
MIC_FLOOR_PERCENTILE   = float(os.getenv("MIC_FLOOR_PERCENTILE", "20"))
MIC_FLOOR_MIN_S        = float(os.getenv("MIC_FLOOR_MIN_S", "1.5"))
# Asymmetric tracking: a room that suddenly gets louder must not deafen ADAM
# on one door slam, but a room that goes quiet should regain sensitivity at
# once. Per recalculation (6/s), so 0.02 ≈ 8s to follow a rise, 0.25 ≈ 0.7s
# to follow a drop.
MIC_FLOOR_RISE         = float(os.getenv("MIC_FLOOR_RISE", "0.02"))
MIC_FLOOR_FALL         = float(os.getenv("MIC_FLOOR_FALL", "0.25"))
# Survive a restart with the room already learned instead of a cold window.
MIC_FLOOR_STATE_PATH      = os.getenv("MIC_FLOOR_STATE_PATH",
                                      str(BASE_DIR / ".mic_floor.json"))
MIC_FLOOR_STATE_MAX_AGE_S = float(os.getenv("MIC_FLOOR_STATE_MAX_AGE_S",
                                            str(7 * 24 * 3600)))
MIC_FLOOR_SAVE_EVERY_S    = float(os.getenv("MIC_FLOOR_SAVE_EVERY_S", "60"))

# Thresholds as ratios of the learned floor. MEASURED, not chosen: on this
# room's recording the floor (p20) is 1512 and the QUIETEST real speech seen
# is 2357 — only +0.7 dB over it. So the open ratio has a hard ceiling of
# 2357/1512 = 1.55, and the 1.9 this used to be put the bar at 2873, ABOVE
# quiet speech: it would have deafened ADAM. 1.25x ≈ +1.9 dB is what the
# end-to-end simulation settled on (adam-tools/_gatesim2.py): with the shape
# vote and a 5-chunk onset it produced ZERO false opens on 25 s of this
# room's noise while still opening on speech at 2357.
# 3.2x ≈ +10 dB is loud enough to be its own evidence and open even if the
# shape vote disagrees (a shout must always work); at floor 1512 that is
# 4838, above the loudest single noise chunk measured (3824).
# 1.06x is the hold rail, = MIC_OPEN_RATIO * 0.85 as simulated. It sits BELOW
# the room's noise p90 on purpose — what actually closes the gate here is the
# shape fraction below, not the level, and the simulation confirms the gate
# always closes (longest open on pure noise: 0.0 s).
# MIC_OPEN_MIN is not a room threshold — it is a rail for digital silence,
# where a ratio of ~0 would otherwise open on the dither in the last bit.
MIC_OPEN_RATIO     = float(os.getenv("MIC_OPEN_RATIO", "1.25"))
MIC_OPEN_STRONG    = float(os.getenv("MIC_OPEN_STRONG", "3.2"))
MIC_OPEN_MIN       = float(os.getenv("MIC_OPEN_MIN", "90"))
MIC_HOLD_RATIO     = float(os.getenv("MIC_HOLD_RATIO", "1.06"))
MIC_HOLD_MAX_RATIO = float(os.getenv("MIC_HOLD_MAX_RATIO", "0.95"))

# ── SPEECH-SHAPE VOTE ───────────────────────────────────────────────────────
# Level-independent by construction, so steady noise reads as not-speech
# however loud it is. Both features come from one 1024-point rFFT of the
# 16 kHz mono (1.02 ms/chunk measured, budget 33.3 ms).
#
# FLATNESS = geometric mean / arithmetic mean of the power spectrum over
# 120-6800 Hz. 1.0 is white noise; voiced speech collapses it. Measured on
# 750 chunks of this room's noise vs synthetic formant-shaped speech at the
# levels really seen here:
#     flat <= 0.40   noise 9.9% false pass   speech 97.2% @2357  100% @3000+
#     flat <= 0.35   noise 4.3% false pass   speech 74.4% @2357  99.7% @3000+
#     flat <= 0.30   noise 2.3% false pass   speech 22.0% @2357  98.8% @3000+
# 0.35 is the knee, and the onset quorum below is what turns its
# residual 4.3% into zero false opens: this room's noise crosses the level
# threshold in runs whose MEDIAN length is 1 chunk and whose longest is 3.
#
# LO/HI = energy 120-1000 Hz over 1000-6800 Hz. Voiced speech is
# bottom-heavy; hiss and fan whine are not. >= 0.60 → 12.9% noise, 97.8%
# speech on its own; it earns its place as the second AND term.
#
# SLACK/HOLD_FRAC: holding uses flat <= flat_max+0.05 and requires only 40% of
# the sustain window to pass, so consonants and inter-syllable dips do not
# truncate a turn while noise (which passes ~5-10% of chunks) still closes it.
MIC_SHAPE_FLAT_MAX   = float(os.getenv("MIC_SHAPE_FLAT_MAX", "0.35"))
MIC_SHAPE_FLAT_SLACK = float(os.getenv("MIC_SHAPE_FLAT_SLACK", "0.05"))
MIC_SHAPE_RATIO_MIN  = float(os.getenv("MIC_SHAPE_RATIO_MIN", "0.60"))
MIC_SHAPE_HOLD_FRAC  = float(os.getenv("MIC_SHAPE_HOLD_FRAC", "0.40"))

# ── SHAPE THRESHOLD, LEARNED PER ROOM ───────────────────────────────────────
# Every number in the block above came from ONE room. Shipping it as a constant
# ships that room's acoustics to a customer who does not have them, and the
# failure is not symmetric: too loose costs a phantom Gemini turn, too tight
# costs EVERY turn — ADAM simply never answers, which is the complaint this
# whole subsystem keeps generating.
#
# So MIC_SHAPE_FLAT_MAX becomes a FLOOR rather than the threshold. ADAM
# measures the flatness of its own room's noise bed (only on chunks the gate
# already judged to be silence, so speech cannot poison it), takes the
# MIC_SHAPE_FLAT_PCTL percentile, backs off by MIC_SHAPE_FLAT_MARGIN, and uses
# that instead whenever it is LOOSER than the measured 0.35:
#
#     flat_max = clamp(p5(noise flatness) * 0.95, MIC_SHAPE_FLAT_MAX, CEIL)
#
# One-sided on purpose. In a hissy room (fan, AC, a PC next to the mic) the
# bed's flatness is high, 0.35 is far stricter than it needs to be, and the
# learned value opens the test up — worth real capture, because speech at low
# SNR is itself flatter than clean speech: the noise fills the spectral
# valleys between the harmonics this statistic exists to see. In a TONAL room
# (a whine, a hum) the bed's flatness can be LOWER than speech's, and a
# two-sided rule would walk the threshold down until nothing passed at all.
# The floor makes that unreachable: worst case is exactly today's behaviour.
# p5 also means ~5% of noise chunks pass the shape test by construction, which
# the onset quorum below is sized to absorb.
# MIC_SHAPE_ADAPT=0 pins it to the constant.
MIC_SHAPE_ADAPT       = os.getenv("MIC_SHAPE_ADAPT", "1").strip().lower() not in (
    "0", "false", "no", "off")
MIC_SHAPE_FLAT_CEIL   = float(os.getenv("MIC_SHAPE_FLAT_CEIL", "0.70"))
MIC_SHAPE_FLAT_MARGIN = float(os.getenv("MIC_SHAPE_FLAT_MARGIN", "0.95"))
MIC_SHAPE_FLAT_PCTL   = float(os.getenv("MIC_SHAPE_FLAT_PCTL", "5"))

# webrtcvad was the FIRST design for the shape vote and it was refuted by
# measurement on this HAT: on 25 s of ordinary room noise it called 100.0% of
# frames "speech" at aggressiveness 0, 1 and 2, and 98.6% at 3. A vote that
# says yes to everything is not a vote, so it is OFF. These knobs survive
# only so a different microphone can be tried without a code change:
# MIC_VAD_BACKEND=webrtc adds it as a further AND term on top of the shape
# test (it can then only make ADAM less sensitive, never more).
MIC_VAD_BACKEND        = os.getenv("MIC_VAD_BACKEND", "off").strip().lower()
MIC_VAD_AGGRESSIVENESS = int(os.getenv("MIC_VAD_AGGRESSIVENESS", "2"))
MIC_VAD_FRAME_MS       = int(os.getenv("MIC_VAD_FRAME_MS", "30"))

# ── LEVEL GATES ─────────────────────────────────────────────────────────────
# Both thresholds are RMS of the FILTERED 16kHz mono audio in int16 units
# (0..32767) — i.e. what audio_utils.rms_pcm16() returns, NOT the raw-S32
# rms_s32() these used to be compared against.
#
# That swap is the fix for "ADAM talks but never hears me". Raw S32 RMS reads
# 68M-108M in a SILENT room on this hardware, ~40x over the old 2_000_000
# floor, so the silence gate and the adaptive noise-floor gate below it could
# never fire: every chunk of pure room noise was forwarded, Gemini received one
# unbroken noise bed, and its turn detection never found a speech onset or
# endpoint. Post-filter these numbers track actual speech.
#
# Calibrated from measurement — re-measure after any change to mic placement or
# MIC_HP_HZ/MIC_LP_HZ/S32_SHIFT, and keep the floor comfortably above measured
# quiet-room RMS but well under measured speech RMS.
#
# ── RE-CALIBRATED against a CLEAN room, then AGAIN against a direct 20s+15s
#    measurement (adam/_floorcal.py, adam/_servodecay.py) ─────────────────────
# Two earlier calibrations were both wrong, for different reasons:
#   * the original numbers ("quiet room 1,580-1,900") were taken while the neck
#     servo held its 50 Hz pulse train forever after a move, so the mic heard
#     coil whine in every "quiet room" sample, AND while the noise trackers were
#     poisoning themselves. Both are fixed now (hardware.py detaches the pin;
#     see MIC_NOISE_LEARN_COOLDOWN_S for the loop).
#   * the replacement numbers came from the live journal's rate-limited "Mic RMS"
#     lines — only n=42 over 100 s — and read p50 773 / p99 1,027. A direct
#     600-chunk measurement through the identical path does NOT reproduce that.
#
# DIRECT MEASUREMENT, two independent runs, same filter chain and the same
# rms_pcm16() the gate compares against:
#     idle, servo never energised   p50 1,039-1,245   p99 1,304-1,405   max 1,506
#     servo energised (holding)     p50 4,658-4,666   max 5,299
#     servo 0-1s after detach()     p50 1,697         max 3,582
#     servo 1-3s after detach()     p50 1,452-1,931
#     servo 5s+ after detach()      p50 1,306-1,422   (== baseline)
#     echo tail after ADAM speaks   1,114 at POST_MUTE_S, worst 1,198 in-window
#     user talking (live journal)   peaks 2,357-2,916
#
# So the usable window is 1,405 (worst idle p99) to 2,357 (quietest measured
# speech peak) — a factor of 1.68, about 4.5 dB, and that is ALL there is. The
# floor is placed at the geometric midpoint, sqrt(1405*2357) = 1,820 -> 1,800:
# equidistant in dB from the noise and from the speech, which is the only
# defensible split when both sides are measured and neither can be moved.
#
# Why not the 2,300 it was: 2,300 sits ABOVE most real speech chunks. It only
# caught the loud peaks, so the journal showed five
# "Speech detected"/"Speech ended" pairs in 100 s and ZERO transcripts — no open
# run ever lasted long enough for Gemini to find both an onset and an endpoint.
# Why not the 1,500 it was briefly: that is 1.07x over the measured idle p99,
# i.e. inside the noise, so the room itself would keep opening the gate.
MIC_SILENCE_FLOOR = float(os.getenv("MIC_SILENCE_FLOOR", "1800"))

# The rolling ambient baseline in listen() adapts to the room; speech must exceed
# ambient * MIC_SPEECH_MARGIN (and the fixed floor above) to OPEN the gate.
# 1.35 = ~2.6 dB over the room floor, which is as much as the 4.5 dB total window
# can spare; the hysteresis + hangover below are what supply the noise immunity a
# bigger margin used to. In this room the ADAPTIVE terms land just under the
# fixed floor (ambient ~1,040 * 1.35 = 1,404, peak ~1,405 * 1.15 = 1,616), so the
# floor is the binding constraint while idle and the adaptive terms take over
# once the room gets genuinely noisier — the intended division of labour.
MIC_SPEECH_MARGIN = float(os.getenv("MIC_SPEECH_MARGIN", "1.35"))
MIC_AMBIENT_INIT  = float(os.getenv("MIC_AMBIENT_INIT", "1200"))  # measured p50
# Runaway clamp on BOTH noise trackers, and the number that decides how bad a
# poisoned tracker is allowed to get. It has to be set from the QUIETEST measured
# speech, not from the floor: the clamp's job is to guarantee that even a fully
# saturated tracker leaves the gate below real speech.
#   at 4,000 (original): worst-case open = 4,000*1.35 = 5,400 — above the entire
#                        speech range. The clamp was decorative; a poisoned
#                        tracker went fully deaf and nothing noticed.
#   at 1,650 (now):      worst-case open = max(1800, 1650*1.35, 1650*1.15)
#                        = 2,228, below the quietest measured speech peak of
#                        2,357. Saturating the tracker now costs sensitivity,
#                        never hearing outright. 1,650 also stays above the
#                        measured idle p99 of 1,405 (1.17x), so it does not
#                        interfere with legitimate adaptation to a noisier room.
# The upper bound is hard: MIC_AMBIENT_MAX * MIC_SPEECH_MARGIN must stay under
# the quietest speech, i.e. MAX < 2357/1.35 = 1,746. Anything above that and the
# clamp stops being a safety net.
# See MIC_NOISE_LEARN_COOLDOWN_S for the feedback loop this is the backstop for.
MIC_AMBIENT_MAX   = float(os.getenv("MIC_AMBIENT_MAX", "1650"))

# ── VAD HYSTERESIS / HANGOVER / PRE-ROLL ────────────────────────────────────
# A per-chunk "is this chunk louder than the gate?" test cannot pass speech
# through intact, no matter how well the threshold is tuned. Speech is not a
# continuous plateau: it has stops, unvoiced consonants and inter-word gaps that
# fall to near the noise floor. In the measured capture, mid-sentence chunks read
# 1,776 / 1,913 / 2,008 sandwiched between 3,860 and 3,895. A plain threshold
# punches holes at exactly those points, so Gemini receives shredded fragments
# with the gaps deleted, finds no coherent onset/endpoint, and returns no
# transcript at all — which is precisely what was observed.
#
# Those dip readings come from a session whose room floor was higher than the
# 1,039-1,245 measured later, so treat them as an UPPER bound on the dips: a dip
# in a quiet moment goes lower, not higher. Either way the requirement is the
# same and it is what the two constants below deliver — a dip must not close the
# gate, whether because it stays above hold_th (1,475 at settled trackers, so
# the as-measured dips clear it) or because it is shorter than the hangover
# (which covers the quieter dips that do not).
#
# The standard fix, and what these three constants implement:
#   RELEASE_RATIO — two thresholds instead of one. Opening the gate needs the
#       full margin; STAYING open only needs this fraction of it. Classic
#       Schmitt-trigger hysteresis, so mid-word dips don't slam the gate.
#   HOLD_MARGIN   — a FLOOR under both thresholds, expressed as a multiple of
#       the tracked NOISE PEAK (session.py keeps a decaying maximum of the
#       non-speech chunks alongside the average). RELEASE_RATIO alone is not
#       safe: 2300 * 0.72 = 1,656, which is BELOW this room's measured quiet
#       range of 1,450-1,990. A hold threshold under the noise floor means every
#       chunk of silence re-arms the hangover, so the gate LATCHES OPEN and
#       streams a continuous noise bed. Observed live: the gate opened at 2,447,
#       held on 1,674 / 1,930 / 1,770 (all pure room noise), and stayed open
#       ~40s — and Gemini answered that noise with hallucinated transcripts in
#       random languages ("안녕하세요", "luego").
#       Against the PEAK rather than the average on purpose: the average is
#       biased low by design (fast-fall/slow-rise, so speech cannot poison it),
#       so it settles near the quiet end of the noise and under-reads the peaks
#       that actually trip the gate. 1.05 clears the measured 1,990 peak at
#       ~2,090 while leaving a real 2,090-2,300 hysteresis gap below open_th.
#   MAX_OPEN_S    — latch watchdog. The noise trackers freeze while the gate is
#       open (speech must not teach them what silence sounds like), so a latched
#       gate is self-sustaining: no update, no threshold movement, no escape.
#       If nothing has re-cleared the OPEN threshold for this long, whatever is
#       holding the gate is not speech — force it shut and resync the estimates
#       up to the level that fooled them. 15s exceeds any single conversational
#       utterance, so it never truncates real speech.
#   HANGOVER_S    — after level finally drops below the release threshold, keep
#       forwarding for this long. Covers gaps between words and the trailing
#       unvoiced consonant that endpointers otherwise clip ("cats" -> "cat").
#       This is load-bearing, not a nicety: the measured mid-word dips (1,776 /
#       1,913 / 2,008) fall INSIDE the quiet-room range, so no level threshold
#       can tell a dip from silence. Only a DURATION test can, which is exactly
#       what the hangover is.
#   PREROLL_S     — onset is only detectable AFTER it has begun, so by the time
#       a chunk clears the threshold the word's attack is already in the
#       previous chunks. Keep a small ring buffer and flush it on open;
#       otherwise "hello" arrives as "ello" and wrecks recognition.
MIC_VAD_RELEASE_RATIO = float(os.getenv("MIC_VAD_RELEASE_RATIO", "0.72"))
MIC_VAD_HOLD_MARGIN   = float(os.getenv("MIC_VAD_HOLD_MARGIN", "1.05"))
# The peak multiplier for the OPEN threshold, and it MUST exceed
# MIC_VAD_HOLD_MARGIN. Both thresholds used to share 1.05, and that single
# shared number silently destroyed the hysteresis: open_th took
# max(FLOOR, ambient*1.35, peak*1.05) while hold_th took
# max(open_th*0.72, peak*1.05), so the moment the peak term won in open_th the
# two expressions became the SAME expression and hold_th == open_th. Observed
# live, in order: 2356/2356, 2764/2764, 2876/2876, 2914/2914, 3007/3007 — a
# Schmitt trigger with zero gap, i.e. a plain threshold. Every mid-word dip then
# slammed the gate, so speech reached Gemini as ~1s slices separated by ~1s
# holes, and Gemini returned NO transcript at all for 20+ consecutive bursts.
# 1.15 keeps the gap open: with the settled peak of 1,973 the peak term is 2,269,
# still under MIC_SILENCE_FLOOR, so a normal room stays floor-driven at 2,300.
MIC_VAD_OPEN_MARGIN   = float(os.getenv("MIC_VAD_OPEN_MARGIN", "1.15"))
# Belt-and-braces ceiling on hold_th, as a fraction of open_th. MIC_VAD_OPEN_MARGIN
# > MIC_VAD_HOLD_MARGIN already implies a gap, but this makes "hold is strictly
# below open" true by construction no matter how the terms are later retuned.
MIC_VAD_MAX_HOLD_RATIO = float(os.getenv("MIC_VAD_MAX_HOLD_RATIO", "0.95"))
# After the gate closes, IGNORE this many seconds of audio for noise-tracking
# purposes. Without it the trackers learn the tail of the utterance that just
# ended — the reverb, the trailing unvoiced consonant, the inter-word gap — all
# of which are far louder than the room. That is a positive feedback loop, and it
# is what made ADAM go deaf mid-conversation: peak climbed 1,800 -> 2,244 ->
# 2,632 -> 2,864 purely on speech tails, dragging open_th from 2,300 up to 3,007,
# past most of the 2,400-5,500 speech range. Higher peak -> more chatter -> more
# speech misfiled as room noise -> higher peak. 1.0s covers the measured reverb
# tail (the 2,693 echo chunk) with margin.
MIC_NOISE_LEARN_COOLDOWN_S = float(os.getenv("MIC_NOISE_LEARN_COOLDOWN_S", "1.0"))
# Hold the gate shut for this long after arecord starts. The trackers begin at
# MIC_AMBIENT_INIT rather than at anything measured, so the very first chunks are
# judged by a guess: live, chunk ~1 read 2,460 against a cold open_th of 2,430,
# opened the gate on nothing, and the ABS_MAX watchdog needed 45s to break it out
# — 45s during which every tracker update was suppressed. Staying shut still
# feeds the trackers, so this is purely "measure the room before judging it".
MIC_WARMUP_S          = float(os.getenv("MIC_WARMUP_S", "1.5"))
MIC_VAD_MAX_OPEN_S    = float(os.getenv("MIC_VAD_MAX_OPEN_S", "15.0"))
# Absolute cap on one continuous open run, for the case MAX_OPEN_S cannot see:
# a room whose noise floor has risen ABOVE open_th, so every chunk legitimately
# clears it and no adaptive threshold under MIC_SILENCE_FLOOR can escape. Firing
# it bounds how much noise reaches Gemini per cycle and logs the level to raise
# the floor above. 45s exceeds any normal conversational turn, so tripping it on
# real speech costs only a ~1.5s refractory window.
MIC_VAD_ABS_MAX_OPEN_S = float(os.getenv("MIC_VAD_ABS_MAX_OPEN_S", "45.0"))
# Length of the rolling median window the gate uses for its SUSTAIN decisions
# (stay open / "something speech-loud happened recently"). ATTACK is unaffected —
# opening still reads the instantaneous chunk plus MIC_VAD_ONSET_CHUNKS.
#
# Measured live with the gate latched open for 40s+ and no "Speech ended":
#   p50 1705 p90 2895 p99 5448 max 9970 | open≥1899 hold≥1731 | opens 1 sent 290
#   p50 1632 p90 1831 p99 2394 max 2558 | open≥1899 hold≥1731 | opens 0 sent 301
# The median sat 1631-1705, BELOW hold≥1731 — the level test wanted to release.
# What held it was arming the hangover off single chunks: p90 1831-2895 means
# 10-25% of noise chunks clear hold_th, i.e. one every 0.13-0.33s, and each one
# restarted the 0.8s hangover. p99 clearing open_th did the same to the 15s soft
# watchdog, leaving only the 45s absolute one — 45s of deafness per cycle, and
# under manual activity detection also 45s during which Gemini is never sent
# activity_end and so never answers.
#
# 0.5s is chosen against the two signals it has to separate:
#   • impulses (1-2 chunks) cannot move a median at all — that needs >50% of the
#     window, i.e. 8 consecutive chunks, which no click is;
#   • intra-word stops in connected speech are 50-150ms, 1.5-4.5 chunks, so they
#     cannot pull the median below hold_th either.
# Raising it makes the gate slower to release (the median lags by half a window);
# lowering it toward 3 chunks converges back on the instantaneous behaviour that
# caused the latch.
MIC_VAD_SUSTAIN_S     = float(os.getenv("MIC_VAD_SUSTAIN_S", "0.5"))
# aplay's --start-delay, in MICROSECONDS: how much audio ALSA buffers before it
# starts the stream. This is dead air at the front of every reply, and with
# SPEAKER_IDLE_CLOSE_S closing the device between turns, every reply pays it.
#
# aplay derives start_threshold from this value, and with the default (0) it uses
# the ENTIRE buffer. Measured on this card with `aplay -v`:
#   --buffer-size=96000 alone  -> buffer 48000, period 24000, start_threshold 48000  (1.00s)
#   + these two settings       -> buffer 62400, period  4800, start_threshold 19200  (0.40s)
# (the driver clamps the requested 96000-frame buffer to 48000 either way).
#
# 0.4s rather than lower because it is also the jitter budget: once the stream is
# running, an audio gap longer than what is still buffered is an underrun, which
# aplay reports and the listener hears as a click. Gemini's stream arrives over
# the Pi Zero 2 W's Wi-Fi, so 0.4s is the margin being bought with 0.4s of
# latency. Lower it if replies still feel late AND the log stays free of
# "buffer underrun(s)".
SPEAKER_START_DELAY_US = int(os.getenv("SPEAKER_START_DELAY_US", "400000"))
# How much audio to assume is still inside aplay's ALSA ring buffer when a reply
# finishes. Two things read it: how long to keep the mic muted after a reply
# (so the tail of ADAM's own sentence is not recorded as the user talking), and
# how long before the aplay process may be torn down (so the tail is not cut off
# mid-word).
#
# Both failure directions have been seen on this hardware: too small clipped the
# last words of replies, too large keeps ADAM deaf for most of a second after it
# stops talking. 0.5s is the value that stopped the clipping. `aplay -v` says the
# driver grants a 62400-frame (1.3s) buffer, so 0.5s is not the worst case — it is
# the measured working point, and raising it toward 1.3s buys tail safety with
# exactly that much extra deafness.
SPEAKER_DRAIN_ALLOWANCE_S = float(os.getenv("SPEAKER_DRAIN_ALLOWANCE_S", "0.5"))
# HANGOVER = how long the gate stays open after the level drops, i.e. how long
# a pause is allowed to be before ADAM decides your turn ended.
#
# 0.6s was too short and it cost recognition accuracy, not just convenience.
# Natural pauses between CLAUSES run 0.5-0.8s, so at 0.6s a normal sentence got
# cut in half: the gate closed, activity_end went out, and the second half
# arrived as its own isolated ~1s turn. A one-second fragment carries no
# grammatical context, which is when the recogniser starts guessing phonetically
# (see STT_LANGUAGE_CODES above — that is where the Portuguese came from).
#
# 1.0s sits above the top of the natural-clause-pause range, so clauses stay
# joined. It is paid for directly in latency: activity_end is what makes Gemini
# start answering, so every reply is ~0.4s later than at 0.6s. That is the
# trade, and it is deliberate — a right answer 0.4s later beats a wrong answer
# in the wrong language. Set MIC_VAD_HANGOVER_S=0.6 to get the snappier,
# choppier behaviour back.
MIC_VAD_HANGOVER_S    = float(os.getenv("MIC_VAD_HANGOVER_S", "1.0"))
# 0.6s is the end-of-speech budget, and under manual activity detection it is
# also pure conversational latency: activity_end is only sent when this expires,
# and Gemini does not start generating until it arrives. Google documents ~500ms
# as the client-side end-of-speech threshold for manual mode, so 0.6s is that
# floor plus one chunk of margin — going lower starts splitting sentences at
# their pauses, which is what the server VAD used to do.
#
# It was 0.8s (up from an original 0.3s) to cover the echo guard, and the
# pre-roll below is what lets it come back down: nothing is lost during the
# guard window because those chunks are ringed and flushed on open.
#
# During the guard window the gate needs ~3,450 rather than 2,300, so a
# quiet reply (measured speech starts at ~2,400) is not declared speech until the
# window expires — but every one of those chunks is sitting in this ring buffer
# and gets flushed the instant the gate does open, so the sentence still reaches
# Gemini from its true first sample. Keep this >= MIC_ECHO_GUARD_S or the guard
# starts eating the front of fast replies. Costs 0.8 * 16000 * 2 = 26 kB of RAM.
MIC_VAD_PREROLL_S     = float(os.getenv("MIC_VAD_PREROLL_S", "0.8"))
# ONSET CONFIRMATION: consecutive chunks that must clear the open threshold
# before the gate is declared open. A duration test, added because the level
# test alone has run out of room.
#
# adam/_hpcal.py ruled out the obvious fix. It scored ten band-pass/high-pass
# detector candidates against the live chain and the best one was WORSE
# (1.67x vs 1.68x usable window, -0.1 dB): once the sub-100 Hz rumble is gone
# the remaining noise is broadband hiss sitting on the speech band, so there
# is nothing left to filter. Threshold tuning and spectral filtering are both
# exhausted at ~4.5 dB of separation.
#
# What that measurement did find: under a 300-3400 band-pass the noise p50
# fell 30% while the p99 fell only 17%. Stationary noise moves both equally.
# A tail that survives band-limiting and sits 1.67x above its own median is
# IMPULSIVE — clicks, creaks, taps, servo bearing ticks. Those are 1-2 chunks
# long; the shortest useful utterance is tens of chunks. So requiring N
# consecutive chunks separates them on a dimension where the gap is enormous
# instead of on level, where it is 4.5 dB.
#
# The cost is zero audio: MIC_VAD_PREROLL_S (0.8s = ~24 chunks) of history is
# already buffered and flushed on open, so N chunks delays the DECISION, not
# the speech. Even 5 chunks = 167 ms sits comfortably inside that 0.8 s.
#
# A LONG duration test is MEASURED, and it is the single biggest false-open
# lever there is. On 25 s of this room's noise, the runs of consecutive chunks
# that cross the open threshold are:
#     1.20x floor  13.7% of chunks   48 runs   longest 7   median run 1
#     1.35x floor   4.0% of chunks   22 runs   longest 3   median run 1
#     1.50x floor   2.1% of chunks   14 runs   longest 2   median run 1
# Median run length 1. Speech sustains for tens of chunks. End-to-end
# (adam-tools/_gatesim2.py), holding everything else fixed:
#     N=2  ~12-22 false opens/min      N=3  4.8-7.2/min      N=5  0.0/min
# and the capture cost of N=5 is only the onset, which the pre-roll covers.
#
# ── WHY "CONSECUTIVE" WAS WRONG, AND WHAT REPLACED IT ───────────────────────
# The 0.0 false opens/min above is real. The "capture cost is only the onset"
# is not, and the reason is a flaw in how it was measured: the simulated
# positives were SUSTAINED SYNTHETIC VOWELS. Real speech is not. "Hey ADAM"
# is /h/ — aspiration, broadband, flatness near 1.0, fails the shape test on
# its own merits — then a vowel, then the /d/ STOP CLOSURE, which is 50-80 ms
# of near-silence, i.e. 2-3 whole chunks BELOW the level threshold. Both are
# speech; neither passes; and under a consecutive rule either one resets the
# counter to zero. The gate then wants 5 clean chunks in a row that ordinary
# English does not contain at conversational level, so `blocked` climbs, and
# `opens` stays 0 unless the user shouts a sustained vowel at the mic. That
# is precisely the reported symptom: ADAM hears a hum but never a sentence.
#
# The duration test is still the right idea — it is what rejects impulses on
# a dimension where the gap is enormous. Only the shape of the test changes,
# from a RUN to a QUORUM: MIC_VAD_ONSET_CHUNKS chunks must pass within the
# last MIC_VAD_ONSET_WINDOW chunks, in any order, gaps allowed.
#
# 3 of 6 (200 ms) is the default, and it costs nothing in false opens because
# the two votes multiply. In the measured room a chunk clears the level
# threshold 12.9% of the time and the shape test 4.3%; independent, that is
# 0.55% per chunk, and the chance of 3 such chunks landing inside any 6-chunk
# window is ~3e-6, i.e. of order 0.01 false opens/min — below the 0.0/min the
# consecutive rule measured only because that number was already at the floor
# of what 25 s can resolve (±2.4/min). What the quorum buys back is the whole
# class of real utterances the run test was silently discarding.
MIC_VAD_ONSET_CHUNKS  = int(os.getenv("MIC_VAD_ONSET_CHUNKS", "3"))
MIC_VAD_ONSET_WINDOW  = int(os.getenv("MIC_VAD_ONSET_WINDOW", "6"))
# Seconds of playback silence after which speaker() CLOSES the ALSA playback
# device instead of holding it open for the whole session.
#
# This is the fix for the actual root cause of "ADAM cannot hear me", measured
# by adam/_amphiss.py through the live capture path, one variable, reversible:
#
#     aplay closed            p50 1082  p99 1519    0% of chunks over the floor
#     aplay open on SILENCE   p50 1726  p99 2163   37% of chunks over the floor
#     aplay closed again      p50 1341  p99 1719    0%
#
# +4.1 dB of noise, produced by ADAM itself. The voiceHAT's class-D amplifier
# idles by switching, and holding the playback device open keeps that switching
# noise coupling into two microphones sitting inches away on the same PCB. It
# is not the room: killing aplay puts the floor back.
#
# What it cost: the quietest real speech ever measured on this unit is 2357,
# so the usable speech-to-noise window collapses from 1.55x to 1.09x — from
# "tight" to "nothing". With 37% of silent chunks over MIC_SILENCE_FLOOR the
# gate latches open on hiss, and the live journal showed exactly that: `sent
# 301` of ~300 chunks per 10s window, i.e. an unbroken wall of noise streamed
# to Gemini, with the 45s hard watchdog firing on silence. A real sentence
# arriving inside that stream has no onset to segment, which is why the
# transcripts were multilingual garbage rather than simply absent.
#
# Raising MIC_SILENCE_FLOOR above the hiss was never an option: it would have
# to go above 2163 (p99) to stop the false opens, and real speech starts at
# 2357 with mid-word dips down to 1776. Removing the noise is the only move
# that leaves any window at all.
#
# The cost of closing the device is one aplay spawn (~50-100ms on a Pi Zero 2 W)
# before the first audio of a reply. That is hidden by the model's own
# time-to-first-audio, and it is paid only after real idleness — never
# mid-sentence (gated on adam_speaking), never during a song (gated on
# song_playing), and never before the drain deadline set by end_of_turn().
# 2.5s is comfortably past the ~0.5s ALSA buffer drain.
#
# THE OTHER SIDE OF THIS TRADE, and the reason for the escape hatch below: the
# voiceHAT is ONE I2S device serving capture and playback, so every teardown is
# a chance for the capture DMA to be left running but delivering zeros (see
# MIC_DEAD_STREAM_S). Closing after every reply turned a once-per-session race
# into a once-per-turn one. There are two mitigations in the code — the
# teardown is graceful (EOF, not SIGTERM) and listen() detects the wedge and
# respawns arecord — but on a unit where the wedge is chronic, the way out is
# to stop tearing the device down at all:
#
#     SPEAKER_IDLE_CLOSE_S=0     → never close; hold the device for the
#                                  session, as ADAM did before this constant
#
# That costs the +4 dB of amp hiss above, permanently. It is survivable now in
# a way it was not when this comment was first written, because the gate no
# longer trusts level alone: the learned floor simply settles onto the hissier
# bed (observe() is skipped only while the amp is HOT, so a permanently-open
# device is learned as the room it now is), and the shape vote rejects hiss on
# its merits — broadband switching noise reads flatness ~1.0 against a 0.35
# limit. Expect reduced sensitivity to quiet speech, not deafness. Prefer the
# default; reach for 0 only if "Capture DEAD" keeps appearing in the journal.
SPEAKER_IDLE_CLOSE_S  = float(os.getenv("SPEAKER_IDLE_CLOSE_S", "2.5"))
# How often listen() reports the mic level DISTRIBUTION (p50/p90/p99/max) plus
# the live thresholds, gate-open count and chunks-sent count. This replaced a
# print of one single chunk's RMS every 4s, which sampled 1 chunk in 120 and so
# could show the median but never the tail — and the tail is what opens the gate.
# Every constant above was calibrated with adam.service stopped (the diagnostics
# need the capture device to themselves), so the distribution UNDER LOAD, with
# aplay holding the amp on and the camera and Vosk resident, was never measured
# at all. 10s is ~300 chunks: enough for a meaningful p99, still one line per
# 10s of journal. Raise it to quieten the log, lower it while calibrating.
MIC_STATS_S           = float(os.getenv("MIC_STATS_S", "10.0"))

# How long a run of EXACT digital silence from arecord counts as a dead
# capture stream rather than a quiet room. The voiceHAT is a single I2S device
# shared by capture and playback, and tearing the playback stream down (which
# SPEAKER_IDLE_CLOSE_S now does after every reply, not once per session) can
# leave the capture DMA running but delivering zeros. Measured live: right
# after a "🔇 Playback idle" close, every chunk came back as
#   📊 Mic 10s: p50 0 p90 0 p99 0 max 0
# and stayed that way — arecord alive, chunks full-size, nothing in the error
# path able to notice, ADAM permanently deaf until a manual restart. An
# INMP441 always has self-noise, so a true 0 cannot come from live hardware;
# a sustained run of it is unambiguous and listen() respawns arecord.
# 3s ~= 90 chunks. Long enough that no transient can trip it, short enough
# that a wedge costs one sentence rather than the rest of the conversation.
MIC_DEAD_STREAM_S     = float(os.getenv("MIC_DEAD_STREAM_S", "3.0"))

# Same detector, shorter fuse, for the window where the wedge is EXPECTED.
# 3s is the right number when the cause is unknown — it has to be long enough
# that nothing transient can trip it. But right after a playback close the
# cause is not unknown: that teardown is the one event known to wedge this
# soundcard's capture DMA. Waiting the full 3s there throws away the two
# seconds immediately after ADAM stops talking, which is exactly when the user
# replies. Inside MIC_DEAD_AFTER_PLAY_WINDOW_S of the device closing, this
# shorter run of exact zeros is enough to declare the stream dead and respawn.
# 0.7s ~= 21 chunks of digital silence; live hardware cannot produce one.
MIC_DEAD_AFTER_PLAY_S        = float(os.getenv("MIC_DEAD_AFTER_PLAY_S", "0.7"))
MIC_DEAD_AFTER_PLAY_WINDOW_S = float(os.getenv("MIC_DEAD_AFTER_PLAY_WINDOW_S",
                                               "3.0"))

# ═════════════════════════════════════════════════════════════════════════════
# SONG / CONCERT PLAYBACK
# ═════════════════════════════════════════════════════════════════════════════

# List of audio files ADAM can play when asked to sing/perform — one is
# picked at random each time. Add/remove/rename paths here freely; must
# be raw PCM WAV files matching PLAYBACK_RATE/PLAYBACK_CHANNELS/16-bit
# (48kHz stereo s16 by default) since playback writes directly into the
# already-open speaker pipe rather than spawning a separate player. Convert
# with:
#   ffmpeg -i input.mp3 -ar 48000 -ac 2 -sample_fmt s16 song1.wav
SONG_FILE_PATHS = [
    str(BASE_DIR / "song1.wav"),
    str(BASE_DIR / "song2.wav"),
    str(BASE_DIR / "song3.wav"),
]

# Song playback pacing. The song is written into speaker()'s aplay pipe from a
# loop, and the loop used to yield with `await asyncio.sleep(0)` — which is not
# a wait at all: it returns on the very next event-loop pass. The loop then
# reads and writes as fast as the 64 KiB pipe will accept, so on this
# single-core Pi it spins against arecord, the Vosk recogniser and the Gemini
# tasks for the whole song, and the audible result is underruns and crackle in
# the thing it is trying to play well.
#
# Nothing is gained by running ahead: aplay consumes at exactly real time. So
# the loop is paced to the audio it just wrote. SONG_CHUNK_FRAMES at
# PLAYBACK_RATE is 4096/48000 = 85.3 ms; sleeping SONG_PACE_FRAC of that per
# chunk keeps roughly one chunk of slack in the pipe (enough that ALSA never
# starves) while handing ~77 ms of every 85 ms back to the rest of the system.
SONG_CHUNK_FRAMES = int(os.getenv("SONG_CHUNK_FRAMES", "4096"))
SONG_PACE_FRAC    = float(os.getenv("SONG_PACE_FRAC", "0.9"))

# ═════════════════════════════════════════════════════════════════════════════
# NECK SERVO (pan only; tilt goes over UART to Pico via ESP32-CAM relay)
# ═════════════════════════════════════════════════════════════════════════════

NECK_GPIO_PIN     = 12
NECK_SERVO_MIN_PW = 0.0005
NECK_SERVO_MAX_PW = 0.0025
NECK_PAN_CENTER   = 90
NECK_TILT_CENTER  = 85
NECK_PAN_MIN      = 30
NECK_PAN_MAX      = 150
NECK_TILT_MIN     = 50
NECK_TILT_MAX     = 120
NECK_SMOOTH_ALPHA = 0.25

# ── SERVO AUTO-DETACH (mic noise!) ──────────────────────────────────────
# Seconds to keep driving PWM after a move, before releasing the pin.
#
# This is an AUDIO constant as much as a servo one. gpiozero's AngularServo
# keeps emitting its 50 Hz pulse train forever once `.angle` is set — the servo
# stays energised, hums, and vibrates the board the two INMP441s are mounted on.
# Re-measured on this build with adam.service stopped (adam/_floorcal.py and
# adam/_servodecay.py, two independent runs), using the same post-filter RMS the
# VAD gates on:
#
#     servo never pulsed        p50 1,039-1,245  p99 1,304-1,405
#     servo attached, holding   p50 4,658-4,666  max 5,299   <- 3.7x, above open_th
#     0-1s after detach()       p50 1,697        max 3,582   <- STILL noisy
#     1-3s after detach()       p50 1,452-1,931
#     5s+ after detach()        p50 1,306-1,422             == baseline
#
# open_th is 1,800, so a holding servo alone pins the gate open. That is the
# whole "ADAM talks but can't hear me" failure: at boot the servo is detached and
# the mic floor is ~1,100, then the first head gesture (or DOA turn, or idle
# re-center) attaches it and the floor quadruples permanently. The gate latches,
# Gemini receives an unbroken hum, and it answers with hallucinated transcripts
# in random languages.
#
# 0.6s is ~2x the travel time of a 9g servo across this build's 120 deg range,
# so the move always completes before the pin is released. After release the
# head holds position on gearbox friction. If YOUR head is heavy enough to droop
# when unpowered, set this to 0 to keep the old always-on behaviour and accept
# the noise floor:  echo 'NECK_SERVO_HOLD_S=0' >> ~/adam/.env
NECK_SERVO_HOLD_S = float(os.getenv("NECK_SERVO_HOLD_S", "0.6"))

# Detaching the pin is NOT the end of the noise — that is the finding the decay
# measurement above exists to record. For roughly two more seconds the servo is
# still mechanically settling (and briefly spikes to 3,582, well over open_th)
# before the floor returns to baseline. `servo_moving` therefore stays SET for
# this long after the detach, so those chunks can neither open the gate nor be
# learned as the room's level. It is deliberately 2.0 and not the ~5s it takes
# for the p50 to land inside 1.1x baseline: while the flag is set the gate cannot
# OPEN, so every extra second is a second in which ADAM cannot notice someone
# starting to talk. 2.0 covers both bad seconds (p50 1,697 then 1,931, max
# 3,582); the residual after that is ~1,400, within 10% of baseline and harmless
# for both the gate and the trackers. An utterance already in progress is never
# truncated by this — see the servo_moving branch in session.py, which keeps
# refreshing the hangover while the flag is set.
NECK_SERVO_SETTLE_S = float(os.getenv("NECK_SERVO_SETTLE_S", "2.0"))

# ── Human-like movement tuning ──────────────────────────────────────────
# Deadzone: minimum degrees the target must shift before the servo moves
# at all — prevents chasing every small DOA fluctuation.
NECK_PAN_DEADZONE_DEG  = 12
# Cooldown: minimum seconds between two servo moves — prevents rapid
# back-to-back corrections that read as jittery/twitchy rather than
# deliberate human-like turns.
NECK_PAN_COOLDOWN_S    = 1.5

# ═════════════════════════════════════════════════════════════════════════════
# DIRECTION-OF-ARRIVAL (DOA) — dual INMP441 mics on v32 BODY board
# ═════════════════════════════════════════════════════════════════════════════

MIC_DISTANCE_M      = 0.065   # 65mm — typical dual-INMP441 spacing
SOUND_SPEED_MPS      = 343.0
DOA_ANGLE_DEADZONE   = 8      # degrees — ignore tiny jitter around center

# ═════════════════════════════════════════════════════════════════════════════
# ESP32-CAM WIRED LINK (Flow 2)
# ═════════════════════════════════════════════════════════════════════════════

PI_UART_PORT = os.getenv("PI_UART_PORT", "/dev/serial0")
PI_UART_BAUD = int(os.getenv("PI_UART_BAUD", "921600"))

# TPM OPTIMIZATION: was 1.0 (1 FPS). Video is the single largest ongoing
# token cost in a Live session — a JPEG frame at VGA resolution can run
# several hundred to 1000+ tokens depending on content, sent continuously
# whenever the camera is on. Confirmed via usage screenshot at 62.31K/65K
# TPM (right at the free-tier ceiling). Halving the send rate to one
# frame every 2s roughly halves video's ongoing token cost with a fairly
# small usability tradeoff.
CAMERA_FPS_INTERVAL = 2.0

# Wire protocol tags — MUST match esp32_cam.ino exactly
TAG_FRAME   = ord('F')
TAG_TOUCH   = ord('T')
TAG_GESTURE = ord('G')

GESTURE_NONE    = 0
GESTURE_ANGRY   = 1   # cheek slap — Touch1 or Touch2
GESTURE_PETTING = 2   # Touch3 + Touch4 together
GESTURE_STOP    = 3   # Touch3 alone — interrupt speech immediately

# ═════════════════════════════════════════════════════════════════════════════
# ATTENTION / IDLE
# ═════════════════════════════════════════════════════════════════════════════

ATTENTION_TIMEOUT_S = 30

# Idle ("sleep") mode. While it is set, every mic chunk goes to the offline
# Vosk wake-word detector and NOTHING goes to Gemini, so ADAM ignores ordinary
# questions BY DESIGN — a fact that is invisible from the outside and has cost
# hours of "the mic is broken" debugging. Both of these are env-overridable
# because the fastest way to rule idle mode out as the cause of silence is to
# turn it off entirely and see whether ADAM starts answering:
#
#     ENABLE_IDLE=0      → never enter idle mode from a timeout OR a tool call
#     IDLE_TIMEOUT_S=…   → seconds of silence before ADAM offers to sleep
#
# The stats line in listen() prints IDLE when this is active; that is the
# authoritative check, not guessing from the face on the screen.
ENABLE_IDLE    = os.getenv("ENABLE_IDLE", "1").strip().lower() not in (
    "0", "false", "no", "off")
IDLE_TIMEOUT_S = float(os.getenv("IDLE_TIMEOUT_S", "90"))

# Hard ceiling on a single idle period. Idle mode routes every mic chunk to
# the offline Vosk detector and NOTHING to Gemini, so while it is set ADAM is
# deaf to conversation by design — and it persists across reconnects. A live
# log showed ADAM overhearing "be quiet" from a phone call, calling
# enter_idle_mode(), and then never coming back: the documented exits (say
# "adam", or Touch3) both need something the room was not giving it — a
# clean wake word in speech-level noise, or a working ESP32-CAM UART link.
# From outside that is indistinguishable from "ADAM is broken". This bounds
# it: after IDLE_MAX_S ADAM resumes listening on its own, quietly, and logs
# why. 0 disables the ceiling and restores the old unbounded behaviour.
IDLE_MAX_S     = float(os.getenv("IDLE_MAX_S", "600"))

_NUDGES = [
    "Still there? Say something — I'm literally just sitting here.",
    "Bhai, main yahan hoon. Camera mein dekh ya naam le.",
    "Either talk or do something interesting. I'm watching you do nothing.",
    "Picture abhi baaki hai mere dost — but only if you say something.",
    "Touch grass, talk to me, or launch the next startup. Pick one.",
]
_nudge_idx = 0


def next_nudge() -> str:
    global _nudge_idx
    n = _NUDGES[_nudge_idx % len(_NUDGES)]
    _nudge_idx += 1
    return n

# ═════════════════════════════════════════════════════════════════════════════
# SEARCH
# ═════════════════════════════════════════════════════════════════════════════

SEARCH_CACHE_TTL = 1800
SEARCH_MIN_GAP_S = 5.0

# ═════════════════════════════════════════════════════════════════════════════
# LAPTOP AGENT — PRODUCTION DISCOVERY (mDNS/Zeroconf, with static fallback)
# ═════════════════════════════════════════════════════════════════════════════

LAPTOP_AGENT_PORT      = int(os.getenv("LAPTOP_AGENT_PORT", "8642"))
LAPTOP_AGENT_TOKEN     = (
    os.getenv("LAPTOP_AGENT_TOKEN") or os.getenv("AGENT_TOKEN") or ""
).strip()
LAPTOP_AGENT_TIMEOUT_S = 4.0
LAPTOP_AGENT_STATIC_IP = os.getenv("LAPTOP_AGENT_IP", "").strip()  # optional manual override
LAPTOP_MDNS_SERVICE    = "_adam-laptop._tcp.local."
LAPTOP_DISCOVERY_TIMEOUT_S = 3.0
LAPTOP_DISCOVERY_TTL_S     = 60.0   # re-verify every 60s in case laptop moved networks
LAPTOP_ACTIONS_TTL_S       = 120.0

# ═════════════════════════════════════════════════════════════════════════════
# VOSK OFFLINE WAKE-WORD
# ═════════════════════════════════════════════════════════════════════════════

VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", str(BASE_DIR / "vosk-model-small-en-us-0.15"))

# ═════════════════════════════════════════════════════════════════════════════
# CONVERSATION HISTORY
# ═════════════════════════════════════════════════════════════════════════════

CONV_MAX_TURNS    = 40   # max turns persisted to disk
# TPM OPTIMIZATION: system_prompt is rebuilt fresh on every single
# reconnect. Re-injecting the full 40-turn history every time was a real,
# avoidable contributor to hitting the 65K TPM free-tier ceiling
# (confirmed via usage screenshot at 62.31K/65K). Full 40-turn history
# stays on disk for continuity across long gaps; only a much shorter
# recent window is actually injected per-session.
CONV_PROMPT_TURNS = int(os.getenv("CONV_PROMPT_TURNS", "4"))

# ═════════════════════════════════════════════════════════════════════════════
# WEBSOCKET FACE SERVER
# ═════════════════════════════════════════════════════════════════════════════

WS_HOST = "localhost"
WS_PORT = 8765