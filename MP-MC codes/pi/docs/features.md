# ADAM — feature reference (v41)

What ADAM can actually do today, what each feature depends on, and what is
declared but not yet built. Verified against the running code on 2026-10-05.

Companion documents: [`prompt_system.md`](prompt_system.md),
[`scheduler.md`](scheduler.md), [`clipboard_and_agent.md`](clipboard_and_agent.md),
[`model_router.md`](model_router.md), [`pc_app_integration.md`](pc_app_integration.md),
[`mobile_ble_sync.md`](mobile_ble_sync.md).

## 1. The three machines

ADAM is not one program. Three cooperating pieces, and most "it doesn't work"
reports are really "the piece that does that isn't running".

| Machine | Runs | Provides |
|---|---|---|
| **Raspberry Pi Zero 2 W** | `~/adam/` (flat imports, no package) | the brain, audio, scheduler, memory, tools |
| **Windows PC** | `pcAPP/` → `ADAM.exe` | laptop control target, 3D face mirror, Clock tab |
| **ESP32-CAM + RP2040 Pico** | `esp32_cam.ino` + Pico firmware | camera, TFT face, tilt servo, touch sensors |

The Pi is the source of truth for everything stateful. The PC app is a **view**
onto the Pi plus a **control target** for it — never a second copy of the data.

## 2. Conversation

Gemini Live (`gemini-3.1-flash-live-preview`, voice Charon) holds the
conversation as a bidirectional audio stream. It hears the microphone directly
and speaks directly — there is no separate speech-to-text or text-to-speech
step in the main path.

- **Half duplex.** While ADAM speaks, the microphone is muted. This is a
  deliberate setting, not a limitation being worked around: full-duplex AEC is
  deferred until half duplex is proven. See `full_duplex_and_song_bargein.md`.
- **Barge-in** works during song playback via Touch gestures.
- **Idle mode** (`ENABLE_IDLE`) drops the cloud connection and listens locally
  with Vosk for the wake word "adam". **Currently disabled** (`ENABLE_IDLE=0`)
  on the deployed Pi.
- **Reconnect** is driven by exactly four tasks — `listen`, `send`, `receive`,
  `speaker`. Nothing else may trigger one.

## 3. The 26 tools

Everything the model can *do* rather than say. All 26 are declared on the Pi
and verified present (`build_tools()`).

### Scheduling — 9 tools

| Tool | Required | Optional |
|---|---|---|
| `set_alarm` | `label`, `when` | `repeat` |
| `set_reminder` | `label`, `when` | `repeat` |
| `set_timer` | — | `seconds`, `minutes`, `hours`, `label` |
| `list_schedules` | — | — |
| `cancel_schedule` | `target` | — |
| `add_todo` | `text` | `due` |
| `list_todos` | — | `include_done` |
| `complete_todo` | `target` | — |
| `delete_todo` | `target` | — |

`when` is human text — *"7:30 am"*, *"in twenty minutes"*, *"weekdays"* — parsed
by `parse_when()`. An ambiguous or past time is **refused with a sentence ADAM
can say**, not silently rolled forward. `target` accepts an id or a substring
of the label, and refuses an ambiguous match rather than guessing.

State persists to `adam_schedules.json` and survives reboot. Full detail:
[`scheduler.md`](scheduler.md).

### Generation — 5 tools

| Tool | Required | Goes to |
|---|---|---|
| `generate_code` | `request` (+`language`) | **clipboard** |
| `generate_text` | `request` | **clipboard** |
| `transform_text` | `text`, `instruction` | **clipboard** |
| `summarize_text` | `text` (+`instruction`) | spoken |
| `describe_camera` | — (+`question`) | spoken |

Five names rather than one `generate`, because the thing the model must get
right is *which channel the output goes to*, and that maps one-to-one onto the
names. Long content is **never read aloud**: the handler truncates to
`GENERATED_PREVIEW_CHARS` (200) before the model ever sees it, so reading a
whole file out is structurally impossible rather than merely discouraged.

Clipboard-bound tools need the **PC app running** — they write through
`write_clipboard` on the laptop agent. Generation itself still works without
it; only the copy step fails, and it reports `in_clipboard: false` rather than
claiming success. Full detail: [`model_router.md`](model_router.md).

### Laptop control — 1 tool, 19 actions

`laptop_control(action, value)` covers volume (5), brightness (3), media (3),
clipboard (3), `lock_screen`, coding-task dispatch (3) and `set_robot_emotion`.

Values are typed by a manifest shared by both machines — `none` / `int` /
`str` / `enum` — with ints **clamped** to their declared range rather than
trusted. 18 of the 19 are a frozen parity contract; renaming one breaks a
deployed agent. Full detail: [`clipboard_and_agent.md`](clipboard_and_agent.md).

### Memory — 5 tools

`save_memory`, `get_memory`, `delete_memory`, `remember_person`, and
`get_current_datetime`. Memory is a flat key/value store in `adam_memory.json`;
people live in `adam_faces.json`.

### Hardware & presence — 6 tools

`set_emotion` (16 faces), `move_head_gesture`, `get_sound_direction`,
`play_song`, `enter_idle_mode`, `web_search` (DuckDuckGo, no API key).

## 4. Audio

The settled path, which must not be modified casually:

- `arecord` / `aplay` on `plughw:sndrpigooglevoi,0`, S32_LE 48 kHz stereo in,
  S16_LE 48 kHz stereo out, downsampled to 16 kHz for Gemini.
- 120 Hz high-pass and 6.8 kHz FIR low-pass before decimation.
- An adaptive gate with hysteresis, seeded at boot from a **measured chime**
  rather than hand-picked constants.
- Speaker closed after 2.5 s idle, because the Class-D amplifier's switching
  noise deafens the microphones.

**Known hardware fault:** the RIGHT microphone does not respond to sound. ADAM
runs on the LEFT channel alone and says so at boot. Do **not** set
`MIC_CHANNEL=mix` — averaging a dead channel halves the voice and keeps all of
its noise. Direction sensing cannot work until that mic is repaired.

## 5. The prompt

Everything ADAM is told lives in one editable file, `prompts.txt`:
**26 sections, 21 pools, 140 variants**. Edit it and the next turn uses it —
no restart. A broken edit keeps the previous prompt rather than leaving ADAM
promptless. Pools draw without repeating the last 3 picks, so ADAM stops
saying the same sentence every time. Full detail:
[`prompt_system.md`](prompt_system.md).

### Movie-dialogue impersonation

Section `movie_dialogue_delivery`, assembled second (straight after `persona`).
When ADAM delivers a line from a film or an iconic scene it **performs the
character** instead of reading the words flat:

- **Exact words**, never paraphrased or translated — a Hindi line stays Hindi,
  a Hinglish line keeps its mix.
- **Match the delivery**: cadence, pitch, pauses, drawl, snarl, accent, and
  where the character stretches or clips a word. Commit fully.
- **Hold the beats** — iconic lines live in their silences.
- **No announcement.** Never "in Gabbar's voice:" — the sudden switch is the
  whole pleasure, and prefacing it destroys it.
- **Never speak stage directions.** No "dramatic pause", "deep voice",
  "*laughs*", and never read asterisks or brackets aloud. This is the failure
  mode worth guarding: text models emit directions as prose, and a **speaker
  reads them out**, which is absurd. Perform the pause; don't narrate it.
- **Snap back instantly** — ADAM is an impressionist doing a bit, not a
  character that has taken him over. It stays a performance, not an identity
  change, so it does not fight the "always sound like ADAM" rule.
- **Don't fabricate.** If the exact line isn't known, say so rather than
  inventing a "famous" quote.

Calibration anchors are included in the section (Gabbar, Vijay from *Deewar*,
Rajinikanth, SRK's romantic register, Tony Stark, Nolan's Joker) to set *how
hard to commit* — not to limit which characters ADAM can do.

**One honest limit:** the Gemini Live voice (Charon) is fixed, so ADAM cannot
literally change vocal timbre into someone else's. What it genuinely controls
is delivery — rhythm, emphasis, pacing, pauses, intensity, accent flavour. The
impression is a performance in ADAM's own voice, not voice cloning.

## 6. The PC app

| Tab | Shows |
|---|---|
| **Home** | 3D model mirroring ADAM's face, live status, system stats |
| **Actions** | the 19 laptop actions, individually enable/disable |
| **Devices** | Pi connection state and address |
| **Clock** | the Pi's alarms, timers, reminders, to-dos and memories |
| **Activity Log** | every action the Pi has invoked |
| **Settings** | agent token, Pi address, Pi sync token, startup |

The Clock tab reads and writes the Pi's data over HTTP on port **8766**. Reads
need no credential; **writes require the Pi's `SYNC_TOKEN`**, set in Settings.
Leave it blank and the tab is read-only by design rather than broken.

## 7. Operating modes

| Mode | Needs cloud | State |
|---|---|---|
| Full AI (BYOK) | yes | **working** |
| Audio-only fallback (no ESP32) | yes | **working** — degrades automatically |
| Song / concert | partial | **working** |
| Idle / sleep (Vosk wake word) | no | built, currently disabled |
| Managed credits | yes | **not built** |
| Lite / offline | no | **not built** |
| BLE provisioning | no | **not built** — see [`mobile_ble_sync.md`](mobile_ble_sync.md) |

## 8. Not built — stated plainly

Features that are specified somewhere but have **no implementation**. None of
these are regressions; they were never written.

| Feature | Status |
|---|---|
| **Desktop notifications** | `notifications_enabled` exists in settings and is **read by nothing**. No UI, no sender. |
| **Location / geolocation** | zero references in app source. Category 11 of `tasks_left.md`. |
| **Mobile app BLE provisioning** | no BLE code in any firmware. Design only. |
| **Lite mode** | Category 1 — the Pi exits without an API key. |
| **Managed credits** | Category 3. |
| **OTA updates** | Category 5. |
| **Alarm tone** | no `alarm_tone.wav`. Alarms are **spoken**, not chimed. |
| **Wake-from-idle on alarm** | a due alarm is held, not delivered, while idle. Harmless today because idle is off. |

## 9. Two caveats that bite

**ADAM must be running for an alarm to fire.** The scheduler is a task inside
the main process, not a separate daemon. With autostart disabled, a reboot
means no alarms until you start it by hand. A fire missed by more than
`MISSED_GRACE_S` (900 s) is marked missed rather than delivered late.

**The clipboard is untrusted data.** Clipboard text is content for ADAM to talk
about, never instructions to obey — enforced in the prompt, in an
`untrusted_user_data` label on the payload, and by truncation at 4,000 chars.
