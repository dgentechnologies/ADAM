# v41 final report

**Date:** 2026-10-02
**Scope:** the v41 master implementation prompt, §53 checkpoint order
**Status:** code complete and offline-verified. **`deployed: NO`** (§38) — nothing in this report has been copied to the Pi.

## 1. What v41 was for

Two problems, stated in the prompt and confirmed in the code:

1. **ADAM said the same sentence every time.** The prompt was Python string literals inside a monolith, so there was nothing to vary and no way to review the wording without reading code.
2. **Every string-typed laptop action silently failed.** The v40 tool layer cast the argument with `int(value)`, so `write_clipboard`, `dispatch_coding_task` and `set_robot_emotion` all arrived at the laptop as `value=None`. Volume and brightness worked, which is exactly why it survived — the two actions anybody tested were the two the bug could not touch.

Around those, v41 added the scheduler, the Pi sync API, the Clock tab, and the model router.

## 2. §53 checkpoint order — where each landed

| # | Checkpoint | Status |
|---|---|---|
| 1 | Plan + gap analysis | done |
| 2 | Central prompt (`prompts.txt` + `prompt_store.py`) | done |
| 3 | Clipboard + typed protocol | done |
| 4 | Scheduler + todo | done |
| 5 | PC-app parity + memory sync | done; **live half not run** (§7) |
| 6 | Multimodal / model router | done; **never called a live model** (§7) |
| 7 | Sync check + documentation | done — this report, the six docs, Part 28 |

## 3. Files changed, by owner

### 3.1 Pi runtime (`pi/adam/`)

| File | Size | What changed |
|---|---|---|
| `prompt_store.py` | 22,357 | **new** — parse, substitute, hot reload, no-repeat pools |
| `prompts.txt` | 45,723 | **new** — 25 sections, 21 pools, 140 variants |
| `scheduler.py` | 47,892 | **new** — store, time model, ticker, nine-tool API |
| `sync_api.py` | 17,128 | **new** — stdlib HTTP API for the companion app |
| `laptop_actions.py` | 21,379 | **new** — shared manifest, parity set, coercion, redaction |
| `model_router.py` | 19,522 | **new** — on-demand code/text/vision/summary/transform |
| `config.py` | — | `MODEL_ROUTER_*`, `PROMPT_*`, `SCHEDULE_FILE`, `CLOCK_JUMP_S`, `MISSED_GRACE_S`, `CLIPBOARD_MAX_CHARS`, `SYNC_*` |
| `tools_schema.py` | — | scheduling (9), laptop (1), generation (5) declarations → **26 total** |
| `tool_handler.py` | — | `_handle_scheduler`, `_handle_laptop_control` rewritten, `_handle_generation`, `_GEN_KIND`, `_preview`, `_latest_camera_frame` |
| `session.py` | — | scheduler ticker, alarm touch precedence, fire delivery via `prompt_store.injection()`, `latest_frame` |
| `main.py` | — | sync API server start, prompt load |
| `adam_smoketest.py` | 31,217 | **new** — 193 assertions across six groups |
| `SystemPrompt.txt` | — | retired to legacy fallback; header records the four dead tool refs |

### 3.2 PC app (`pcAPP/`)

`static/js/clock.js` (new Clock tab), `backend.py` (19 `@action` registrations, `/pi/*` relay proxy), `build_exe.py` bundling.

### 3.3 Shared / prompts / docs

Docs written this cycle: [`prompt_system.md`](prompt_system.md), [`scheduler.md`](scheduler.md), [`clipboard_and_agent.md`](clipboard_and_agent.md), [`pc_app_integration.md`](pc_app_integration.md), [`model_router.md`](model_router.md); Part 28 appended to [`development_log.md`](development_log.md) (3,344 → 3,672 lines, 9 "v41" mentions); Categories 7 and 8 of [`tasks_left.md`](tasks_left.md) rewritten against the code.

## 4. The historical model investigation

The prompt asked which model ADAM should use for on-demand generation. Answered from the record rather than chosen fresh — `adamV29.py:127-129`:

```python
LIVE_MODEL        = "gemini-3.1-flash-live-preview"
GEN_MODEL_CASCADE = ["gemini-3.1-flash-lite-preview", "gemini-3.1-flash-live-preview"]
GEN_RETRIES       = 2
```

| Job | Model | Why |
|---|---|---|
| conversation, vision | `gemini-3.1-flash-live-preview` | the live model already sees the camera |
| code, text, summary, transform | `gemini-3.1-flash-lite-preview` | text work does not need the live model |
| retries | 2 (3 attempts) | matches `GEN_RETRIES` |

Defaults, not hardcoded values — overridable from `.env`. The API key is reused from `config.API_KEY` rather than duplicated, so there is one secret to rotate.

## 5. PC-app gap analysis (§16)

**Static — verified.** AST walk over `@action` decorators in `pcAPP/backend.py`:

- 19 registrations; all 18 `LIVE_PARITY_ACTIONS` **present**, zero missing.
- `value_type`: after resolving it the way the decorator actually does — at runtime, from `laptop_actions.spec(name)`, not from the decorator literal — **all 19 agree** with the manifest, zero mismatches.
- One action outside the parity set: `clipboard_paste`. Informational, not drift — it is canonical but deliberately not required.

**A trap worth recording.** Reading `value_type` as a literal out of the decorator args shows five *apparent* mismatches (`volume_set`, `brightness_set`, `write_clipboard`, `dispatch_coding_task`, `set_robot_emotion`). They are not real: `@action` leaves the literal blank on purpose and fills it from the shared manifest. My first pass flagged them as a genuine defect; the code was right and the parse was wrong. Declaring the type in one place on both machines is the whole design, and the blank literal is that design working.

**Live — not run.** Nothing was listening on 8642 during this work (`WinError 10061`, connection refused), so `GET /actions` against a real agent was never performed and `parity_report()` was never exercised against a live reply. The static diff is the right thing to diff statically, but it is not the same as a deployed agent answering. To close it, with the PC app open:

```bash
curl -s http://127.0.0.1:8642/actions
```

and compare against `LIVE_PARITY_ACTIONS`.

## 6. Tests run

```
193 passed, 0 failed, 0 skipped
```

| Group | Assertions | Covers |
|---|---|---|
| `imports` | 14 | every module imports cleanly on a flat-import Pi |
| `prompt` | 83 | parse, hot reload + last-known-good fallback, 300-draw collision check per pool, ≥5 variants |
| `laptop` | 65 | manifest typing, coercion/clamping, aliases, parity set, redaction |
| `memory` | 4 | store round-trip |
| `router` | 27 | envelope on both paths, `_scrub()` against three key shapes, fail-soft returns, tool-wiring agreement |

**19 PC-app counts:** canonical 19, required 18, aliases 5. **26** tool declarations. Files changed and SHA-verified below.

**19 offline by design.** No group spends an API call or needs the network — the question "is this install sane" has to be answerable on a Pi with no Wi-Fi.

Also run: `py_compile` clean on `config.py`, `tools_schema.py`, `tool_handler.py`, `model_router.py`, `adam_smoketest.py`; `build_tools()` returns 26 declarations with all five generation tools carrying correct properties.

## 7. Untested — stated plainly (§0.9)

| Item | Why it is untested | Risk if wrong |
|---|---|---|
| **Any live model call** | no `generate_content` request has been made from this tree | latency, token counts and vision quality are unmeasured; the `code`/`text`/`vision` model names are unverified against the live API |
| **Live PC-app parity** | nothing listening on 8642 | a deployed agent could differ from source |
| **Alarm firing end to end** | needs a running session and a real clock advance | the fire path is verified by reading code |
| **Clock-jump resync** | needs NTP correcting a wrong boot clock | the ticker could fire stale schedules at once |
| **Duplicate-fire suppression** | needs a restart between decide and speak | an alarm could double-deliver |
| **`parse_when()` edge cases** | no `scheduler` smoke group exists | a misparsed time is refused rather than silently wrong, so this is contained |

The first and the last two are the ones I would test first on the Pi. `scheduler.md` §13 names the missing `scheduler` smoke group as the obvious next test to write.

## 8. Known risks

1. **Duplicate-fire suppression is the subtle one.** It depends on `last_fired` being written before the item is spoken. Simplify the ticker and it can regress silently.
2. **The clock-jump check can be folded away.** It has its own constant and comment precisely because a naive `now >= due` loop fires every stale schedule the instant NTP corrects the boot clock.
3. **No locks by construction.** `sync_api`, `main.py` and the ticker share one event loop, which makes `replace_all()`'s read-modify-write atomic. **Adding a thread pool later would silently break that** and needs a real lock. A comment on the read path says so.
4. **The 18 parity actions are frozen.** Renaming one breaks an agent the user has not updated; new capability gets a new name and the old spelling keeps resolving through `ALIASES`.
5. **Secrets are now spoken, not just logged.** `model_router._scrub()` exists because error text goes back to the model, which says it out loud. An SDK exception can echo the request URL with the key as a query parameter.
6. **Stale `v40` headers** remain in `adam.service:2` and some module docstrings (e.g. `tool_handler.py` line 2). Cosmetic; nothing reads them.
7. **`adam.py` (188,143 bytes) is still present** alongside the split `adam/` package. Deletion needs the user's say-so.

## 9. Deployment

```
deployed: NO
```

None of the v41 code has been copied to the Pi. The order that avoids a half-deployed window is: the seven new modules first, then `config.py`, then `tools_schema.py` and `tool_handler.py`, then run `adam_smoketest.py` on the Pi before restarting `adam.service`. The lazy-import pattern means a partially-copied tree loses individual features rather than the whole tool layer — a Pi without `scheduler.py` answers every other tool call and reports "that isn't available" for alarms.

The Windows executable *was* rebuilt this cycle: `dist/ADAM.exe`, 37,294,029 bytes (35.57 MB), verified to contain the updated `static/js/clock.js`.

## 10. Constraints honoured

- **§0.3 — the audio path was not touched.** No change to `arecord`/`aplay`, AdaptiveGate, the filters, half-duplex/drain logic, `write_all()`, the device name, or reconnect behaviour. Every v41 feature is additive and isolated.
- **§0.4 — reconnect triggers unchanged:** only `listen`, `send`, `receive`, `speaker`.
- **§0.7 — no secrets exposed.** No `.env` printed, no key logged, none copied into docs or fixtures. The router reuses the existing key rather than adding a second one.
- **§18 — clipboard is data.** Enforced three times: the permanent `clipboard_safety` prompt section, the `untrusted_user_data` label on the result, and truncation at 4,000 chars.
- **§0.9 / §49 — nothing claimed that was not tested.** §7 above is the full list of what was not, including the live model call.
- **§0.12 — no ground-rule conflict arose,** so no stop-and-ask was needed.
