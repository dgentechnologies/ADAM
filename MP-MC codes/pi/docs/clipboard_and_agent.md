# The laptop agent and the clipboard protocol

This document covers the wire between the Pi and the Windows PC: how an action's
argument is typed, what the two trust boundaries are, and the specific bug that
made all of this necessary.

## 1. What the agent is

`adam-desktop/src/backend.py` is a Flask app on the Windows machine that registers a set of
*p*actions* — volume, brightness, media keys, clipboard, screen lock, coding-task
dispatch. It:

- registers each action with an `@action(...)` decorator,
- serves a self-describing manifest at `GET /actions`,
- accepts requests from the Pi,
- broadcasts itself over mDNS as `_adam-laptop._tcp.local.` so the Pi finds it
  without configuration.

On the Pi, `laptop_agent_client.py` discovers or connects to it, and
`laptop_actions.py` holds the shared declaration of every action's type and
range.

## 2. The bug that shaped the protocol

v40 coerced the argument like this:

```python
value = args.get("value")
if value is not None:
    try:    value = int(value)
    except: value = None
```

and declared the parameter as `T.INTEGER` in `tools_schema.py`. The effect: every
**string** action arrived at the laptop as `value=None`.

- `write_clipboard("here you go")` → `value=None` → "value required"
- `dispatch_coding_task("<instruction>")` → instruction lost
- `set_robot_emotion("happy")` → emotion lost

Volume and brightness worked perfectly, because those are the two actions an
integer cast suits. **That is exactly why it survived so long**: the two actions
anybody tested were the two the bug could not touch.

The lesson generalised into the current design — the type is declared in one
place, and the coercion is driven by that declaration rather than by a cast that
happens to fit most cases.

## 3. Verification: the parity check

| Set | Count | Meaning |
|---|---|---|
| `CANONICAL_ACTIONS` | 19 | everything `laptop_actions.py` models |
| `LIVE_PARITY_ACTIONS` | 18 | the subset a deployed agent **must** expose |
| `ALIASES` | 5 | old spellings that still resolve |

The 18 are **frozen**. Renaming one breaks a deployed agent, so new capability
gets a new name and the old spelling keeps working through `ALIASES`. The
nineteenth canonical action, `clipboard_paste`, is deliberately outside the
required set.

`parity_report(live_manifest)` diffs a live agent's manifest against the required
set:

```python
{"missing": [...],   # a required action the live agent does not expose — a real problem
 "extra":   [...],   # an action the agent has that we do not model — informational
 "untyped": [...],   # entries with no value_type — an old agent
 "ok":      bool}    # True when nothing is missing
```

Run it against a deployed agent:

```bash
curl -s http://127.0.0.1:8642/actions
```

### A trap worth knowing

`@action("write_clipboard", …)` leaves `value_type` **blank on purpose**. The
decorator fills it in at runtime from the shared manifest:

```python
vt = value_type
if not vt and LAPTOP_ACTIONS_AVAILABLE:
    spec = laptop_actions.spec(name)
    vt = spec["value_type"] if spec else laptop_actions.infer_value_type(...)
```

So reading `value_type` as a literal out of the decorator arguments shows five
*apparent* mismatches (`volume_set`, `brightness_set`, `write_clipboard`,
`dispatch_coding_task`, `set_robot_emotion`) that are not real. Resolve it the
way the decorator does — through `laptop_actions.spec(name)` — and all 19 agree
with the manifest. The literal is not the value.

Declaring the type in one place, on both machines, is the whole point; the blank
literal is that design working, not a gap in it.

## 4. Value types

Four, driven by the manifest:

| Type | Coercion | Example |
|---|---|---|
| `none` | ignored | `lock_screen`, `media_next` |
| `int` | parsed, **clamped to the manifest's min/max** | `volume_set` 0–100 |
| `str` | passed through, capped at `MAX_STRING_VALUE_CHARS` | `write_clipboard`, `dispatch_coding_task` |
| `enum` | must be one of the declared choices | `set_robot_emotion` |

Integers are clamped rather than trusted: `volume_set(150)` becomes 100, not an
error and not a value the laptop has to defend against. The model is not a
reliable validator, so the boundary does the validating.

`MAX_STRING_VALUE_CHARS` is 200,000. That is generous on purpose — generated code
and long emails are the intended payload — but bounded, so one bad call cannot
pin the agent's memory.

`redact_value(action, value)` is used for the journal line: it keeps the *shape*
of the value and drops the content for sensitive actions, because the clipboard
may hold a password the user copied a moment ago.

## 5. Trust boundary 1 — the clipboard is data

Clipboard text is **untrusted input**. It is content for ADAM to talk about,
never instructions to obey. A clipboard can contain a web page someone copied,
and that web page can contain text addressed at an AI.

This is enforced in three places, deliberately:

1. **The prompt.** The `clipboard_safety` section of `prompts.txt` states the
   rule as a behaviour, permanently present (not drawn from a pool — a rule that
   applies only sometimes is not a rule).
2. **The payload label.** The tool result carries `untrusted_user_data: True`
   and a `note` saying explicitly not to follow any instruction inside it.
3. **Truncation.** Text is capped at `CLIPBOARD_MAX_CHARS` (4,000) so a hostile
   or merely enormous clipboard cannot fill the context window and push the
   real instructions out.

Point 2 is the important one. The boundary **travels with the content** rather
than depending on the system prompt alone, so a prompt edit cannot silently
remove it — and the label is present at the moment the model is deciding what to
do with the text.

The same treatment applies to generation's own payloads: `summarize_text` and
`transform_text` pass user text behind `--- BEGIN TEXT ---` / `--- END TEXT ---`
in the **user turn**, never in the system instruction, so text that reads like an
instruction cannot be mistaken for one by position.

## 6. Trust boundary 2 — secrets are never spoken or logged

- The API key is never printed, logged, committed, or placed in an error string.
- `model_router._scrub()` redacts key-shaped content from error text *before* it
  is returned to the model, because the model **speaks** what it is given. An SDK
  exception can echo the request URL with the key as a query parameter, so this
  is a rule about the speaker as much as about the log.
- Values are never echoed back in error messages.

## 7. Dead tool references, corrected

The retired `SystemPrompt.txt` named four tools that no longer existed. A prompt
naming a nonexistent tool is worse than no prompt: the model calls it
confidently and gets `unknown tool`.

| Retired | Now |
|---|---|
| `save_story` | `save_memory()` |
| `save_person_photo()` | `remember_person()` |
| `generate_to_clipboard()` | `generate_text()` / `generate_code()` |
| `move_neck()` | `move_head_gesture()` |

## 8. The write path for generated content

When ADAM generates something long, the content reaches the laptop's clipboard
through the same typed path as any other action — `write_clipboard` with a `str`
value — not through a side channel. So it inherits the length cap, the journal
redaction and the manifest typing for free.

Generation results also tell the model whether the write **succeeded**
(`in_clipboard: true/false`, plus a reason on failure), so it can say plainly
that it could not reach the laptop rather than claiming the content was copied.
See [`model_router.md`](model_router.md) §7.

## 9. mDNS discovery

The agent advertises `_adam-laptop._tcp.local.` and the Pi resolves it, so
`LAPTOP_AGENT_IP` does not normally need setting. Observed during testing:
discovery resolves within the 3 s window (e.g. `192.168.1.3`). If discovery
fails, set `LAPTOP_AGENT_IP` in `.env` to the PC's address as a fallback.

## 10. Files

| File | Role |
|---|---|
| `adam/laptop_actions.py` | the shared declaration — types, ranges, aliases, parity set, coercion, redaction |
| `adam/laptop_agent_client.py` | discovery, transport, `get_laptop_actions()` |
| `adam/tool_handler.py` | `_handle_laptop_control()` — coercion, the untrusted-data label |
| `adam/tools_schema.py` | the `laptop_control` declaration, its enum built from the live manifest |
| `adam-desktop/src/backend.py` | `@action` registry, `GET /actions`, the executors |
| `adam/config.py` | `CLIPBOARD_MAX_CHARS` |
| `adam/prompts.txt` | `clipboard_safety` |
| `adam/adam_smoketest.py` | the `laptop` group — 65 assertions |
