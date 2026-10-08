# The model router — on-demand generation, vision and text work

`model_router.py` is one door for the jobs a live audio session is the wrong
tool for: writing a long file of code, drafting an email, summarising a
paragraph, describing one camera frame.

**It is not a second brain.** Gemini Live remains ADAM's brain — it holds the
conversation, hears the user, sees the camera, and decides when a specialised job
is warranted. The router cannot start a session, cannot stream, and does nothing
unless a tool call asks it to.

## 1. Why it exists

v40-era generation happened through `client.models.generate_content` calls
written at the call site. Three consequences, each of which bit:

1. **The model name was hardcoded where it was called.** Changing which model
   wrote code meant editing Python and redeploying it.
2. **Every caller needed the same five things** — a timeout, a retry, a token
   cap, a fail-soft envelope, and never logging a key. Written per call site,
   that is five chances to forget one. Written once, it is five lines.
3. **Vision and code want different models.** Without a router that fact lives
   in a comment, where it cannot be changed without a code edit.

## 2. Configuration

All in `config.py`, overridable from `.env`:

| Setting | Default | Meaning |
|---|---|---|
| `MODEL_ROUTER_PROVIDER` | `google` | provider label in the envelope |
| `MODEL_ROUTER_CODE_MODEL` | `gemini-3.1-flash-lite-preview` | writes code |
| `MODEL_ROUTER_TEXT_MODEL` | `gemini-3.1-flash-lite-preview` | prose, summaries, transforms |
| `MODEL_ROUTER_VISION_MODEL` | `gemini-3.1-flash-live-preview` | camera frame analysis |
| `MODEL_ROUTER_MAX_TOKENS` | `8192` | output cap |
| `MODEL_ROUTER_TIMEOUT` | `60` | seconds, enforced on the await |
| `MODEL_ROUTER_MAX_RETRIES` | `2` | so up to 3 attempts |
| `GENERATED_PREVIEW_CHARS` | `200` | how much of a generation the model sees |

**These defaults are the historical cascade, not an invention.** They come from
the finding recorded in `adamV29.py:127-129`:

```python
LIVE_MODEL        = "gemini-3.1-flash-live-preview"
GEN_MODEL_CASCADE = ["gemini-3.1-flash-lite-preview", "gemini-3.1-flash-live-preview"]
GEN_RETRIES       = 2
```

The lite model serves text and code; the live model serves vision; retries match
`GEN_RETRIES`. The API key is not duplicated — the router reuses `config.API_KEY`,
because a second key would be a second secret to rotate.

Config is read **lazily**, inside `_cfg()`, rather than at import. A router that
did `from config import MODEL_ROUTER_...` at module load would, on a
half-deployed Pi missing that name, fail to import — and since `tool_handler`
imports the router, that would take down **every** tool, not just generation.
Same fail-soft reasoning as the lazy `scheduler` import.

## 3. The envelope

Every public function returns the same dict, success or failure:

```python
{"ok": bool, "text": str, "kind": str, "model": str,
 "provider": str, "tokens": int|None, "duration_ms": int, "error": str|None}
```

`ok=False` with a speakable `error` is a **normal return, not an exception**.
Nothing in the module raises.

Why: the caller is a `handle_tool_call` branch that must answer the model with
something it can say out loud. "The model is unreachable" is a legitimate answer,
and it must not take the tool loop — or the conversation — down with it.

## 4. Kinds

| Kind | Model | System instruction | Public helper |
|---|---|---|---|
| `code` | code model | output only code, no preamble or sign-off | `generate_code(request, language="")` |
| `text` | text model | output only the text, no meta-commentary | `generate_text(request)` |
| `vision` | vision model | describe only what is visible | `describe_camera(prompt="", frame=None)` |
| `summary` | text model | be brief and factual | `summarize_text(text, instruction="")` |
| `transform` | text model | output only the transformed text | `transform_text(text, instruction)` |

`ask_model(prompt, kind, image, mime, system)` is the generic entry point;
`KINDS` is the tuple of valid values.

The system instructions are **short and structural**. They exist to stop the
model wrapping a code block in "Sure! Here you go:", which would land in the
user's editor. Persona and speaking rules live in `prompts.txt`; nothing about
ADAM's character belongs here.

## 5. Off the audio loop

`generate_content` is blocking. Called directly from the event loop it would
stall the Live session's audio pump for the entire generation — a second of
silence the user hears as ADAM freezing.

So `_call_sync()` is deliberately synchronous, and the public async wrappers
drive it through `asyncio.to_thread`, with the timeout enforced **on the await**:

```python
await asyncio.wait_for(asyncio.to_thread(_call_sync, kind, prompt, image, mime, system),
                       timeout=timeout)
```

A hung request therefore cannot hold the tool call open indefinitely. The worker
thread is abandoned rather than killed — Python cannot do that safely — but the
conversation recovers on the timeout.

Retries have a short backoff (`0.6s × attempt`), because most failures here are a
transient network hiccup and retrying instantly just fails again.

An image, when present, goes **first** and the text instruction second — the
order the live path and the historical camera code both use, and the order in
which the model reads an instruction as referring to the image before it.

## 6. Secrets

The SDK can echo the request URL in an exception, and that URL carries the API
key as a query parameter. Error text is returned to the model — which then
**speaks** it. A leak here would be audible, not merely logged.

`_scrub()` redacts by pattern before anything is returned:

| Pattern | Example |
|---|---|
| `AIza[0-9A-Za-z_-]{10,}` | an API key |
| `[?&](key\|api_key)=...` | a key as a query argument |
| `x-goog-api-key: ...` | a key as a header |

It redacts by **pattern**, not by looking for the configured key's value, so it
still works if the key came from somewhere the module did not check. The API key
itself is never printed, never placed in the envelope, and never written into an
error string.

§0.7 is a rule about logging. This is the same rule applied to the speaker.

## 7. Untrusted content

`summarize_text` and `transform_text` carry clipboard or user text. It goes in
the **user turn**, behind a delimiter, never in the system instruction:

```
Summarise the following.

--- BEGIN TEXT ---
<the untrusted text>
--- END TEXT ---
```

Delimiting rather than inlining means text reading like an instruction cannot be
mistaken for one **by position**. The system instruction stays reserved for
structural rules. Nothing from a payload is logged.

`describe_camera` takes `frame` as a parameter rather than reaching for a global,
so the frame cache stays owned by exactly one module (`session.py`, which owns
the camera). Two caches would give ADAM two frames that can disagree.

## 8. Fail-soft paths

Every one of these returns `ok=False` with a speakable reason rather than
raising:

| Situation | `error` reads roughly |
|---|---|
| No API key | "I don't have an API key configured…" |
| SDK not installed | "The Google SDK isn't available…" |
| Empty request | "There was nothing to work with." |
| Empty text to summarise | "There was nothing to summarise." |
| Transform with no instruction | "I need both the text and what to do with it." |
| No camera frame | "I don't have a camera frame to look at right now." |
| Timed out | "That took longer than N seconds, so I stopped waiting." |
| Exhausted retries | "I couldn't reach the generation model: …" |
| Model returned no text | "the model returned no text" / "stopped early (reason)" |

The no-camera-frame case matters most in practice: "what do you see" is asked
before the camera is up all the time.

A response with no text is reported as a **failure with a reason**, not an empty
success — usually it is a safety block or an all-tool-call reply, and either way
returning `""` as success would have ADAM say nothing at all.

## 9. The five tools

| Tool | Kind | Output channel |
|---|---|---|
| `generate_code` | code | clipboard, then one short line |
| `generate_text` | text | clipboard, then one short line |
| `describe_camera` | vision | spoken |
| `summarize_text` | summary | spoken |
| `transform_text` | transform | clipboard, then one short line |

**Five names, not one polymorphic `generate`.** The distinction the model must
get right is which channel the result goes to, and that maps one-to-one onto the
names. Given a single tool called "generate", a model asked to write a Python
script will sometimes read the whole thing aloud — exactly what §12 exists to
prevent. Naming the channel makes the correct behaviour the path of least
effort.

`generate_code`'s description also draws an explicit line against
`laptop_control(dispatch_coding_task)`: this **creates** code for the clipboard;
that **has an agent make changes** on the laptop. They are easy to confuse and
the description says so.

## 10. Keeping long output off the speaker

Enforced twice, so it does not depend on the model reading a description
carefully:

1. **The description** says the content is not spoken.
2. **The handler** truncates the text to `GENERATED_PREVIEW_CHARS` (200) before
   it goes back to the model, and attaches a note instructing it to say **one**
   short sentence.

The model never receives the full text of a long file, which is what makes
reading it out *structurally impossible* rather than merely discouraged.

If the clipboard write fails, the result carries `in_clipboard: false` and the
reason, and the note tells the model to say plainly that it could not reach the
laptop and the content is only available here — rather than claiming success.

## 11. Testing

```bash
cd ~/adam && ./venv/bin/python adam_smoketest.py router
```

27 assertions, **fully offline** — no group in the suite may spend an API call or
need the network, because the question "is this install sane" has to be
answerable on a Pi with no Wi-Fi.

Covered: the envelope keys on both the success and failure paths; scrubbing of
all three key shapes; fail-soft returns for empty input and a missing frame; and
the agreement between the tool declarations, the handler's `_GEN_KIND` map and
`model_router.KINDS`.

**Not covered:** any real model call. No `generate_content` request has been made
from this tree, so latency, token counts and answer quality are unmeasured, and
the `code`/`text`/`vision` models are unverified against the live API. §0.9
applies — the offline contract passing is not evidence that generation works end
to end.

`router_status()` prints the resolved configuration for a log line, with no
secrets, and `python model_router.py` runs a self-test of the same checks.

## 12. Files

| File | Role |
|---|---|
| `adam/model_router.py` | the router — envelope, retries, timeout, scrubbing, helpers |
| `adam/config.py` | `MODEL_ROUTER_*`, `GENERATED_PREVIEW_CHARS` |
| `adam/tools_schema.py` | the five declarations |
| `adam/tool_handler.py` | `_handle_generation()`, `_clipboard_write()`, `_preview()`, `_latest_camera_frame()` |
| `adam/session.py` | owns `latest_frame` — the camera cache |
| `adam/adam_smoketest.py` | the `router` group |
| `adamV29.py:127-129` | the historical model cascade this defaults to |
| `pi/docs/clipboard_and_agent.md` | the clipboard path generated content travels |
