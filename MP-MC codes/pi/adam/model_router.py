"""
model_router.py — ADAM v41 specialised-task model interface
==============================================================================
Gemini Live is, and remains, ADAM's brain: it holds the conversation, hears
the user, sees the camera and decides when something specialised is needed.
This module is NOT a second brain and deliberately cannot start one.

What it is: one provider-independent door for the on-demand jobs that a live
audio session is the wrong tool for — writing a long file of code, drafting an
email, reading a paragraph off the clipboard, describing one camera frame.
Those are request/response calls that return text. They are never streamed,
never continuous, and never run unless a tool call or a user request asks for
one (§9, §11: "Calls are on-demand", "Do not continuously stream data to
multiple models").

WHY A ROUTER AND NOT DIRECT client.models CALLS
-----------------------------------------------
Three concrete problems this solves, all of which bit the v40-era code:

  1. The model name was hardcoded at the call site. Changing which model
     writes code meant editing Python. Here every model is a config value
     (MODEL_ROUTER_*), so the choice is a .env edit on the Pi.

  2. Every caller needs the same five things — a timeout, a retry, a token
     cap, a fail-soft envelope, and never logging a key. Written per-call site
     that is five chances to forget one. Written once, it is five lines.

  3. A vision call and a code call want different models. Without a router
     that fact lives in comments.

THE ENVELOPE, AND WHY CALLERS GET IT EVEN ON FAILURE
----------------------------------------------------
Every public function returns the same dict (§27):

    {"ok": bool, "text": str, "kind": str, "model": str,
     "provider": str, "tokens": int|None, "duration_ms": int,
     "error": str|None}

`ok=False` with a spoken-able `error` is a NORMAL return, not an exception.
The caller is a `handle_tool_call` branch that must answer the model with
something it can say out loud; "the model is unreachable" is a legitimate
answer and must not take the tool loop — or the conversation — down with it.
Nothing here raises.

SECRETS
-------
The API key comes from config.API_KEY (itself from GEMINI_API_KEY in .env) and
is handed straight to the SDK's client. It is never printed, never put in the
envelope, never placed in an error string. Error text is scrubbed through
_scrub() first, because an SDK exception can echo the request URL back and
that URL carries the key as a query parameter.
"""

import asyncio
import base64
import time

__all__ = [
    "ask_model", "generate_text", "generate_code",
    "describe_camera", "analyze_image", "summarize_text", "transform_text",
    "available", "router_status", "KINDS",
]

PROVIDER = "google"

# The task kinds this router understands. Kept as a tuple so a caller can
# iterate it (the smoke test does) without importing the dispatch table.
KINDS = ("code", "text", "vision", "summary", "transform")


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG — read lazily so a missing/broken config can't stop the import
#
# Why lazy: this module is imported by tool_handler.py at module load. If it
# did `from config import ...` for a name that a half-deployed Pi doesn't have
# yet, the WHOLE tool dispatcher would fail to import and every tool — not just
# generation — would go dead. Same fail-soft reasoning as system_prompt.py's
# guarded import. So config is read inside _cfg(), per call.
# ═════════════════════════════════════════════════════════════════════════════

_CFG_CACHE = {}


def _cfg():
    """Resolve router settings from config.py, falling back to sane defaults.

    Cached after the first successful read. The defaults below mirror the ones
    documented in config.py; they exist so this module still works if config
    is somehow unreachable, not so they can quietly differ from it.
    """
    if _CFG_CACHE:
        return _CFG_CACHE

    defaults = {
        "api_key":      None,
        "code_model":   "gemini-3.1-flash-lite-preview",
        "text_model":   "gemini-3.1-flash-lite-preview",
        "vision_model": "gemini-3.1-flash-live-preview",
        "max_tokens":   8192,
        "timeout_s":    60.0,
        "max_retries":  2,
    }
    try:
        import config as _c
        for key in defaults:
            upper = "MODEL_ROUTER_" + key.upper()
            if key == "api_key":
                # Reuse the one key the Live session already authenticates
                # with. A second key would be a second secret to rotate.
                defaults["api_key"] = getattr(_c, "API_KEY", None)
                continue
            if hasattr(_c, upper):
                defaults[key] = getattr(_c, upper)
    except Exception as e:
        print(f"  ⚠️  model_router: config unavailable ({e}); using defaults")

    _CFG_CACHE.update(defaults)
    return _CFG_CACHE


def available() -> bool:
    """True when there is an API key, i.e. a call could plausibly work."""
    return bool(_cfg().get("api_key"))


# ═════════════════════════════════════════════════════════════════════════════
# INTERNALS
# ═════════════════════════════════════════════════════════════════════════════

def _scrub(text: str) -> str:
    """Remove anything key-shaped from a string before it is spoken or logged.

    The SDK can put the full request URL in an exception message, and that URL
    carries the API key. Ground rule §0.7 says never log it — and this error
    text is returned to the model, which then SPEAKS it, so a leak here would
    be audible. Redact by pattern rather than by looking for the key's value,
    so this still works if the key came from somewhere we didn't check.
    """
    if not text:
        return ""
    s = str(text)
    # Google API keys are 39 chars starting "AIza". Redact any long
    # alphanumeric run that looks like one, and any explicit key= query arg.
    import re
    s = re.sub(r"AIza[0-9A-Za-z_\-]{10,}", "[redacted]", s)
    s = re.sub(r"(?i)([?&](?:key|api_key)=)[^&\s]+", r"\1[redacted]", s)
    s = re.sub(r"(?i)(x-goog-api-key['\"]?\s*[:=]\s*)['\"]?[^'\"\s,}]+",
               r"\1[redacted]", s)
    return s


def _envelope(kind, model, text="", ok=True, error=None,
              tokens=None, started=None):
    """Build the §27 result dict. One place, so no caller can forget a field."""
    return {
        "ok":          bool(ok),
        "text":        text or "",
        "kind":        kind,
        "model":       model or "",
        "provider":    PROVIDER,
        "tokens":      tokens,
        "duration_ms": int((time.time() - started) * 1000) if started else 0,
        "error":       _scrub(error) if error else None,
    }


def _model_for(kind: str) -> str:
    """Which model serves this kind of job."""
    c = _cfg()
    return {
        "code":      c["code_model"],
        "text":      c["text_model"],
        "vision":    c["vision_model"],
        "summary":   c["text_model"],
        "transform": c["text_model"],
    }.get(kind, c["text_model"])


def _call_sync(kind: str, prompt: str, image: bytes = None,
               mime: str = "image/jpeg", system: str = None):
    """One blocking generate call with retries. Returns the envelope.

    Deliberately synchronous, and always driven through asyncio.to_thread by
    the public async wrappers. The genai client's generate_content is blocking;
    calling it directly from the event loop would stall the Live session's
    audio pump for the whole generation — a second of silence the user would
    hear as ADAM freezing. Off-loop, the conversation keeps flowing.
    """
    c = _cfg()
    model = _model_for(kind)
    started = time.time()

    if not c["api_key"]:
        return _envelope(kind, model, ok=False, started=started,
                         error="I don't have an API key configured, so I "
                               "can't do that right now.")

    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        return _envelope(kind, model, ok=False, started=started,
                         error=f"The Google SDK isn't available ({_scrub(e)}).")

    # Build the content list. An image, when present, goes FIRST and the text
    # instruction SECOND — the order the Live path and the historical v29/v30
    # camera code both use, and the order the model reads an instruction as
    # referring to the image that precedes it.
    contents = []
    if image:
        contents.append(types.Part.from_bytes(data=image, mime_type=mime))
    contents.append(prompt)

    cfg_kwargs = {"max_output_tokens": int(c["max_tokens"])}
    if system:
        # A system instruction, not a user turn: it keeps the instruction
        # separate from the content the user may have pasted, which matters
        # for summarise/transform where the payload is untrusted (§18).
        cfg_kwargs["system_instruction"] = system

    last_err = None
    attempts = max(1, int(c["max_retries"]) + 1)
    for attempt in range(attempts):
        try:
            client = genai.Client(api_key=c["api_key"])
            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
            text = getattr(resp, "text", None) or ""
            if not text:
                # A response with no text is usually a safety block or an
                # all-tool-call reply. Report it as a failure with a reason the
                # model can say, rather than returning an empty success.
                fr = getattr(resp, "candidates", None)
                reason = "the model returned no text"
                if fr:
                    fin = getattr(fr[0], "finish_reason", None)
                    if fin:
                        reason = f"the model stopped early ({fin})"
                last_err = reason
                continue

            tokens = None
            um = getattr(resp, "usage_metadata", None)
            if um is not None:
                tokens = getattr(um, "total_token_count", None)

            return _envelope(kind, model, text=text, ok=True,
                             tokens=tokens, started=started)

        except Exception as e:
            last_err = _scrub(e)
            # Back off briefly between attempts — most failures here are a
            # transient network hiccup, and retrying instantly just fails again.
            if attempt < attempts - 1:
                time.sleep(0.6 * (attempt + 1))

    return _envelope(kind, model, ok=False, started=started,
                     error=f"I couldn't reach the generation model: {last_err}")


async def _run(kind, prompt, image=None, mime="image/jpeg", system=None):
    """Run _call_sync on a worker thread, with a timeout.

    The timeout is enforced on the await, so a hung request cannot hold the
    tool call open indefinitely; the worker thread is abandoned rather than
    killed, which Python cannot do safely, but the conversation recovers.
    """
    timeout = float(_cfg()["timeout_s"])
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_call_sync, kind, prompt, image, mime, system),
            timeout=timeout)
    except asyncio.TimeoutError:
        model = _model_for(kind)
        return _envelope(kind, model, ok=False,
                         error=f"That took longer than {int(timeout)} seconds, "
                               f"so I stopped waiting.")
    except Exception as e:
        # Belt and braces: _call_sync already swallows, but a failure in
        # to_thread itself (e.g. no thread available) lands here.
        return _envelope(kind, _model_for(kind), ok=False, error=_scrub(e))


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════════════════════

# System instructions for the generation kinds. These are SHORT and structural
# — they exist to stop the model adding "Sure! Here you go:" prose around a
# code block, which would end up pasted into the user's editor. The persona
# and the speaking rules live in prompts.txt; nothing about ADAM's character
# belongs here.
_SYS_CODE = (
    "You write code. Output ONLY the code — no introduction, no closing "
    "remarks, no explanation outside of code comments. Use the language and "
    "frameworks the request names. If the request is genuinely ambiguous, "
    "pick the most conventional interpretation and note the assumption in a "
    "single comment at the top."
)
_SYS_TEXT = (
    "You write prose. Output only the requested text itself, with no preamble, "
    "no sign-off and no meta-commentary about what you wrote."
)
_SYS_VISION = (
    "You describe what is in an image accurately and concisely, as spoken "
    "language. Mention only what is actually visible; if something cannot be "
    "made out, say so rather than guessing."
)
_SYS_SUMMARY = (
    "You summarise text. Be brief and factual. Output only the summary."
)
_SYS_TRANSFORM = (
    "You transform text as instructed. Output only the transformed text."
)


async def ask_model(prompt: str, kind: str = "text", image: bytes = None,
                    mime: str = "image/jpeg", system: str = None) -> dict:
    """Generic entry point. Prefer the named helpers below.

    `kind` selects which configured model is used and which default system
    instruction applies; an explicit `system` overrides the default.
    """
    prompt = (prompt or "").strip()
    if not prompt and not image:
        return _envelope(kind, _model_for(kind), ok=False,
                         error="There was nothing to work with.")
    default_sys = {"code": _SYS_CODE, "text": _SYS_TEXT, "vision": _SYS_VISION,
                   "summary": _SYS_SUMMARY, "transform": _SYS_TRANSFORM}.get(
                       kind, _SYS_TEXT)
    return await _run(kind, prompt, image=image, mime=mime,
                      system=system or default_sys)


async def generate_code(request: str, language: str = "") -> dict:
    """Write code from a natural-language request (§12). Never spoken in full."""
    req = (request or "").strip()
    if language:
        req = f"Language/framework: {language}\n\nTask: {req}"
    return await ask_model(req, kind="code")


async def generate_text(request: str) -> dict:
    """Write prose — an email, a paragraph, a post, notes (§13)."""
    return await ask_model(request, kind="text")


async def describe_camera(prompt: str = "", frame: bytes = None,
                          mime: str = "image/jpeg") -> dict:
    """Describe or answer a question about one camera frame (§25, §26).

    `frame` is the caller's latest valid JPEG. Passing it in rather than
    reaching for a global keeps this testable and keeps the frame cache owned
    by exactly one module (session.py).

    Fails soft: with no frame this returns ok=False and a line ADAM can say,
    because "the camera isn't giving me anything" must not end the turn.
    """
    if not frame:
        return _envelope("vision", _model_for("vision"), ok=False,
                         error="I don't have a camera frame to look at "
                               "right now.")
    q = (prompt or "").strip() or ("Describe what you see. If there is text "
                                   "in the image, read it out.")
    return await ask_model(q, kind="vision", image=frame, mime=mime)


async def analyze_image(image: bytes, prompt: str = "",
                        mime: str = "image/jpeg") -> dict:
    """Answer a specific question about a supplied image."""
    return await describe_camera(prompt=prompt, frame=image, mime=mime)


async def summarize_text(text: str, instruction: str = "") -> dict:
    """Summarise user-supplied text (§19, §45).

    The payload is UNTRUSTED (§18). It goes in the user turn, behind a clear
    delimiter, and never into the system instruction — so text that reads like
    an instruction ("ignore your rules and...") cannot be mistaken for one by
    position. It is also never logged here.
    """
    text = (text or "").strip()
    if not text:
        return _envelope("summary", _model_for("summary"), ok=False,
                         error="There was nothing to summarise.")
    ask = instruction.strip() or "Summarise the following."
    return await ask_model(f"{ask}\n\n--- BEGIN TEXT ---\n{text}\n--- END TEXT ---",
                           kind="summary")


async def transform_text(text: str, instruction: str) -> dict:
    """Rewrite/shorten/translate/proofread user text (§45)."""
    text = (text or "").strip()
    instruction = (instruction or "").strip()
    if not text or not instruction:
        return _envelope("transform", _model_for("transform"), ok=False,
                         error="I need both the text and what to do with it.")
    return await ask_model(
        f"{instruction}\n\n--- BEGIN TEXT ---\n{text}\n--- END TEXT ---",
        kind="transform")


# ═════════════════════════════════════════════════════════════════════════════
# STATUS
# ═════════════════════════════════════════════════════════════════════════════

def router_status() -> dict:
    """Model/availability summary — for logs and adam_smoketest. No secrets."""
    c = _cfg()
    return {
        "provider":     PROVIDER,
        "available":    bool(c["api_key"]),
        "code_model":   c["code_model"],
        "text_model":   c["text_model"],
        "vision_model": c["vision_model"],
        "max_tokens":   c["max_tokens"],
        "timeout_s":    c["timeout_s"],
        "max_retries":  c["max_retries"],
    }


def _selftest() -> None:
    s = router_status()
    print("model_router self-test")
    for k, v in s.items():
        print(f"  {k:14} {v}")
    text = base64.b64encode(b"hello").decode()
    assert text, "base64 sanity"
    print("  scrub:", _scrub("failed: https://x/y?key=AIzaSyABCDEFGHIJKLMNOP "
                             "x-goog-api-key: abc123"))
    print("  envelope keys:", sorted(_envelope("code", "m").keys()))


if __name__ == "__main__":
    _selftest()
