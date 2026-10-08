"""
system_prompt.py — ADAM v41 system-prompt assembler (thin loader)
==============================================================================
build_system_prompt() assembles the full instruction block sent to Gemini at
the start of every session. It is rebuilt fresh on every (re)connect so the
injected date/time, memory, known faces, schedule and recent conversation
window are always current.

v41 CHANGE — THIS FILE NO LONGER CONTAINS ANY PROMPT PROSE.
Every word now lives in `prompts.txt` and is read through prompt_store. This
module's only job is to:

  1. ask prompt_store which sections to assemble, and in what order
  2. render the live data (date/time, memory, faces, schedules, history) that
     the prose has {placeholders} for
  3. skip data sections whose data is empty, so an empty memory does not
     produce a dangling "━━━ YOUR MEMORY ━━━" header with nothing under it

Previously this file held a hardcoded fallback persona plus the date/time,
search-policy and language-policy blocks as Python string literals, while the
persona itself came from SystemPrompt.txt. That split meant the same policy
could be stated twice, differently, with the Python copy quietly winning. There
is now exactly one source of truth and it is editable without touching code.

Reads the live `memory`, `faces`, and `conv_log` objects from memory_store by
reference — because those are mutated in place elsewhere, this always sees the
latest state at call time.
"""

import datetime

from config import CONV_PROMPT_TURNS
from memory_store import memory, faces, conv_log
import prompt_store

# Sections whose body is pure data injection. If the rendered data is empty the
# whole section is dropped rather than emitting a bare header. Maps section
# name -> placeholder name it consumes.
_DATA_SECTIONS = {
    "memory_header":    "memory",
    "faces_header":     "faces",
    "schedules_header": "schedules",
    "history_header":   "history",
}


def _render_memory() -> str:
    if not memory:
        return ""
    return "\n".join(f"  {k}: {v}" for k, v in memory.items())


def _render_faces() -> str:
    if not faces:
        return ""
    return "\n".join(
        f"  [{pid}] {info.get('name', '?')} — {info.get('notes', '')}"
        for pid, info in faces.items()
    )


def _render_schedules() -> str:
    """Upcoming alarms/reminders/timers and open todos, as passive context.

    Imported lazily and guarded: the scheduler is an additive v41 feature and
    the prompt must still build correctly on an install where it is absent or
    failed to start. A missing scheduler drops the section; it never raises
    into prompt building.
    """
    try:
        import scheduler
    except Exception:
        return ""
    try:
        return scheduler.render_for_prompt()
    except Exception as e:
        print(f"[prompt] scheduler context unavailable: {e}", flush=True)
        return ""


def _render_history() -> str:
    """Recent conversation turns, with generic-AI-disclaimer replies scrubbed.

    A single slip into the "I'm just a language model" voice, once persisted to
    adam_conversations.json, was being replayed verbatim into every subsequent
    session prompt and reinforcing the pattern into completely unrelated later
    conversations. Dropping the ADAM side of such a turn (while keeping the
    user's words, which are still useful context) also cleans up lines already
    on disk from before this guard existed, not just future ones.

    The marker list itself now lives in `===POOL disclaimer_markers===` in
    prompts.txt so it can be extended without a code change.
    """
    if not conv_log:
        return ""
    markers = prompt_store.disclaimer_markers()
    lines = []
    for turn in conv_log[-CONV_PROMPT_TURNS:]:
        ts = turn.get("ts", "")
        u = turn.get("user", "").strip()
        a = turn.get("adam", "").strip()
        if a and any(m in a.lower() for m in markers):
            a = ""   # drop the disclaimer reply, keep the user's turn
        if u:
            lines.append(f"  [{ts}] User: {u}")
        if a:
            lines.append(f"  [{ts}] ADAM: {a}")
    return "\n".join(lines)


def build_system_prompt() -> str:
    """The complete system instruction for a new Gemini Live session."""
    # Pick up any edit to prompts.txt made since the last connect. Once per
    # session build, so the stat() is free in practice.
    prompt_store.reload_if_changed()

    now_dt = datetime.datetime.now()
    subs = {
        "date_time": now_dt.strftime("%A, %d %B %Y, %I:%M %p"),
        "memory":    _render_memory(),
        "faces":     _render_faces(),
        "schedules": _render_schedules(),
        "history":   _render_history(),
    }

    parts = []
    for name in prompt_store.assembly_order():
        placeholder = _DATA_SECTIONS.get(name)
        if placeholder is not None and not subs.get(placeholder):
            continue    # no data — skip the header entirely
        text = prompt_store.section(name, **subs).strip()
        if text:
            parts.append(text)

    if not parts:
        # Cannot happen with a valid prompts.txt (prompt_store falls back to a
        # built-in persona before returning an empty snapshot), but an empty
        # system_instruction would make ADAM a blank generic assistant — the
        # single worst failure mode for this product. Belt and braces.
        print("[prompt] assembled prompt was empty — using emergency persona",
              flush=True)
        return prompt_store._EMERGENCY_PERSONA

    return "\n\n".join(parts)
