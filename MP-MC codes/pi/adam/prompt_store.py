"""
prompt_store.py — ADAM v41 central prompt loader
==============================================================================
Parses `prompts.txt` — the single authoritative source for every word ADAM is
told — and hands pieces of it to the rest of the program.

WHY THIS EXISTS
---------------
Before v41, ADAM's prompt prose lived in four places at once:

  * SystemPrompt.txt            — the persona (218 lines)
  * system_prompt.py            — a hardcoded fallback persona, plus the
                                  date/time, search-policy and language-policy
                                  blocks as Python string literals
  * session.py                  — five inline "[SYSTEM: ...]" injections for
                                  the refusal loop, STOP gesture, cheek slap,
                                  petting, and idle nudge
  * config.py                   — the _NUDGES list

Changing what ADAM says therefore meant editing up to four files and knowing
which one won a conflict. Worse, the canned reaction lines were single strings,
so ADAM said the identical sentence every time it saved a memory or finished a
task — which is exactly the "no creativity, same stuffs, boring" problem this
rewrite was asked to fix.

Now: one file, one parser, one place to edit. Anything in Python that needs
words asks this module for them.

DESIGN NOTES
------------
* FORMAT IS LINE-ORIENTED, NOT YAML. PyYAML is installed on the Pi, so YAML
  was possible — but the content is prose dense with colons, quotes, em-dashes,
  emoji, Devanagari and ━ box-drawing characters. In YAML all of that has to
  live inside indentation-sensitive block scalars, where one stray space
  invalidates the whole document and the error points at a line far from the
  mistake. Here, prose is literal: every line that is not a column-0 directive
  is content, verbatim, with no escaping and no indentation rules. A bad edit
  can only break the one directive line it touched, and prompts_check.py names
  that line number.

* PLACEHOLDERS ARE SUBSTITUTED BY EXPLICIT .replace(), NOT str.format(). The
  prose legitimately contains braces (code examples, "{}" in generated text
  discussions). str.format() would raise KeyError/ValueError on those and take
  down prompt building. Only declared placeholders that were actually supplied
  are replaced; everything else is left exactly as written.

* FAILURE IS NEVER FATAL. A parse that produces no sections leaves the last
  known-good snapshot in place and prints a warning. If the very first parse
  fails (file missing or empty on a fresh install), the loader falls back to
  SystemPrompt.txt and then to a minimal built-in persona, so ADAM still boots
  and still talks. A typo in prompts.txt must never be able to brick the robot.

* ANTI-REPETITION IS STRUCTURAL, NOT A PROMPT REQUEST. pick() remembers the
  last few lines returned per pool and will not return them again until the
  ring rotates, so "never the same line twice in a row" is guaranteed by code
  rather than hoped for from the model.
"""

import os
import random
import re
import sys
import threading

from config import (
    PROMPTS_FILE, SYSTEM_PROMPT_FILE, PROMPT_HOT_RELOAD,
    PROMPT_POOL_NO_REPEAT,
)

# ═════════════════════════════════════════════════════════════════════════════
# FORMAT
# ═════════════════════════════════════════════════════════════════════════════

# ===SECTION name===  /  ===POOL name===   at column 0, lowercase/digits/_ name.
_DIRECTIVE_RE = re.compile(r"^===(SECTION|POOL)\s+([A-Za-z_][A-Za-z0-9_]*)\s*===\s*$")

# Comment lines: "#!" at column 0 only. A "#" anywhere else is literal prose
# (the persona text uses it), and "#!" indented is literal too — this keeps the
# comment marker unambiguous and impossible to trip over accidentally.
_COMMENT_PREFIX = "#!"

# The special section naming the assembly order. Not prose; one section name
# per line. Listed here so prompts_check.py and the loader agree.
ORDER_SECTION = "_assembly_order"

# Every placeholder the loader knows how to substitute. A placeholder in the
# file that is not in this set is a typo — prompts_check.py flags it, and at
# runtime it is simply left as literal text (visible in the prompt, which makes
# the mistake obvious rather than silent).
KNOWN_PLACEHOLDERS = frozenset({
    "date_time", "memory", "faces", "history", "schedules",
    "elapsed", "nudge", "label", "kind", "line", "status",
})

# Sections that must exist for the assembled prompt to be coherent. Missing
# any of these is an error from prompts_check.py, not a warning.
REQUIRED_SECTIONS = (
    ORDER_SECTION,
    "persona",
    "datetime_header",
    "search_policy",
    "language_policy",
    "anti_repetition",
    "banned_phrases",
    "clipboard_safety",
    "laptop_control_safety",
    "memory_header",
    "faces_header",
    "history_header",
    "inject_refusal_correction",
    "inject_stop_gesture",
    "inject_cheek_slap",
    "inject_petting",
    "inject_idle_nudge",
)

# Pools that must exist, each with at least PROMPT_POOL_MIN_VARIANTS entries.
# These are the 20 situations master prompt §4 enumerates, plus the
# disclaimer-scrub list (which is data, and exempt from the minimum).
REQUIRED_POOLS = (
    "task_completed", "reminder_fired", "alarm_fired", "timer_completed",
    "greeting", "idle_nudge", "gesture_reaction", "cant_hear_you",
    "confirmation", "clipboard_read", "clipboard_written",
    "generated_code_ready", "generated_text_ready",
    "coding_task_dispatched", "coding_task_completed",
    "coding_task_cancelled", "error_failure", "scheduler_confirmation",
    "todo_confirmation", "camera_response",
)

# Pools holding data rather than spoken variants — exempt from the ≥5 rule.
DATA_POOLS = ("disclaimer_markers",)

# Last-resort persona if prompts.txt AND SystemPrompt.txt are both unusable.
# Deliberately short: this is a "the robot still boots and still sounds like
# itself" safety net, not a second copy of the persona to keep in sync.
_EMERGENCY_PERSONA = (
    "You are ADAM (Autonomous Desktop AI Module), a witty physical desk robot "
    "built by DGEN Technologies, Kolkata. You are hardware, not software — "
    "never say you are 'just a language model'. Keep replies short and "
    "conversational, and always reply in the same language the user just "
    "spoke. Your central prompt file could not be read, so you are running on "
    "a minimal fallback personality."
)


# ═════════════════════════════════════════════════════════════════════════════
# PARSER  (pure function — prompts_check.py imports and reuses it)
# ═════════════════════════════════════════════════════════════════════════════

def parse_text(text: str):
    """Parse prompts.txt content.

    Returns (sections, pools, errors):
      sections : dict[str, str]        section name -> prose, trailing ws stripped
      pools    : dict[str, list[str]]  pool name    -> variant lines
      errors   : list[str]             human-readable "line N: ..." problems

    Never raises. An unparseable file yields whatever was parseable plus a list
    of errors, so a partial file still produces a usable prompt and a precise
    complaint.
    """
    sections: dict = {}
    pools: dict = {}
    errors: list = []

    cur_kind = None
    cur_name = None
    buf: list = []

    def flush():
        if cur_name is None:
            return
        if cur_kind == "SECTION":
            sections[cur_name] = "\n".join(buf).strip("\n").rstrip()
        else:
            variants = []
            for raw in buf:
                v = raw.strip()
                if not v:
                    continue
                # Allow an optional "- " bullet so a pool can be written as a
                # list without the dash becoming part of the spoken line.
                if v.startswith("- "):
                    v = v[2:].strip()
                if v:
                    variants.append(v)
            pools[cur_name] = variants

    for lineno, raw in enumerate(text.splitlines(), start=1):
        if raw.startswith(_COMMENT_PREFIX):
            continue

        m = _DIRECTIVE_RE.match(raw)
        if m:
            flush()
            cur_kind, cur_name = m.group(1), m.group(2)
            if cur_kind == "SECTION" and cur_name in sections:
                errors.append(f"line {lineno}: duplicate SECTION '{cur_name}' "
                              f"— the later one silently wins; rename or merge")
            if cur_kind == "POOL" and cur_name in pools:
                errors.append(f"line {lineno}: duplicate POOL '{cur_name}' "
                              f"— the later one silently wins; rename or merge")
            buf = []
            continue

        # A line that LOOKS like a directive but did not match the pattern is
        # almost always a typo ("===SECTION persona==" / "=== SECTION x ===" /
        # a capital letter or space in the name). Silently treating it as prose
        # would dump "===SECTION persona==" into ADAM's actual instructions, so
        # call it out by line number.
        if raw.lstrip().startswith("===") and raw.strip().endswith("=="):
            errors.append(
                f"line {lineno}: malformed directive {raw.strip()!r} — expected "
                f"exactly ===SECTION name=== or ===POOL name=== at column 0, "
                f"name using letters/digits/underscore only"
            )
            continue

        if cur_name is None:
            # Content before the first directive. Blank lines are fine (the
            # header comment block ends with them); real text is a mistake.
            if raw.strip():
                errors.append(f"line {lineno}: text before the first "
                              f"===SECTION/===POOL directive is ignored")
            continue

        buf.append(raw)

    flush()
    return sections, pools, errors


def _parse_order(sections: dict) -> list:
    """Section names, in order, from the _assembly_order section."""
    raw = sections.get(ORDER_SECTION, "")
    order = []
    for line in raw.splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        order.append(name)
    return order


def find_placeholders(text: str) -> set:
    """Every {placeholder} token in a block of text."""
    return set(re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", text))


# ═════════════════════════════════════════════════════════════════════════════
# LIVE STATE  (last-known-good snapshot + hot reload)
# ═════════════════════════════════════════════════════════════════════════════

_lock = threading.Lock()

_sections: dict = {}
_pools: dict = {}
_order: list = []
_errors: list = []
_stamp = None           # (mtime, size) of the file the snapshot came from
_loaded_from = None     # "prompts.txt" | "SystemPrompt.txt" | "builtin"
_warned_errors = None   # so a persistent error is printed once, not per build

# Per-pool ring of recently returned variants, for anti-repetition.
_recent: dict = {}


def _file_stamp(path):
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _install_legacy_fallback():
    """Populate a minimal snapshot from SystemPrompt.txt (or the built-in).

    Only reached when prompts.txt is missing or produced zero sections — i.e.
    an install that has not been updated, or a catastrophic edit on a machine
    with no previous good parse to fall back to. ADAM boots either way.

    SystemPrompt.txt was retired in v41 and its shipped content is now a
    pointer-to-prompts.txt notice, which would make a terrible persona. The
    RETIRED marker on its first line is how we tell "someone's real custom
    prompt" from "our own tombstone" — if the file still holds a real prompt
    (an install that hasn't been updated, or a user who put their own text
    there) it is used; if it is the tombstone, we skip straight to the
    built-in emergency persona.
    """
    global _sections, _pools, _order, _loaded_from
    persona = ""
    if SYSTEM_PROMPT_FILE.exists():
        try:
            text = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
            if not text.startswith("RETIRED"):
                persona = text
        except Exception:
            persona = ""
    if persona:
        _loaded_from = SYSTEM_PROMPT_FILE.name
    else:
        persona = _EMERGENCY_PERSONA
        _loaded_from = "builtin"
    _sections = {"persona": persona}
    _pools = {}
    _order = ["persona"]


def reload_if_changed(force: bool = False) -> bool:
    """Re-parse prompts.txt if it changed on disk. Returns True if reloaded.

    Called once per prompt build (i.e. per reconnect), not per turn, so the
    stat() cost is irrelevant. Keeps the previous good snapshot on failure.
    """
    global _sections, _pools, _order, _errors, _stamp, _loaded_from, _warned_errors

    stamp = _file_stamp(PROMPTS_FILE)

    with _lock:
        have_snapshot = bool(_sections)
        if not force and have_snapshot:
            if not PROMPT_HOT_RELOAD:
                return False
            if stamp == _stamp:
                return False

        if stamp is None:
            # File absent. Only fall back if we have nothing at all; an
            # existing good snapshot outlives a file that briefly vanishes
            # (e.g. mid-rsync during deployment).
            if not have_snapshot:
                print(f"[prompt] {PROMPTS_FILE.name} not found — "
                      f"falling back to legacy prompt source", flush=True)
                _install_legacy_fallback()
                _stamp = None
                return True
            return False

        try:
            text = PROMPTS_FILE.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[prompt] cannot read {PROMPTS_FILE.name}: {e} — "
                  f"keeping previous prompt", flush=True)
            if not have_snapshot:
                _install_legacy_fallback()
                return True
            return False

        sections, pools, errors = parse_text(text)

        if not sections:
            print(f"[prompt] {PROMPTS_FILE.name} parsed to zero sections — "
                  f"keeping previous prompt", flush=True)
            if not have_snapshot:
                _install_legacy_fallback()
                return True
            return False

        order = _parse_order(sections)
        if not order:
            # No explicit order: fall back to every non-underscore,
            # non-injection section in file order. Keeps a hand-trimmed file
            # working rather than producing an empty prompt.
            order = [n for n in sections
                     if not n.startswith("_") and not n.startswith("inject_")]
            errors.append(f"section '{ORDER_SECTION}' missing or empty — "
                          f"using file order for all non-inject sections")

        _sections, _pools, _order, _errors = sections, pools, order, errors
        _stamp = stamp
        _loaded_from = PROMPTS_FILE.name
        _recent.clear()   # pools may have changed shape; start the rings over

        n_var = sum(len(v) for v in pools.values())
        print(f"[prompt] loaded {PROMPTS_FILE.name}: {len(sections)} sections, "
              f"{len(pools)} pools ({n_var} variants)"
              + (f", {len(errors)} warning(s)" if errors else ""), flush=True)

        # Print the actual warnings once per distinct error set, so a standing
        # problem is visible at boot without spamming every reconnect.
        sig = tuple(errors)
        if errors and sig != _warned_errors:
            for e in errors:
                print(f"[prompt]   warning: {e}", flush=True)
            print(f"[prompt]   run: python prompts_check.py   for full validation",
                  flush=True)
        _warned_errors = sig
        return True


def _ensure_loaded():
    if not _sections:
        reload_if_changed(force=True)


# ═════════════════════════════════════════════════════════════════════════════
# SUBSTITUTION
# ═════════════════════════════════════════════════════════════════════════════

def _substitute(text: str, subs: dict) -> str:
    """Replace only the declared placeholders that were actually supplied.

    Explicit .replace() rather than str.format(): the prose contains literal
    braces (code examples), which str.format() would treat as fields and raise
    on. Anything not supplied is left as literal text — visible in the prompt,
    which makes a typo obvious instead of silent.
    """
    if not subs:
        return text
    for key, val in subs.items():
        token = "{" + key + "}"
        if token in text:
            text = text.replace(token, "" if val is None else str(val))
    return text


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═════════════════════════════════════════════════════════════════════════════

def section(name: str, default: str = "", **subs) -> str:
    """One section's prose, with placeholders substituted. "" if absent."""
    _ensure_loaded()
    raw = _sections.get(name)
    if raw is None:
        return default
    return _substitute(raw, subs)


def injection(name: str, **subs) -> str:
    """A mid-session "[SYSTEM: ...]" injection string.

    Same storage as a section; separate function because the call sites read
    better and because a missing injection is worth a one-line complaint — an
    empty inject() would silently turn a feature off.
    """
    _ensure_loaded()
    raw = _sections.get(name)
    if raw is None:
        print(f"[prompt] missing injection section '{name}' in "
              f"{PROMPTS_FILE.name} — nothing will be sent", flush=True)
        return ""
    return _substitute(raw, subs).strip()


def pool(name: str) -> list:
    """All variants of a pool, as stored. [] if absent."""
    _ensure_loaded()
    return list(_pools.get(name, ()))


def pick(name: str, **subs) -> str:
    """One variant from a pool, never the same one twice in a row.

    Keeps a short ring of recently-returned lines per pool and excludes them
    from the next draw. With a pool of N variants the ring is capped at N-1 so
    there is always at least one legal candidate — a 2-line pool simply
    alternates, which is still strictly better than the single hardcoded string
    this replaced.
    """
    _ensure_loaded()
    variants = _pools.get(name)
    if not variants:
        return ""
    if len(variants) == 1:
        return _substitute(variants[0], subs)

    ring = _recent.setdefault(name, [])
    cap = max(1, min(PROMPT_POOL_NO_REPEAT, len(variants) - 1))
    candidates = [v for v in variants if v not in ring] or list(variants)
    chosen = random.choice(candidates)

    ring.append(chosen)
    del ring[:-cap]
    return _substitute(chosen, subs)


def assembly_order() -> list:
    """Section names to concatenate, in order."""
    _ensure_loaded()
    return list(_order)


def disclaimer_markers() -> tuple:
    """Lowercase substrings that mark a generic-AI-disclaimer reply.

    Used to scrub such replies out of conversation history before they are
    replayed into a new session prompt and reinforce the pattern. Falls back to
    the historical hardcoded tuple if the pool is missing, because an empty
    scrub list would silently re-enable the exact regression this guards.
    """
    _ensure_loaded()
    markers = tuple(m.lower() for m in _pools.get("disclaimer_markers", ()))
    if markers:
        return markers
    return (
        "just a language model", "just an ai", "just a chatbot",
        "i'm an ai", "i am an ai", "as an ai", "i don't have a physical",
        "i do not have a physical", "large language model",
        "can't help with that", "cannot help with that",
    )


def status() -> dict:
    """Loader state, for /status, the PC app and the smoke test."""
    _ensure_loaded()
    return {
        "source": _loaded_from,
        "path": str(PROMPTS_FILE),
        "sections": len(_sections),
        "pools": len(_pools),
        "variants": sum(len(v) for v in _pools.values()),
        "warnings": list(_errors),
        "hot_reload": bool(PROMPT_HOT_RELOAD),
    }


if __name__ == "__main__":
    # `python prompt_store.py` — quick eyeball check without the full
    # validation of prompts_check.py.
    reload_if_changed(force=True)
    st = status()
    print(f"source   : {st['source']}")
    print(f"sections : {st['sections']}")
    print(f"pools    : {st['pools']} ({st['variants']} variants)")
    print(f"order    : {' -> '.join(assembly_order())}")
    if st["warnings"]:
        print("warnings :")
        for w in st["warnings"]:
            print(f"  - {w}")
    sys.exit(0)
