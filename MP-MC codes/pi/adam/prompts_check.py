#!/usr/bin/env python3
"""
prompts_check.py — validator for ADAM's central prompt file
==============================================================================
Run this after editing prompts.txt. It is the safety net that makes a
hand-editable prompt file safe to hand-edit:

    python prompts_check.py              # check prompts.txt
    python prompts_check.py other.txt    # check a specific file

Exit code 0 = clean (warnings allowed), 1 = errors found.

WHAT IT CHECKS
--------------
  1. Syntax           — malformed ===SECTION/===POOL directives, by line number
  2. Duplicates       — a repeated section/pool name (the later one silently
                        wins, which is how you lose an edit without noticing)
  3. Required sections— the blocks the assembler and the injection call sites
                        actually ask for
  4. Required pools    — the 20 anti-repetition situations from master prompt §4
  5. Pool size         — every situational pool has at least 5 variants
  6. Pool duplicates   — the same line twice in one pool (quietly weakens the
                        anti-repetition ring)
  7. Placeholders      — unknown {tokens} (typos), and placeholders used in a
                        section that is never supplied with them
  8. Assembly order    — names an existing section, no section left unreferenced
  9. Empty blocks      — a section or pool with no content
 10. Leaked secrets    — an API-key-shaped string in the prompt file, which must
                        never happen (master prompt §0.7)

WHY A SEPARATE SCRIPT
--------------------
prompt_store's runtime behaviour is deliberately forgiving: it keeps the last
known-good snapshot and warns, because a typo must never take the robot off the
air mid-conversation. That forgiveness means a mistake can go unnoticed until
someone reads the logs. This script is the strict counterpart — run it before
deploying and the forgiving path never has to save you.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompt_store import (                                  # noqa: E402
    parse_text, find_placeholders, _parse_order,
    ORDER_SECTION, KNOWN_PLACEHOLDERS, REQUIRED_SECTIONS,
    REQUIRED_POOLS, DATA_POOLS,
)

try:
    from config import PROMPTS_FILE, PROMPT_POOL_MIN_VARIANTS   # noqa: E402
except Exception:
    PROMPTS_FILE = Path(__file__).resolve().parent / "prompts.txt"
    PROMPT_POOL_MIN_VARIANTS = 5

# Which placeholders the assembler actually supplies to which section. A section
# using a placeholder outside its entry here would emit a literal "{foo}" into
# ADAM's instructions.
SUPPLIED = {
    "datetime_header":   {"date_time"},
    "memory_header":     {"memory"},
    "faces_header":      {"faces"},
    "schedules_header":  {"schedules"},
    "history_header":    {"history"},
    "inject_idle_nudge": {"elapsed", "nudge"},
    "inject_alarm_fired": {"label", "kind", "line"},
    "inject_coding_task_done": {"status", "line"},
}

# Placeholders a POOL line may legitimately use — these are filled in by the
# caller at pick() time.
POOL_PLACEHOLDERS = {"label", "kind", "status", "elapsed"}

# Anything shaped like a real credential has no business in a prompt file.
SECRET_RE = re.compile(
    r"(AIza[0-9A-Za-z_\-]{30,}"            # Google API key
    r"|sk-[0-9A-Za-z]{20,}"                # OpenAI-style key
    r"|ghp_[0-9A-Za-z]{30,}"               # GitHub token
    r"|GEMINI_API_KEY\s*=\s*\S+"           # an actual assignment
    r"|[0-9a-f]{32,}\b)"                   # long hex blob (token_hex)
)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(PROMPTS_FILE)

    if not path.exists():
        print(f"FAIL  {path} does not exist")
        return 1

    text = path.read_text(encoding="utf-8")
    sections, pools, errors = parse_text(text)

    problems: list = []    # hard errors -> exit 1
    warnings: list = []    # worth fixing, still usable

    # ── 1 & 2: syntax + duplicates, straight from the parser ────────────────
    problems.extend(errors)

    # ── 10: leaked secrets (checked early; the loudest failure) ─────────────
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.startswith("#!"):
            continue
        m = SECRET_RE.search(line)
        if m:
            problems.append(
                f"line {lineno}: looks like a credential in the prompt file "
                f"({m.group(1)[:6]}…) — secrets must never appear here"
            )

    # ── 3: required sections ────────────────────────────────────────────────
    for name in REQUIRED_SECTIONS:
        if name not in sections:
            problems.append(f"required section '{name}' is missing")

    # ── 9: empty blocks ─────────────────────────────────────────────────────
    for name, body in sections.items():
        if not body.strip():
            problems.append(f"section '{name}' is empty")
    for name, variants in pools.items():
        if not variants:
            problems.append(f"pool '{name}' is empty")

    # ── 4 & 5: required pools, and the ≥5-variant rule ──────────────────────
    for name in REQUIRED_POOLS:
        if name not in pools:
            problems.append(f"required pool '{name}' is missing")
            continue
        n = len(pools[name])
        if n < PROMPT_POOL_MIN_VARIANTS:
            problems.append(
                f"pool '{name}' has {n} variant(s), needs at least "
                f"{PROMPT_POOL_MIN_VARIANTS} — too few and ADAM starts "
                f"repeating itself"
            )

    # ── 6: duplicate lines inside a pool ────────────────────────────────────
    for name, variants in pools.items():
        seen = {}
        for v in variants:
            key = v.strip().lower()
            seen[key] = seen.get(key, 0) + 1
        dupes = [k for k, c in seen.items() if c > 1]
        if dupes:
            warnings.append(
                f"pool '{name}' repeats {len(dupes)} line(s) "
                f"(e.g. {dupes[0][:40]!r}) — duplicates shrink the effective "
                f"pool without looking like it"
            )

    # ── 7: placeholders ─────────────────────────────────────────────────────
    for name, body in sections.items():
        if name == ORDER_SECTION:
            continue
        used = find_placeholders(body)
        unknown = used - KNOWN_PLACEHOLDERS
        for ph in sorted(unknown):
            problems.append(
                f"section '{name}' uses unknown placeholder '{{{ph}}}' — it "
                f"will appear literally in ADAM's instructions"
            )
        allowed = SUPPLIED.get(name, set())
        for ph in sorted((used & KNOWN_PLACEHOLDERS) - allowed):
            problems.append(
                f"section '{name}' uses '{{{ph}}}' but the assembler never "
                f"supplies it there — it will appear literally"
            )
        # The reverse: a section declared to receive data but not using it.
        for ph in sorted(allowed - used):
            warnings.append(
                f"section '{name}' is supplied '{{{ph}}}' but does not use it "
                f"— that data will not reach the model"
            )

    for name, variants in pools.items():
        if name in DATA_POOLS:
            continue
        for i, v in enumerate(variants, start=1):
            for ph in sorted(find_placeholders(v)):
                if ph not in POOL_PLACEHOLDERS:
                    problems.append(
                        f"pool '{name}' variant {i} uses '{{{ph}}}' — pools "
                        f"may only use {sorted(POOL_PLACEHOLDERS)}"
                    )

    # ── 8: assembly order ───────────────────────────────────────────────────
    order = _parse_order(sections)
    if not order:
        problems.append(f"section '{ORDER_SECTION}' lists no sections")
    for name in order:
        if name not in sections:
            problems.append(
                f"'{ORDER_SECTION}' names '{name}', which is not a section "
                f"in this file"
            )
        if name.startswith("inject_"):
            problems.append(
                f"'{ORDER_SECTION}' names '{name}' — inject_* sections are "
                f"sent mid-session, they must not be part of the assembled "
                f"system prompt"
            )
    if len(order) != len(set(order)):
        problems.append(f"'{ORDER_SECTION}' lists a section twice")

    referenced = set(order) | {ORDER_SECTION}
    for name in sections:
        if name in referenced or name.startswith("inject_") or name.startswith("_"):
            continue
        warnings.append(
            f"section '{name}' exists but is not in '{ORDER_SECTION}' — it is "
            f"dead text and ADAM will never see it"
        )

    # ── report ──────────────────────────────────────────────────────────────
    n_var = sum(len(v) for v in pools.values())
    print(f"{path.name}: {len(sections)} sections, {len(pools)} pools, "
          f"{n_var} variants")

    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  ~ {w}")

    if problems:
        print(f"\n{len(problems)} error(s):")
        for p in problems:
            print(f"  ! {p}")
        print("\nFAIL")
        return 1

    print("\nOK" + (f" ({len(warnings)} warning(s))" if warnings else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
