# The prompt system — `prompts.txt` and `prompt_store.py`

This document covers where ADAM's personality, rules and phrasing live, how to
edit them safely, and what happens when an edit goes wrong.

It exists because v40 kept the prompt as Python string literals inside
`adam.py`. That meant a personality change was a code change, the prompt could
not be reviewed on its own, and — the reason v41 was started — ADAM said the
same sentence every single time, because there was nothing to vary.

## 1. The split

| Half | Lives in | Changed by |
|---|---|---|
| Prompt text, rules, phrasing | `prompts.txt` | editing a text file |
| Loading, parsing, substitution, hot reload | `prompt_store.py` | code change |

`prompt_store.py` is 22 KB and changes rarely. `prompts.txt` is the file you
edit. Nothing in this document requires touching Python.

## 2. File format

Three directives. That is the whole grammar.

```
#! A comment. Lines starting with #! are ignored by the parser,
#! which is how this file documents itself.

===SECTION identity===
You are ADAM. You are a small desktop robot. You speak in short
sentences and you never lecture.

===POOL confirmation===
Right, done.
All set.
Got it.
Consider it handled.
That's sorted.
```

- `===SECTION name===` starts a section. The body runs until the next
  directive. `section()` returns it as one string.
- `===POOL name===` starts a pool. **Each line is one complete variant.**
  `pick()` returns one at random.
- `#!` at the start of a line is a comment.

Blank lines separate nothing structurally — they are just whitespace — but they
are how the file stays readable, so keep using them.

### Placeholders

Sections and pool entries may contain `{placeholders}`, filled at call time:

```python
prompt_store.section("memory_recall", key="favourite colour", value="blue")
prompt_store.pick("scheduler_confirmation", label="standup")
```

`find_placeholders(text)` lists what a given text needs. A placeholder with no
value supplied is left as-is rather than being replaced with `None` or an empty
string, so a missing substitution is visible in the prompt rather than silently
swallowing a word.

## 3. The three guarantees, and how each is tested

These are the properties that make a text file safe to change on a running
robot. Every one has a test in `adam_smoketest.py`'s `prompt` group.

### Hot reload with a last-known-good fallback

`reload_if_changed()` re-parses when the file's mtime changes, so you can edit
`prompts.txt` on the Pi and the next turn uses it — no restart.

**If the new file fails to parse, the old prompt stays live.** A parse that
yields zero sections is treated as a broken edit, the previous prompt is kept,
and a warning is recorded. The smoke test asserts this by feeding the loader a
deliberately empty file and confirming the prompt survives:

```
[prompt] prompts.txt parsed to zero sections — keeping previous prompt
[prompt] loaded prompts.txt: 26 sections, 21 pools (140 variants)
```

This is the single most important behaviour in the system. The alternative —
taking the file at its word — means one stray `===` while editing turns ADAM
mute or, worse, promptless, and you find out by talking to it.

Set `PROMPT_HOT_RELOAD=0` to disable reloading entirely.

### Pools that do not repeat

`pick(name)` draws without replacing the last `PROMPT_POOL_NO_REPEAT` draws
(default 3). Consecutive identical lines therefore cannot come out of the same
pool.

The smoke test draws 300 times per pool and counts collisions; any collision at
all fails the group. A pool of one variant is a collision every time, which is
how the check catches a pool that has been edited down to nothing useful.

### A minimum number of variants

`PROMPT_POOL_MIN_VARIANTS` (default 5, from `PROMPT_POOL_MIN_VARIANTS` in the
environment) is enforced by the smoke test for every pool in
`REQUIRED_POOLS`. The reasoning: a pool of two is not anti-repetition, it is a
coin flip, and the user hears the same line half the time.

Current state: **26 sections, 21 pools, 140 variants.**

## 4. Writing a pool that actually sounds varied

The mechanical requirement is ≥5 lines. The useful requirement is that the
lines differ in *shape*, not just in wording. These do not read as varied:

```
Alarm set.
Alarm is set.
I have set the alarm.
Your alarm is set.
The alarm has been set.
```

They are the same sentence five ways, and the user notices within a week. What
works is varying the register:

```
Right, that's set for 7.
7 am it is.
Done — 7 o'clock.
I'll wake you at 7.
Set. Anything else?
```

Rules of thumb, learned from using the thing:

- **Vary length**, not just vocabulary. A two-word line next to a nine-word line
  reads as a different speaker; a nine-word line next to a ten-word one does not.
- **Let some entries be flat and some be warm.** Uniform warmth is its own kind
  of monotony.
- **Do not put a placeholder in every entry.** If every line says `{label}`, the
  repetition is in the structure.
- **Check it out loud.** These lines are spoken, not read. Punctuation that
  reads fine can produce odd prosody — a dash-heavy line often lands better than
  a comma-heavy one.

## 5. Pools versus sections

The distinction matters when editing:

| | `SECTION` | `POOL` |
|---|---|---|
| Retrieved by | `section(name)` | `pick(name)` |
| Returns | one string | one random variant |
| Use for | rules, persona, behaviour | confirmations, nudges, errors |

**Behaviour rules belong in sections, never in pools.** A rule that is only
present some of the time is not a rule. If you want ADAM to always treat
clipboard text as untrusted data, that goes in a section — see
[`clipboard_and_agent.md`](clipboard_and_agent.md) for how that particular rule
is enforced twice, once in the prompt and once in code.

## 6. Placeholders and substitution order

`_substitute()` runs over the assembled text after the section body is chosen.
Substitution is a single pass — a value that itself contains `{braces}` is not
re-substituted, which prevents a memory value containing braces from being
interpreted as a placeholder.

## 7. Assembly

`assembly_order()` returns the section order. `injection(name, **subs)` returns
a section for the *middle* of a conversation — the scheduler uses this to
deliver a reminder into the live session, and `render_for_prompt()` in
`scheduler.py` builds what goes there.

Dynamic state — memory contents, current time, upcoming alarms — is injected at
assembly time, not written into `prompts.txt`. The file holds only text that is
true regardless of what the robot currently knows. Nothing that changes
hourly belongs in a file you edit by hand.

## 8. `SystemPrompt.txt` is retired

It is kept only as a legacy fallback (`SYSTEM_PROMPT_FILE`) for a Pi that might
be mid-update across the change, and it is no longer the active source. The
smoke test asserts that the active source is `prompts.txt`:

```
c.ok("prompts.txt is the active source", st["source"] == PROMPTS_FILE.name)
```

Its header records the four dead tool references it used to contain and what
replaced each. This mattered: a prompt that names tools that do not exist is
worse than no prompt, because the model confidently calls them and gets
`unknown tool`.

| Retired reference | Replaced by |
|---|---|
| `save_story` | `save_memory()` |
| `save_person_photo()` | `remember_person()` |
| `generate_to_clipboard()` | `generate_text()` / `generate_code()` |
| `move_neck()` | `move_head_gesture()` |

## 9. Editing checklist

1. Edit `prompts.txt`.
2. Keep `===SECTION===` / `===POOL===` lines exactly on their own line.
3. Any pool you touch: ≥5 variants, genuinely different shapes.
4. Run the prompt group:

```bash
cd ~/adam && ./venv/bin/python adam_smoketest.py prompt
```

5. If it reports a pool below minimum or a collision, fix the file — do not
   lower `PROMPT_POOL_MIN_VARIANTS` to make it pass.
6. No restart needed; the next turn picks it up. `prompt_store.status()` reports
   what is loaded, which sections parsed, and any warnings.

## 10. Debugging

| Symptom | Cause | Check |
|---|---|---|
| Old prompt still used | Broken parse; fallback engaged | `status()["warnings"]`; look for "keeping previous prompt" |
| Same line twice | Pool under `PROMPT_POOL_NO_REPEAT`, or only one variant parsed | `pool(name)` and check its length |
| Placeholder visible in speech | No value passed | `find_placeholders()` on the section; check the call site |
| Section missing entirely | Typo in the directive, or a stray `===` above it | `status()["sections"]` |
| Edits not picked up | `PROMPT_HOT_RELOAD=0`, or mtime unchanged | `reload_if_changed(force=True)`; mtime is the trigger |

## 11. Files

| File | Role |
|---|---|
| `adam/prompts.txt` | the prompt — the file you edit |
| `adam/prompt_store.py` | parse, substitute, hot reload, pools, status |
| `adam/SystemPrompt.txt` | retired; legacy fallback only |
| `adam/config.py` | `PROMPTS_FILE`, `PROMPT_HOT_RELOAD`, `PROMPT_POOL_NO_REPEAT`, `PROMPT_POOL_MIN_VARIANTS` |
| `adam/adam_smoketest.py` | the `prompt` group — 83 assertions |
