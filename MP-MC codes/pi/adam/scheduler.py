"""
scheduler.py — ADAM v41 local alarms, timers, reminders and todos
==============================================================================
Everything here runs on the Pi, offline, and survives a reboot. The PC app is
a live VIEW of this file, not a second copy of the truth: the laptop syncs
from here and caches the result, so a Wi-Fi drop freezes the UI rather than
letting the two ends disagree about what is scheduled.

WHAT IT STORES
--------------
One JSON file (config.SCHEDULE_FILE, ~/adam/adam_schedules.json):

    {"version": 1,
     "schedules": [ {...}, ... ],
     "todos":     [ {...}, ... ]}

A schedule is one of three kinds:
    alarm     fires at a clock time, optionally repeating on chosen weekdays
    reminder  identical to an alarm in behaviour; differs only in how ADAM
              is told to deliver it (a reminder names a task, an alarm is a
              wake-up) and in the prompt section it fires into
    timer     fires once, N seconds from when it was set

THE TIME MODEL — why this file is careful about clocks
------------------------------------------------------
Two decisions here are the difference between an alarm that works and one
that misbehaves in ways nobody can reproduce.

1. EVERYTHING IS LOCAL WALL-CLOCK TIME. Next-fire times are computed with
   datetime.now(), not from UTC or time.monotonic(). So an alarm set for
   "07:00" rings at 07:00 on the wall clock the user is looking at, including
   the morning after a DST change — which is what a person means by "7 am".
   (The trade-off is that a 24-hour timer spanning a DST shift takes 24h of
   wall time rather than 86400 SI seconds. That is the right trade for a
   bedside alarm.)

2. THE CLOCK CAN JUMP, AND WE NOTICE. The Pi has no RTC, so it boots with a
   wrong clock and corrects itself over NTP — a jump of minutes or hours,
   usually within the first minute of uptime. A naive "now >= due" loop would
   fire every stale schedule at once the moment that correction lands. So the
   ticker compares the wall clock against a monotonic reference and treats a
   disagreement larger than CLOCK_JUMP_S as a resync: it recomputes next-fire
   times from the new clock and fires nothing. Missed-fire handling below then
   decides, on its own, whether anything was genuinely missed.

MISSED FIRES
------------
A schedule whose time passed while ADAM was off is delivered on boot if it
was due within MISSED_GRACE_S, and silently marked as missed if it was due
longer ago than that. The reasoning: a reminder from eleven minutes ago is
still useful, a wake-up alarm from eleven hours ago is not. Either way a
one-shot is never left pending to surprise the user days later, and a
repeating alarm resumes at its next occurrence rather than replaying the ones
it slept through.

DUPLICATE FIRES
---------------
Each schedule records `last_fired` (a local-time string). Firing is
conditional on that not already matching the occurrence being fired, so a
restart between "decide to fire" and "speak it" cannot double-deliver the same
alarm. This is the bug class that makes a robot announce "time's up" twice
twenty seconds apart, which reads as a malfunction.

REPEAT RULE — no double conversion
-----------------------------------
For one-shot schedules `at` is already the resolved next occurrence, and
repeat is None. For repeating schedules `time_of_day` + `repeat.weekdays` are
authoritative and `at` is a cache that this module alone recomputes from them
— never read back as the source of truth. Keeping those two representations
from both claiming authority is why next_due() exists in one place only.

DELIVERY
--------
This module never speaks and never touches the speaker path (§0.3). Firing is
published to the session's existing out_q / injection mechanism via the three
hooks main.py/session.py already expose:
      drain_pending_fires()   -> what fired, once (session.py delivers it)
      alarm_ringing()         -> whether Touch1/2/3 currently mean snooze/dismiss
      snooze_current()/dismiss_current()
So the audio path stays exactly as it was and this stays additive.

Config.Imports only from config, like memory_store.py.
"""

import asyncio
import datetime as dt
import time
import uuid

from config import SCHEDULE_FILE, MISSED_GRACE_S, CLOCK_JUMP_S, TICK_INTERVAL_S
from memory_store import load_json, save_json

__all__ = [
    "set_alarm", "set_timer", "set_reminder", "list_schedules",
    "cancel_schedule", "add_todo", "list_todos", "complete_todo",
    "delete_todo", "render_for_prompt", "start_scheduler", "stop_scheduler",
    "drain_pending_fires", "alarm_ringing", "snooze_current", "dismiss_current",
    "snapshot", "SCHEMA_VERSION", "KINDS", "WEEKDAY_NAMES",
]

SCHEMA_VERSION = 1
KINDS = ("alarm", "reminder", "timer")
WEEKDAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_SNOOZE_SECONDS = 300           # Touch1/Touch2 give five more minutes

# How many times a transient (one-shot) item may be snoozed before ADAM gives
# up ringing it. Without a ceiling a forgotten alarm rings until the battery
# dies, which is worse than a missed one.
_MAX_SNOOZES = 12

# Schedules whose next fire is further out than this are not rendered into the
# prompt. The model does not need next month's birthdays in every session.
_PROMPT_HORIZON_DAYS = 14
_PROMPT_MAX_TODOS = 25

# ═════════════════════════════════════════════════════════════════════════════
# STATE
# ═════════════════════════════════════════════════════════════════════════════

_store = load_json(SCHEDULE_FILE, {"version": SCHEMA_VERSION,
                                   "schedules": [], "todos": []})

# A corrupt or hand-edited file is repaired rather than trusted — a malformed
# entry that survived into the ticker would throw on every tick, and the
# symptom would be "my alarms stopped working" with no obvious cause.
if not isinstance(_store, dict):
    print("⚠️  adam_schedules.json is not an object — starting fresh")
    _store = {"version": SCHEMA_VERSION, "schedules": [], "todos": []}
_store.setdefault("version", SCHEMA_VERSION)
_store.setdefault("schedules", [])
_store.setdefault("todos", [])


def _valid_schedule(s) -> bool:
    if not isinstance(s, dict):
        return False
    if s.get("kind") not in KINDS:
        return False
    if not s.get("id") or not isinstance(s.get("label"), str):
        return False
    # A repeating entry needs time_of_day + weekdays; a one-shot needs `at`.
    if s.get("repeat"):
        return bool(s.get("time_of_day")) and isinstance(s["repeat"].get("weekdays"), list)
    return bool(s.get("at"))


def _valid_todo(t) -> bool:
    return (isinstance(t, dict) and t.get("id")
            and isinstance(t.get("text"), str) and t["text"].strip())


def _sanitise() -> None:
    before = (len(_store["schedules"]), len(_store["todos"]))
    _store["schedules"] = [s for s in _store["schedules"] if _valid_schedule(s)]
    _store["todos"] = [t for t in _store["todos"] if _valid_todo(t)]
    after = (len(_store["schedules"]), len(_store["todos"]))
    if before != after:
        print(f"⚠️  schedules: dropped {before[0]-after[0]} malformed schedule(s) "
              f"and {before[1]-after[1]} malformed todo(s)")


_store.setdefault("tombstones", [])
_sanitise()

# Fires waiting to be delivered to the model. A single-element-list mailbox
# rather than a plain list attribute, matching the idiom in tool_handler.py.
_pending: list = []
_ringing: list = []          # [schedule_id] currently ringing, or []

_sched_task = None
_ticks_seen = [0]

print(f"✅ Scheduler: {len(_store['schedules'])} schedule(s) | "
      f"{len(_store['todos'])} todo(s) loaded")


def _save() -> None:
    save_json(SCHEDULE_FILE, _store)


def _now() -> dt.datetime:
    return dt.datetime.now()


def _iso(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M")


# Sync metadata. `_iso` is minute-resolution because that is what a user means
# by a time; `updated_at` needs SECONDS or two edits in the same minute tie and
# last-write-wins cannot order them.
def _iso_s(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S")


def _touch(entry: dict) -> dict:
    """Stamp a record as modified now. Call on every create and mutation."""
    entry["updated_at"] = _iso_s(_now())
    return entry


# How long a deletion is remembered. A client that has been offline longer than
# this will not learn about the delete and may push its stale copy back, so this
# is the real bound on "how long may a device stay away".
TOMBSTONE_KEEP_DAYS = 30


def _tombstone(kind: str, entry: dict) -> None:
    """Record that something was deliberately deleted.

    Kept in its own list rather than flagged on the row: the live schedules and
    todos lists stay exactly as every existing reader expects them, so there is
    no way for a deleted alarm to survive into the firing path.
    """
    _store.setdefault("tombstones", []).append({
        "id": entry.get("id", ""),
        "kind": kind,                       # "schedule" | "todo"
        "label": str(entry.get("label") or entry.get("text") or "")[:120],
        "deleted_at": _iso_s(_now()),
    })


def _prune_tombstones() -> None:
    rows = _store.get("tombstones") or []
    if not rows:
        return
    cutoff = _now() - dt.timedelta(days=TOMBSTONE_KEEP_DAYS)
    kept = []
    for r in rows:
        try:
            when = dt.datetime.strptime(r.get("deleted_at", ""), "%Y-%m-%dT%H:%M:%S")
        except Exception:
            continue                        # undated tombstone is unusable
        if when >= cutoff:
            kept.append(r)
    if len(kept) != len(rows):
        print(f"  🧹 pruned {len(rows) - len(kept)} tombstone(s) older than "
              f"{TOMBSTONE_KEEP_DAYS} days")
    _store["tombstones"] = kept


# Pruned HERE, not up beside _sanitise(). The store is loaded before this point
# but the function is not: calling it earlier raised NameError at import, which
# py_compile does not catch because it is a runtime ordering fault, not a syntax
# one. Keep this call below the definition.
_prune_tombstones()


def _parse(s: str):
    try:
        return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M")
    except Exception:
        return None


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ═════════════════════════════════════════════════════════════════════════════
# TIME PARSING — the model resolves relative phrases itself and passes a
# concrete value, but these accept the shapes it actually tends to send.
# ═════════════════════════════════════════════════════════════════════════════

def parse_time_of_day(text: str):
    """'07:00' | '7:00' | '07:00:30' | '7am' | '7:30 pm' -> (hour, minute)."""
    if not text:
        return None
    s = str(text).strip().lower().replace(".", "").replace(" ", "")
    ampm = None
    if s.endswith("am") or s.endswith("pm"):
        ampm = s[-2:]
        s = s[:-2]
    if ":" in s:
        parts = s.split(":")
        if len(parts) < 2:
            return None
        try:
            hh, mm = int(parts[0]), int(parts[1].split(":")[0])
        except ValueError:
            return None
    else:
        try:
            hh, mm = int(s), 0
        except ValueError:
            return None
    if ampm == "pm" and hh < 12:
        hh += 12
    elif ampm == "am" and hh == 12:
        hh = 0
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return None
    return hh, mm


def parse_when(text: str):
    """A datetime the user meant, from the shapes the model sends.

    Accepts 'YYYY-MM-DDTHH:MM', 'YYYY-MM-DD HH:MM', 'HH:MM' (today, or
    tomorrow if that time has already passed), and RFC3339 with a trailing Z
    or offset. Returns a naive LOCAL datetime, or None if unparseable.
    """
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None
    # RFC3339 with offset -> convert to local naive.
    try:
        parsed = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            return parsed.astimezone().replace(tzinfo=None)
        return parsed
    except ValueError:
        pass
    hm = parse_time_of_day(s)
    if hm is None:
        return None
    now = _now()
    cand = now.replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)
    if cand <= now:
        cand += dt.timedelta(days=1)
    return cand


def parse_weekdays(value):
    """'weekdays' | 'mon,wed,fri' | ['mon','tue'] -> ['mon','tue',...] or None.

    None means "not a repeat" — a one-shot. An unrecognised token is an error
    (returns False) rather than being quietly dropped, because silently
    turning "every day" into "once, today" is how an alarm goes missing.
    """
    if value in (None, "", [], ()):
        return None
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("none", "once", "one-shot", "oneshot", "no", "false"):
            return None
        if low in ("weekday", "weekdays", "workdays"):
            return ["mon", "tue", "wed", "thu", "fri"]
        if low in ("weekend", "weekends"):
            return ["sat", "sun"]
        if low in ("daily", "everyday", "every day", "all", "*"):
            return list(WEEKDAY_NAMES)
        items = [p.strip() for p in low.replace(";", ",").split(",") if p.strip()]
    elif isinstance(value, (list, tuple)):
        items = [str(p).strip().lower() for p in value if str(p).strip()]
    else:
        return False
    out = []
    for it in items:
        key = it[:3]
        if key not in WEEKDAY_NAMES:
            return False
        if key not in out:
            out.append(key)
    return sorted(out, key=WEEKDAY_NAMES.index) if out else None


# ═════════════════════════════════════════════════════════════════════════════
# NEXT-DUE COMPUTATION — the single place that decides when anything fires.
# ═════════════════════════════════════════════════════════════════════════════

def next_due(s: dict, after: dt.datetime = None):
    """When `s` should next fire, strictly after `after` (default: now).

    Repeating: next matching weekday at time_of_day. One-shot: its `at`.
    Returns None for a repeating entry with no remaining match in the search
    window, or for a one-shot whose time is in the past.
    """
    ref = after or _now()

    # A pending snooze outranks the entry's own schedule — it is the next time
    # this thing will actually make a noise, so it is what the UI and the
    # prompt must show.
    snoozed = _parse(s.get("snooze_until") or "")
    if snoozed is not None and snoozed > ref:
        return snoozed

    if s.get("repeat"):
        hm = parse_time_of_day(s.get("time_of_day", ""))
        days = s["repeat"].get("weekdays") or []
        if hm is None or not days:
            return None
        for offset in range(0, 8):          # today..+7 covers any weekday set
            day = ref + dt.timedelta(days=offset)
            if WEEKDAY_NAMES[day.weekday()] not in days:
                continue
            cand = day.replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)
            if cand > ref:
                return cand
        return None
    at = _parse(s.get("at", ""))
    if at is None:
        return None
    return at if at > ref else None


def _describe(s: dict) -> str:
    """A short human string for logs and the PC app — never the model's voice."""
    kind = s.get("kind", "?")
    if s.get("repeat"):
        days = s["repeat"].get("weekdays") or []
        when = "daily" if len(days) == 7 else ",".join(days)
        return f"{kind} {s.get('time_of_day')} ({when})"
    return f"{kind} at {s.get('at')}"


def _friendly(s: dict) -> str:
    """The time as a person would say it, for the confirmation line."""
    if s.get("repeat"):
        hm = parse_time_of_day(s.get("time_of_day", ""))
        days = s["repeat"].get("weekdays") or []
        hh, mm = (hm if hm else (0, 0))
        h12 = hh % 12 or 12
        suffix = "AM" if hh < 12 else "PM"
        clock = f"{h12}:{mm:02d} {suffix}" if mm else f"{h12} {suffix}"
        if len(days) == 7:
            return f"every day at {clock}"
        if days == ["mon", "tue", "wed", "thu", "fri"]:
            return f"every weekday at {clock}"
        return f"at {clock} on {', '.join(days)}"
    at = _parse(s.get("at", ""))
    if at is None:
        return "at that time"
    h12 = at.hour % 12 or 12
    suffix = "AM" if at.hour < 12 else "PM"
    clock = f"{h12}:{at.minute:02d} {suffix}" if at.minute else f"{h12} {suffix}"
    if at.date() == _now().date():
        return f"today at {clock}"
    if at.date() == (_now() + dt.timedelta(days=1)).date():
        return f"tomorrow at {clock}"
    return f"{at.strftime('%d %b')} at {clock}"


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API — what the tool handlers call. All return the same envelope:
#   {"ok": bool, "reason": str, "data": {...}}
# so tool_handler.py can shape the model's reply without special cases.
# ═════════════════════════════════════════════════════════════════════════════

def _ok(**data):
    return {"ok": True, "reason": "", "data": data}


def _err(reason):
    return {"ok": False, "reason": reason, "data": {}}


def set_alarm(label: str, when: str = "", repeat=None):
    """An alarm at a clock time, optionally repeating on given weekdays."""
    label = (label or "").strip() or "alarm"
    days = parse_weekdays(repeat)
    if days is False:
        return _err("I didn't understand which days — try 'daily' or 'weekdays'.")

    now = _now()
    if days:
        hm = parse_time_of_day(when)
        if hm is None:
            return _err("I need a clock time for a repeating alarm, like 07:00.")
        entry = {
            "id": _new_id("al"), "kind": "alarm", "label": label,
            "time_of_day": f"{hm[0]:02d}:{hm[1]:02d}",
            "repeat": {"weekdays": days},
            "enabled": True, "last_fired": None, "snoozes": 0,
            "created": _iso(now), "updated_at": _iso_s(now),
        }
        at = next_due(entry, after=now)
        entry["at"] = _iso(at) if at else None
        if not at:
            return _err("That repeat pattern has no upcoming occurrence.")
    else:
        at = parse_when(when)
        if at is None:
            return _err("I need a time for that alarm, like '07:00' or a date.")
        if at <= now:
            return _err("That time has already passed — give me one in the future.")
        entry = {
            "id": _new_id("al"), "kind": "alarm", "label": label,
            "at": _iso(at), "repeat": None,
            "enabled": True, "last_fired": None, "snoozes": 0,
            "created": _iso(now), "updated_at": _iso_s(now),
        }

    _store["schedules"].append(entry)
    _save()
    print(f"  ⏰ alarm set: {_describe(entry)} — {label!r}")
    return _ok(id=entry["id"], kind="alarm", label=label,
               spoken_time=_friendly(entry), at=entry.get("at"),
               repeat=(days or None))


def set_reminder(label: str, when: str = "", repeat=None):
    """A thing to be said at a time. Behaves as an alarm; differs in delivery."""
    r = set_alarm(label, when, repeat)
    if r["ok"]:
        for s in _store["schedules"]:
            if s["id"] == r["data"]["id"]:
                s["kind"] = "reminder"
                break
        _save()
        print(f"  📌 reminder set: {r['data']['spoken_time']} — {label!r}")
        r["data"]["kind"] = "reminder"
    return r


def set_timer(seconds=None, label: str = "", minutes=None, hours=None):
    """A countdown. Fires once, N seconds from now."""
    total = 0.0
    if seconds is not None:
        try:
            total += float(seconds)
        except (TypeError, ValueError):
            return _err("That timer length wasn't a number.")
    if minutes is not None:
        try:
            total += float(minutes) * 60
        except (TypeError, ValueError):
            return _err("That timer length wasn't a number.")
    if hours is not None:
        try:
            total += float(hours) * 3600
        except (TypeError, ValueError):
            return _err("That timer length wasn't a number.")
    if total <= 0:
        return _err("How long should the timer run for?")
    if total > 60 * 60 * 24 * 7:
        return _err("That's longer than a week — set an alarm instead?")

    now = _now()
    entry = {
        "id": _new_id("ti"), "kind": "timer",
        "label": (label or "").strip() or f"{int(round(total/60)) or 1} minute timer",
        "at": _iso(now + dt.timedelta(seconds=total)),
        "repeat": None, "enabled": True, "last_fired": None, "snoozes": 0,
        "created": _iso(now), "updated_at": _iso_s(now), "duration_s": int(total),
    }
    _store["schedules"].append(entry)
    _save()
    mins, secs = divmod(int(total), 60)
    hh, mins = divmod(mins, 60)
    parts = ([f"{hh} hour" + ("s" if hh != 1 else "")] if hh else []) + \
            ([f"{mins} minute" + ("s" if mins != 1 else "")] if mins else []) + \
            ([f"{secs} second" + ("s" if secs != 1 else "")] if secs and not hh else [])
    spoken = " ".join(parts) or f"{int(total)} seconds"
    print(f"  ⏳ timer set: {spoken} — {entry['label']!r}")
    return _ok(id=entry["id"], kind="timer", label=entry["label"],
               spoken_time=f"in {spoken}", at=entry["at"])


def list_schedules(include_past: bool = False):
    """Upcoming alarms/reminders/timers, soonest first."""
    now = _now()
    rows = []
    for s in _store["schedules"]:
        if not s.get("enabled", True):
            continue
        due = next_due(s, after=now)
        if due is None:
            if not include_past:
                continue
        rows.append({
            "id": s["id"], "kind": s.get("kind"),
            "label": s.get("label", ""),
            "next": _iso(due) if due else None,
            "spoken_time": _friendly(s) if due else "already passed",
            "repeats": bool(s.get("repeat")),
            "weekdays": (s.get("repeat") or {}).get("weekdays") or [],
        })
    rows.sort(key=lambda r: r["next"] or "9999")
    return _ok(count=len(rows), schedules=rows)


def cancel_schedule(target: str = ""):
    """Cancel one schedule by id, or by (partial) label. Ambiguity is refused."""
    target = (target or "").strip()
    if not target:
        return _err("Which one should I cancel?")

    match = [s for s in _store["schedules"] if s["id"] == target]
    if not match:
        low = target.lower()
        match = [s for s in _store["schedules"]
                 if low in str(s.get("label", "")).lower()]
    if not match:
        return _err(f"I couldn't find a schedule matching '{target}'.")
    if len(match) > 1:
        names = "; ".join(f"{s.get('label')} ({_friendly(s)})" for s in match[:4])
        return _err(f"That matches more than one — which: {names}?")
    dead = match[0]
    _store["schedules"] = [s for s in _store["schedules"] if s["id"] != dead["id"]]
    _tombstone("schedule", dead)
    if _ringing and _ringing[0] == dead["id"]:
        _ringing.clear()
    _save()
    print(f"  🗑️  cancelled: {_describe(dead)} — {dead.get('label')!r}")
    return _ok(id=dead["id"], label=dead.get("label", ""), kind=dead.get("kind"))


def add_todo(text: str, due: str = ""):
    text = (text or "").strip()
    if not text:
        return _err("What should I add to the list?")
    if len(text) > 500:
        text = text[:500]
    due_iso = None
    if due:
        parsed = parse_when(due)
        if parsed is None:
            return _err("I didn't understand that due date.")
        due_iso = _iso(parsed)
    entry = {"id": _new_id("td"), "text": text, "done": False,
             "created": _iso(_now()), "updated_at": _iso_s(_now()),
             "due": due_iso, "done_at": None}
    _store["todos"].append(entry)
    _save()
    print(f"  📝 todo added: {text!r}")
    return _ok(id=entry["id"], text=text, due=due_iso,
               open_count=sum(1 for t in _store["todos"] if not t["done"]))


def list_todos(include_done: bool = False):
    rows = [t for t in _store["todos"] if include_done or not t.get("done")]
    rows.sort(key=lambda t: (t.get("done", False), t.get("created", "")))
    return _ok(count=len(rows), todos=[
        {"id": t["id"], "text": t["text"], "done": bool(t.get("done")),
         "due": t.get("due"), "created": t.get("created")} for t in rows])


def _find_todo(target: str):
    target = (target or "").strip()
    if not target:
        return None, "Which todo?"
    exact = [t for t in _store["todos"] if t["id"] == target]
    if exact:
        return exact[0], ""
    low = target.lower()
    part = [t for t in _store["todos"] if low in str(t.get("text", "")).lower()]
    if not part:
        return None, f"I couldn't find a todo matching '{target}'."
    if len(part) > 1:
        names = "; ".join(t["text"][:40] for t in part[:4])
        return None, f"That matches more than one todo — which: {names}?"
    return part[0], ""


def complete_todo(target: str = "", text: str = ""):
    todo, err = _find_todo(target or text)
    if err:
        return _err(err)
    todo["done"] = True
    todo["done_at"] = _iso(_now())
    _touch(todo)
    _save()
    print(f"  ✅ todo done: {todo['text']!r}")
    return _ok(id=todo["id"], text=todo["text"],
               open_count=sum(1 for t in _store["todos"] if not t["done"]))


def delete_todo(target: str = "", text: str = ""):
    todo, err = _find_todo(target or text)
    if err:
        return _err(err)
    _store["todos"] = [t for t in _store["todos"] if t["id"] != todo["id"]]
    _tombstone("todo", todo)
    _save()
    print(f"  🗑️  todo deleted: {todo['text']!r}")
    return _ok(id=todo["id"], text=todo["text"])


# ── Called by the PC app's sync API (sync_api.py), which is the only writer
# from off-device. Kept here so the file format has exactly one owner. ──────

def replace_all(schedules=None, todos=None):
    """Replace the store with what the PC app sends.

    The Pi stays authoritative in the sense that it owns the format and the
    timestamps, but the user's phone/laptop UI must be able to edit the list
    too, so this accepts a full replacement. Every incoming entry is
    re-validated through the same checks as a locally-created one, and the
    next-fire cache is recomputed rather than trusted — otherwise a client
    could hand us an `at` that never fires.
    """
    if schedules is not None:
        if not isinstance(schedules, list):
            return _err("schedules must be a list")
        clean = []
        for s in schedules:
            if not isinstance(s, dict):
                continue
            if not s.get("id"):
                s["id"] = _new_id(s.get("kind", "sc")[:2] or "sc")
            s.setdefault("kind", "alarm")
            s.setdefault("enabled", True)
            s.setdefault("snoozes", 0)
            s.setdefault("last_fired", None)
            s.setdefault("created", _iso(_now()))
            s.setdefault("updated_at", _iso_s(_now()))
            if s.get("repeat"):
                days = parse_weekdays(s["repeat"].get("weekdays"))
                if days in (None, False):
                    continue
                s["repeat"] = {"weekdays": days}
            if not _valid_schedule(s):
                continue
            due = next_due(s)
            if s.get("repeat"):
                s["at"] = _iso(due) if due else None
            clean.append(s)
        _store["schedules"] = clean

    if todos is not None:
        if not isinstance(todos, list):
            return _err("todos must be a list")
        clean = []
        for t in todos:
            if not isinstance(t, dict):
                continue
            if not t.get("id"):
                t["id"] = _new_id("td")
            t.setdefault("done", False)
            t.setdefault("created", _iso(_now()))
            t.setdefault("updated_at", _iso_s(_now()))
            t.setdefault("done_at", None)
            t.setdefault("due", None)
            if _valid_todo(t):
                clean.append(t)
        _store["todos"] = clean

    _save()
    print(f"  🔄 store replaced from PC app: "
          f"{len(_store['schedules'])} schedule(s), {len(_store['todos'])} todo(s)")
    return _ok(schedules=len(_store["schedules"]), todos=len(_store["todos"]))


def snapshot() -> dict:
    """Everything the PC app needs, in one object, for its local cache."""
    now = _now()
    scheds = []
    for s in _store["schedules"]:
        due = next_due(s, after=now)
        scheds.append({
            "id": s["id"], "kind": s.get("kind"), "label": s.get("label", ""),
            "at": _iso(due) if due else s.get("at"),
            "time_of_day": s.get("time_of_day"),
            "repeat": s.get("repeat"),
            "enabled": bool(s.get("enabled", True)),
            "last_fired": s.get("last_fired"),
            "spoken_time": _friendly(s) if due else "already passed",
            "overdue": due is None,
        })
    scheds.sort(key=lambda r: r["at"] or "9999")
    return {
        "version": SCHEMA_VERSION,
        "generated": _iso(now),
        "now": now.strftime("%Y-%m-%dT%H:%M"),
        "weekday": WEEKDAY_NAMES[now.weekday()],
        "schedules": scheds,
        "todos": [{"id": t["id"], "text": t["text"], "done": bool(t.get("done")),
                   "due": t.get("due"), "created": t.get("created"),
                   "done_at": t.get("done_at")} for t in _store["todos"]],
    }


# ═════════════════════════════════════════════════════════════════════════════
# FIRING
# ═════════════════════════════════════════════════════════════════════════════

def _fire(s: dict, occurrence: dt.datetime, late_by_s: float = 0.0,
          dedupe_stamp: str = None) -> None:
    """Mark `s` as fired and queue the announcement.

    `occurrence` is the time being announced. `dedupe_stamp` is what gets
    written to last_fired to stop a re-fire, and is only passed separately for
    a SNOOZE fire on a repeating alarm: there, last_fired must keep pointing
    at the original 07:00 occurrence. If the 07:05 snooze overwrote it, the
    repeating branch would compare today's 07:00 stamp against "07:05",
    see a mismatch, and ring the same alarm a third time.
    """
    stamp = dedupe_stamp if dedupe_stamp is not None else \
        f"{occurrence.strftime('%Y-%m-%dT%H:%M')}#{s['id']}"
    s["last_fired"] = stamp
    if dedupe_stamp is None:
        # A fresh occurrence resets the snooze budget. A snooze fire must not,
        # or _MAX_SNOOZES never trips and a forgotten alarm rings forever.
        s["snoozes"] = 0

    if s.get("repeat"):
        nxt = next_due(s, after=occurrence + dt.timedelta(seconds=1))
        s["at"] = _iso(nxt) if nxt else None
    else:
        # A one-shot has no next occurrence. Disabling rather than deleting
        # keeps it visible in the UI afterwards, so an alarm that already
        # rang is still recognisable rather than mysteriously gone.
        s["enabled"] = False

    if s.get("kind") == "alarm":
        _ringing[:] = [s["id"]]

    _pending.append({
        "id": s["id"], "kind": s.get("kind"), "label": s.get("label", ""),
        "at": occurrence.strftime("%Y-%m-%dT%H:%M"),
        "late_by_minutes": int(late_by_s // 60),
    })
    late = f" (late by {int(late_by_s//60)} min)" if late_by_s >= 60 else ""
    print(f"  🔔 {s.get('kind')} fired{late}: {s.get('label')!r}")


def _check(now: dt.datetime, monotonic_elapsed: float,
           wall_elapsed: float) -> bool:
    """One pass over the schedules. Returns True if a clock jump was seen."""
    jumped = abs(wall_elapsed - monotonic_elapsed) > CLOCK_JUMP_S
    if jumped:
        print(f"  ⚠️  clock jumped {wall_elapsed - monotonic_elapsed:+.0f}s "
              f"(wall vs monotonic) — resyncing schedules")
        for s in _store["schedules"]:
            if s.get("repeat"):
                due = next_due(s, after=now)
                s["at"] = _iso(due) if due else None
        _save()
        return True

    changed = False
    for s in _store["schedules"]:
        if not s.get("enabled", True):
            continue

        # A snooze is a one-shot override that outranks the entry's own
        # schedule. It is checked FIRST and for every kind, because a snoozed
        # repeating alarm must ring again in five minutes without its
        # time_of_day being rewritten — if the snooze lived in `at` instead,
        # tomorrow's 07:00 alarm would permanently become 07:05.
        snooze_at = _parse(s.get("snooze_until") or "")
        if snooze_at is not None:
            if now < snooze_at:
                continue
            late = (now - snooze_at).total_seconds()
            s.pop("snooze_until", None)
            if late > MISSED_GRACE_S:
                # Snoozed, then ADAM was off for hours. Don't ring now.
                changed = True
                if not s.get("repeat"):
                    s["enabled"] = False
                continue
            _fire(s, snooze_at, late_by_s=late,
                  dedupe_stamp=(s.get("last_fired") if s.get("repeat") else None))
            changed = True
            continue

        if s.get("repeat"):
            hm = parse_time_of_day(s.get("time_of_day", ""))
            days = (s.get("repeat") or {}).get("weekdays") or []
            if hm is None or WEEKDAY_NAMES[now.weekday()] not in days:
                continue
            occurrence = now.replace(hour=hm[0], minute=hm[1],
                                     second=0, microsecond=0)
            if now < occurrence:
                continue
            late = (now - occurrence).total_seconds()
            if late > MISSED_GRACE_S:
                # Slept through it entirely. Advance the cache so it does not
                # fire on the next tick, and say nothing.
                nxt = next_due(s, after=now)
                s["at"] = _iso(nxt) if nxt else None
                changed = True
                continue
            stamp = f"{occurrence.strftime('%Y-%m-%dT%H:%M')}#{s['id']}"
            if s.get("last_fired") == stamp:
                continue
            _fire(s, occurrence, late_by_s=late)
            changed = True
            continue

        at = _parse(s.get("at", ""))
        if at is None or now < at:
            continue
        late = (now - at).total_seconds()
        stamp = f"{at.strftime('%Y-%m-%dT%H:%M')}#{s['id']}"
        if s.get("last_fired") == stamp:
            continue
        if late > MISSED_GRACE_S:
            # Too old to be useful. Retire it silently so a week-old alarm
            # does not greet the user on the next boot.
            s["enabled"] = False
            s["last_fired"] = stamp
            changed = True
            print(f"  …  skipped stale {s.get('kind')}: {s.get('label')!r} "
                  f"(due {int(late//60)} min ago)")
            continue
        _fire(s, at, late_by_s=late)
        changed = True

    if changed:
        _save()
    return False


async def _ticker() -> None:
    """The only thing that decides anything is due. One task for the process."""
    last_wall = _now()
    last_mono = time.monotonic()
    while True:
        try:
            await asyncio.sleep(TICK_INTERVAL_S)
            now = _now()
            mono = time.monotonic()
            _check(now, mono - last_mono, (now - last_wall).total_seconds())
            last_wall, last_mono = now, mono
            _ticks_seen[0] += 1
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # One bad entry must never kill the ticker; if it did, the symptom
            # would be "none of my alarms work any more".
            print(f"⚠️  scheduler tick error: {e}")
            await asyncio.sleep(1)


async def start_scheduler():
    """Start the ticker. Call ONCE, from main.py, outside the reconnect loop —
    a reconnect must not start a second ticker (§0.4)."""
    global _sched_task
    if _sched_task and not _sched_task.done():
        return _sched_task
    _sched_task = asyncio.create_task(_ticker())
    print(f"✅ Scheduler started (tick {TICK_INTERVAL_S}s, "
          f"missed-fire grace {int(MISSED_GRACE_S)}s)")
    return _sched_task


async def stop_scheduler() -> None:
    global _sched_task
    if _sched_task and not _sched_task.done():
        _sched_task.cancel()
        try:
            await _sched_task
        except (asyncio.CancelledError, Exception):
            pass
    _sched_task = None


# ═════════════════════════════════════════════════════════════════════════════
# THE THREE HOOKS session.py CALLS
# ═════════════════════════════════════════════════════════════════════════════

def drain_pending_fires() -> list:
    """Take everything that has fired since the last call. Exactly once each."""
    if not _pending:
        return []
    out = list(_pending)
    _pending.clear()
    return out


def alarm_ringing() -> bool:
    """True while an alarm is awaiting snooze/dismiss, so the session knows
    Touch1/2 (snooze) and Touch3 (dismiss) mean the alarm and not idle-exit."""
    return bool(_ringing)


def snooze_current(minutes: int = 5):
    """Push the ringing alarm out by `minutes`. Returns the new time, or None."""
    if not _ringing:
        return None
    s = next((x for x in _store["schedules"] if x["id"] == _ringing[0]), None)
    _ringing.clear()
    if s is None:
        return None
    s["snoozes"] = int(s.get("snoozes", 0)) + 1
    if s["snoozes"] > _MAX_SNOOZES:
        s["enabled"] = False
        _save()
        print(f"  …  snoozed {s['snoozes']}x, giving up on {s.get('label')!r}")
        return None
    nxt = _now() + dt.timedelta(minutes=minutes)
    # One representation for both kinds: a snooze is always an override field,
    # never a rewrite of `at` or `time_of_day`. next_due() reads it, so the
    # PC app and the prompt both show the snoozed time rather than the
    # original one. A one-shot is re-enabled because _fire() disabled it.
    s["snooze_until"] = _iso(nxt)
    if not s.get("repeat"):
        s["enabled"] = True
    _save()
    print(f"  😴 snoozed {minutes} min: {s.get('label')!r} -> {_iso(nxt)}")
    return _iso(nxt)


def dismiss_current():
    """Stop the ringing alarm without rescheduling it."""
    if not _ringing:
        return None
    s = next((x for x in _store["schedules"] if x["id"] == _ringing[0]), None)
    _ringing.clear()
    if s is None:
        return None
    s.pop("snooze_until", None)
    _save()
    print(f"  ✔️  dismissed: {s.get('label')!r}")
    return s.get("label", "")


# ═════════════════════════════════════════════════════════════════════════════
# PROMPT CONTEXT — system_prompt.py calls this; a raise here would break
# prompt building, so the whole body is defensive.
# ═════════════════════════════════════════════════════════════════════════════

def render_for_prompt() -> str:
    """Upcoming schedules and open todos, as passive context for the model."""
    try:
        now = _now()
        horizon = now + dt.timedelta(days=_PROMPT_HORIZON_DAYS)
        lines = []

        upcoming = []
        for s in _store["schedules"]:
            if not s.get("enabled", True):
                continue
            due = next_due(s, after=now)
            if due is not None and due <= horizon:
                upcoming.append((due, s))
        upcoming.sort(key=lambda p: p[0])

        if upcoming:
            lines.append("Alarms, timers and reminders already set (local time):")
            for due, s in upcoming[:12]:
                lines.append(f"  • {s.get('label','')} — {_friendly(s)} "
                             f"[{s.get('kind')}, id {s['id']}]")

        open_todos = [t for t in _store["todos"] if not t.get("done")]
        if open_todos:
            lines.append("")
            lines.append(f"Open todos ({len(open_todos)}):")
            for t in open_todos[:_PROMPT_MAX_TODOS]:
                extra = f" (due {t['due']})" if t.get("due") else ""
                lines.append(f"  • {t['text']}{extra} [id {t['id']}]")
            if len(open_todos) > _PROMPT_MAX_TODOS:
                lines.append(f"  … and {len(open_todos) - _PROMPT_MAX_TODOS} more.")

        return "\n".join(lines)
    except Exception as e:
        print(f"[prompt] scheduler render failed: {e}", flush=True)
        return ""


# ═════════════════════════════════════════════════════════════════════════════
# SELF-TEST — python scheduler.py
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  scheduler.py self-test")
    print("=" * 70)

    ok = fail = 0

    def check(label, cond):
        global ok, fail
        if cond:
            ok += 1
            print(f"  PASS  {label}")
        else:
            fail += 1
            print(f"  FAIL  {label}")

    check("parse '07:00'", parse_time_of_day("07:00") == (7, 0))
    check("parse '7:30 pm'", parse_time_of_day("7:30 pm") == (19, 30))
    check("parse '12am' is midnight", parse_time_of_day("12am") == (0, 0))
    check("parse '12pm' is noon", parse_time_of_day("12pm") == (12, 0))
    check("parse junk is None", parse_time_of_day("half past") is None)
    check("weekdays 'workdays'", parse_weekdays("workdays") ==
          ["mon", "tue", "wed", "thu", "fri"])
    check("weekdays 'daily' is 7", len(parse_weekdays("daily")) == 7)
    check("weekdays bad token is False", parse_weekdays("mondayish") is False)
    check("weekdays None means one-shot", parse_weekdays("once") is None)

    t = parse_when("09:30")
    check("parse_when '09:30' is future", t is not None and t > _now())

    now = _now()
    rep = {"kind": "alarm", "id": "x", "label": "t",
           "time_of_day": "07:00", "repeat": {"weekdays": ["mon", "wed"]}}
    nd = next_due(rep, after=now)
    check("repeating next_due lands on mon or wed",
          nd is not None and WEEKDAY_NAMES[nd.weekday()] in ("mon", "wed"))
    check("repeating next_due is in the future", nd > now)

    past = {"kind": "alarm", "id": "y", "label": "t",
            "at": _iso(now - dt.timedelta(days=2)), "repeat": None}
    check("stale one-shot has no next_due", next_due(past, after=now) is None)

    check("valid schedule accepted", _valid_schedule(rep))
    check("schedule with no kind rejected", not _valid_schedule({"id": "z"}))
    check("todo needs text", not _valid_todo({"id": "1", "text": "  "}))
    check("todo ok", _valid_todo({"id": "1", "text": "buy milk"}))

    r = list_todos()
    check("list_todos returns an envelope", r["ok"] and "todos" in r["data"])
    r = list_schedules()
    check("list_schedules returns an envelope", r["ok"] and "schedules" in r["data"])

    err = cancel_schedule("definitely-not-a-real-thing-xyz")
    check("cancelling an unknown schedule is refused", not err["ok"])

    check("bad timer length refused", not set_timer(seconds="soon")["ok"])
    check("negative timer refused", not set_timer(seconds=-5)["ok"])
    check("absurd timer refused", not set_timer(hours=24 * 30)["ok"])

    snap = snapshot()
    check("snapshot has the four keys",
          {"version", "schedules", "todos", "now"} <= set(snap))
    check("alarm_ringing is False at rest", alarm_ringing() is False)
    check("no fires pending at rest", drain_pending_fires() == [])
    check("drain is idempotent", drain_pending_fires() == [])

    # Firing, without touching the real store's live entries.
    probe = {"id": "probe1", "kind": "alarm", "label": "probe",
             "at": _iso(now - dt.timedelta(seconds=5)), "repeat": None,
             "enabled": True, "last_fired": None, "snoozes": 0}
    _store["schedules"].append(probe)
    _check(now, monotonic_elapsed=1.0, wall_elapsed=1.0)
    fired = drain_pending_fires()
    check("a just-due one-shot fires", len(fired) == 1 and fired[0]["id"] == "probe1")
    check("firing raises alarm_ringing", alarm_ringing() is True)
    check("firing does not re-fire on the next tick",
          (_check(now, 1.0, 1.0), drain_pending_fires()) == (False, []))
    check("fired one-shot is disabled, not deleted",
          probe["enabled"] is False and probe in _store["schedules"])

    d = dismiss_current()
    check("dismiss clears the ring", d == "probe" and alarm_ringing() is False)

    probe2 = {"id": "probe2", "kind": "alarm", "label": "probe2",
              "at": _iso(now - dt.timedelta(seconds=5)), "repeat": None,
              "enabled": True, "last_fired": None, "snoozes": 0}
    _store["schedules"].append(probe2)
    _check(now, 1.0, 1.0)
    drain_pending_fires()
    new_at = snooze_current(5)
    check("snooze returns a time about 5 minutes out",
          new_at is not None and abs(
              (_parse(new_at) - (_now() + dt.timedelta(minutes=5))).total_seconds()) < 5)
    check("snooze clears the ring", alarm_ringing() is False)
    check("snoozed entry is re-enabled", probe2["enabled"] is True)
    check("snooze does not rewrite `at`",
          probe2.get("snooze_until") is not None)
    check("next_due reports the snoozed time",
          next_due(probe2) == _parse(probe2["snooze_until"]))

    # The snooze actually rings when it comes due, exactly once.
    probe2["snooze_until"] = _iso(now - dt.timedelta(seconds=2))
    _check(now, 1.0, 1.0)
    f2 = drain_pending_fires()
    check("a due snooze fires", len(f2) == 1 and f2[0]["id"] == "probe2")
    check("the snooze override is consumed",
          probe2.get("snooze_until") is None)
    _check(now, 1.0, 1.0)
    check("a consumed snooze does not fire twice", drain_pending_fires() == [])
    dismiss_current()

    # The bug this file's _fire docstring describes: a snoozed REPEATING alarm
    # must not let its own time_of_day occurrence ring a second time.
    rep_probe = {
        "id": "probe4", "kind": "alarm", "label": "probe4",
        "time_of_day": now.strftime("%H:%M"),
        "repeat": {"weekdays": [WEEKDAY_NAMES[now.weekday()]]},
        "enabled": True, "last_fired": None, "snoozes": 0,
    }
    rep_probe["at"] = _iso(next_due(rep_probe) or now)
    _store["schedules"].append(rep_probe)
    _check(now, 1.0, 1.0)
    check("a repeating alarm due now fires",
          any(f["id"] == "probe4" for f in drain_pending_fires()))
    stamp_before = rep_probe["last_fired"]
    snooze_current(5)
    rep_probe["snooze_until"] = _iso(now - dt.timedelta(seconds=2))
    _check(now, 1.0, 1.0)
    check("the repeating alarm's snooze fires",
          any(f["id"] == "probe4" for f in drain_pending_fires()))
    check("a snooze fire preserves the dedupe stamp",
          rep_probe["last_fired"] == stamp_before)
    _check(now, 1.0, 1.0)
    check("the repeating occurrence does NOT ring again after a snooze",
          drain_pending_fires() == [])
    check("a snooze fire does not reset the snooze budget",
          rep_probe["snoozes"] == 1)
    dismiss_current()

    stale = {"id": "probe3", "kind": "alarm", "label": "stale",
             "at": _iso(now - dt.timedelta(days=3)), "repeat": None,
             "enabled": True, "last_fired": None, "snoozes": 0}
    _store["schedules"].append(stale)
    _check(now, 1.0, 1.0)
    check("a 3-day-old one-shot does NOT fire",
          all(f["id"] != "probe3" for f in drain_pending_fires()))

    jumped = _check(now, monotonic_elapsed=1.0, wall_elapsed=3600.0)
    check("a clock jump is detected and resyncs", jumped is True)

    txt = render_for_prompt()
    check("render_for_prompt returns a string", isinstance(txt, str))

    # Remove the probes so the self-test never leaves junk behind.
    _store["schedules"] = [s for s in _store["schedules"]
                           if not str(s.get("id", "")).startswith("probe")]
    _save()

    print("=" * 70)
    print(f"  {ok} passed, {fail} failed")
    print("=" * 70)
    raise SystemExit(1 if fail else 0)
