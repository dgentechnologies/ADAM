"""Add sync metadata to the Pi scheduler: updated_at stamps and tombstones.

Prerequisite for cross-device sync (docs/CLOUD_DATA_SCHEMA.md §6.3). Without a
per-record modification time, last-write-wins has nothing to compare and a
bridge has to guess which copy is newer — guessing wrong silently discards a
real edit.

Design note on tombstones: deletions are recorded in a SEPARATE list, not as a
`deleted: true` flag on the row. Flagging in place would mean every reader —
list_schedules, snapshot, next_due, drain_pending_fires — has to filter, and a
single missed filter means a deleted alarm still rings. Keeping the live lists
exactly as they are makes that class of bug impossible.
"""
from pathlib import Path

p = Path(r"D:\Dgen Technologies Pvt. Ltd\ADAM\MP-MC codes\pi\adam\scheduler.py")
s = p.read_text(encoding="utf-8")
orig = s


def once(old, new, label):
    global s
    if old not in s:
        raise SystemExit(f"ANCHOR MISSING ({label}) - aborting with no changes")
    if s.count(old) != 1:
        raise SystemExit(f"ANCHOR AMBIGUOUS ({label}, {s.count(old)}x) - aborting")
    s = s.replace(old, new, 1)


# ── 1. helpers ────────────────────────────────────────────────────────────
once(
'''def _parse(s: str):''',
'''# Sync metadata. `_iso` is minute-resolution because that is what a user means
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


def _parse(s: str):''', "helpers")

# ── 2. prune on load, and guarantee the key exists ────────────────────────
once(
'''_sanitise()
''',
'''_store.setdefault("tombstones", [])
_sanitise()
_prune_tombstones()
''', "load")

# ── 3. stamp creations ────────────────────────────────────────────────────
# One global replace covers all three literals that carry "created": the two
# in set_alarm and the timer's, which spells it as a prefix of a longer line.
s_before = s
s = s.replace('"created": _iso(now),', '"created": _iso(now), "updated_at": _iso_s(now),')
n_created = s_before.count('"created": _iso(now),')
if not n_created:
    raise SystemExit("ANCHOR MISSING (schedule creations)")
print(f"  stamped {n_created} schedule creation literal(s)")

once('''             "created": _iso(_now()), "due": due_iso, "done_at": None}''',
     '''             "created": _iso(_now()), "updated_at": _iso_s(_now()),
             "due": due_iso, "done_at": None}''', "todo create")

# ── 4. stamp mutations ────────────────────────────────────────────────────
once('''    todo["done"] = True
    todo["done_at"] = _iso(_now())
    _save()''',
'''    todo["done"] = True
    todo["done_at"] = _iso(_now())
    _touch(todo)
    _save()''', "complete_todo")

# ── 5. tombstone deletions ────────────────────────────────────────────────
once('''    _store["todos"] = [t for t in _store["todos"] if t["id"] != todo["id"]]
    _save()''',
'''    _store["todos"] = [t for t in _store["todos"] if t["id"] != todo["id"]]
    _tombstone("todo", todo)
    _save()''', "delete_todo")

once('''    _store["schedules"] = [s for s in _store["schedules"] if s["id"] != dead["id"]]
    if _ringing and _ringing[0] == dead["id"]:''',
'''    _store["schedules"] = [s for s in _store["schedules"] if s["id"] != dead["id"]]
    _tombstone("schedule", dead)
    if _ringing and _ringing[0] == dead["id"]:''', "cancel_schedule")

# ── 6. replace_all (the sync API's wholesale writer) ──────────────────────
once('''            s.setdefault("created", _iso(_now()))''',
     '''            s.setdefault("created", _iso(_now()))
            s.setdefault("updated_at", _iso_s(_now()))''', "replace_all schedules")

once('''            t.setdefault("created", _iso(_now()))''',
     '''            t.setdefault("created", _iso(_now()))
            t.setdefault("updated_at", _iso_s(_now()))''', "replace_all todos")

assert s != orig
p.write_text(s, encoding="utf-8")
print("scheduler.py patched: updated_at stamps + tombstones")
