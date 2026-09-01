"""
memory_store.py — ADAM v40 persistent memory
==============================================================================
JSON persistence for ADAM's three long-lived stores:

  • memory     — key/value facts the model saves via save_memory()
  • faces      — people ADAM has been told to remember (remember_person())
  • conv_log   — rolling window of recent conversation turns, so a FRESH
                 (non-resumed) session still has context instantly

All three are plain module-level objects that the rest of the codebase
mutates IN PLACE (memory[k]=v, faces[pid]=..., conv_log.append(...)). That
is deliberate: importing them elsewhere (`from memory_store import memory`)
binds to the same live object, so every module sees the same state without
any getter/setter plumbing. Nothing here ever REBINDS these names, only
mutates them, which is what makes that work.

Imports only from config — sits low in the dependency graph.
"""

import os
import json
import time
import datetime
from pathlib import Path

from config import (
    MEMORY_FILE,
    FACE_MEMORY_FILE,
    CONV_MEMORY_FILE,
    CONV_MAX_TURNS,
)


def load_json(path: Path, default):
    """Load JSON with corruption resilience — a power-loss mid-write on a
    Pi's SD card is common enough in a physical product that this must not
    crash the whole robot on boot. A corrupt file is backed up (for later
    forensics) rather than silently deleted, and the caller gets a clean
    default so ADAM boots with empty-but-functional memory instead of
    refusing to start."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠️  {path.name} is corrupt/unreadable ({e}) — "
              f"backing up and starting fresh")
        try:
            backup = path.with_suffix(path.suffix + f".corrupt.{int(time.time())}")
            path.rename(backup)
            print(f"    (corrupt file preserved at {backup.name} for inspection)")
        except Exception as e2:
            print(f"    ⚠️  could not back up corrupt file: {e2}")
        return default


def save_json(path: Path, data) -> None:
    """Atomic write — write to a temp file in the same directory, fsync it,
    then os.replace() onto the real path. os.replace is atomic on POSIX, so
    a power loss or crash mid-write leaves either the OLD complete file or
    the NEW complete file, never a half-written/corrupt one. This matters a
    lot more on a robot that can lose power ungracefully (unplugged, brownout
    from servo current draw, etc.) than on a normal server."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"⚠️  Save {path.name}: {e}")
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


memory = load_json(MEMORY_FILE, {})
faces  = load_json(FACE_MEMORY_FILE, {})
print(f"✅ Memory: {len(memory)} entries | Faces: {len(faces)} known")

# ── Rolling conversation history — lets a FRESH (non-resumed) session
# pick up context instantly instead of starting blank. Persisted to disk
# so it survives a full process restart too. ─────────────────────────
conv_log: list = load_json(CONV_MEMORY_FILE, [])
print(f"✅ Conversation history: {len(conv_log)} turns loaded")


def save_conversation_log() -> None:
    if len(conv_log) > CONV_MAX_TURNS:
        del conv_log[:-CONV_MAX_TURNS]
    save_json(CONV_MEMORY_FILE, conv_log)


def append_conversation_turn(user_text: str, adam_text: str) -> None:
    u = (user_text or "").strip()
    a = (adam_text or "").strip()
    if not u and not a:
        return
    conv_log.append({
        "ts":   datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "user": u,
        "adam": a,
    })
    save_conversation_log()
