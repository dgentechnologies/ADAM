"""
sync_api.py — small HTTP API so the PC app can read and edit the Pi's data
==============================================================================
A stdlib-only HTTP/1.1 server (no Flask) on SYNC_HOST:SYNC_PORT (default
0.0.0.0:8766) that serves the four stores the companion app needs:

    GET  /api/ping          is this the Pi, and what is it running
    GET  /api/snapshot      EVERYTHING in one object — the app's normal call
    GET  /api/schedules     alarms, timers, reminders
    GET  /api/todos
    GET  /api/memories
    GET  /api/conversations
    PUT  /api/schedules     replace the schedule list
    PUT  /api/todos         replace the todo list
    POST /api/todos         add one (same as the add_todo tool)
    POST /api/schedules     add one (same as set_alarm/set_timer)

WHY THE PI IS THE SOURCE OF TRUTH
--------------------------------
The user asked for the list to live on the Pi and be synced down to the
laptop when the app connects. That ordering matters: ADAM can set an alarm by
voice with no laptop anywhere in the room, so if the laptop held the only
copy, an alarm set while the app was closed would simply not exist. The Pi
owns the data; the app keeps a cached copy so the tab still renders when the
Pi is unreachable, and shows its age rather than pretending to be live.

CONCURRENCY — why this is safe without locks
--------------------------------------------
asyncio.start_server runs in the SAME event loop as main.py and
scheduler.py. So a request handler and a scheduler tick can never run at the
same instant, and the store's read-modify-write in replace_all() is atomic
with respect to the ticker by construction rather than by locking. Adding a
thread pool later would silently break that and would need a lock; there is a
comment on the read path saying so.

SECURITY
--------
Writes require the SYNC_TOKEN header and are refused 403 without it. An empty
configured token makes the whole API READ-ONLY rather than open. Values are
never echoed in errors, and the token never reaches a log line. Requests are
size-capped so a malformed Content-Length cannot make the Pi allocate its way
into an OOM. This binds to the LAN by default — appropriate for a home robot
that must work without configuration, and the token is what protects the
write path.
"""

import asyncio
import json
import time

from config import SYNC_HOST, SYNC_PORT, SYNC_TOKEN
# conv_log is imported rather than re-read from CONV_MEMORY_FILE: the file is
# only rewritten when a session ends or a turn is appended, so reading disk
# here would serve a stale conversation list mid-session. memory_store mutates
# this list in place, so this binds to the live object.
from memory_store import memory, conv_log
import scheduler
import touch_controls
import laptop_pairing

__all__ = ["start_sync_api", "stop_sync_api", "API_VERSION"]

API_VERSION = 1

# A snapshot of the app's own conversation log is capped — the file grows
# without bound over months, and the PC app only ever renders a recent view.
_CONV_LIMIT = 60

# 1 MiB. The largest legitimate body is a full todos+schedules replacement,
# which is kilobytes. Anything bigger is a bug or an attack.
_MAX_BODY = 1024 * 1024

_server = None


# ═════════════════════════════════════════════════════════════════════════════
# HTTP PLUMBING
# ═════════════════════════════════════════════════════════════════════════════

_REASON = {200: "OK", 400: "Bad Request", 403: "Forbidden", 404: "Not Found",
           405: "Method Not Allowed", 413: "Payload Too Large",
           500: "Internal Server Error"}


async def _reply(writer, status: int, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    head = (
        f"HTTP/1.1 {status} {_REASON.get(status, '')}\r\n"
        f"Content-Type: application/json; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Access-Control-Allow-Origin: *\r\n"
        f"Access-Control-Allow-Headers: Content-Type, X-ADAM-Token\r\n"
        f"Access-Control-Allow-Methods: GET, POST, PUT, OPTIONS\r\n"
        f"Cache-Control: no-store\r\n"
        f"Connection: close\r\n\r\n"
    ).encode("ascii")
    writer.write(head + body)
    try:
        await writer.drain()
    except Exception:
        pass


def _authorised(headers: dict) -> bool:
    """Writes need the token. An unset token means read-only, not open."""
    if not SYNC_TOKEN:
        return False
    supplied = headers.get("x-adam-token", "")
    if not supplied:
        auth = headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()
    # compare_digest keeps the comparison from leaking the token's length and
    # prefix through timing, which a plain == would.
    import hmac
    return hmac.compare_digest(supplied, SYNC_TOKEN)


async def _read_body(reader, headers: dict):
    """Read and parse a JSON body, or return (None, error_status)."""
    try:
        length = int(headers.get("content-length", "0") or "0")
    except ValueError:
        return None, 400
    if length > _MAX_BODY:
        return None, 413
    if length <= 0:
        return {}, None
    raw = await reader.readexactly(length)
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, 400
    if not isinstance(data, dict):
        return None, 400
    return data, None


_ROUTES_GET = {
    "/api/ping":          "_r_ping",
    "/api/snapshot":      "_r_snapshot",
    "/api/schedules":     "_r_schedules",
    "/api/todos":         "_r_todos",
    "/api/memories":      "_r_memories",
    "/api/conversations": "_r_conversations",
    "/api/touch/assignments": "_r_touch_assignments",
}
_ROUTES_WRITE = {
    "/api/schedules":    "_w_schedules",
    "/api/todos":        "_w_todos",
    "/api/todos/done":   "_w_todo_done",
    "/api/todos/delete": "_w_todo_delete",
    "/api/schedules/cancel": "_w_schedule_cancel",
    "/api/touch/assignments": "_w_touch_assignments",
    "/api/laptops/pair": "_w_laptop_pair",
    "/api/laptops/unpair": "_w_laptop_unpair",
}


async def _handle(reader, writer) -> None:
    try:
        request_line = await asyncio.wait_for(reader.readline(), timeout=10)
        if not request_line:
            return
        try:
            method, target, _ = request_line.decode("latin-1").split(" ", 2)
        except ValueError:
            await _reply(writer, 400, {"error": "bad request line"})
            return

        headers = {}
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=10)
            if line in (b"\r\n", b"\n", b""):
                break
            if b":" in line:
                k, v = line.decode("latin-1").split(":", 1)
                headers[k.strip().lower()] = v.strip()

        path = target.split("?", 1)[0].rstrip("/") or "/"
        method = method.upper()

        if method == "OPTIONS":
            await _reply(writer, 200, {"ok": True})
            return

        if method == "GET":
            fn = _ROUTES_GET.get(path)
            if fn is None:
                await _reply(writer, 404, {"error": "unknown path", "path": path})
                return
            await _reply(writer, 200, await globals()[fn]())
            return

        if method in ("PUT", "POST"):
            fn = _ROUTES_WRITE.get(path)
            if fn is None:
                await _reply(writer, 404, {"error": "unknown path", "path": path})
                return
            if not _authorised(headers):
                # Say WHY in one line, without hinting at the token's value.
                why = ("no SYNC_TOKEN configured — this Pi is read-only"
                       if not SYNC_TOKEN else "bad or missing X-ADAM-Token")
                await _reply(writer, 403, {"error": "forbidden", "reason": why})
                return
            body, err = await _read_body(reader, headers)
            if err:
                await _reply(writer, err, {"error": _REASON.get(err, "bad body")})
                return
            status, payload = await globals()[fn](body, method)
            await _reply(writer, status, payload)
            return

        if method in ("PATCH", "DELETE"):
            await _reply(writer, 405, {
                "error": "not supported",
                "hint": "Send the whole updated list with PUT, or use the "
                        "todo/schedule tools by voice."})
            return

        await _reply(writer, 405, {"error": f"method {method} not allowed"})

    except asyncio.TimeoutError:
        pass
    except asyncio.IncompleteReadError:
        pass
    except Exception as e:
        print(f"  ⚠️  sync api error: {e}")
        try:
            await _reply(writer, 500, {"error": "internal error"})
        except Exception:
            pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════════════════════
# READ HANDLERS
#
# These touch the scheduler store and the in-memory `memory` dict directly.
# That is only safe because this server shares the event loop with them — see
# the module docstring. Do not move this onto a thread pool without adding a
# lock around every read-modify-write below.
# ═════════════════════════════════════════════════════════════════════════════

async def _r_ping() -> dict:
    return {"ok": True, "app": "adam", "kind": "pi", "api": API_VERSION,
            "readonly": not bool(SYNC_TOKEN), "time": int(time.time()),
            "local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "capabilities": {**touch_controls.capabilities(),
                             "laptop_pairing": laptop_pairing.available()}}


async def _r_touch_assignments() -> dict:
    return {"ok": True, "api": API_VERSION, "assignments": touch_controls.read_assignments(),
            "sensors": touch_controls.SENSOR_INFO}


async def _w_touch_assignments(body: dict, method: str):
    try:
        assignments = await touch_controls.save_assignments(body.get("assignments"))
        return 200, {"ok": True, "api": API_VERSION, "assignments": assignments}
    except ValueError as error:
        return 400, {"ok": False, "error": str(error)}
    except Exception:
        return 500, {"ok": False, "error": "Touch preferences could not be saved on ADAM"}


async def _w_laptop_pair(body: dict, method: str):
    try:
        return 200, {"api": API_VERSION, **await laptop_pairing.pair(body)}
    except ValueError as error:
        return 400, {"ok": False, "error": str(error)}
    except Exception:
        return 500, {"ok": False, "error": "The laptop connection could not be saved"}


async def _w_laptop_unpair(body: dict, method: str):
    try:
        return 200, {"api": API_VERSION, **await laptop_pairing.unpair()}
    except Exception:
        return 500, {"ok": False, "error": "The laptop connection could not be revoked"}


async def _r_snapshot() -> dict:
    snap = scheduler.snapshot()
    snap["memories"] = dict(memory)
    return {"ok": True, "api": API_VERSION, "data": snap}


async def _r_schedules() -> dict:
    return {"ok": True, "api": API_VERSION,
            "data": scheduler.snapshot()["schedules"]}


async def _r_todos() -> dict:
    return {"ok": True, "api": API_VERSION,
            "data": scheduler.snapshot()["todos"]}


async def _r_memories() -> dict:
    return {"ok": True, "api": API_VERSION,
            "data": {k: v for k, v in memory.items()}}


async def _r_conversations() -> dict:
    log = conv_log if isinstance(conv_log, list) else []
    return {"ok": True, "api": API_VERSION, "total": len(log),
            "data": log[-_CONV_LIMIT:]}


# ═════════════════════════════════════════════════════════════════════════════
# WRITE HANDLERS
# ═════════════════════════════════════════════════════════════════════════════

async def _w_schedules(body: dict, method: str):
    if method == "POST":
        rows = body.get("schedules")
        one = body if rows is None else (rows[0] if rows else {})
        kind = (one.get("kind") or "alarm").lower()
        if kind == "timer":
            env = scheduler.set_timer(
                seconds=one.get("seconds"), minutes=one.get("minutes"),
                hours=one.get("hours"), label=one.get("label", ""))
        elif kind == "reminder":
            env = scheduler.set_reminder(one.get("label", ""),
                                         one.get("when") or one.get("at", ""),
                                         one.get("repeat"))
        else:
            env = scheduler.set_alarm(one.get("label", ""),
                                      one.get("when") or one.get("at", ""),
                                      one.get("repeat"))
    else:
        rows = body.get("schedules")
        if rows is None and isinstance(body.get("data"), list):
            rows = body["data"]
        if not isinstance(rows, list):
            return 400, {"ok": False, "error": "expected {'schedules': [...]}"}
        env = scheduler.replace_all(schedules=rows)

    if not env.get("ok"):
        return 400, {"ok": False, "error": env.get("reason", "rejected")}
    return 200, {"ok": True, "api": API_VERSION, "data": env.get("data", {}),
                 "schedules": scheduler.snapshot()["schedules"]}


async def _w_todos(body: dict, method: str):
    if method == "POST":
        env = scheduler.add_todo(body.get("text", ""), body.get("due", "") or "")
    else:
        rows = body.get("todos")
        if rows is None and isinstance(body.get("data"), list):
            rows = body["data"]
        if not isinstance(rows, list):
            return 400, {"ok": False, "error": "expected {'todos': [...]}"}
        env = scheduler.replace_all(todos=rows)

    if not env.get("ok"):
        return 400, {"ok": False, "error": env.get("reason", "rejected")}
    return 200, {"ok": True, "api": API_VERSION, "data": env.get("data", {}),
                 "todos": scheduler.snapshot()["todos"]}


# ── Fine-grained edits ────────────────────────────────────────────────────
# These exist so the PC app NEVER has to reconstruct an entry to change it.
# That matters because snapshot() is a DISPLAY view, not a storage view: for a
# repeating schedule its "at" is the recomputed next occurrence rather than
# the stored one, and it omits "snoozes" entirely. Feeding a snapshot row back
# through replace_all() could therefore drop a recurring alarm or reset a
# snooze count. Routing the edit to the scheduler's own mutation instead means
# the entry keeps every field it had.
#
# "target" is an id ("td_ab12cd34", "al_...") or a text/label substring —
# scheduler._find_todo and cancel_schedule already accept both and refuse an
# ambiguous match rather than guessing.

async def _w_todo_done(body: dict, method: str):
    env = scheduler.complete_todo(
        (body or {}).get("id") or (body or {}).get("target") or "")
    if not env.get("ok"):
        return 404, {"ok": False, "error": env.get("reason", "not found")}
    return 200, {"ok": True, "api": API_VERSION, "data": env.get("data", {}),
                 "todos": scheduler.snapshot()["todos"]}


async def _w_todo_delete(body: dict, method: str):
    env = scheduler.delete_todo(
        (body or {}).get("id") or (body or {}).get("target") or "")
    if not env.get("ok"):
        return 404, {"ok": False, "error": env.get("reason", "not found")}
    return 200, {"ok": True, "api": API_VERSION, "data": env.get("data", {}),
                 "todos": scheduler.snapshot()["todos"]}


async def _w_schedule_cancel(body: dict, method: str):
    env = scheduler.cancel_schedule(
        (body or {}).get("id") or (body or {}).get("target") or "")
    if not env.get("ok"):
        return 404, {"ok": False, "error": env.get("reason", "not found")}
    return 200, {"ok": True, "api": API_VERSION, "data": env.get("data", {}),
                 "schedules": scheduler.snapshot()["schedules"]}


# ═════════════════════════════════════════════════════════════════════════════
# LIFECYCLE
# ═════════════════════════════════════════════════════════════════════════════

async def start_sync_api():
    """Start the HTTP API. Call ONCE from main.py, beside start_ws_server().

    Returns the asyncio.Server, or None if the port is taken / unavailable.
    Never raises: the companion app losing its data view must not stop the
    robot from listening, and the WS face channel is unaffected either way.
    """
    global _server
    if _server is not None:
        return _server
    try:
        _server = await asyncio.start_server(_handle, SYNC_HOST, SYNC_PORT)
    except Exception as e:
        print(f"⚠️  Sync API unavailable on {SYNC_HOST}:{SYNC_PORT}: {e}")
        return None
    mode = "read-only (no SYNC_TOKEN set)" if not SYNC_TOKEN else "read-write"
    print(f"✅ Sync API  → http://{SYNC_HOST}:{SYNC_PORT}  [{mode}]")
    if not SYNC_TOKEN:
        print("   Set SYNC_TOKEN in ~/adam/.env to let the PC app save "
              "changes back to the Pi.")
    return _server


async def stop_sync_api() -> None:
    global _server
    if _server is not None:
        _server.close()
        try:
            await _server.wait_closed()
        except Exception:
            pass
        _server = None
