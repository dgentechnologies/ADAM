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
import hmac
import ipaddress
import json
import os
import time

from config import SYNC_HOST, SYNC_PORT, SYNC_TOKEN, BASE_DIR, APP_VERSION
# conv_log is imported rather than re-read from CONV_MEMORY_FILE: the file is
# only rewritten when a session ends or a turn is appended, so reading disk
# here would serve a stale conversation list mid-session. memory_store mutates
# this list in place, so this binds to the live object.
from memory_store import memory, conv_log
import scheduler
import touch_controls
import laptop_pairing
import desktop_pairing

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
           405: "Method Not Allowed", 408: "Request Timeout",
           413: "Payload Too Large", 500: "Internal Server Error"}


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
    supplied = headers.get("x-adam-token", "")
    if desktop_pairing.authenticate(supplied):
        return True
    if desktop_pairing.identity() or not SYNC_TOKEN:
        return False
    if not supplied:
        auth = headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()
    # Compare as BYTES, not str. hmac.compare_digest raises TypeError on str
    # containing non-ASCII ("comparing strings with non-ASCII characters is not
    # supported"), and headers are attacker-controlled — so a token header with
    # one accented character used to escape into the generic 500 handler and log
    # a line, instead of being the plain 403 it is. Encoding first makes every
    # malformed token take the same path as a wrong one.
    #
    # compare_digest (not ==) keeps the comparison from leaking the token's
    # length and prefix through timing.
    return hmac.compare_digest(supplied.encode("utf-8", "replace"),
                               SYNC_TOKEN.encode("utf-8", "replace"))


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
    # Timeout the body read. Every other read in _handle is wrapped in
    # wait_for(10s), but this one was not: a client could declare a legal
    # Content-Length, send nothing, and hold this coroutine plus its socket
    # open forever. IncompleteReadError only fires if the peer actually
    # disconnects, so a peer that simply stalls was never caught. On a 512 MB
    # Pi, accumulating stalled connections leaks file descriptors.
    try:
        raw = await asyncio.wait_for(reader.readexactly(length), timeout=10)
    except asyncio.TimeoutError:
        return None, 408
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, 400
    if not isinstance(data, dict):
        return None, 400
    return data, None


# Reads that expose PERSONAL data and therefore need the token, unlike the
# schedule/todo reads which stay open so the companion app can show something
# before it has been paired.
#
# Without this the Pi served the user's real name, saved memories and the last
# 60 conversation turns to ANY device on the LAN, unauthenticated, over plain
# HTTP on 0.0.0.0 — fine on a trusted home network, not fine on office, hostel
# or hotel Wi-Fi, where `curl http://adam-pi.local:8766/api/conversations` from
# any phone was enough.
_SENSITIVE_GET = {"/api/memories", "/api/conversations", "/api/tombstones"}

_ROUTES_GET = {
    "/api/ping":          "_r_ping",
    "/api/snapshot":      "_r_snapshot",
    "/api/schedules":     "_r_schedules",
    "/api/todos":         "_r_todos",
    "/api/memories":      "_r_memories",
    "/api/conversations": "_r_conversations",
    "/api/touch/assignments": "_r_touch_assignments",
    "/api/pair/info":     "_r_pair_info",
    "/api/tombstones":    "_r_tombstones",
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
    "/api/pair/claim":   "_w_pair_claim",
    "/api/pair/release": "_w_pair_release",
}

# /api/pair/claim is the ONE write that cannot require the token, because
# handing over that token is the whole point of it. It is gated on state
# instead of on a secret: it answers only while this unit is unclaimed. See
# _w_pair_claim for the trust model.
_UNAUTHENTICATED_WRITE = {"/api/pair/claim"}


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

        if path == "/api/pair/desktop" and method == "POST":
            body, err = await _read_body(reader, headers)
            if err:
                await _reply(writer, err, {"ok": False})
                return
            try:
                payload = desktop_pairing.claim(body)
                await _reply(writer, 200, payload)
            except ValueError:
                await _reply(writer, 403, {"ok": False, "error": "Authorization required"})
            return
        if path in ('/api/sync/records', '/api/sync/apply'):
            grant = desktop_pairing.authenticate(headers.get('x-adam-token', ''))
            if not grant:
                await _reply(writer, 403, {'ok': False, 'error': 'Authorization required'})
                return
            import record_sync
            try:
                if path == '/api/sync/records' and method == 'GET':
                    payload = record_sync.snapshot()
                elif path == '/api/sync/apply' and method == 'POST':
                    body, err = await _read_body(reader, headers)
                    if err:
                        await _reply(writer, err, {'ok': False})
                        return
                    payload = record_sync.apply(body)
                else:
                    raise ValueError('Unsupported sync operation')
                await _reply(writer, 200, payload)
            except (ValueError, KeyError, TypeError):
                await _reply(writer, 409, {'ok': False, 'error': 'Record rejected; refresh sync or check timezone and schema.'})
            return
        if path == "/api/pair/session" and method == "GET":
            grant = desktop_pairing.authenticate(headers.get("x-adam-token", ""))
            await _reply(writer, 200 if grant else 403, {"ok": bool(grant), **(grant or {})})
            return

        if method == "GET":
            fn = _ROUTES_GET.get(path)
            if fn is None:
                await _reply(writer, 404, {"error": "unknown path", "path": path})
                return
            auth = _authorised(headers)
            if (path in _SENSITIVE_GET or desktop_pairing.identity() and path not in ("/api/ping", "/api/pair/info")) and not auth:
                await _reply(writer, 403, {
                    "error": "forbidden",
                    "reason": ("this read needs X-ADAM-Token — set SYNC_TOKEN "
                               "in ~/adam/.env and pair the app")})
                return
            await _reply(writer, 200, await globals()[fn](auth))
            return

        if method in ("PUT", "POST"):
            fn = _ROUTES_WRITE.get(path)
            if fn is None:
                await _reply(writer, 404, {"error": "unknown path", "path": path})
                return
            if path not in _UNAUTHENTICATED_WRITE and not _authorised(headers):
                # Say WHY in one line, without hinting at the token's value.
                why = ("no SYNC_TOKEN configured — this Pi is read-only"
                       if not SYNC_TOKEN else "bad or missing X-ADAM-Token")
                await _reply(writer, 403, {"error": "forbidden", "reason": why})
                return
            body, err = await _read_body(reader, headers)
            if err:
                await _reply(writer, err, {"error": _REASON.get(err, "bad body")})
                return
            # Stamp the peer from the SOCKET, never from the body. _w_pair_claim
            # refuses non-private addresses, and a caller must not be able to
            # spoof that by sending its own "_peer".
            try:
                body["_peer"] = (writer.get_extra_info("peername") or ("", 0))[0]
            except Exception:
                body["_peer"] = ""
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
# ZERO-CONFIG PAIRING
#
# The goal: the user opens the app, presses "Find ADAM", picks the one that
# appears, and is connected. No IP address, no port, no token typed by hand.
#
# mDNS (discovery.py) solves "where is it". These endpoints solve "how does the
# app get the token", which is the part that cannot be advertised — a TXT
# record is readable by every device on the network.
#
# TRUST MODEL — trust on first use, and say so out loud.
#
#   While this unit is UNCLAIMED, /api/pair/claim hands the sync token to
#   whoever asks from a private address. Once claimed it refuses everyone until
#   the owner releases it (an authenticated call) — so the window is open only
#   between first boot and first pairing, not permanently.
#
#   This is the same bargain a printer or a smart speaker makes, and it is a
#   real bargain, not a free lunch: during that window, another device on the
#   same Wi-Fi could claim ADAM first. Three things keep that honest:
#     1. the window closes permanently at the first claim,
#     2. the claimer must be on a private/loopback address, so nothing off the
#        LAN can take it,
#     3. ADAM says the pairing out loud, so a claim the user did not make is
#        noticed rather than silent.
#
#   The alternative — making the user read a code off the robot — is more
#   secure and was explicitly not wanted. This is the documented trade.
# ═════════════════════════════════════════════════════════════════════════════

_PAIR_FILE = BASE_DIR / ".paired.json"
_pair_state = {"paired": False, "peer": "", "at": 0}


def _load_pair_state() -> None:
    """Read the claim record. A corrupt file leaves the unit UNCLAIMED rather
    than permanently locked — a user who cannot pair has a brick, whereas the
    worst case of re-opening the window is re-pairing on their own LAN."""
    global _pair_state
    try:
        if _PAIR_FILE.exists():
            raw = json.loads(_PAIR_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and raw.get("version") == 1:
                _pair_state = {"paired": bool(raw.get("paired")),
                               "peer": str(raw.get("peer", ""))[:64],
                               "at": int(raw.get("at", 0))}
    except Exception as e:
        print(f"  ⚠️  pairing record unreadable ({e}) — starting unpaired")


def _save_pair_state() -> None:
    try:
        _PAIR_FILE.write_text(json.dumps({"version": 1, **_pair_state}),
                              encoding="utf-8")
        try:
            os.chmod(_PAIR_FILE, 0o600)
        except Exception:
            pass
    except Exception as e:
        print(f"  ⚠️  could not persist pairing record: {e}")


def _is_private(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_private or                ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


async def _r_pair_info(authorised: bool = False) -> dict:
    """Unauthenticated identity, so the app can list what it found.

    Deliberately contains NOTHING secret: a name, a stable id, and whether this
    unit is already taken. Enough to render a picker, useless to an attacker.
    """
    try:
        import discovery
        ident = discovery.short_id()
    except Exception:
        ident = "ADAM"
    return {"ok": True, "api": API_VERSION, "id": ident, "name": "ADAM",
            "paired": _pair_state["paired"], "version": APP_VERSION,
            "needs_token": not bool(SYNC_TOKEN), "wsPort": 8765, **desktop_pairing.public_info()}


async def _w_pair_claim(body: dict, method: str):
    """Legacy LAN-first claims cannot prove possession; never return the shared key."""
    return 403, {"ok": False, "error": "Use authenticated desktop pairing with a code from ADAM."}


async def _w_pair_release(body: dict, method: str):
    """Release the claim so a different device can pair. Authenticated: only
    whoever currently holds the token may give it up."""
    _pair_state.update({"paired": False, "peer": "", "at": 0})
    _save_pair_state()
    try:
        import discovery
        await discovery.update_paired(False)
    except Exception:
        pass
    print("  🔓 Pairing released — ADAM is open to pair again")
    return 200, {"ok": True, "api": API_VERSION, "paired": False}


# ═════════════════════════════════════════════════════════════════════════════
# READ HANDLERS
#
# These touch the scheduler store and the in-memory `memory` dict directly.
# That is only safe because this server shares the event loop with them — see
# the module docstring. Do not move this onto a thread pool without adding a
# lock around every read-modify-write below.
# ═════════════════════════════════════════════════════════════════════════════

async def _r_ping(authorised: bool = False) -> dict:
    return {"ok": True, "app": "adam", "kind": "pi", "api": API_VERSION,
            "readonly": not bool(SYNC_TOKEN), "time": int(time.time()),
            "local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "capabilities": {**touch_controls.capabilities(),
                             "laptop_pairing": laptop_pairing.available()}}


async def _r_touch_assignments(authorised: bool = False) -> dict:
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


async def _r_snapshot(authorised: bool = False) -> dict:
    snap = scheduler.snapshot()
    # Gating /api/memories alone would have been theatre: the snapshot embeds
    # the same dict, so an unauthenticated caller could just read it here.
    # Omit the key entirely rather than send an empty one, so a client can tell
    # "not allowed to see this" from "there are no memories".
    if authorised:
        snap["memories"] = dict(memory)
        snap["tombstones"] = list(scheduler._store.get("tombstones", []))
    return {"ok": True, "api": API_VERSION, "data": snap}


async def _r_schedules(authorised: bool = False) -> dict:
    return {"ok": True, "api": API_VERSION,
            "data": scheduler.snapshot()["schedules"]}


async def _r_todos(authorised: bool = False) -> dict:
    return {"ok": True, "api": API_VERSION,
            "data": scheduler.snapshot()["todos"]}


async def _r_tombstones(authorised: bool = False) -> dict:
    """Deletions the Pi has recorded, newest last.

    A bridge needs these to tell "the user deleted this" from "this device has
    not heard about it yet". Without them, deleting a todo on the phone lets an
    offline Pi push its copy back and the row returns from the dead.
    """
    return {"ok": True, "api": API_VERSION,
            "keep_days": getattr(scheduler, "TOMBSTONE_KEEP_DAYS", None),
            "data": list(scheduler._store.get("tombstones", []))}


async def _r_memories(authorised: bool = False) -> dict:
    return {"ok": True, "api": API_VERSION,
            "data": {k: v for k, v in memory.items()}}


async def _r_conversations(authorised: bool = False) -> dict:
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
    _load_pair_state()
    if desktop_pairing.identity():
        import memory_sync
        memory_sync.recover()
    asyncio.create_task(desktop_pairing.display_codes())
    try:
        _server = await asyncio.start_server(_handle, SYNC_HOST, SYNC_PORT, ssl=desktop_pairing.ssl_context())
    except Exception as e:
        print(f"⚠️  Sync API unavailable on {SYNC_HOST}:{SYNC_PORT}: {e}")
        return None
    mode = "read-only (no SYNC_TOKEN set)" if not SYNC_TOKEN else "read-write"
    print(f"✅ Sync API  → http://{SYNC_HOST}:{SYNC_PORT}  [{mode}]")
    print(f"   Pairing: {'claimed' if _pair_state['paired'] else 'OPEN — the app can claim this unit'}")
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
