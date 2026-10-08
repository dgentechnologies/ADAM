# PC app integration

The Windows companion app (`pcAPP/`) and the Pi talk over two independent
channels that are easy to confuse:

| Channel | Port | Direction | Carries |
|---|---|---|---|
| Laptop **agent** | 8642 | Pi → PC | *commands* — volume, clipboard, media, coding tasks |
| Pi **sync API** | 8766 | PC → Pi | *data* — schedules, todos, memories, conversations |

They are separate servers owned by separate machines, and neither is a proxy for
the other. A command goes Pi→PC on 8642; a data read goes PC→Pi on 8766.

## 1. The Pi sync API (`sync_api.py`)

A stdlib-only HTTP/1.1 server — **no Flask on the Pi** — on `SYNC_HOST:SYNC_PORT`
(default `0.0.0.0:8766`).

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/ping` | is this the Pi, and what is it running |
| `GET` | `/api/snapshot` | **everything in one object** — the app's normal call |
| `GET` | `/api/schedules` | alarms, timers, reminders |
| `GET` | `/api/todos` | to-do list |
| `GET` | `/api/memories` | memory store |
| `GET` | `/api/conversations` | recent conversation log |
| `PUT` | `/api/schedules` | replace the schedule list |
| `PUT` | `/api/todos` | replace the to-do list |
| `POST` | `/api/todos` | add one — same as the `add_todo` tool |
| `POST` | `/api/schedules` | add one — same as `set_alarm`/`set_timer` |

`/api/snapshot` exists because the app's normal need is *all of it at once*, and
four round trips to render one tab is three more chances to be half-loaded.

The conversation list is capped at `_CONV_LIMIT` (60): the file grows without
bound over months and the app only ever renders a recent view.

### Security model

- Writes require the `SYNC_TOKEN` header; without it they are refused `403`.
- **An empty configured token makes the whole API read-only** rather than open.
  This is the important default: a missing `SYNC_TOKEN` in `.env` disables
  writes instead of exposing them to the LAN.
- Values are never echoed in error messages, and the token never reaches a log.
- Bodies are capped at `_MAX_BODY` (1 MiB) so a malformed `Content-Length`
  cannot make the Pi allocate its way into an OOM. The largest legitimate body
  is a full todos+schedules replacement, which is kilobytes.
- It binds to the LAN by default — appropriate for a home robot that must work
  without configuration. The token is what protects the write path; the read
  path is deliberately open so the app can show *something* before setup.

### Concurrency

`asyncio.start_server` runs in the **same event loop** as `main.py` and
`scheduler.py`. A request handler and a scheduler tick therefore cannot run at
the same instant, so `replace_all()`'s read-modify-write is atomic by
construction rather than by locking. **Adding a thread pool later would silently
break that** and would need a real lock — there is a comment on the read path
saying so.

## 2. The laptop agent (`pcAPP/backend.py`)

Flask on Windows, serving `GET /actions` — the self-describing manifest the Pi
reads to learn what this machine can do. Full detail in
[`clipboard_and_agent.md`](clipboard_and_agent.md).

There is also a `/pi/*` proxy in the PC app's backend: the **browser does not
call the Pi directly**. The Clock tab fetches `/pi/snapshot` and writes to
`/pi/write/<target>` on its own backend, which relays to the Pi. This keeps the
token and the Pi's address on the desktop side instead of exposing them to
whatever page the webview loads.

## 3. Parity — the check that catches drift

`laptop_actions.parity_report(live_manifest)` diffs a live agent's manifest
against the 18 actions a deployed agent must expose.

**Static result (verified 2026-10-02, `pcAPP/backend.py`):**

- 19 `@action` registrations, all 18 required actions present — zero missing.
- After resolving `value_type` the way the decorator actually does (at runtime,
  from the shared manifest, not from the decorator literal), **all 19 agree with
  `laptop_actions.py`** — zero mismatches.
- One action outside the parity set: `clipboard_paste`. It is in the Pi's
  canonical set, so this is informational, not drift.

**Live result: not run.** No agent was listening on 8642 during this work, so
`GET /actions` against a running agent was not performed. The static diff is the
right thing to diff statically, but it is not the same as a deployed agent
answering. To close it, with the app open:

```bash
curl -s http://127.0.0.1:8642/actions
```

and compare against `LIVE_PARITY_ACTIONS`.

## 4. Four bugs fixed on 2026-10-05

All four were found while diagnosing one report: *"the Alarm tab says not
connected, and clicking the left panel does nothing."* They were independent.

### 4.1 The whole dashboard was dead — a JavaScript syntax error

`static/js/dashboard.js` had an unclosed `if (eyeL && eyeR) {` in
`renderStatus()`. The indentation hid it: the `}` on the following line closed
the inner `else`, and every function after it was silently nested one level
deeper.

One missing brace means the file **fails to parse, so none of it ever runs** —
no nav handlers, no header clock, no status polling. The sidebar links were
plain `<a href="#">` elements with nothing bound, so clicking them genuinely
did nothing. Verify with:

```bash
node --check pcAPP/static/js/dashboard.js
```

This is worth adding to any pre-build check: PyInstaller bundles `static/`
without parsing it, so a syntax error ships silently into the exe.

### 4.2 Navigation was hostage to the 3D viewport

`DOMContentLoaded` ran `init3D()` first and `setupPanelTabs()` sixth, as a bare
sequence. `init3D()` constructs a `THREE.WebGLRenderer`, which **throws** when
WebGL is unavailable — routine in pywebview/WebView2 without GPU acceleration.
That throw aborted the whole handler before navigation was ever wired.

Each step is now wrapped individually and navigation is wired **first**. A
decorative 3D model must not be able to take down the app's menu.

### 4.3 A stale Pi address was never corrected

`_note_remote_client_ip()` saved the Pi's address only
`if not settings.get("pi_ip")` — once. The Pi is on DHCP and does move (it has
gone `.9` → `.11`), so a saved address went stale and **stayed** stale: the app
kept dialling `192.168.0.128`, an address on a subnet the Pi had left, while
the live address sat in the very variable being ignored.

Now persisted whenever it **changes**. `_pi_call()` also falls back to the
`pi_host` setting (default `adam-pi.local`, which avahi answers) and remembers
whichever address replied.

### 4.4 The sync token could not be set from the UI

The Clock tab's writes need the Pi's `SYNC_TOKEN`, but there was no field for
it in Settings and `/settings` did not accept the key — so enabling writes
meant hand-editing `settings.json`. The tab looked broken when it was merely
locked. Settings now has **Pi Sync Token** and **Pi Hostname (fallback)**
fields, and the backend accepts `sync_token`, `pi_host` and `pi_sync_port`.

### Also corrected

The Devices tab said *"Searching via mDNS…"* when no address was known. The Pi
**publishes no mDNS service record**, so nothing was searching and nothing ever
would. It now reads "Waiting for ADAM to connect…". The real fix is to make the
Pi advertise — see [`mobile_ble_sync.md`](mobile_ble_sync.md) §6.

## 5. The Clock tab

`pcAPP/static/js/clock.js` renders the Pi's schedules, todos and memories.

Design decisions worth keeping:

- **Polls only while visible.** A `setInterval` checks
  `#tab-clock.classList.contains('active')` before fetching, so the Pi is not
  hit every five seconds while the user looks at the 3D view.
- **Never optimistic.** Every write (`togglePiTodo`, `deletePiSchedule`) relays
  to the Pi and then re-fetches from its reply. A failed write must not leave the
  UI claiming the change landed.
- **Single-item verbs.** Ticking a to-do sends `{toggle: id}` rather than
  rewriting the whole list, so fields the display snapshot omits — a recurring
  alarm's stored time, a snooze count — can never be lost in transit.
- **Offline is a state, not an error.** `payload.status !== 'ok'` shows an
  Offline badge and a reason, the same as the catch branch. The Pi being
  unreachable is expected, not exceptional.
- **Separate from `dashboard.js`'s `setupClock()`**, which drives the header's
  local date and time. Different data, different failure modes.

## 6. Building the executable

`build_exe.py` bundles `static/` **wholesale**, so a front-end change only
reaches the shipped app after a rebuild. A successful build does not by itself
prove the new asset was bundled — verify it, because the failure is silent:

```bash
python -c "import PyInstaller.archive.readers as r; a=r.CArchiveReader('dist/ADAM.exe'); print([n for n in a.toc if 'clock.js' in str(n)])"
```

Two traps encountered doing this:

- **Windows arcnames use backslashes.** The archive key is `static\js\clock.js`,
  not `static/js/clock.js`; looking up the forward-slash form raises
  `KeyError: No entry named ... found in the archive!`.
- **The exe can be locked.** `PermissionError: [WinError 5] Access is denied:
  'dist\ADAM.exe'` means a running `ADAM.exe` holds the file, not that the build
  is broken. Close it, or from Git Bash:

```
taskkill //F //IM ADAM.exe
```

The doubled slashes are required: MSYS rewrites `/F` into a drive path
(`F:/`), which `taskkill` rejects as `Invalid argument/option`.

## 7. Files

| File | Role |
|---|---|
| `pcAPP/backend.py` | action registry, `/actions`, `/pi/*` proxy, mDNS |
| `pcAPP/static/js/clock.js` | the Clock tab |
| `pcAPP/static/js/dashboard.js` | header clock, settings, the rest of the UI |
| `pcAPP/build_exe.py` | PyInstaller packaging |
| `adam/sync_api.py` | the Pi's HTTP API |
| `adam/laptop_actions.py` | shared action manifest, parity set, coercion |
| `adam/laptop_agent_client.py` | Pi-side discovery and transport |
| `pi/docs/clipboard_and_agent.md` | the command channel in full |
| `pi/docs/scheduler.md` | what the Clock tab renders |

## 8. Desktop pairing and touch shortcuts (source update, 2026-10-07)

The data API now advertises capabilities after their process-lifetime services
start. Authenticated `POST /api/laptops/pair` and `/api/laptops/unpair` let the
desktop authorize its own generated control key through the existing Pi sync
key. See [laptop_pairing.md](laptop_pairing.md) for verification, persistence,
revocation and the single-active-laptop limit.

Authenticated `POST /api/touch/assignments` atomically saves per-pad tap, double
tap and hold bindings. `GET /api/touch/assignments` exposes the confirmed mapping
and firmware sensor labels. See [touch_controls.md](touch_controls.md) for exact
numbering, safety priority and the telemetry contract. The Pi dispatches mapped
actions over the existing laptop command channel; the PC must not execute touch
telemetry again.

These changes are covered by isolated local tests and have not been deployed or
tested on physical hardware. The mobile hardware setup path remains a separate
provisioning responsibility; Firebase account sync is not a device-control key.
