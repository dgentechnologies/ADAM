# ADAM Pi-Side Upgrade Requirements
**Companion doc to:** `ADAM_Windows_App_PLAN.md` (Antigravity Phase — Windows Laptop App)
**Purpose:** Everything that needs to change in the Pi codebase (`MP-MC codes/pi/adam/`) for the Windows app's phases to actually work end-to-end — mainly Phase 4 (3D model status) and Phase 5 (coding agent status), since Phases 1/2/3/6 are laptop-only and need nothing from the Pi.

This is a requirements list, not code — matching how the app plan doc was written.

---

## 0. What Already Works, No Changes Needed

Worth stating plainly so nothing gets rebuilt that doesn't need to be:

- **`ws_server.py`** already broadcasts emotion/speaking state over `ws://<pi-ip>:8765`. Phase 4's 3D model panel can subscribe to this exactly as it is today. Zero Pi-side change required for basic emotion mirroring.
- **`laptop_agent_client.py`**'s mDNS discovery (`_adam-laptop._tcp.local.`), IP caching, and `/actions` manifest refresh already work and don't care whether the laptop side is a terminal script or a tray app — the wrapping done in Phase 1/2 is invisible to the Pi.
- **`tools_schema.py`**'s `build_laptop_control_declaration()` already dynamically builds the volume/brightness tool schema from whatever `/actions` returns — so *if* coding-agent actions were added as plain `@action()` entries (not a separate system), the Pi would pick them up with zero schema code changes. This matters for the design decision in section 3 below.

---

## 1. WebSocket Broadcast — Extend, Don't Replace

**Why:** Phase 4 needs more than emotion/speaking to make the 3D model feel "live," and Phase 5 needs a channel to push coding-task state changes to the dashboard without the laptop having to poll.

**What's missing today:** `ws_broadcast()` in `tool_handler.py` is only called from `set_emotion`. Nothing broadcasts:
- Idle-mode enter/exit (`idle_mode.set()`/`.clear()` in `session.py` — currently a local state change with no outward signal)
- `adam_speaking` transitions (the speaker task already has the moment `adam_speaking.set()`/`.clear()` fires — it just never tells `ws_broadcast`)
- Connection/reachability heartbeat (there's `heartbeat.py` writing to `/dev/shm` for the local watchdog, but nothing pushes "I'm alive" externally over the WebSocket — a laptop app has no equivalent of reading `/dev/shm` on the Pi)
- Head pan/tilt angle (optional, for the stretch goal of rotating the 3D model to match)

**Upgrade required:**
- Add `ws_broadcast({"type": "speaking", "active": bool})` calls at the two existing `adam_speaking.set()`/`.clear()` sites in `session.py`'s `speaker()`/`receive()` tasks.
- Add `ws_broadcast({"type": "idle", "active": bool})` at the existing idle-mode transition points (there are several — STOP gesture, voice request, wake word, `IDLE_MAX_S` timeout — all already funnel through `_idle_mode_persistent[0]` toggles, so this is one helper function called from each site, not new logic).
- Add a periodic low-frequency broadcast (`{"type": "heartbeat", "ts": ...}`) — a few-second interval task, similar in shape to the existing `laptop_agent_healthcheck()` task, so a laptop app watching the socket can tell "Pi is alive but quiet" apart from "Pi is unreachable."
- (Optional, Phase 4 stretch) broadcast pan angle from `camera()`'s existing `_last_commanded_pan[0]` whenever it changes.

**Sizing:** small — this is additive calls into an already-working broadcast function, not new infrastructure.

---

## 2. Reachability From Outside the LAN's Happy Path

**Why:** the dashboard's "Connected to ADAM / Waiting for ADAM" status pill (Phase 3) needs to distinguish *Pi is off*, *Pi is on but WebSocket server didn't start*, and *Pi is on and streaming* — three different states that currently only distinguish as "did the socket connect or not."

**What's missing today:** `ws_server.py` degrades silently to a no-op if the `websockets` package or port isn't available (`except Exception: print(...); return None`) — correct for not crashing the robot, but it means a laptop app has no way to tell "ADAM chose not to run the face server" apart from "ADAM is unreachable." Same story for `esp_link.connected` (UART-only, not exposed anywhere off-Pi).

**Upgrade required:**
- No behavior change to the degrade-gracefully logic — just make sure `start_ws_server()`'s success/failure is itself one of the pieces of state a laptop could learn about, most simply by folding it into the heartbeat payload from section 1 (`{"type": "heartbeat", "ws_ok": true, "esp_link_connected": true, ...}`) rather than building a second reporting path.

**Sizing:** trivial — piggybacks on the heartbeat broadcast already being added in section 1.

---

## 3. Coding-Agent Tool — the Actual New Surface

**Why:** this is the one genuinely new capability. Everything else in this doc is "expose more of what already exists"; this is new.

**Design decision, stated plainly:** the laptop-side coding-agent dispatch (Phase 5 of the app plan) was designed to reuse the existing `@action()` registry in `laptop_agent.py` rather than invent a parallel system. That decision has a direct, and favorable, consequence for the Pi side: **if coding-agent actions are registered as ordinary `@action()` entries** (e.g. `dispatch_coding_task`, `check_coding_task_status`), then `tools_schema.py`'s existing `build_laptop_control_declaration()` already builds them into the Gemini tool schema automatically, through the exact same `laptop_control` tool ADAM already calls for volume/brightness. In that case, **no new FunctionDeclaration and no new tool name are strictly required** — `laptop_control(action="dispatch_coding_task", value=...)` would just work today, mechanically, the moment the laptop agent registers those actions.

That said, two things push toward a small amount of dedicated Pi-side work rather than relying on the generic path as-is:

### 3a. `laptop_control`'s `value` parameter is a bare integer — coding tasks need a string prompt
**What's missing:** `tools_schema.py`'s `laptop_control` declaration types `value` as `S(type=T.INTEGER, ...)`. A coding task dispatch needs to pass a free-text prompt (and a project path), which doesn't fit an integer field.

**Upgrade required:** either (a) widen `laptop_control`'s `value` field to accept a string as well as an int (a small schema change, backward compatible since existing volume/brightness actions still pass ints), or (b) add a dedicated `dispatch_coding_task` FunctionDeclaration in `tools_schema.py` with proper `prompt: string` and `project_path: string` fields instead of overloading `value`. **(b) is the cleaner path** — coding dispatch is a different enough shape of call (a prompt, not a percentage) that giving it its own declaration avoids stretching `laptop_control`'s contract, and keeps the volume/brightness action list's simplicity intact for what it was designed for.

### 3b. System-text injection when a task needs input or completes
**Why:** this is the actual "ADAM tells the user" behavior discussed earlier — status sitting in the laptop dashboard alone doesn't get spoken. ADAM's existing pattern for "something happened in the background, tell the user about it right now" is the `inject()` helper in `session.py`, already used for GoAway handling, gesture reactions, and idle nudges.

**Upgrade required:**
- A new background-watching task in `session.py`, same shape as the existing `laptop_agent_healthcheck()` task — polls (or, once section 1's heartbeat/WebSocket plumbing is in place, subscribes to) the laptop agent's coding-task status.
- On a state transition to `needs_input` or `done`, call the existing `inject()` with a `[SYSTEM: ...]` message, exactly the same mechanism already used for `"[SYSTEM: User pressed STOP...]"` and idle nudges — no new injection mechanism needed, just a new caller of the one that exists.
- A mailbox flag pair in `tool_handler.py`, matching the existing `_play_song_requested` / `_idle_mode_requested` single-element-list pattern, so the module-level tool handler (which dispatches `dispatch_coding_task`) can signal the session-local watcher without a rebind.

### 3c. Voicing a "needs input" question and relaying the answer back
**Why:** the harder half of status-awareness is round-tripping — ADAM says "Claude Code is asking: should I overwrite the existing config?", the user answers out loud, and that answer needs to reach the CLI subprocess's stdin on the laptop, not just get spoken into the void.

**Upgrade required:** this needs a small conversational-state addition roughly parallel to how `idle_mode` or `song_playing` are tracked today — a `coding_task_awaiting_input` flag that, while set, routes the next user utterance (from the normal Gemini Live turn, not a special tool call) into a `respond_to_coding_task` action call instead of a normal spoken reply, then clears itself once the laptop agent confirms the input was delivered. This is the one piece of new *conversational logic* in this whole doc — everything else in this section is wiring existing patterns to a new destination; this is a genuinely new turn-taking behavior and deserves its own design pass before being built, since getting the boundary wrong (deciding whether an utterance is "answering Claude Code" vs. "just talking to ADAM about something else while a task happens to be pending in the background") is a real ambiguity, not a mechanical one.

**Sizing:** 3a/3b are small, following established patterns closely. 3c is the one item in this whole document that isn't just "reuse an existing mechanism for a new source" — it's worth scoping and reviewing on its own before implementation, most likely with an explicit trigger phrase or touch gesture to disambiguate "this is for Claude Code" from ordinary conversation, rather than silently intercepting the next utterance.

---

## 4. Nothing Required for Phases 1, 2, 3, or 6

Worth stating explicitly, since it bounds the scope of this document: the app-shell wrapping, Windows Startup-folder install, dashboard UI/styling, and installer packaging are entirely laptop-side concerns. They read from the Pi (via the WebSocket and the existing `/actions`/`/control`/`/ping` HTTP surface) but require no Pi code changes beyond what's listed above. A person could build and ship those four phases against the Pi exactly as it exists today.

---

## Summary Table

| Area | Change | New mechanism? | Size |
|---|---|---|---|
| Emotion broadcast | none — already exists | no | — |
| Speaking-state broadcast | add 2 `ws_broadcast()` calls | no (reuses `ws_broadcast`) | trivial |
| Idle-state broadcast | add `ws_broadcast()` calls at existing transition points | no | small |
| Heartbeat/reachability broadcast | new periodic task | new task, existing pattern (`laptop_agent_healthcheck`-shaped) | small |
| `ws_ok`/`esp_link` status | fold into heartbeat payload | no | trivial |
| Coding task dispatch schema | new `dispatch_coding_task` FunctionDeclaration | new declaration, existing dispatch pattern | small |
| Coding task status → spoken injection | new watcher task + `inject()` calls | no (reuses `inject()`) | small |
| Coding task mailbox flags | new `_coding_task_*` single-element lists | no (existing mailbox pattern) | trivial |
| Voice answer routed to Claude Code | new turn-interception logic while `coding_task_awaiting_input` | **yes — new conversational behavior** | needs its own design pass |
