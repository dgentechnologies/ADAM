# ADAM Windows Companion App — Phase-Wise Build Plan
> Historical engineering plan. The current version0.01 behavior is documented
> in [README.md](README.md) and [UPDATE_REPORT.md](docs/UPDATE_REPORT.md).
> Current implementation uses Windows DPAPI for credentials, Firebase for
> optional account sync, explicit startup opt-in, and a portable executable;
> conflicting details below describe earlier planning decisions.

**Codename:** Antigravity Phase
**Company:** DGEN Technologies Pvt. Ltd.
**Scope:** Rebuild `laptop_agent.py` from a headless Flask script into a proper Windows desktop app — real UI, background service behavior, startup-folder install, coding-agent (Claude Code / Codex) status awareness, all wrapped in the same visual language as the ADAM mobile app (`DESIGN.md`, logo above).

---

## 0. Ground Rules (apply to every phase)

- **No Railway, no cloud service.** Everything here runs on the user's own Windows machine. Zero recurring cost to DGEN.
- **Keep the existing action-registry pattern.** `@action(...)` decorator + `ACTIONS` dict + `/actions` + `/control` stays exactly as-is — it already works and the Pi already depends on its shape. Nothing in this plan touches that contract.
- **Additive, not a rewrite of the brain.** The Flask backend becomes one thread/process inside a bigger app; it does not get replaced.
- **Design system is locked to `DESIGN.md`.** True Black (`#000000`) background, Near Black (`#0A0A0A`) / Charcoal (`#1C1C1E`) surfaces, Pure White (`#FFFFFF`) primary accent, Michroma typography, pill buttons, 24px radius cards, 1px hairline borders, halftone "digital skin" texture at 2–5% opacity. The ADAM wordmark (uploaded logo) is the app icon, splash mark, and tray icon (monochrome, inverted for light contexts if ever needed).
- **One mDNS identity.** The service name `_adam-laptop._tcp.local.` does not change — the Pi's existing discovery code keeps working with zero changes on the Pi side.

---

## Phase 1 — Foundation: Turn the Script Into an App Shell

**Goal:** `laptop_agent.py`'s Flask server keeps running exactly as it does today, but it now lives inside a real Windows application process instead of a terminal window.

**What happens:**
- Wrap the existing Flask app + mDNS broadcast in a background thread inside a desktop app shell (system tray resident, no visible window by default).
- Tray icon = ADAM wordmark mark, monochrome, matching the uploaded logo.
- Tray right-click menu: `Open Dashboard`, `Pause Agent`, `Restart Agent`, `View Logs`, `Quit`.
- All existing behavior (volume/brightness actions, `/actions`, `/control`, `/ping`, mDNS) is preserved byte-for-byte in logic — only the process wrapper changes.
- A local logging layer is added (rotating file log) since a background app has no visible console to read errors from anymore — this replaces the current `print()`-to-terminal behavior, which becomes invisible once this stops running in a terminal window.

**Exit criteria:** App launches, sits in the tray, Pi can still discover and control volume/brightness exactly as before. No dashboard yet — just proof the backend survives the move into an app shell.

---

## Phase 2 — Windows Integration: Startup, Background, Lifecycle

**Goal:** The app behaves like a proper installed Windows background utility, not a script someone has to remember to run.

**What happens:**
- Installer/setup step places a shortcut into the Windows **Startup folder** (`shell:startup`), so the app launches automatically on login — no scheduled task fuss, no admin-only service install, matches how most lightweight tray companions ship.
- App starts **minimized to tray** on launch — never pops a window on login.
- Single-instance lock: launching it twice (e.g. double-click after it's already running) just brings the existing tray/dashboard forward instead of spawning a second Flask server on the same port.
- Graceful shutdown path: `Quit` from the tray menu unregisters mDNS cleanly (already exists in the code as `stop_mdns_broadcast()`) and stops the Flask thread before exiting — same discipline the Pi side already has for its own shutdown (SIGTERM handling in `main.py`).
- Config (`AGENT_TOKEN`, `AGENT_PORT`) is no longer a bare `.env` a non-technical user has to hand-edit — Phase 3's Settings panel becomes the place to set this, with `.env` as the underlying storage so nothing about the token/port mechanism changes.

**Exit criteria:** Fresh Windows machine, run the installer once, reboot — ADAM agent is running in the tray with no manual step. Laptop restarts don't require re-launching anything.

---

## Phase 3 — The Dashboard UI

**Goal:** A real window, opened from the tray, styled exactly to `DESIGN.md`, that gives a human something to look at and control.

**Visual direction (pulling directly from DESIGN.md):**
- True Black background, halftone digital-skin texture at low opacity behind content.
- Michroma for all headings/labels; body copy in white/light-grey per the type scale already defined.
- Charcoal (`#1C1C1E`) cards, 24px radius, 1px hairline (`#2C2C2E`) borders — same card language as the mobile app.
- Pill-shaped buttons: solid white/black for primary actions, hairline-outline for secondary.
- Status dot vocabulary reused as-is: solid white = active, hollow ring = offline, pulsing white↔grey = processing. This vocabulary already exists in the design system and should mean the same thing here as it does on mobile, so a user reading either app learns one visual language.

**Dashboard layout (single window, no page navigation needed at this size):**

1. **Header band** — ADAM wordmark (top-left, small), connection status pill (top-right: "Connected to ADAM" / "Waiting for ADAM" using the same status-dot states).
2. **Centerpiece: the 3D ADAM model panel** (this is Phase 4 in detail — placeholder card here in Phase 3, wired to real state in Phase 4). Occupies the visual anchor of the window, Charcoal card, generous negative space around it per the spacing system.
3. **System status row** below/beside the model — small stat tiles (volume level, brightness level, mic/camera reachability if surfaced later) in the same card language, secondary-grey labels, white values.
4. **Coding Agent status card** (Phase 5 ties in here) — shows idle / running / needs input / done, with the pulsing-processing dot when a Claude Code or Codex job is active.
5. **Activity log panel** — recent actions ADAM has triggered on this laptop (volume changes, brightness changes, coding tasks dispatched), newest first, quiet secondary-grey text, replacing the old print-to-terminal feed with something a non-technical person in the office can actually glance at.
6. **Settings** — token, port, startup toggle, "restart agent" — pill secondary buttons, opens as an overlay/modal with the glassmorphism treatment DESIGN.md specifies for floating panels (backdrop blur 20–40px, 60% opacity dark backing).

**Exit criteria:** Opening the dashboard from the tray shows a window that looks unmistakably like it belongs to the same product family as the mobile app — same palette, same type, same shape language — and reflects live agent status, not mock data.

---

## Phase 4 — The 3D ADAM Model (Live Status Visualization)

**Goal:** The dashboard's centerpiece is a 3D representation of ADAM that visually reflects the *robot's actual current state*, not just the laptop agent's state — turning the dashboard into a real mirror of the physical unit.

**What "actual status" means here, concretely, mapped from what the Pi already tracks and already has a vocabulary for (see `hardware.py` / `session.py` / `tools_schema.py` emotion set):**
- **Emotion/face state** — the same enum ADAM's TFT face already uses (`happy, sad, surprised, angry, thinking, excited, love, blush, confused, smug, sleep, rizz, panic, shy, reconnecting`) is mirrored onto the 3D model's expression/color state, sourced from the existing `ws_server.py` WebSocket broadcast (`ws_broadcast({"type": "emotion", ...})`) that ADAM already emits on every `set_emotion()` call — this app just becomes another subscriber to that same feed, no new data path needed on the Pi.
- **Speaking / listening / idle** — mirrors `adam_speaking` / idle_mode state, same WebSocket channel.
- **Connection state** — is the Pi even reachable right now (mDNS/heartbeat), shown as the model dimming/going monochrome-static when offline vs. animated when live.
- **Head position** — optional stretch: pan/tilt angle reflected as actual model rotation, since that data already exists (`servo_pan`, `NECK_PAN_CENTER` etc.) — nice-to-have, not required for v1.

**Implementation shape:**
- A lightweight 3D viewport embedded in the dashboard window (glTF/GLB model of ADAM's physical form, rendered with a real-time 3D engine embed rather than pre-rendered video, so state changes are instant).
- Model material follows the achromatic palette — white/grey/charcoal materials, no color, consistent with "strictly monochrome palette to emphasize form" per DESIGN.md, with the *emotion state* expressed through the pulsing/status-dot vocabulary and subtle motion (idle sway, "speaking" mouth/face-panel animation) rather than through color, keeping the whole app on-brand.
- This app connects to the Pi's existing `ws://<pi-ip>:8765` face WebSocket (already built, already broadcasting) as a client — it does not need the Pi to build anything new for Phase 4 to work. This is the single biggest reason to sequence this app work after confirming that WebSocket stays alive on the Pi side; it already does.

**Exit criteria:** When ADAM (the physical robot) changes emotion, speaks, goes idle, or drops offline, the laptop dashboard's 3D model reflects it within roughly a second, with no polling delay a person would notice.

---

## Phase 5 — Coding Agent Status Addition (Claude Code / Codex)

**Goal:** Extend the existing action-registry backend so ADAM can dispatch a coding task to Claude Code or Codex CLI running on this laptop, and so the laptop agent tracks and reports task state (running / needs input / done / errored) back to ADAM and to the dashboard.

**Design, following the existing `@action` pattern exactly — no parallel system invented:**
- New endpoints alongside the existing `/actions`, `/control`, `/ping`: a small, self-contained addition, same auth token, same Flask app, same port.
- `dispatch_coding_task` — spawns the CLI (`claude -p "..." --output-format stream-json` or the Codex equivalent) as a subprocess, tagged with a task ID.
- A background watcher thread parses the CLI's structured stream-json output and classifies each task into one of four states: `running`, `needs_input` (permission prompt / clarification detected), `done`, `error`.
- `check_coding_task_status` — the Pi-side tool call ADAM already has a pattern for (mirrors `laptop_control`'s existing shape) polls this.
- When state flips to `needs_input` or `done`, the laptop agent doesn't just wait to be polled — it also pushes the change onto the same WebSocket/status channel the dashboard is already listening to (Phase 4's plumbing), so the **Coding Agent status card** in the dashboard updates live, and ADAM's Pi-side session (once its matching tool + system-text injection is added — a Pi-side change, out of scope for this laptop app but the natural next step) can voice "Claude Code needs your input" the moment it happens rather than only when asked.
- Task history (last N dispatched tasks, their outcome, duration) shown in the dashboard's Activity Log panel from Phase 3 — same card, same list, coding tasks just become another entry type alongside volume/brightness actions.

**Exit criteria:** From the dashboard (and eventually from ADAM's own voice, once the matching Pi-side tool ships), a coding task can be dispatched, its live state is visible on the laptop screen, and the moment it needs a human or finishes, that's reflected in the UI without needing to alt-tab into a terminal to check.

---

## Phase 6 — Packaging & Distribution

**Goal:** A single installer a non-engineer at DGEN can run.

**What happens:**
- Build the app into a standalone Windows executable (no "install Python first" step for end users).
- Installer handles: placing the app, writing the Startup-folder shortcut (Phase 2), first-run token generation/setup (Phase 3 settings), and an uninstall path that also cleans up the Startup shortcut and unregisters mDNS gracefully.
- App icon and installer branding use the ADAM wordmark exactly as uploaded, on the True Black background per DESIGN.md.
- Version string shown in the dashboard settings panel, for support purposes as more Founder Edition units ship.

**Exit criteria:** One `.exe`/installer, double-click, done — matches the bar of any consumer desktop companion app, appropriate for the Founder Edition units going to non-technical early adopters.

---

## Sequencing Notes

- Phases 1→3 can be built and shipped even before Phase 4/5 exist — the dashboard is useful the moment it shows real volume/brightness state and a connection pill, so this doesn't have to be one giant release.
- Phase 4 depends on nothing changing on the Pi — the WebSocket face broadcast already exists and already carries everything needed.
- Phase 5's laptop-side half is fully independent of the Pi; the Pi-side half (a new tool declaration + injection point in `tool_handler.py`/`session.py`, discussed earlier) is a separate, smaller follow-up once this app's status-tracking plumbing exists to talk to.
- Nothing in this plan requires Railway, a cloud database, or any recurring hosting cost — the entire system stays LAN-local between the Pi and this laptop app, consistent with how `laptop_agent_client.py` already discovers and talks to it today.
