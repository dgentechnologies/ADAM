# ADAM Laptop Companion App — Setup & Feature Spec
**DGEN Technologies Pvt. Ltd.** | v1.0 | For engineering handoff — build reference

---

## 0. Purpose

This document specifies the **desktop companion app** for Windows/macOS/Linux that upgrades the existing `laptop_agent.py` script into a real installable app with a UI, tray presence, and auto-discovery — this is what lets ADAM control the user's laptop (volume, brightness, and future actions) and is the desktop counterpart to the mobile app in `ADAM_Mobile_App_Setup_Spec.md`.

**Design direction:** Lightweight, mostly-invisible utility app. Lives in the system tray/menu bar. Opens a small window only when the user wants to check status or change settings — never a heavyweight always-on-screen app. Same visual language as the mobile app (dark, minimal, ADAM's accent color) so the two feel like one product.

---

## 1. What This App Replaces / Extends

Today, `laptop_agent.py` is a Flask script the user runs manually (`python laptop_agent.py`), which:
- Exposes `/actions`, `/control`, `/ping` on the LAN.
- Broadcasts itself via mDNS/Zeroconf as `_adam-laptop._tcp.local.`.
- Requires a `.env` file with `AGENT_TOKEN` + `AGENT_PORT` set up by hand.

The app wraps all of this into a real installer + background service + tray UI + first-run pairing flow, so a non-technical user never touches a terminal or `.env` file.

---

## 2. High-Level Flow Map

```
Download & Install → Launch → Sign in with Google (same account as mobile app)
  → Auto-Discover ADAM on LAN (or manual pairing code) → Pair & Generate Token
  → Permission Grants (mic/screen/accessibility as needed per OS)
  → Background Service Running (tray icon) → Settings / Dashboard (on demand)
```

---

## 3. Screen-by-Screen Flow

### 3.1 Download & Install
- Distributed as a signed installer per platform: `.exe`/`.msi` (Windows), `.dmg`/`.pkg` (macOS), `.AppImage`/`.deb` (Linux).
- Installer sets the app to launch at login by default (toggle-able later in Settings) since this is meant to run continuously in the background.

### 3.2 Launch — First Run Welcome
- Short welcome screen: "This app lets ADAM control things on your laptop — volume, brightness, and more coming soon."
- CTA: **"Sign in with Google"** — same account system as the mobile app, so this laptop is linked to the same DGEN account/device record as the user's ADAM unit.

### 3.3 Auto-Discover ADAM
- App starts its local Flask/HTTP service in the background immediately (this is the actual `laptop_agent.py` core, now bundled and managed rather than run manually) and broadcasts via mDNS, exactly as today.
- Screen shows: **"Looking for your ADAM on this network…"** with a live spinner and the ADAM face animation.
- Two outcomes:
  1. **Found automatically** (ADAM's Pi discovers this agent via the existing `_discover_laptop_agent_ip()` mDNS flow) → show "Found ADAM-<name>! Pairing…" and proceed.
  2. **Not found** → fallback screen with:
     - A manual **6-digit pairing code** displayed on this laptop's screen, which the user reads aloud to ADAM or enters in the mobile app's "Link a laptop" screen (mobile app → Settings → Connected Laptops → Add). This covers networks where mDNS is blocked by router isolation (common on guest networks/some mesh routers).
     - A manual IP entry field as a last resort, mirroring the existing `LAPTOP_AGENT_IP` env fallback already supported in `adam.py`.

### 3.4 Pairing & Token Generation
- On successful discovery/pairing, the app generates a secure random token (replaces manually editing `AGENT_TOKEN` in `.env`) and registers it with the DGEN backend against the user's account + this ADAM's device ID, and shares it with ADAM directly over the local pairing handshake — never typed by hand.
- Confirmation screen: "Paired with ADAM-<name>. He can now control volume, brightness, and more on this laptop."
- Multiple laptops can pair to the same ADAM (e.g., work laptop + personal laptop) — each gets its own token, listed and individually revocable in the mobile app's "Connected Laptops" screen and in this app's own Settings.

### 3.5 Permission Grants (OS-specific)
Different OSes require explicit user grants for the underlying actions `laptop_agent.py` already performs — surface these clearly instead of letting silent OS-level failures confuse the user later:

- **Windows:** No special permission needed for volume/brightness (`pycaw`/`screen_brightness_control` work out of the box); flag if running with restricted permissions.
- **macOS:** Requires **Accessibility** and possibly **Automation** permission grants (for `osascript` volume control) — app should detect if these are missing and deep-link directly to `System Settings → Privacy & Security → Accessibility` with a clear "Enable this so ADAM can control your volume" explainer, rather than a cryptic failure the first time ADAM tries an action.
- **Linux:** Depends on `amixer`/`screen_brightness_control` availability — app should check and report missing system dependencies plainly at setup time, not on first failed command.

### 3.6 Background Service Running
- After setup, the app minimizes to the **system tray (Windows/Linux) / menu bar (macOS)**.
- Tray icon shows connection state at a glance: green dot (ADAM connected & control active), grey (ADAM offline/not on this network), red (error — e.g., permission revoked, token invalid).
- Tray menu (right-click / click):
  - "ADAM: <name> — Connected" (status line, non-clickable)
  - "Open Dashboard"
  - "Pause laptop control" (temporarily stop responding to ADAM's commands without fully quitting)
  - "Settings"
  - "Quit"

### 3.7 Dashboard (opened on demand from tray)
Small window, single page, sections:

- **Status card**: Connected ADAM name/serial, connection quality, last command received + timestamp (e.g., "Volume set to 40% — 2 min ago") — this visibility matters so users trust what's happening on their machine.
- **Available Actions list**: Mirrors the existing `/actions` manifest — Volume Up/Down/Set/Mute/Unmute, Brightness Up/Down/Set, and any future actions (Lock Screen, Open App, Mute Spotify, etc. — per the extensibility already built into `laptop_agent.py`'s `@action()` decorator pattern). Each row shows a toggle to **enable/disable that specific action** — e.g., a user may want ADAM controlling volume but not locking their screen. This requires a small protocol addition: an `enabled_actions` allow-list checked in `/control` before dispatch.
- **Activity Log**: Last ~20 commands ADAM sent to this laptop, for transparency/debugging (e.g., "brightness_set(70) — success", "volume_mute — failed: no active audio device").
- **Manual test buttons**: Let the user trigger any action once from this UI directly (useful for verifying permissions work without needing ADAM to say it).

### 3.8 Settings
- **Account**: Signed-in Google account, sign out.
- **Connected ADAM**: Device name/serial, "Unpair this laptop" (revokes token both locally and on backend).
- **Startup**: Launch at login toggle.
- **Network**: Manual IP override field (fallback if mDNS is unreliable on this network), port override.
- **Permissions**: Re-check/re-request OS permissions (re-run the §3.5 flow on demand).
- **Notifications**: Toggle desktop notifications for actions performed (e.g., a subtle toast "ADAM set volume to 60%") — off by default to avoid being annoying, on for users who want visibility.
- **About**: App version, link to support, "Check for updates."

---

## 4. Feature Backlog Beyond Volume/Brightness (extends the existing `@action()` registry — no architecture change needed, just new decorated functions)

Design the Dashboard's Available Actions list to scale to these without a UI rework:

- Lock screen
- Open a specific app / URL
- Media controls (play/pause/skip — Spotify/system media keys)
- Take a screenshot and send it to ADAM's Gallery (mobile app tie-in)
- Shutdown/sleep/restart (with confirmation dialog — destructive action, needs its own extra-confirm layer distinct from other toggles)
- Read clipboard content aloud (ties into the `generate_to_clipboard` pattern already in ADAM's system prompt — reverse direction: ADAM reading FROM clipboard)
- "Focus mode" — mute notifications on the laptop when ADAM detects the user is deep in work (future, ties into presence/camera signals)

---

## 5. OTA / App Updates

- Standard desktop auto-update pattern (e.g., Squirrel/Electron auto-updater equivalent, or a simple "check DGEN's release manifest, download, prompt to restart" flow — consistent with the OTA approach described for the Pi/mobile side).
- Since this app is a background service, prefer **silent download + "Restart to update" tray notification** over forced interruption.

---

## 6. Security Notes for Engineering

- Token-based auth between ADAM and the laptop agent already exists (`AGENT_TOKEN` check in `/control`) — the app should generate this token cryptographically (not user-typed), rotate it on re-pairing, and store it in the OS's secure credential store (Windows Credential Manager / macOS Keychain / Linux Secret Service) rather than a plaintext `.env` file, which is the biggest security gap in the current script-based version.
- `/control` and `/actions` should remain LAN-only (no port-forwarding, no cloud relay for this specific channel) — matches the existing security note in `laptop_agent.py`'s own docstring. The app should actively warn the user if it detects the port might be exposed externally (e.g., UPnP auto-port-forward is common on consumer routers and could accidentally expose this).
- Add a rate-limit / anomaly check on `/control` (e.g., more than N action calls per second) as cheap defense-in-depth, independent of the token check.

---

## 7. Build Priority

**Must-have for launch:**
Install → Sign in → Auto-discover/pair → Tray running → Dashboard with existing volume/brightness actions → basic Settings (unpair, startup toggle).

**Fast-follow:**
Per-action enable/disable toggles, Activity Log, manual IP fallback UI, OS permission deep-links (especially macOS Accessibility).

**Later:**
Extended action set (lock screen, media controls, app launching), OTA auto-update for the app itself, multi-laptop management surfaced richly in the mobile app.
