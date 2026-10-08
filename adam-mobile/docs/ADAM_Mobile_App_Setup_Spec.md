# ADAM Companion App — Mobile Setup & Feature Spec
**DGEN Technologies Pvt. Ltd.** | v1.0 | For engineering handoff — build reference

---

## 0. Purpose

This document defines the complete first-run setup flow and ongoing feature set for the **ADAM Companion App** (iOS + Android). It is written so design/dev can start wireframing and building screens directly from it. Every screen below should be treated as a minimum spec — copy/visuals can be refined, but the flow order, data captured, and states handled should not be skipped.

**Design direction:** Premium, minimal, dark-first UI (matches ADAM's TFT face aesthetic — black background, single accent color, generous whitespace). Feels like setting up an Apple/Sonos device, not filling out a form. Every step should have a reason ADAM gives in his own voice (first-person, deadpan, GenZ-confident — matches the established Instagram voice), not just generic instructional copy.

---

## 1. High-Level Flow Map

```
Splash → Account (Sign in with Google) → Find My ADAM (BLE/Wi-Fi pairing)
  → Wi-Fi Credential Handoff → Device Claim & Naming
  → AI Brain Setup (BYOK / Managed Credits / Skip → Lite Mode)
  → Personality & Voice Preferences → Face/Camera Permission (recognize me)
  → Home Integrations (optional, can skip) → Tutorial / First Conversation
  → Home Dashboard (ongoing app)
```

Total forced steps to a *working* ADAM: **Account → Pairing → Wi-Fi → Naming → AI Brain choice**. Everything else is skippable and revisitable later from Settings. Never block a user from finishing setup because of an optional step — Lite Mode must always be reachable.

---

## 2. Screen-by-Screen Flow

### 2.1 Splash / Welcome
- DGEN logo animation → ADAM logo/face animation (idle blink).
- Single CTA: **"Set up my ADAM"**.
- Secondary link: "I already have an ADAM set up" → jumps straight to sign-in → Home Dashboard (for a second phone/re-install case).

### 2.2 Account — Sign in with Google
- **Primary auth: Google Sign-In** (OAuth). This is the account that:
  - Owns the device record in DGEN's backend (Supabase).
  - Is used for managed AI credits / Razorpay purchase history.
  - Is used for OTA update opt-in and push notifications.
- Secondary option: Email + OTP (for users who decline Google, or corporate/B2B buyers without a personal Google account).
- Screen copy (ADAM voice): *"Before we begin — who am I working for?"*
- Legal: Terms of Service + Privacy Policy checkbox (required), short one-liner about what data leaves the device (see §6).

### 2.3 Find My ADAM — Device Discovery
- Prompt: **"Power on ADAM and wait for the eyes to open."** Show a short looping video/animation of the physical power-on sequence and the boot face.
- Two discovery paths, tried in order:
  1. **BLE**: Pi Zero 2W (or ESP32-CAM, whichever holds BLE role) advertises a provisioning service on first boot / after factory reset. App scans and lists nearby ADAM units by serial number (printed on the base + on the box).
  2. **Fallback — ADAM's own hotspot**: If BLE isn't available/reliable on that hardware rev, ADAM boots into a temporary Wi-Fi AP (`ADAM-Setup-XXXX`). App instructs user to manually join that network via OS Wi-Fi settings (with a deep-link button that opens Wi-Fi settings on Android; iOS requires manual join due to platform restriction — show clear numbered steps + the exact SSID/password to copy).
- Screen shows a live "Searching…" state with the ADAM face animation "looking around."
- On found: show device serial + a photo/icon, confirm **"Is this your ADAM?"**

### 2.4 Wi-Fi Credential Handoff
- App shows the phone's currently-known Wi-Fi networks (reads from a manual list — OS APIs don't expose passwords, so the user picks their home network from a scan list and types the password once).
- Data is sent to ADAM over the temporary BLE/local-AP channel only (never over the internet in plaintext) — encrypted handshake, then ADAM connects to real Wi-Fi and drops its provisioning AP.
- Screen shows connection progress: `Sending credentials… → ADAM connecting… → Confirmed online ✅`
- Failure states to design explicitly:
  - Wrong password → clear retry, no cryptic error.
  - 2.4GHz-only hardware note (Pi Zero 2W Wi-Fi is 2.4GHz) — if user selects a 5GHz-only SSID, warn *before* attempting: "ADAM's Wi-Fi radio only supports 2.4GHz networks — pick the 2.4GHz version of your network if you have both."
  - Timeout after 60s → offer retry or "Start over."

### 2.5 Device Claim & Naming
- Once online, app fetches the device's serial + hardware batch info from the backend and links it to the signed-in account.
- User names their unit (default suggestion: "ADAM"). This name is used in voice wake context and app UI, not necessarily the wake word itself (wake word stays "ADAM" / "hey ADAM" for the voice model unless a future custom-wake-word feature ships).
- If this is a **Founder Edition unit (#001–#010)**, show a special congratulatory screen with the serial number and a note about the Founder perks (Discord badge, lifetime credit priority, etc.) — pulled from the backend's batch metadata.

### 2.6 AI Brain Setup — the BYOK / Managed / Lite decision
This is the most important screen in the whole flow — presented clearly, no dark patterns, real explanation of trade-offs.

**Three cards, side by side or stacked:**

| Option | What happens | Cost |
|---|---|---|
| **Bring Your Own Key (recommended)** | User pastes/creates a free Google Gemini API key. ADAM uses it directly — DGEN never sees the key's usage. | Free (Google's own free tier) |
| **DGEN Managed Credits** | DGEN handles the AI connection via ephemeral tokens; billed against a purchased credit pack. | ₹599–₹11,999 packs |
| **Skip for now (Lite Mode)** | ADAM works with clock, alarms, smart home, pre-recorded responses, local face recognition — but no live conversation. | Free |

- **BYOK flow:** In-app guided steps (not just a link):
  1. "Tap below to open Google AI Studio and create a free key" (deep link to `aistudio.google.com/apikey`).
  2. User copies the key, returns to app (auto-detect clipboard copy of an API-key-shaped string and offer "Paste key we found?" as a convenience, with explicit user confirmation before it's used).
  3. Key is sent to ADAM over the local network only (device-to-device or via the backend as an encrypted pass-through — key is stored encrypted on the Pi, not required to persist server-side) and validated with a live test call. Show a green check on success.
  4. Short explainer screen: "Your key, your quota, your data goes to Google under your own account — DGEN never touches it."
- **Managed Credits flow:** Show pack options (Trial ₹599 / Starter ₹1,499 / Standard ₹2,999 / Value ₹5,499 / Pro ₹11,999) with estimated active-minutes for each, Razorpay checkout embedded, confirmation screen, auto-provision ephemeral token backend for this device.
- **Skip flow:** Explain Lite Mode plainly, and place a persistent (but non-nagging) "Unlock full AI" entry point in the app's Home Dashboard for later.
- This choice can be changed anytime later in **Settings → AI Brain**.

### 2.7 Personality & Voice Preferences
- Voice selection (if/when multiple TTS voices are supported — currently "Charon"; design the picker to scale to more voices later).
- Language preference (English / Hindi / Bengali / Hinglish auto-detect) — sets a hint for ADAM's language-matching behavior, not a hard lock.
- Personality intensity slider (optional, future-facing): Sarcasm level, from "dry professional" to "full roast mode." Ships with a sensible default even if not launched day one — reserve the UI slot.
- Wake sensitivity: default / more sensitive / less sensitive (maps to attention/VAD thresholds).

### 2.8 Face & People Recognition Permission
- Explain camera use plainly: "ADAM's camera helps him recognize you and react to your expressions. Frames are processed locally on the device — nothing is streamed or stored unless you explicitly save a face."
- CTA: **"Let ADAM meet you"** — triggers an in-app or on-device guided "look at the camera" moment, confirms detection, prompts for the user's name to associate with the face (`remember_person` tool equivalent).
- Skippable — camera/face features degrade gracefully if declined (ADAM just won't personalize by face).

### 2.9 Home Integrations (Optional — Smart Home Setup)
- Card grid: **Smart Lights / Smart Plugs / Smart Fan / (More coming soon)**.
- Each card opens a standard "Add Device" flow (local MQTT broker discovery on the LAN, or vendor-specific pairing if using a cloud API like a Tuya/Google Home bridge — to be decided per integration partner).
- Explicitly skippable with **"I'll do this later"** — routes to Home Dashboard → Integrations tab.

### 2.10 Tutorial / First Conversation
- Short 3–4 card swipe tutorial: "Just talk to him", "Touch his cheek if he's annoying you", "Say 'ADAM, stay quiet' to go idle", "Ask him to sing."
- Ends with a live mic-open moment in-app (or a nudge to just talk to the physical unit) so the very first thing that happens post-setup is ADAM actually responding — the emotional payoff moment. Don't skip this; it's the "wow" beat that converts a setup into a fan.

### 2.11 Home Dashboard (ongoing app, post-setup)
Bottom tab bar, 4–5 tabs:

1. **Home** — ADAM's current status (online/offline, current emotion face live-mirrored, battery/power state if applicable, quick actions: mute, put to sleep, wake).
2. **Gallery** — Photos ADAM has captured (see §3).
3. **Smart Home** — Manage connected devices, scenes, routines (see §4).
4. **Memory** — View/edit what ADAM remembers (people, saved facts) — surfaces `save_memory`/`remember_person` data with edit/delete controls. Transparency matters here; users should be able to see and delete anything ADAM "knows" about them.
5. **Settings** — Account, AI Brain (BYOK/credits/usage), Wi-Fi, Voice/Personality, Notifications, OTA updates, About/Support, Factory reset.

---

## 3. Gallery Feature (Photos ADAM Has Taken)

- ADAM's ESP32-CAM can capture stills (on request, on face-recognition events, or periodic "moments" if that feature ships).
- App syncs these from local storage on the Pi (over LAN when on the same network) or via the backend if cloud-backup is enabled (opt-in, off by default — respects the "data stays local unless you choose otherwise" principle established in the billing doc).
- Basic gallery grid, tap to view full-size, share, delete. Group by date. Optional "starred" moments.
- Storage note for engineering: define a retention policy default (e.g., 30 days rolling on-device, longer if cloud-backup opted in) and surface it in Settings.

## 4. Smart Home Tab

- **Add Device** flow per category: Lights, Plugs, Fans, (extensible list — designed so adding a new category later is just a new card, not a re-architecture).
- **Scenes**: user-defined groups of actions (e.g., "Movie Night" = dim lights + fan on low), triggerable by voice through ADAM or by tapping in-app.
- **Routines**: time or presence-based automation (e.g., "Turn off everything when ADAM detects no one home for 30 min" — ties into RCWL-0516 presence sensing).
- Backend: local MQTT broker on the home network is the default transport (matches the existing DGEN architecture used for ADAM's own smart-home tool calls) — cloud bridges are a stretch goal per platform (Google Home / Alexa interoperability) not required for v1.

## 5. OTA Software Updates

- Settings → **Software Update** screen: shows current firmware/software version running on the Pi, available update (if any), changelog (pulled from a DGEN-hosted release feed), and a single **"Update Now"** button.
- Update mechanics (engineering notes):
  - Pi checks a DGEN-hosted manifest endpoint (version, changelog, signed package URL) on a schedule and on manual "Check for updates" tap.
  - Download package, verify signature, apply via a staged/atomic swap (avoid bricking on power loss mid-update — matches the atomic-write philosophy already used for `adam_memory.json`/`adam_conversations.json` in the codebase).
  - Show progress in-app (downloading → verifying → installing → rebooting → back online) since a full OTA may take a few minutes and the physical robot will visibly restart.
  - Auto-update toggle (default: notify-only, not silent auto-install, until OTA has a proven track record — safety-critical for a robot with moving servo parts).
  - Rollback: keep the previous known-good version so a failed/bad update can auto-revert (A/B partition style, or a simple "last known good" snapshot if full A/B imaging is out of scope for v1).

## 6. Data & Privacy Notes (surface plainly in-app, not just in ToS)

- BYOK: API key stored encrypted on-device; DGEN backend never needs to see it for BYOK mode.
- Camera frames: processed on-device by default; nothing streamed to DGEN unless the user opts into cloud photo backup.
- Conversation audio: only leaves the device to Google's Gemini Live API (per the user's own key, in BYOK mode) or via DGEN's ephemeral-token relay (managed mode) — never stored raw by DGEN beyond what's needed for the managed-tier session.
- Memory (`adam_memory.json`, faces, conversation history) lives on the device; the app reads/writes it over the local network or a synced backend copy if the user wants cross-device access to their Memory tab.

## 7. Error / Edge States Checklist (design must cover all of these)

- ADAM already claimed by another account (found via BLE but backend says it's owned by someone else) → block with a clear "This ADAM is already set up on another account. Contact support if this is a mistake."
- Wi-Fi network requires a captive portal / hotel-style login → out of scope for v1, show a clear "This type of network isn't supported yet — try a home Wi-Fi network."
- Backend unreachable during setup (no internet on phone) → allow local-only BLE pairing + Wi-Fi handoff to proceed, defer account linking until connectivity returns.
- User closes the app mid-setup → resume exactly where they left off next launch (persist setup-state locally, not just in memory).
- API key invalid/revoked later (post-setup) → push notification + Home Dashboard banner: "ADAM's brain disconnected — tap to reconnect", not a silent failure.
- Factory reset → confirmation dialog with consequence explained ("This clears ADAM's Wi-Fi, memory, and API key. You'll need to set him up again.") before executing.

---

## 8. Build Priority (suggested phasing for v1 app)

**Must-have for launch (Batch 1 Founders):**
Account → Pairing → Wi-Fi → Naming → AI Brain (BYOK + Skip only, Managed Credits can follow) → basic Home Dashboard → Settings (Wi-Fi/AI Brain/Factory Reset).

**Fast-follow (within first 1–2 months):**
Face recognition setup, Gallery, Managed Credits + Razorpay, OTA updates.

**Later:**
Smart Home tab, Scenes/Routines, personality sliders, multi-voice picker, cloud photo backup.
