# ADAM — Production Setup, Operating Modes & Tasks Left

**Document Purpose:** Comprehensive reference document synthesizing the current production setup status, multi-mode operational architecture (including Lite Mode, Managed Credits, BYOK, and Audio-Only), and the prioritized roadmap of tasks left to achieve complete production readiness for ADAM on Raspberry Pi Zero 2 W.

**Target Path:** `MP-MC codes/pi/docs/tasks_left.md`  
**Related Guides:**
- [`setup.md`](file:///d:/Dgen%20Technologies%20Pvt.%20Ltd/ADAM/MP-MC%20codes/pi/docs/setup.md) — Tested bring-up guide for Pi Zero 2 W (Debian 13 Trixie)
- [`ADAM_Mobile_App_Setup_Spec.md`](file:///d:/Dgen%20Technologies%20Pvt.%20Ltd/ADAM/mobileAPP/docs/ADAM_Mobile_App_Setup_Spec.md) — Mobile onboarding & Lite Mode spec
- [`ADAM_App_Technical_Build_Spec.md`](file:///d:/Dgen%20Technologies%20Pvt.%20Ltd/ADAM/mobileAPP/docs/ADAM_App_Technical_Build_Spec.md) — Mobile app & Capacitor build architecture

---

## 1. Production Architecture Overview

The production system runs on a **Raspberry Pi Zero 2 W** (512MB RAM, Quad-core ARM Cortex-A53 @ 1GHz) running Debian 13 (Trixie) 64-bit in headless console mode (`multi-user.target`).

```
                              ┌──────────────────────── Raspberry Pi Zero 2 W ────────────────────────┐
   INMP441 x2 (I2S)  ───────▶ │  arecord (plughw:sndrpigooglevoi,0 · S32_LE 48k 2ch)                   │
                              │        │                                                               │
                              │        ▼   audio_utils DSP (S32→S16 mono 16k, GCC-PHAT DOA)            │
   MAX98357A Amp (I2S) ◀────── │  aplay (plughw:sndrpigooglevoi,0 · S16_LE 48k 2ch)                    │
                              │        ▲                                                               │
                              │   ┌────┴──── session.py (asyncio tasks) ───────────────────────────────┤
                              │   │         ├── google-genai ⇄ Gemini Live API (Full AI Mode)          │
                              │   │         └── Local Intent Engine (Lite / Offline Mode) [PENDING]    │
                              │   │                                                                    │
   Pan Servo (PWM)     ◀────── │  hardware.servo_pan() (GPIO 12 · 0.6s hold + auto-detach)             │
                              │                                                                        │
   ESP32-CAM           ◀─UART─▶ │  esp32_link.py (/dev/serial0 @ 921600 via PL011 ttyAMA0)              │
      │                       │     • RECV: 'F'+JPEG (Vision), 'T' (Touch 1-4), 'G' (Gestures)         │
      └─▶ RP2040 Pico         │     • SEND: "EMO:<face>", "TILT:<deg>", "CAM:ON/OFF"                   │
         (TFT Face + Tilt)    └────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Operating Modes Matrix

ADAM is designed to operate under multiple software and hardware states depending on network connectivity, account credentials, and hardware availability:

| Mode | Trigger / Condition | Cloud API Required? | Vision / Camera | Voice / TTS | Movement & Face | Description & Capabilities |
|---|---|---|---|---|---|---|
| **0. BLE Setup / Provisioning** | First boot or factory reset (unconfigured Wi-Fi) | **No** (Local BLE / SoftAP) | Off | Silent / Chime | Searching eyes pulse (`blink`) | Initial onboarding mode: advertises `ADAM-XXXX` over BLE or `ADAM-Setup-XXXX` Wi-Fi SoftAP for mobile app discovery and credential handoff. |
| **1. Full AI (BYOK)** | User provides Gemini API Key in `.env` or companion app | **Yes** (Gemini Live WebSocket) | Active (1 frame / 2s duty cycle) | Full bidirectional streaming voice (Charon) | Pan/tilt tracking, live TFT emotions | Default live experience: conversation, web search, tool execution, memory, pop-culture banter. |
| **2. Managed Credits** | Device authenticated via DGEN backend token | **Yes** (Gemini Live via ephemeral tokens) | Active (duty-cycled) | Full bidirectional streaming voice | Full actuation & TFT face | Same as BYOK, but uses session tokens issued by `apps/api` with credit balance metering. |
| **3. Lite Mode (Offline / Free)** | User skips AI Key during setup OR no internet/quota | **No** (Local execution only) | Optional (local presence / face detect) | Pre-recorded audio clips / local chime | Idle animations, clock face, basic gestures | Offline desktop buddy: digital clock, alarms, timer, local touch reactions, smart home relay. |
| **4. Audio-Only Fallback** | ESP32-CAM disconnected / UART inactive | **Yes** (Gemini Live) | Disabled | Full bidirectional voice | Direct Pan servo only (no tilt/TFT) | Automatic degradation when camera/display UART fails. Conversation and audio tools continue uninterrupted. |
| **5. Idle / Sleep Mode** | User says *"be quiet"*, taps Touch3, or idle timeout | **No** (Vosk local wake word only) | Off (`CAM:OFF`) | Silent | Resting face (`sleep`), center neck | Low power, mic muted to cloud. Offline Vosk listens for *"adam"* wake word to resume. |
| **6. Concert / Song Mode** | User asks ADAM to sing or perform | Partial (triggered via tool call) | Active | Local WAV playback (`song1-3.wav`) | Reaction dance / nod | Plays 48kHz stereo WAVs directly into `aplay` stream; temporarily mutes mic to prevent feedback loop. |

---

## 3. Production Readiness: What is Completed vs Missing

### A. Verified & Working (Completed)
- [x] **VoiceHAT Audio Pipeline:** Native ALSA `arecord` / `aplay` pipelines mapped by stable card name (`plughw:sndrpigooglevoi,0`).
- [x] **DSP Anti-Aliasing & Filtering:** 120 Hz high-pass (de-rumble) and 6.8 kHz FIR low-pass (anti-alias) preventing decimation aliasing.
- [x] **VAD Hysteresis & Echo Guard:** Calibrated Schmitt trigger (`MIC_SILENCE_FLOOR=1800`, onset confirmation N=2, post-mute echo guard) eliminating false triggering on room hiss and speaker echo.
- [x] **Speaker Idle Release:** Closing `aplay` after 2.5s of silence (`SPEAKER_IDLE_CLOSE_S`) to shut off Class-D amplifier switching noise that deafened microphones.
- [x] **Pan Servo Noise Elimination:** GPIO 12 PWM driving with 0.6s hold timer and auto-detaching to stop 50Hz coil hum from vibrating microphones.
- [x] **High-Speed UART Link:** Reliable 921600 baud serial on dedicated PL011 (`ttyAMA0` via `disable-bt`) with separate non-blocking write daemon.
- [x] **Offline Wake-Word Preloading:** Vosk model preloaded once at boot to eliminate runtime memory fragmentation and prevent Pi Zero 2W OOM brownouts.
- [x] **Headless Resource Optimization:** Multi-user console boot (~140MB active RSS out of 415MB RAM), OS package bloat purged.
- [x] **Systemd Auto-Restart:** Service unit `adam.service` with graceful SIGTERM handler to park servos, save memory, and shut off camera before exit.

---

## 4. Comprehensive Checklist of Tasks Left

### Category 1: Lite Mode Implementation (High Priority)
Lite Mode is specified in `ADAM_Mobile_App_Setup_Spec.md` (§2.6) as the zero-cost offline state, but currently the Pi code loops attempting to connect to Gemini Live and exits if no API key is present.

- [ ] **1.1. Create `lite_mode.py` Fallback Engine:**
  - Build a lightweight local state machine that runs when `GEMINI_API_KEY` is missing or when network connectivity is lost.
  - Implement basic rule-based reactions for:
    - Time/Date announcements (using local system clock).
    - Touch reactions (e.g., Touch 1/2 slap -> display angry face + servo recoil; Touch 3/4 pet -> display happy face + gentle nod).
    - Offline timer and alarm notifications.
- [ ] **1.2. Pre-recorded Voice / Sound Packs for Lite Mode:**
  - Add short pre-rendered audio feedback clips for common interactions (e.g., *"Online"*, *"Good morning"*, *"Alarm ringing"*, chime beeps).
- [ ] **1.3. Seamless Mode Transition:**
  - Allow runtime switching between Lite Mode and Full AI Mode without restarting the systemd service when an API key is provided via the mobile app.

---

### Category 2: BLE Mode, Provisioning & App Communication Protocol (High Priority)
Production units must allow out-of-the-box pairing with the **ADAM Companion App** without requiring SSH, HDMI monitors, or manual `.env` file editing.

#### 2.1. Device Naming & Identity Specification
* **Hardware Serial Number:** `DGEN-ADAM-XXXX` (e.g., `DGEN-ADAM-0007`), laser-engraved on the base and printed on the box packaging. Founder Edition units are `DGEN-ADAM-0001` through `DGEN-ADAM-0010`.
* **BLE Advertisement Name:** `ADAM-[0-9A-F]{4}` (e.g., `ADAM-3F2A`), matching `DeviceShortId` in the mobile app types (`mobileAPP/packages/types/src/common.ts`). The 4-character suffix is derived from the last 2 bytes of the unit's hardware MAC address.
* **Wi-Fi Hotspot Fallback SSID:** `ADAM-Setup-[0-9A-F]{4}` (e.g., `ADAM-Setup-3F2A`), matching `SetupSsid`. Broadcast as a 2.4GHz SoftAP if BLE is unavailable or pairing fails after 60s.

#### 2.2. Hardware BLE Controller Architecture
* **The Pi Zero 2 W vs. ESP32-CAM Trade-off:**
  * In production, the Pi Zero 2 W has `dtoverlay=disable-bt` in `/boot/firmware/config.txt` to assign the high-accuracy PL011 hardware UART (`ttyAMA0`) to `/dev/serial0` at 921600 baud for camera streaming and motor control.
  * **Recommended Solution (ESP32-CAM BLE Host):** The ESP32-CAM holds the BLE Peripheral role on initial unprovisioned boot. It advertises `ADAM-XXXX`, receives the GATT pairing payload from the mobile app, and relays it to the Pi over UART:
    * `PROV:WIFI:<ssid>:<password>`
    * `PROV:KEY:<gemini_api_key>`
    * `PROV:MODE:<byok|managed|lite>`
  * **Alternative Solution (Pi-Side BLE):** Use `miniuart-bt` overlay to keep Bluetooth active on the Pi while locking core frequency, running `python-bluezero` / BlueZ GATT service on Linux.

#### 2.3. BLE GATT Service & Characteristic Specifications
* **Custom Primary Service UUID:** `19B10000-E8F2-537E-4F6C-D104768A1214` (ADAM Provisioning Service)
* **GATT Characteristics Table:**

| Characteristic | UUID | Properties | Format / Payload | Description |
|---|---|---|---|---|
| **Device Name** | `0x2A00` | Read | UTF-8 String (e.g. `ADAM-3F2A`) | Standard Bluetooth Device Name characteristic. |
| **Wi-Fi Provisioning** | `0xAD01` | Write | JSON: `{"ssid": "Home-WiFi", "password": "secret"}` | Mobile app writes the target 2.4GHz Wi-Fi credentials. |
| **Connection Status** | `0xAD02` | Read, Notify | JSON: `{"status": "connecting\|connected\|failed", "ip": "192.168.1.50", "err": ""}` | Emits progress events to app: `validating` → `connected` with assigned LAN IP. |
| **Pairing Nonce / Proof** | `0xAD03` | Read | JSON: `{"serial": "DGEN-ADAM-0007", "nonce": "8f3b2a1c"}` | One-time proof-of-possession token required by mobile app for `POST /devices/claim`. |
| **AI Brain Config** | `0xAD04` | Write | JSON: `{"mode": "byok\|managed\|lite", "key": "AIzaSy..."}` | Transmits BYOK Gemini key or sets Managed/Lite mode locally on device. |
| **Location Handoff** | `0xAD05` | Write | JSON: `{"city": "Kolkata", "region": "WB", "country": "IN", "lat": 22.57, "lon": 88.36}` | Initial user location sync during onboarding for localized search context. |

#### 2.4. Post-Provisioning Local LAN Communication (App ⇄ ADAM)
Once ADAM joins the home Wi-Fi and drops BLE provisioning:
* **mDNS / Zeroconf Discovery:**
  * ADAM broadcasts service `_adam._tcp.local.` on port `8765` (hostname `adam.local`).
  * The mobile app uses Zeroconf/NSD to discover ADAM's local IP address on the home subnet without user configuration.
* **Bi-directional WebSocket Telemetry (`ws://adam.local:8765` via `ws_server.py`):**
  * **ADAM → App (State Mirroring):** Live face emotion updates (`{"type": "emotion", "value": "happy"}`), speaking indicator, and system vitals (CPU temp, RAM usage, Wi-Fi RSSI).
  * **App → ADAM (Remote Actions):** Put to sleep (`{"cmd": "sleep"}`), wake up (`{"cmd": "wake"}`), volume adjustment (`{"cmd": "volume", "level": 80}`), mode switching (`{"cmd": "set_brain", "mode": "lite"}`), and dynamic location updates.
* **Local REST Endpoints (Lightweight HTTP Server on Port 8765):**
  * `GET /api/status`: Device health, firmware version, and active AI mode.
  * `POST /api/location`: Dynamic phone location update whenever user travels or moves networks.
  * `GET /api/memories` & `DELETE /api/memories/:id`: Local memory inspection and deletion for user privacy.
  * `POST /api/wifi/reconfigure`: Allows switching Wi-Fi networks from app settings without a full hardware factory reset.

#### 2.5. Tasks Left for Provisioning & App Communication:
- [ ] **2.5.1. Implement BLE GATT Provisioning Firmware:**
  - Build the BLE peripheral service inside `esp32_cam.ino` (or Pi BlueZ service) with service UUID `19B10000-E8F2-537E-4F6C-D104768A1214` and characteristics `0xAD01`–`0xAD04`.
  - Format UART relay messages `PROV:WIFI:...` from ESP32 to Pi.
- [ ] **2.5.2. NetworkManager Automation Script on Pi:**
  - Create a Python handler in `pi/adam/provisioning.py` that receives UART credentials and runs `nmcli dev wifi connect "<ssid>" password "<pass>"`.
  - Send status result (`connected` + IP, or `bad_password` / `timeout`) back over UART to update BLE characteristic `0xAD02`.
- [ ] **2.5.3. Wi-Fi SoftAP Fallback Portal:**
  - If no BLE connection is established within 60 seconds of initial boot, spin up `create_ap` or `hostapd` with SSID `ADAM-Setup-XXXX` serving an embedded HTTP setup page on `192.168.4.1`.
- [ ] **2.5.4. Expand `ws_server.py` for Bi-Directional App Commands:**
  - Extend the existing emotion-broadcast WebSocket server to accept incoming control messages from the mobile app (sleep, wake, mute, volume, memory management).

---

### Category 3: Managed Credits & Backend Token Integration (Medium Priority)
For users who do not provide their own Gemini key and buy credit packs.

- [ ] **3.1. Ephemeral Token Fetcher:**
  - Implement a client in `session.py` to authenticate with DGEN backend (`https://api.dgentechnologies.com/v1/devices/token`) using the device serial and secret token.
  - Obtain short-lived Gemini Live session tokens instead of relying on a master API key.
- [ ] **3.2. Quota & Credit Balance Monitor:**
  - Handle backend credit exhaustion events gracefully by playing a polite in-character notification (*"Looks like we're out of credits, bhai. Top up in the app!"*) and dropping to Lite Mode.

---

### Category 4: Operating System Hardening & SD Card Durability (High Priority)
Pis running continuous I/O on microSD cards can suffer filesystem corruption during sudden power drops or power cuts.

- [ ] **4.1. OverlayFS / Read-Only Root Filesystem:**
  - Configure Raspberry Pi OS OverlayFS (`raspi-config` non-destructive overlay) so the root partition is mounted read-only during normal operation.
  - Mount a dedicated persistent partition (or tmpfs) for `adam_memory.json`, `adam_faces.json`, and logs.
- [ ] **4.2. Hardware Watchdog Integration:**
  - Enable Broadcom hardware watchdog `/dev/watchdog` (`dtparam=watchdog=on` in `config.txt` and `watchdog.service`) with a 15-second heartbeat to automatically reboot the Pi if the OS freezes.
- [ ] **4.3. Brownout & Power Glitch Protection:**
  - Add voltage brownout detection logic: monitor `vcgencmd get_throttled` for undervoltage bits (`0x1`, `0x10000`). If undervoltage occurs during high servo/speaker load, log a warning and throttle servo movement speed.

---

### Category 5: OTA (Over-The-Air) Firmware Updates (Medium Priority)
Ensuring deployed desktop units can receive bug fixes and feature updates seamlessly.

- [ ] **5.1. Update Client Service (`adam-updater.service`):**
  - Background periodic check against DGEN's release manifest endpoint.
  - Download signed `.tar.gz` or Git tag release.
- [ ] **5.2. A/B Partitioning or Safe Staging Rollback:**
  - Stage updates in a separate folder (`~/adam_next`), run `adam_smoketest.py`.
  - If tests pass, atomically swap directory symlinks and restart `adam.service`. If boot crashes 3 times, roll back automatically.

---

### Category 6: Local Face Recognition & Person Memory (Medium Priority)
- [ ] **6.1. Edge Face Detection via ESP32-CAM:**
  - Confirm whether ESP32-CAM firmware can perform local Haar/MobileNet face bounding box detection before sending frames over UART, reducing unnecessary UART bandwidth when nobody is in view.
- [ ] **6.2. Face ID Embedding Storage:**
  - Populate `adam_faces.json` with embeddings captured during the mobile app onboarding *"Let ADAM meet you"* screen.

---

### Category 7: Reminders, Scheduling, Alarms & Timers Engine (High Priority)
Enabling ADAM to act as an active desktop executive assistant with both voice and companion app schedule management.

- [ ] **7.1. Offline-Compatible Alarms & Timers (Lite Mode & Full AI):**
  - Implement a persistent schedule daemon/task (`scheduler.py`) that runs regardless of cloud connectivity.
  - Store active alarms, timers, and reminders in `adam_schedules.json` (persisted across reboots).
  - Fire local audio alarms (`aplay alarm_tone.wav`) when a timer or alarm expires.
  - Physical interaction:
    - Tap Touch1 or Touch2: Snooze alarm for 5 minutes.
    - Tap Touch3: Dismiss/Stop alarm or timer immediately.
- [ ] **7.2. Natural Language Scheduling Tools (Full AI Live Mode):**
  - Add function declarations to `tools_schema.py` and dispatch in `tool_handler.py`:
    - `set_alarm(time: str, label: str, recurring: bool)` (e.g. *"Set an alarm for 7:30 AM every weekday"*)
    - `set_timer(duration_seconds: int, label: str)` (e.g. *"Set a 15-minute tea timer"*)
    - `set_reminder(text: str, trigger_time: str)` (e.g. *"Remind me at 4 PM to check the solar panel telemetry"*)
    - `list_schedules()` (e.g. *"What alarms or reminders do I have today?"*)
    - `cancel_schedule(schedule_id: str)` (e.g. *"Cancel my 4 PM reminder"*)
- [ ] **7.3. Proactive Voice Wake & Delivery:**
  - When a reminder time arrives, if ADAM is in sleep/idle mode, automatically wake the unit, display an alert/excited face on the TFT display, sound an ascending chime, and speak the reminder proactively:
    - *"Bhai, reminder: Call Sukomal about the Auralis batch now."*
- [ ] **7.4. Mobile Companion App Schedule Synchronization:**
  - Expose `/api/schedules` via `ws_server.py` REST handler so users can view, create, edit, and delete alarms/reminders directly from the mobile app's Home dashboard.

---

### Category 8: Advanced Laptop Companion Control & Clipboard Sync (Medium Priority)
Extending the modular `laptop_agent.py` running on the user's laptop to turn ADAM into an active desktop productivity co-pilot.

- [ ] **8.1. Laptop Clipboard Read & Write Integration:**
  - Integrate `pyperclip` into `laptop_agent.py` to register clipboard actions in the `@action` registry:
    - `@action("clipboard_get", "Read the text currently copied on the laptop's clipboard.")`:
      - Allows ADAM to answer *"What's on my clipboard?"* or summarize/analyze text currently copied on the user's computer.
    - `@action("clipboard_set", "Copy specified text to the laptop's clipboard.", needs_value=True)`:
      - Allows the user to say *"ADAM, copy that code snippet to my laptop clipboard"* or *"Copy that summary"*, and ADAM writes it directly into the laptop's OS clipboard.
    - `@action("clipboard_paste", "Simulate paste keystroke into the active laptop window.")`:
      - Simulates `Ctrl+V` (Windows/Linux) or `Cmd+V` (macOS) into the active window.
- [ ] **8.2. Extended Laptop Automation Actions:**
  - `@action("lock_screen", "Lock the laptop screen immediately")`: For quick privacy when walking away from the desk (*"ADAM, lock my laptop"*).
  - `@action("media_play_pause", "Toggle playback on Spotify/media player")` / `media_next` / `media_prev`.
  - `@action("open_url", "Open a website or link in default browser", needs_value=True)`: (*"ADAM, open GitHub on my laptop"*).
  - `@action("sleep_display", "Turn off laptop displays to save power")`.
- [ ] **8.3. Laptop Agent Connection Resilience:**
  - Enhance `laptop_agent_client.py` on the Pi to re-verify Zeroconf mDNS every 60 seconds and gracefully inform the user if the laptop went to sleep or changed Wi-Fi networks.

---

### Category 9: Desktop Productivity, Health Nudges & Daily Briefing (Medium Priority)
Desktop-centric ambient intelligence features that leverage ADAM's physical presence, camera, and microphones.

- [ ] **9.1. Desk Posture & Screen-Break Nudges:**
  - Use ESP32-CAM presence detection and a 45-minute continuous desk session timer.
  - When the user has been seated continuously for 45+ minutes, ADAM performs a head tilt gesture and delivers a gentle/witty voice nudge:
    - *"Bhai, you've been hunched over code for 45 minutes straight. Sit up, stretch, and drink some water."*
- [ ] **9.2. Pomodoro Focus Mode:**
  - Add `start_pomodoro(focus_minutes: int = 25, break_minutes: int = 5)` tool.
  - During focus intervals: ADAM displays a concentrated/focused face, mutes non-urgent idle nudges, and can automatically set the laptop to "Do Not Disturb" via the laptop agent.
  - Chimes and announces break intervals when the timer completes.
- [ ] **9.3. Morning / Daily Executive Briefing:**
  - Upon first face detection of the day in front of the camera:
    - Greet user by name (from `adam_faces.json`).
    - Announce current time, date, local weather summary (via `web_search`), and any pending reminders from `adam_schedules.json`.
    - End with a custom in-character Bollywood or Tony Stark motivational line.

---

### Category 10: Voice Photo Snap, Mobile Framing/Filters & Dual-Gallery Export (High Delight Feature)
Users can ask ADAM to take a photo/selfie by voice; ADAM captures it, triggers a countdown sequence, streams it to the companion app, applies aesthetic frames/filters, and saves it both in the app and the phone's native camera roll.

- [ ] **10.1. Voice Tool & Hardware Shutter Sequence (`take_photo`):**
  - Add `take_photo(caption: str = "", countdown_seconds: int = 3)` tool in `tools_schema.py` and `tool_handler.py`.
  - Verbal acknowledgment & countdown:
    - ADAM responds in character: *"Say cheese, bhai!"* or *"Striking a pose? Hold still for 3 seconds..."*
    - Neck pan/tilt centers the user's face using DOA and face tracking.
    - TFT Face changes to `camera` / `winking` eye animation, with a visual 3-2-1 countdown on screen.
    - Plays audio shutter click sound effect through speaker (`aplay shutter.wav`).
  - ESP32-CAM captures high-resolution still frame (UXGA/SXGA JPEG) and pushes it to `/home/pi/adam/gallery/<timestamp>.jpg`.
- [ ] **10.2. Real-time Handoff to Mobile Companion App:**
  - Pi notifies mobile app over WebSocket (`ws://adam.local:8765`) with payload `{"type": "new_photo", "photo_id": "<id>", "url": "http://adam.local:8765/api/gallery/<id>.jpg", "timestamp": "..."}`.
  - Mobile app fetches the full-resolution image over LAN via local REST endpoint.
- [ ] **10.3. Mobile App Aesthetic Filters & Framing Engine:**
  - Mobile app processes the image client-side (via Canvas / WebGL filter pipeline) before saving:
    - **Polaroid / Retro Border:** Clean white Polaroid border with handwritten-style date, location, and *"Snapped by ADAM"*.
    - **Cyberpunk / Holographic HUD:** Futuristic neon cyan/amber HUD markings, targeting reticles, battery icons, and DGEN logo.
    - **Monochrome / Noir:** High-contrast black & white aesthetic matching ADAM's hardware design.
    - **Founder Edition Gold Badge:** Exclusive gold watermark badge for units #001–#010.
- [ ] **10.4. Dual-Gallery Storage & Native Phone Export:**
  - **In-App ADAM Gallery (`/gallery`):** Displays photos in the companion app grouped by date with full-screen viewer, favorites, and share actions.
  - **Native Phone Camera Roll / Gallery:** Uses Capacitor `@capacitor/filesystem` and `@capacitor-community/media-library` to automatically save the framed photo directly to the phone's native **Photos / Gallery** app under an **"ADAM Moments"** album.
  - ADAM delivers final voice confirmation: *"Photo saved to your phone gallery! Looking sharp as always."*

---

### Category 11: Geolocation Handoff & Location-Aware Search Intelligence (High Utility Feature)
Enabling the mobile companion app to share the user's current city and coordinates with ADAM so all web searches, weather lookups, local news, and time-sensitive queries are immediately localized without requiring the user to manually specify their city.

- [ ] **11.1. Companion App Geolocation Capture:**
  - Request OS Location permissions (`@capacitor/geolocation`) during mobile onboarding (or under Settings → Location).
  - On permission granted: retrieve GPS coordinates (`latitude`, `longitude`) and reverse-geocode into city, region/state, and country (e.g., *"Kolkata, West Bengal, India"*).
  - Transmit location to ADAM during BLE onboarding (Characteristic `0xAD05`) or dynamically over Wi-Fi (`POST /api/location`).
- [ ] **11.2. Local Storage & Persistence on Pi:**
  - Persist the received location in `adam_memory.json` under key `user_location`:
    ```json
    {
      "user_location": {
        "city": "Kolkata",
        "region": "West Bengal",
        "country": "India",
        "lat": 22.5726,
        "lon": 88.3639,
        "updated_at": "2026-09-04 18:00"
      }
    }
    ```
- [ ] **11.3. Ambient System Prompt Injection (`system_prompt.py`):**
  - Inject the location directly into the Gemini Live instruction block on every session connect:
    ```
    ━━━ CURRENT USER LOCATION & CONTEXT ━━━
      User is physically located in: Kolkata, West Bengal, India (22.5726° N, 88.3639° E).
      Always default to this city/region when answering questions about weather, nearby places,
      local events, movie timings, traffic, petrol prices, or local news — never ask
      "which city are you in?" unless the user specifically asks for another place.
    ```
- [ ] **11.4. Location-Augmented Web Search (`web_search.py`):**
  - When the model calls `web_search()` for geographically sensitive topics (e.g. *"what's the weather"*, *"best biryani near me"*, *"AQI level"*, *"local news"*), automatically bias or suffix the search query with the user's city/locality.
  - Set DuckDuckGo's regional search parameter (e.g., `region="in-en"`) based on the detected country code for highly accurate local results.
- [ ] **11.5. Dynamic Travel & Network Relocation Sync:**
  - When the user travels with ADAM (e.g., to an office, hotel, or another city), the companion app detects the change in phone location/Wi-Fi and automatically syncs the updated location to ADAM in the background via WebSocket or REST.

---

## 5. Prioritized Step-by-Step Implementation Roadmap

```mermaid
flowchart TD
    subgraph Phase 1: Core Voice, Offline & Lite Mode
        T1["1. Lite Mode Engine (lite_mode.py)"]
        T2["2. Local Sound Pack & Offline Chimes"]
        T3["3. Graceful Keyless Fallback in main.py"]
        T4["4. Alarms & Reminders Engine (scheduler.py)"]
    end

    subgraph Phase 2: Hardware Reliability & SD Protection
        T5["5. Enable Hardware Watchdog (/dev/watchdog)"]
        T6["6. OverlayFS / Persistent Partition Split"]
        T7["7. Undervoltage / Throttling Monitoring"]
    end

    subgraph Phase 3: Desktop Companion & Ecosystem
        T8["8. BLE / Local AP Provisioning Daemon"]
        T9["9. Laptop Clipboard & Extended Controls"]
        T10["10. Desktop Health Nudges & Pomodoro Mode"]
        T11["11. Voice Photo Snap & Mobile Filter Engine"]
        T12["12. Geolocation Handoff & Local Search Context"]
        T13["13. App Memory, Alarms & Telemetry WebSync"]
    end

    subgraph Phase 4: Production Scale & OTA
        T14["14. Managed Credits Ephemeral Token Proxy"]
        T15["15. Safe Staged OTA Auto-Updater"]
        T16["16. Long-Run Thermal & Soak Testing"]
    end

    Phase 1 --> Phase 2 --> Phase 3 --> Phase 4
```

1. **Sprint 1 (Immediate Core):** Implement `lite_mode.py`, keyless fallback, and the local `scheduler.py` engine for alarms, timers, and reminders.
2. **Sprint 2 (Hardware Hardening):** Setup hardware watchdog `/dev/watchdog` and split persistent files onto a dedicated partition for SD-card wear protection.
3. **Sprint 3 (Desktop & Laptop Companion):** Implement BLE provisioning responder, geolocation handoff for localized search, expand `laptop_agent.py` with clipboard and media actions, implement `take_photo` with mobile framing/gallery export, and add posture/Pomodoro nudges.
4. **Sprint 4 (Cloud & OTA):** Connect the ephemeral token client for Managed Credits mode and configure the automated OTA update service.
