# ADAM Companion App — Comprehensive Status & AI Agent Roadmap

> **Target Audience:** AI Coding Agents & System Engineers  
> **Source Documents Synthesized:**  
> - `docs/ADAM_App_Technical_Build_Spec.md` (Monorepo, Static Export & Capacitor Architecture)  
> - `docs/ADAM_Mobile_App_Setup_Spec.md` (Complete Hardware Setup, Provisioning & Feature Spec)  
> - `docs/ADAM_Stitch_UI_Prompt.md` (Achromatic Design Tokens & Screen Prompts)  
> - `docs/DESIGN.md` (Michroma typography, surface tokens & components)  
> - `docs/FEEDBACK.md` (Known edge-cases, copy decisions & placeholders)  
> **Workspace:** `mobileAPP/`  
> **Last Verified:** August 2026

---

## 1. Project Architecture Overview

The **ADAM Companion App** is a cross-platform mobile companion designed for the ADAM AI desk robot (DGEN Technologies). It follows a **Turborepo monorepo** architecture using **pnpm workspaces**:

```
mobileAPP/
├── apps/
│   ├── web/             # Next.js 15 App Router (Static Export Client for Capacitor)
│   ├── api/             # Fastify backend service (REST & WebSockets contract)
│   └── mobile-shell/    # Capacitor wrapper compiling to Android APK / iOS
├── packages/
│   ├── ui/              # Achromatic Design System (Monochrome tokens, Michroma fonts, Framer Motion)
│   ├── types/           # Shared TypeScript contracts & Zod validation schemas
│   └── config/          # Shared Tailwind, TypeScript, ESLint configurations
├── docs/                # Architecture, setup specifications, and UI prompt archives
└── ref/                 # Stitch reference exports
```

---

## 2. Completed Work (Done [x])

### 2.1 Workspace & Monorepo Infrastructure
- [x] **Turborepo & Workspace Linking**: Fully linked package workspaces (`@adam/ui`, `@adam/types`, `@adam/config`, `@adam/web`, `@adam/api`, `@adam/mobile-shell`).
- [x] **Strict TypeScript Contracts (`packages/types`)**:
  - `auth.ts`: Session, Google OAuth, Email OTP types.
  - `device.ts`: Robot telemetry, hardware status, Founder Edition metadata.
  - `wifi.ts`: Scanning, signal metrics, 3-phase handoff state machine.
  - `setup.ts`: 13-stage setup sequence state types.
  - `memory.ts`: Facts, user memory entries, conversation logs.
  - `gallery.ts`: Captured media moments and photo metadata.
  - `ota.ts`: Staged firmware updates and changelog models.
  - `credits.ts` & `preferences.ts`: BYOK keys, managed credit packages, voice settings.
- [x] **Achromatic Design System (`packages/config` & `packages/ui`)**:
  - Pure monochrome design tokens (`--adam-black: #000000`, `--adam-charcoal: #1C1C1E`, `--adam-grey-*`, `--adam-white: #FFFFFF`) with no saturated hues.
  - Michroma and monospace digital skin textures implemented.
  - Core UI components: `AdamFaceMark`, `RadarSweep`, `StepProgress`, `Screen`, `ScreenHeader`, `Button`, `Card`, `OptionCard`, `TextField`, `Toggle`, `SegmentedControl`, `StatusDot`, `EmptyState`, `Wordmark`, `NotYetDesigned`.
- [x] **TypeScript Strict Validation**: `pnpm typecheck` passes 100% cleanly across all packages.

### 2.2 First-Run Setup Flow (`apps/web/src/app/(setup)`)
*All 13 screens defined in the Setup Spec are implemented with animations and navigation:*
- [x] `/splash`: 2-second animated waking-up state with `AdamFaceMark` and brand wordmark.
- [x] `/welcome`: Primary entry with unboxing CTA (`Set up my ADAM`) and returning device link.
- [x] `/sign-in`: Google Sign-In & Email OTP with terms confirmation.
- [x] `/discover`: Pulsing `RadarSweep` scanning for BLE advertisement (`ADAM-XXXX`).
- [x] `/device-found`: Device identification, serial confirmation, and pairing validation.
- [x] `/wifi-select` & `/wifi-password`: 2.4GHz network filtering, signal indicators, and password entry.
- [x] `/connecting`: 3-phase sequential handshake visualizer (Sending credentials -> Connecting -> Confirmed online).
- [x] `/name-device`: Personalized robot naming.
- [x] `/founder-reveal`: Recognition badge for Founder Edition units (#001–#010).
- [x] `/ai-brain`: Three-way branching decision: **BYOK (Gemini)**, **Managed Credits**, or **Lite Mode**.
- [x] `/byok`: Guided API key setup with deep links and clipboard detection.
- [x] `/credits`: Tiered credit pack selector (Trial, Starter, Standard, Value, Pro).
- [x] `/camera-permission` & `/face-capture`: Local on-device biometric face capture screen for owner recognition.

### 2.3 Main Application & Post-Setup Screens (`apps/web/src/app/(app)`)
- [x] `/home`: Live status dashboard (online/offline indicator, battery, emotion face, sleep/wake quick controls).
- [x] `/gallery`: Photo grid captured by ADAM with date groupings and detail view.
- [x] `/memory`: Searchable knowledge stream with edit/delete controls for stored user facts.
- [x] `/settings`: Central preferences list.
- [x] `/settings/software-update`: OTA check, changelog display, and atomic update progress simulator.
- [x] `/settings/laptops`: Paired companion laptop/developer machine management.
- [x] `/settings/about`: Hardware batch, serials, and firmware metadata.

### 2.4 State Management, Static Export & Mocks
- [x] **Zustand Setup Store (`stores/setup-store.ts`)**: LocalStorage-persisted state tracking active step, device info, and completion timestamp.
- [x] **Mock Fixture Engine (`lib/mock/api.ts` & `fixtures.ts`)**: Typed mock implementations of all BLE discovery, Wi-Fi handoff sequencing, memory logs, and OTA firmware lifecycles.
- [x] **Next.js Static Export Config**: `output: 'export'` properly configured in `apps/web/next.config.mjs` generating static HTML/JS bundles into `apps/web/out`.

---

## 3. Pending / Remaining Work (To Do [ ])

### 3.1 Unfinished Post-Setup Screens (Currently `NotYetDesigned` Placeholders)
*As noted in `docs/FEEDBACK.md` and `docs/ADAM_Mobile_App_Setup_Spec.md`, these 5 screens are currently placeholder stubs:*
- [ ] `/smart-home` (Spec §4):
  - Room management grid with per-device toggles (Lights, Plugs, Fans).
  - Scene shortcuts ("Movie Night", "Wind Down").
  - Routine automation editor (presence-based triggers via ADAM RCWL-0516 sensor).
- [ ] `/settings/ai-brain` (Spec §2.6 & §3.4):
  - In-app switcher to swap between BYOK Gemini key, DGEN Managed Credits, and Lite Mode post-onboarding.
  - Key replacement/removal interface.
- [ ] `/settings/wifi` (Spec §2.4 & §3.4):
  - Current network details, signal quality, and re-provisioning trigger to change Wi-Fi networks without full factory reset.
- [ ] `/settings/voice` (Spec §2.7 & §3.4):
  - Voice selection (Charon, etc.), language hint (English, Hindi, Bengali, Hinglish), wake sensitivity slider, and personality sarcasm slider.
- [ ] `/settings/account` (Spec §2.2 & §3.4):
  - Profile details, Google account unlink, data export, device ownership transfer, and sign-out.

### 3.2 Native Capacitor Plugins & Hardware Bridges (`apps/mobile-shell`)
*Currently, native features use browser mock fallbacks. Real native plugin wiring is required:*
- [ ] **Bluetooth LE (BLE) Provisioning**:
  - Install and configure `@capacitor-community/bluetooth-le`.
  - Scan for `ADAM-XXXX` provisioning service UUID over BLE.
  - Write Wi-Fi credentials to ADAM's ESP32-CAM / Pi Zero 2W characteristic.
- [ ] **mDNS / Local Network Discovery**:
  - Implement Zeroconf discovery for locating `adam.local` on the local Wi-Fi subnet.
- [ ] **Native Biometric Camera**:
  - Integrate `@capacitor/camera` for native camera permissions and hardware frame capture on `/face-capture`.
- [ ] **Capacitor Secure Storage**:
  - Implement `@capacitor-community/secure-storage` for on-device encrypted storage of API keys, tokens, and credentials.
- [ ] **Android Shell Build & Gradle Configuration**:
  - Run `pnpm cap add android` to instantiate `apps/mobile-shell/android`.
  - Configure `network-security-config.xml` to permit local cleartext HTTP/WS to `192.168.x.x` and `adam.local`.
  - Compile and verify signed/debug APK.

### 3.3 Backend API & Persistence (`apps/api`)
*Fastify server currently contains only a `/health` route.*
- [ ] **REST Endpoints Implementation**:
  - `POST /auth/google`, `POST /auth/otp`, `GET /auth/me`
  - `GET /device/status`, `POST /device/rename`, `POST /device/claim`
  - `GET /memory`, `POST /memory`, `DELETE /memory/:id`, `POST /memory/search`
  - `GET /gallery`, `DELETE /gallery/:id`, `POST /gallery/backup`
  - `GET /ota/manifest`, `POST /ota/apply`
  - `POST /credits/checkout` (Razorpay integration), `GET /credits/balance`
- [ ] **Database Setup**:
  - Implement Prisma models (`User`, `Device`, `CreditLedger`, `CloudMemoryBackup`) with SQLite/PostgreSQL.
- [ ] **Client API Integration**:
  - Wire `apps/web` TanStack Query hooks to consume `apps/api` in production mode while retaining mock fixtures in preview mode.

### 3.4 Direct Robot Real-Time Telemetry Bridge
- [ ] **WebSocket Communication**:
  - Establish a persistent WebSocket connection between the mobile app and ADAM's Python core (`adamV29.py` / `adamV30.py`).
  - Stream live emotion state changes, audio active states, and system vitals (CPU, temp, battery) to the dashboard.
- [ ] **Live Media Streaming (WebRTC / MJPEG)**:
  - Low-latency camera view streaming from ADAM's ESP32-CAM to the mobile app for live vision preview.

### 3.5 Polish, Edge-Case Handling & E2E Testing
- [ ] **Setup Flow Error Handlers**:
  - BLE disconnect mid-pairing retry modal.
  - Wi-Fi wrong password detection and 2.4GHz network alert.
  - Bluetooth disabled prompt with deep link to OS Settings.
- [ ] **Hydration & Extension Safety**:
  - Resolve potential extension attribute mismatches (`data-new-gr-c-s-check-loaded`) noted in `docs/FEEDBACK.md` using `suppressHydrationWarning`.
- [ ] **Automated Testing**:
  - Playwright E2E test suite running the entire static web onboarding flow.

---

## 4. Priority Roadmap for Next Steps

```
Phase 1: Design & Complete 5 Remaining App Screens
         └── /smart-home, /settings/ai-brain, /settings/wifi, /settings/voice, /settings/account

Phase 2: Wire Native Capacitor Plugins
         └── BLE Provisioning, Camera Plugin, Secure Storage, Android Shell Generation (APK)

Phase 3: Implement Fastify Backend & Prisma DB
         └── Authentication, Razorpay Credits, Device Claiming & Cloud Sync

Phase 4: Real-Time WebSocket Telemetry
         └── Bridge mobile dashboard with ADAM Python robot daemon (adamV29/adamV30)

Phase 5: Field Testing, Error Hardening & E2E Tests
         └── Bluetooth dropouts, Wi-Fi retry flows, Playwright test coverage
```

---

## 5. Development Command Reference

| Action | Command |
|---|---|
| Start entire monorepo in dev mode | `pnpm dev` |
| Start Next.js web client only | `pnpm dev:web` |
| Start Fastify backend only | `pnpm dev:api` |
| Run complete workspace typecheck | `pnpm typecheck` |
| Build static web bundle (`apps/web/out`) | `pnpm build:web` |
| Sync static build to Capacitor mobile shell | `pnpm cap:sync` |
| Open native Android project in Android Studio | `pnpm cap:android` |