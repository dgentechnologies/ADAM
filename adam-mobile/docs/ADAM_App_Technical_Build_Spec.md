# ADAM Companion App — Technical Build Spec (Web-First, Wrapped to APK)
**For: AI coding agent implementation reference**
**Stack: React + Next.js (frontend) + Node.js (backend) → wrapped into an installable Android APK (and iOS later)**

This document is the engineering source of truth for building the ADAM Companion App as a website first, then packaging it into a native-installable app. Read this fully before writing any code. It assumes the UI design already exists (see `ADAM_Mobile_App_Setup_Spec.md` and `ADAM_Stitch_UI_Prompt.md` for flow/visual reference) — this document is about **architecture, stack, and how the "website becomes an app" mechanism works.**

---

## 1. High-Level Concept — "Website That Opens Like an App"

The plan: build ONE Next.js web application (responsive, mobile-first) that is:
1. Hosted normally on the web (e.g., `app.dgentechnologies.com`), and
2. Wrapped into a native Android **APK** so it installs from a file / Play Store like a real app, and
3. Later wrapped into an iOS app the same way, with zero/minimal code changes.

This is NOT "build a website and slap a WebView on it carelessly." Done correctly (via **Capacitor**, explained below), the wrapped app can still access native device features (Bluetooth, camera, local network/mDNS discovery, push notifications, secure storage) that a plain browser tab cannot. This matters a lot for this specific app because ADAM's setup flow needs BLE/local-network device discovery, which a normal website in a browser cannot do.

**Recommended wrapping tool: Capacitor (by Ionic).** Reasons over alternatives:
- Capacitor takes an existing web app (Next.js output) and wraps it in a real native shell (Android/iOS project), while exposing native plugins (Bluetooth LE, Wi-Fi info, filesystem, push notifications, camera, secure storage) to the web code via a JS bridge.
- Unlike a simple "Trusted Web Activity" (TWA) or a bare WebView APK, Capacitor gives real native API access — required for BLE device pairing and local Wi-Fi handoff in the setup flow.
- Unlike React Native, it does NOT require rewriting the UI in a different component system — the exact same Next.js/React code (compiled to static output) runs inside it. This matches the explicit requirement: "make the whole mobile app in react next js... then map a apk file which will open that."

**Do NOT use:** a plain WebView-only wrapper with no native bridge (cannot do BLE/Wi-Fi setup), and do NOT use a server-rendering-dependent Next.js mode inside the wrapped app (the wrapped app needs a fully static/exportable build — see §4).

---

## 2. Repository / Project Structure

Use a **monorepo** with three top-level packages:

```
adam-app/
├── apps/
│   ├── web/                 # Next.js app — THE app (UI + client logic)
│   ├── api/                 # Node.js backend (Express or Fastify)
│   └── mobile-shell/        # Capacitor project wrapping apps/web's static export
├── packages/
│   ├── ui/                  # Shared React component library (buttons, cards, etc.)
│   ├── types/                # Shared TypeScript types (API contracts, device models)
│   └── config/               # Shared ESLint/TS/Tailwind config
├── package.json               # Monorepo root (pnpm workspaces or turborepo)
├── turbo.json                 # If using Turborepo for build orchestration
└── README.md
```

Use **pnpm workspaces + Turborepo** (or Nx) to manage this — keeps the web app, API, and mobile shell in one repo with shared types (critical: the mobile app and backend must agree on API contracts, and a shared `packages/types` folder enforces that at compile time).

---

## 3. Frontend — `apps/web` (Next.js + React)

### 3.1 Framework & Version
- **Next.js 15+ (App Router)**, TypeScript strict mode.
- **React 19**.
- **Tailwind CSS** for styling — matches the black/white/monochrome design system exactly (define the full greyscale palette as CSS variables / Tailwind theme tokens, e.g. `--adam-black: #000000`, `--adam-charcoal: #1C1C1E`, `--adam-grey-mid: #3A3A3C`, `--adam-grey-light: #8E8E93`, `--adam-white: #FFFFFF` — never introduce hue tokens).
- **Framer Motion** for the subtle animations described in the design spec (breathing face animation, radar pulse, step transitions).
- **shadcn/ui** as the headless component base (buttons, dialogs, sheets, inputs) — restyle every component to the monochrome design system; do not ship shadcn's default color theme.

### 3.2 CRITICAL CONSTRAINT — Static Export Mode
Because this app will be wrapped by Capacitor into a mobile shell that has **no Node.js server running on the device**, the Next.js app must be built in **static export mode** (`output: 'export'` in `next.config.js`) for everything that runs inside the wrapped app.

Implications for the AI agent writing code:
- **No Next.js API routes** inside `apps/web` for anything the mobile app depends on at runtime — all backend logic lives in `apps/api` (a separately hosted Node service) and is called via `fetch`/REST or WebSocket from the client.
- **No Server Components that fetch data server-side at request time** for pages used inside the wrapped app — use Client Components with `useEffect`/data-fetching libraries (TanStack Query recommended) instead, since there's no server at runtime on-device.
- Server Components / SSR are still fine for the **separately deployed marketing/web version** of the same app (if `app.dgentechnologies.com` is meant to also work as a full website in a browser) — but that requires either (a) a separate Next.js deployment target (SSR mode) that is NOT what gets bundled into Capacitor, or (b) keeping the entire app static-export-compatible everywhere for simplicity. **Recommendation: keep the whole app static-export-compatible everywhere.** Simpler mental model, one build artifact for both web and app.
- Dynamic routes must use `generateStaticParams` or be avoided in favor of client-side routing state where the route segments aren't known at build time.

### 3.3 Data Fetching & State
- **TanStack Query (React Query)** for all server communication (device status, memory list, gallery, settings) — handles caching, retries, and works cleanly in a fully client-rendered static app.
- **Zustand** for local/global UI state that isn't server data (setup flow progress, currently selected device, theme mode) — lightweight, no boilerplate, plays well with Next.js static export.
- **React Hook Form + Zod** for all forms (Wi-Fi password entry, API key entry, naming, credit pack checkout) — Zod schemas double as the shared validation contract with the backend (put shared Zod schemas in `packages/types`).

### 3.4 Routing Structure (App Router, static-export compatible)

```
app/
├── (setup)/
│   ├── splash/page.tsx
│   ├── welcome/page.tsx
│   ├── sign-in/page.tsx
│   ├── discover/page.tsx
│   ├── device-found/page.tsx
│   ├── wifi-select/page.tsx
│   ├── wifi-password/page.tsx
│   ├── connecting/page.tsx
│   ├── name-device/page.tsx
│   ├── founder-reveal/page.tsx
│   ├── ai-brain/page.tsx
│   ├── byok/page.tsx
│   ├── credits/page.tsx
│   ├── camera-permission/page.tsx
│   └── face-capture/page.tsx
├── (app)/
│   ├── home/page.tsx
│   ├── gallery/page.tsx
│   ├── smart-home/page.tsx
│   ├── memory/page.tsx
│   └── settings/
│       ├── page.tsx
│       ├── account/page.tsx
│       ├── ai-brain/page.tsx
│       ├── wifi/page.tsx
│       ├── voice/page.tsx
│       ├── laptops/page.tsx
│       ├── software-update/page.tsx
│       └── about/page.tsx
├── layout.tsx
└── globals.css
```
Setup flow is a linear wizard — implement it with a shared `SetupLayout` that renders the step progress indicator and handles forward/back navigation via Zustand-tracked step state, not just raw Next.js routing history (so "resume where you left off" — an explicit requirement in the mobile spec — works even after the app is force-closed; persist current step to `localStorage`/Capacitor Preferences).

### 3.5 Native Feature Access (via Capacitor plugins, called from React)
These are the native capabilities the web code must call through Capacitor's plugin bridge (NOT achievable in a normal browser tab):

| Feature | Capacitor Plugin | Used In |
|---|---|---|
| Bluetooth LE scan/connect | `@capacitor-community/bluetooth-le` | Device discovery (§ discover/page.tsx) |
| Wi-Fi network info / join hotspot | `@capacitor-community/wifi` (Android) or custom native plugin | Wi-Fi handoff fallback flow |
| Camera | `@capacitor/camera` | Face capture screen |
| Push notifications | `@capacitor/push-notifications` | OTA update alerts, "ADAM disconnected" alerts |
| Secure storage | `@capacitor/preferences` + `capacitor-secure-storage-plugin` | Storing auth tokens, cached BYOK key reference |
| Local network / mDNS discovery | Custom native plugin (Bonjour/NSD wrapper) — see §6 | Finding ADAM/laptop agent on LAN post-setup |
| Google Sign-In | `@codetrix-studio/capacitor-google-auth` | Sign-in screen |
| Filesystem (for OTA/gallery caching) | `@capacitor/filesystem` | Gallery, OTA download progress |

**Agent implementation note:** wrap every native call in a small abstraction layer (e.g., `lib/native/bluetooth.ts`, `lib/native/wifi.ts`) that checks `Capacitor.isNativePlatform()` first and gracefully no-ops or shows a "this feature requires the app" message when running in a plain browser — since the same codebase may also be opened in a desktop/mobile browser directly for marketing/preview purposes.

---

## 4. Backend — `apps/api` (Node.js)

### 4.1 Framework
- **Node.js 20+ LTS**, TypeScript.
- **Fastify** (preferred over Express for this — better TypeScript support and built-in schema validation, which pairs well with the shared Zod contracts).
- **Prisma ORM** against **PostgreSQL** (Supabase, matching the existing infra already chosen in the DGEN billing docs — reuse the same Supabase project/instance already planned for credit-pack tracking).

### 4.2 Responsibilities of the backend
The backend is the source of truth for anything that must be trusted/centralized — it is NOT in the hot path for device-to-app local communication (BLE/local network stays device-to-device where possible for latency and offline resilience). Backend responsibilities:

1. **Account & device registry** — Google OAuth verification, linking a signed-in user to one or more ADAM device serials, Founder Edition batch metadata lookup.
2. **Ephemeral token issuance** — for Managed Credits mode, implements the flow already designed in `ADAM_Gemini_Billing_Monetisation_Strategy_v1.docx` (Section 4.3 Node.js skeleton) — this backend IS that Express/Fastify service, just now formally part of this monorepo as `apps/api`.
3. **Credit ledger** — track purchased packs, deduct on session use, expose balance to the app.
4. **Razorpay integration** — create orders, verify payment webhooks, update credit ledger.
5. **OTA release manifest** — serves the current firmware/software version metadata + changelog + signed package URL that both the Pi (`adam.py`'s future OTA checker) and the app's Settings → Software Update screen read from.
6. **Push notification dispatch** — triggers via Firebase Cloud Messaging (FCM) for Android when e.g. an OTA update is available or ADAM goes offline unexpectedly.
7. **Gallery cloud backup (opt-in only)** — signed upload URLs to object storage (Supabase Storage or S3-compatible) for users who opt into cloud photo backup, per the privacy principle already established (local-first by default).
8. **Laptop-agent pairing token registry** — issues/revokes the secure tokens described in `ADAM_PC_Laptop_App_Spec.md` §6, so a laptop and a phone can both authenticate against the same ADAM under one account.
9. **Memory sync (optional)** — if the user wants their Memory tab to sync across devices, this is the backend endpoint for that; otherwise Memory reads happen directly from the Pi over LAN.

### 4.3 What the backend explicitly does NOT do
- Does NOT proxy live audio/video from ADAM's Gemini Live session (that's the Pi ↔ Google direct connection, per the existing `adam.py` architecture).
- Does NOT store a user's BYOK Gemini API key server-side (it is generated by the user, sent device-to-device from phone to Pi during setup, and stored encrypted on the Pi only — see §7 security notes).
- Does NOT handle Wi-Fi credential transfer (that's BLE/local-AP device-to-device between phone and Pi during setup, never touches the internet).

### 4.4 API Contract Pattern
Define every endpoint's request/response shape as a Zod schema in `packages/types`, imported by both `apps/web` (for React Hook Form validation + TanStack Query typing) and `apps/api` (for Fastify route schema validation). This is the single most important discipline for an AI agent to follow consistently — never hand-write duplicate interface shapes on both sides.

Example endpoints to implement first:
```
POST   /auth/google                — exchange Google ID token for session
GET    /devices                    — list devices linked to account
POST   /devices/claim               — claim a newly-paired device (serial + account)
GET    /devices/:id                 — device detail (name, status, batch info)
PATCH  /devices/:id                 — rename device
POST   /devices/:id/ai-brain        — set BYOK/managed mode (key never stored raw)
GET    /credits/packs                — list available credit packs + pricing
POST   /credits/purchase             — create Razorpay order
POST   /credits/webhook              — Razorpay payment webhook
GET    /credits/balance/:deviceId    — current credit balance
POST   /tokens/ephemeral             — issue ephemeral Gemini Live token (managed mode)
GET    /ota/manifest                 — latest firmware/app version + changelog
POST   /laptops/pair                 — register a new laptop-agent pairing token
GET    /laptops/:deviceId            — list laptops paired to a device
DELETE /laptops/:pairingId           — revoke a laptop pairing
GET    /gallery/:deviceId            — list cloud-backed-up photos (if opted in)
POST   /gallery/upload-url            — signed upload URL for a photo
```

### 4.5 Realtime/local communication note (important architecture point)
The **phone app talks to the Pi directly over the local network or BLE** for anything latency-sensitive or setup-related (device discovery, Wi-Fi handoff, live status polling on the Home dashboard, memory read/write, gallery sync from local Pi storage) — it does **not** round-trip through `apps/api` for these. `apps/api` is the cloud backend for account/billing/OTA/cross-device concerns only. The AI agent should build a `lib/adam-device-client.ts` module in `apps/web` that talks HTTP/WebSocket directly to the Pi's local IP (discovered via mDNS/BLE, same as the existing `_discover_laptop_agent_ip()` pattern already used in `adam.py` for the laptop agent) — this mirrors the same mDNS service-discovery approach already proven in the codebase, just now also used for phone-to-Pi communication, not only Pi-to-laptop.

---

## 5. From Next.js Static Export → Android APK (Capacitor Setup)

### 5.1 Build Pipeline
```
1. cd apps/web && next build            # produces static export in apps/web/out
2. cd apps/mobile-shell
3. npx cap sync android                 # copies apps/web/out into the native project + syncs plugins
4. npx cap open android                 # opens Android Studio
5. Build → Generate Signed APK/AAB      # produces the installable .apk / .aab for Play Store
```

### 5.2 `apps/mobile-shell` setup
- Initialize once with `npx cap init "ADAM" "com.dgentechnologies.adam" --web-dir=../web/out`.
- Add Android platform: `npx cap add android`.
- Add iOS platform later (same web code, zero rewrite): `npx cap add ios`.
- `capacitor.config.ts` sets `webDir: '../web/out'` and configures splash screen / status bar to match the black theme (`backgroundColor: '#000000'`, `androidSplashResourceName`, status bar style `dark`/light-content).
- Install and register every native plugin listed in §3.5 in this project (`npm install` + `npx cap sync`).

### 5.3 Signing & Distribution
- Generate a proper Android keystore for release signing (`keytool -genkey -v -keystore adam-release.keystore ...`) — store this keystore + credentials securely (NOT in the git repo; use a secrets manager or CI secret store).
- For direct APK distribution (Founder Drop / early testers, sideloading before Play Store approval): produce a signed release APK, host it on `dgentechnologies.com/download` or distribute via a direct link/QR code in the unboxing materials.
- For public launch: publish as an AAB (Android App Bundle) to the Google Play Store (required format for Play Store since 2021) — this is a small additional step (`Generate Signed Bundle` instead of APK in Android Studio) but should be the actual long-term distribution channel; sideloaded APKs are a stopgap for the first 10 Founder units only.
- iOS: will require an Apple Developer account, TestFlight for beta, App Store submission for public release — same Capacitor `ios` project, out of scope for the very first Batch 1 milestone but architected for from day one by keeping everything Capacitor-compatible.

### 5.4 Deep Linking / App Association
- Configure Android App Links (and iOS Universal Links later) so links like `https://app.dgentechnologies.com/setup` open directly in the installed app instead of a browser, if the app is installed — needed for the "scan a QR code on the box to start setup" unboxing flow mentioned in the GTM strategy.

---

## 6. Local Network Discovery — The One Genuinely Hard Native Piece

This is the part of the app most likely to need custom native code (not just an off-the-shelf Capacitor plugin), so the AI agent should scope this explicitly rather than assume a plugin exists that does exactly this:

- The existing Python codebase already does mDNS discovery Pi→laptop (`_discover_laptop_agent_ip()` in `adam.py`, `_adam-laptop._tcp.local.` service type) and needs the mirror-image capability added: **Pi advertises its own service** (e.g., `_adam-device._tcp.local.`) so the **phone app** can discover it on the same LAN post-Wi-Fi-setup (for the Home Dashboard's live status, Memory tab, Gallery sync, etc.).
- On Android, this requires the **Network Service Discovery (NSD) API** — either write a small custom Capacitor plugin wrapping `android.net.nsd.NsdManager`, or use an existing community plugin if one proves reliable (evaluate `@theopensource-company/capacitor-nsd` or similar at implementation time — verify actively maintained status before depending on it, since Capacitor community plugins vary in upkeep).
- On iOS, equivalent is Bonjour via `NetServiceBrowser` — also may need a small custom plugin (`capacitor-nsd` community plugins sometimes cover both, verify).
- **For initial setup pairing specifically** (before the phone even knows the Pi's Wi-Fi credentials), use **BLE** instead of mDNS, since the Pi isn't on the home Wi-Fi network yet at that point — this matches the existing mobile spec's "Find My ADAM" flow (§2.3 of `ADAM_Mobile_App_Setup_Spec.md`), which already specifies BLE as the primary discovery method with a temporary-hotspot fallback.
- **Firmware-side work required in parallel** (flag this to Tirthankar — this is Pi/Python side, not app side): the Pi needs a BLE GATT peripheral service for provisioning (likely via BlueZ + a Python BLE library, or a lightweight companion process) and an mDNS responder (e.g., Python `zeroconf` library, already a dependency in `adam.py`) advertising a NEW service type for the phone app to find post-setup, separate from the existing laptop-agent service type.

---

## 7. Security Notes (carried over and made concrete for this stack)

- **Auth tokens**: store in Capacitor `Preferences`/secure storage plugin, never in `localStorage` alone (localStorage inside a WebView is not secure storage).
- **BYOK API key**: never sent to or stored on `apps/api`. Transmitted directly from phone to Pi over the local BLE/Wi-Fi channel established during setup, encrypted in transit (use a simple asymmetric handshake — Pi generates a keypair at first boot, phone encrypts the API key with the Pi's public key before sending over BLE, matching the "stored encrypted in the Pi's local storage" requirement already stated in the billing strategy doc).
- **Laptop pairing tokens**: generated server-side (`apps/api`), never user-typed, stored in OS-native secure credential stores on the laptop side (already specified in `ADAM_PC_Laptop_App_Spec.md` §6) and in Capacitor secure storage on the phone side.
- **CORS**: `apps/api` should only accept requests from the app's known origins (the deployed web origin + the Capacitor app's custom scheme, e.g. `capacitor://localhost` on iOS / `http://localhost` on Android — configure Fastify CORS accordingly, this is a common Capacitor gotcha).
- **Certificate pinning** (stretch goal, post-launch): consider for the `apps/api` connection once out of early beta, to harden against MITM on public Wi-Fi during setup.

---

## 8. Testing & Environments

- **Local dev**: `apps/web` runs via `next dev` in a normal browser for UI iteration (native features stub/no-op gracefully per §3.5's abstraction layer requirement) — this is the fast feedback loop; native-specific flows (BLE, Wi-Fi) are only truly testable in the Capacitor-built app on a real device or emulator with the plugin installed.
- **Capacitor Live Reload**: configure `capacitor.config.ts`'s `server.url` to point at the local `next dev` server during active development so native testing doesn't require a full rebuild per change.
- **Staging backend**: separate Fastify + Supabase project instance from production, so setup-flow testing (device claiming, credit purchase with Razorpay test mode) doesn't touch real user data.
- **Device lab**: at minimum, test on one mid-range and one budget Android device physically, plus the Android Studio emulator — BLE/NSD behavior varies meaningfully across real OEM Android skins (Xiaomi/Samsung background process restrictions are a known source of BLE scanning bugs).

---

## 9. Build Order for the AI Agent (suggested implementation sequence)

1. Scaffold the monorepo (pnpm workspaces + Turborepo), shared `packages/types` and `packages/ui` with the Tailwind monochrome theme tokens.
2. Build `apps/web`'s static screens for the full setup flow (§3.4 routes) using mock/local data first — no backend or native calls yet, pure UI matching the Stitch-generated designs.
3. Stand up `apps/api` with auth + device registry endpoints; wire the sign-in screen and device claim screen to real endpoints.
4. Initialize `apps/mobile-shell`, get a bare Capacitor build running on an Android emulator showing the static-exported `apps/web` output — validate the export pipeline works end-to-end before adding native plugins.
5. Integrate BLE plugin + build the real device-discovery/pairing flow against actual Pi firmware (coordinate with firmware side for the BLE GATT service — this is the highest-risk integration point, start it early).
6. Integrate Wi-Fi handoff, then the rest of the setup wizard screens against real device communication.
7. Build the Home Dashboard + Memory/Gallery/Settings against the local-network Pi API + `apps/api` cloud endpoints as applicable.
8. Add Razorpay + credits flow, OTA update screen, push notifications last (these depend on `apps/api` maturity, not on the Pi).
9. Signing, release APK build, sideload-test on the Founder Edition units.

---

## 10. Key Decisions Already Made (do not re-litigate without reason)

- Framework: Next.js (App Router) + React + TypeScript, static export mode, wrapped via Capacitor — not React Native, not a bare WebView, not Flutter.
- Styling: Tailwind CSS, strict black/white/greyscale token system, shadcn/ui as headless base.
- Backend: Node.js + Fastify + Prisma + Supabase Postgres, mirroring infra already planned in DGEN's billing strategy doc.
- Payments: Razorpay (already the chosen processor per DGEN's GTM doc).
- Monorepo: pnpm workspaces + Turborepo.
- Local device communication bypasses the cloud backend entirely (BLE for pairing, mDNS/local HTTP for ongoing phone↔Pi communication) — the cloud backend is for account/billing/OTA/cross-device sync only.
