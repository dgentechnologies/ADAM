# ADAM Companion App — Developer Guide

Everything you need to open this repo cold: what the pieces are, how to run them, and
where the rules live. Scope of this guide is the `mobileAPP/` workspace only — the rest
of the `ADAM` folder is unrelated firmware/Python.

---

## 1. What this project is

A companion mobile app for the ADAM AI desk robot (DGEN Technologies Pvt. Ltd.).
It does two things:

1. **Provisioning** — a 15-step setup wizard that finds the unit, hands it Wi-Fi
   credentials, names it, chooses how its AI is powered, and optionally teaches it
   your face.
2. **Day-to-day control** — home dashboard, captured moments, smart-home, memory
   (what ADAM knows about you), and settings including OTA updates.

It ships as a **Next.js static export** wrapped by **Capacitor** into an Android/iOS
app. There is no server rendering at runtime: every route is prerendered HTML plus
client JS, so anything that would normally be a server redirect is a client decision.

**Current state:** the entire UI layer is built and runs on mock data. `apps/api` is
scaffolding only — no real Supabase, Razorpay, BLE, mDNS, or camera wiring yet.
The whole setup flow is walkable end-to-end with no backend, no Pi, and no internet
beyond loading the app (see §6).

---

## 2. Layout

```
mobileAPP/
├── apps/
│   ├── web/            # Next.js 15 App Router, React 19 — the actual app
│   ├── api/            # Fastify backend (scaffolding, not wired up)
│   └── mobile-shell/   # Capacitor wrapper → Android APK / iOS
├── packages/
│   ├── ui/             # @adam/ui — the design system (framework-agnostic React)
│   ├── types/          # @adam/types — Zod schemas; the single source of contracts
│   └── config/         # shared Tailwind preset + tsconfig base
├── docs/               # specs (see §7) — these override the Stitch mockups
└── ref/                # Stitch UI export; visual reference, NOT production code
```

Inside `apps/web/src`:

| Path | What lives there |
|---|---|
| `app/(setup)/*` | the 15 wizard screens; route group adds no URL segment |
| `app/(app)/*` | the post-setup app (home, gallery, memory, smart-home, settings) |
| `app/page.tsx` | root route — decides where to send you on cold start |
| `lib/setup-flow.ts` | `SETUP_ORDER`, `nextStep()` — all wizard branching |
| `lib/mock/` | `fixtures.ts` (fake device, networks, packs) + `api.ts` (fake latency) |
| `lib/native/` | Capacitor abstractions: platform, preferences, secure storage |
| `stores/` | Zustand stores; `setup-store.ts` is persisted and Zod-validated |
| `components/` | app-specific shell pieces (app bar, tab bar, setup shell/transition) |

---

## 3. Prerequisites

- Node.js **>= 20.11.0**
- pnpm **>= 9** (`packageManager` pins `pnpm@9.12.3`)
- Android Studio only if you want to build an APK

```bash
pnpm install
```

Run it from `mobileAPP/` — that is the pnpm workspace root.

---

## 4. Running it

All commands below are from `mobileAPP/`.

**The web app (what you want 95% of the time):**

```bash
pnpm dev:web
```

Serves on http://localhost:3000. Hot reload covers `apps/web`, `packages/ui`, and
`packages/types` — the UI package is transpiled, not prebuilt, so editing a component
refreshes immediately.

**Everything at once (web + api):**

```bash
pnpm dev
```

**Just the API** (only useful once real endpoints exist):

```bash
pnpm dev:api
```

### Port already in use

`apps/web` hardcodes `-p 3000`. To use another port:

```bash
pnpm --dir apps/web exec next dev -p 4000
```

### First compile is slow

Next 15 + React 19 + this dependency set takes several minutes for a cold
`dev` compile or a production build on a modest Windows machine. It is not hung —
watch CPU, not the log, which buffers spinner output when redirected to a file.

---

## 5. Build, check, ship

```bash
pnpm typecheck                     # tsc --noEmit across every package
pnpm --dir apps/web run lint       # next lint
pnpm build                         # turbo build; static export lands in apps/web/out
```

TypeScript runs in strict mode with `noUncheckedIndexedAccess`, `noUnusedLocals`, and
`noUnusedParameters`. `next.config.mjs` sets `eslint.ignoreDuringBuilds: false` and
`typescript.ignoreBuildErrors: false`, so a lint error or an orphaned import fails the
build. Fix them; don't bypass them.

**Preview the real static export** (this is what Capacitor ships, and the best way to
demo without a dev server):

```bash
npx -y serve apps/web/out -l 4321
```

**Android APK:**

```bash
pnpm build          # must run first — Capacitor copies apps/web/out
pnpm cap:sync
pnpm cap:android    # opens Android Studio
```

---

## 6. Demo / pitch mode

The app is fully walkable on mock data. Nothing calls out to a network.

- Fake unit: serial `DGEN-ADAM-0007`, short id `ADAM-3F2A`, flagged Founder Edition.
- Fake networks include `DGEN_STUDIO_5G` (5 GHz, shown as unsupported and not
  selectable) and `Coffee_Shop_Free` (open — skips the password screen entirely).
- Credit packs, OTA manifest, memory entries, and gallery items are all fixtures.
- Mock calls carry realistic delays (device scan ~2.6 s, Wi-Fi handoff ~4.2 s) so the
  loading states are visible during a demo.

Three branches to show, all of which end at `/home`:

| Branch | Path through the wizard |
|---|---|
| BYOK | welcome → sign-in → discover → device-found → wifi-select → (password) → connecting → name-device → founder-reveal → ai-brain → **byok** → camera-permission → face-capture → home |
| Managed credits | … → ai-brain → **credits** → camera-permission → face-capture → home |
| Lite | … → ai-brain → **skip** → camera-permission → face-capture → home |

Two behaviours worth demonstrating deliberately: picking the open network skips
`/wifi-password`, and reloading the app mid-flow resumes on the step you left
(the root route waits for the persisted store to rehydrate before deciding).

**Resetting between demos:** the wizard persists to `adam.setup.v1` (localStorage in a
browser, Capacitor Preferences on device). Clear that key and reload to start fresh.

---

## 7. Rules that are not obvious from the code

- **Greyscale only.** The Tailwind preset replaces the `colors` key with a 10-step
  achromatic ramp, so any stray hue utility fails the build. Semantic tokens are
  `var(--adam-*)`, which means **no alpha modifiers on semantic colours** — use a ramp
  step (`bg-charcoal/60`) or the `.chrome-blur` utility instead.
- **Typography split.** Inter for all UI; Michroma (`font-display`) only for the ADAM
  wordmark, eyebrow labels, and the Founder Edition reveal.
- **Specs beat mockups.** `ref/code.html` and the Stitch screenshots are visual
  reference. Where they contradict `docs/`, the docs win — the export contains three
  different tab bars, stray logo chips, and mislabelled titles.
- **The BYOK key never touches our servers.** It goes phone → Pi, encrypted with the
  Pi's public key, and is stored encrypted on the Pi only. Do not persist it in a
  store, in preferences, or in any API call.
- **Auth tokens go in Capacitor secure storage**, never plain `localStorage`.
- **`@adam/types` is the contract.** Every mock is schema-valid, so swapping to real
  endpoints is a transport change, not a refactor.
- **Wizard branching lives in one place** — `lib/setup-flow.ts`. Don't give a screen
  its own opinion about where it leads.

Further reading in `docs/`: `ADAM_App_Technical_Build_Spec.md` (architecture),
`ADAM_Mobile_App_Setup_Spec.md` (feature/flow spec), `DESIGN.md` (tokens and
components), `PROJECT_STATUS.md` (what is built vs. pending), `FEEDBACK.md` (copy
decisions and known edge cases).

---

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `pnpm lint` prompts "How would you like to configure ESLint?" | ESLint config exists only under `apps/web`; run `pnpm --dir apps/web run lint` |
| Build sits on "Creating an optimized production build" | Slow, not stuck. Check for a stale `next build` process competing for CPU and kill it; `rm -rf apps/web/.next apps/web/out` for a clean run |
| Cold start lands on `/welcome` when you expected to resume | Clear `adam.setup.v1`, or check the persisted blob still matches the `SetupState` schema — an invalid blob resets the wizard by design |
| Paste chip on `/byok` does nothing | The Clipboard API is permission-gated and missing in some webviews; type or keyboard-paste the key |
| `@adam/types` import errors in the editor | The package uses NodeNext specifiers; `next.config.mjs` maps them via webpack `resolve.extensionAlias` — restart the TS server after touching that file |
