# ADAM Companion for Android

ADAM's private mobile companion by DGEN Technologies. Next.js static export + Capacitor; the Android app bundles its screens and does not need a developer laptop or the development API to run.

Version 0.2.1 adds optional account sync with the Windows companion while keeping working phone features and the interactive ADAM hardware demo. Discovery, pairing, device status and laptop controls use clearly marked sample devices; real ADAM BLE transport remains outside this release. Managed billing is not connected.

## Features

- Optional account sign-in with Firebase; local use works without signing in.
- Persistent memories and people: add, edit, search and delete.
- Explicit account sync for memories, people, voice, wake word and AI selections: Settings > Your profile > Sync with desktop. The same Firebase user signs in on both apps. Choose Enable account sync, confirm the phone-data import, then tap Sync now. Photos, face profiles, notifications and private keys stay on the phone.
- Camera and photo import, a private local gallery, Android sharing and deletion.
- Local face profile with front/side photos; no recognition or robot transfer claims.
- Home Assistant REST integration for device states and light/switch/fan/scene controls.
- Local profile, light/dark appearance, voice and AI preferences.
- Validated memory backup/restore, local data erasure, privacy notice and account deletion.
- Android Keystore encryption for API/access tokens and Firebase session persistence.
- Guided ADAM demo: discovery, naming, sample Wi-Fi setup, pairing, expressions, brightness, volume, sleep/wake, disconnect/re-pair and laptop media controls. Demo settings persist independently of real phone data.
- An opt-in notification inbox with Android-managed background reading, pause and clear controls. Access is requested from a dedicated screen only after tapping Enable; camera access is requested only when taking a photo.

## Build

Requirements: Node 20.11+, pnpm 9, Java 17, Android SDK 35. Android minimum SDK is 23 (Android 6); a current Android System WebView is required. Physical-device compatibility must be tested across the intended release device range.

```powershell
pnpm install --frozen-lockfile
pnpm --filter @adam/web build
pnpm --filter @adam/api exec tsx --test ../../tests/local-data.test.ts ../../tests/home-assistant.test.ts ../../tests/mobile-sync.test.ts
./scripts/build-android.ps1 -Configuration Release -SkipWebBuild
```

The release script requires private signing credentials. See [Android release guide](docs/ANDROID_RELEASE.md). Output: `releases/ADAM-0.2.1-release.apk` and its SHA-256 checksum. Filenames are derived from Android's `versionName`; previous versioned APKs are retained.

For browser development, run `pnpm dev:web`; for the exact exported bundle, run `node apps/web/serve-out.mjs` and open `http://localhost:3000`. Home Assistant browser access requires CORS; native Android uses Capacitor HTTP.

## Workspace

`apps/web` contains product screens and local services. `apps/mobile-shell/android` contains the Android shell and native Companion plugin. `packages/ui`, `packages/types` and `packages/config` contain shared components, schemas and tokens. `apps/api` is development scaffolding and is not a shipped app dependency.

[Current implementation and release notes](docs/ANDROID_RELEASE.md) supersede the older mock/provisioning roadmap in `DEVELOPMENT.md` and `docs/PROJECT_STATUS.md`.
