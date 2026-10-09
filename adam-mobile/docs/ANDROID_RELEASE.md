# ADAM Android 0.2.2

Current release: 9 October 2026, package `com.dgentechnologies.adam`, version code
10. Use `releases/ADAM-0.2.2-release.apk` with Windows `adamV0.02.exe`.

## Changes in 0.2.2

- Bounded startup hydration removes the splash-screen race. Damaged setup JSON
  is backed up before recovery; unavailable storage has a Retry action.
- The launcher icon, splash and sign-in branding use the desktop logo.
  `scripts/sync-branding.py` regenerates the assets from the desktop ICO.
- All setup screens remain. Device naming is restored between connecting and
  the Founder/AI screens; Back navigation follows the selected branch.
- Native Google sign-in initializes its plugin before use. Failed or cancelled
  login stays on sign-in. Email/password login, registration and password reset
  are distinct actions. Google/Firebase live acceptance is still required.
- The shared version-2 envelope adds to-dos, alarms/timers/reminders and multiple
  named devices. Rename/delete and simulated BLE use the same IDs, validation,
  merge rules and tombstones on Windows and Android. Update both clients
  together; older clients reject version 2 without overwriting its records.
- Account sync starts only after explicit import confirmation and Sync now;
  later edits sync while open and online. See the
  [canonical protocol](../../shared/COMPANION_PROTOCOL.md).
- The animated background limits rendering work, respects reduced motion and
  releases GPU resources when its screen closes.
- The Android notification listener declaration is restored so the existing
  opt-in notification screen can work. No permission is requested at startup.

The 0.2.2 verification record is in
[`RELEASE_0.2.2.md`](RELEASE_0.2.2.md). Historical release evidence below describes
the artifacts tested at those dates; it is not a claim of current live account
or physical-device verification.

## Historical 0.2.1 behavior and evidence

Release work: 7 October 2026. Package: `com.dgentechnologies.adam`, version code 8. Minimum Android 6 (API 23), target/compile Android 15 (API 35). A current Android System WebView is required. The APK contains portable Java/Dex code and web assets with no required ARM-only native library. Version 0.2.0 is preserved in `releases` with its original checksum and native test record.

## Product behavior

The app starts without an account or an ADAM robot. Memories, people, name, voice preferences and onboarding are validated and persisted through Capacitor Preferences. Gallery photos are resized and stored as IndexedDB blobs in the app sandbox. Face profile images are resized to 640 pixels and kept locally. None of these features calls the developer API.

Home Assistant is a real optional REST integration. A connection must return valid states before it is saved; controls send service requests and reload the reported state. A stored access token is bound to its configured URL. Public addresses require HTTPS; local home servers may use HTTP. Browser testing requires Home Assistant CORS configuration; the Android app uses native Capacitor HTTP.

Firebase supports optional email/password and Google sign-in, reset email, profile editing, sign-out and account deletion. Cloud profile access is optional for successful sign-in. Version 0.2.1 adds explicit account sync for memories, people, voice, wake word and AI selections. Sign in with the same Firebase user as the PC app, open Settings > Your profile > Sync with desktop, choose Enable account sync, confirm importing this phone's records, and tap Sync now. Signing in or confirming enable alone does not upload phone data. Android session credentials and Home Assistant/Gemini keys use AES-GCM encryption with a non-exportable Android Keystore key; browser service keys are held only in memory.

Sync patches only `users/{uid}.companion` through a Firestore transaction in the existing `adam-ai1` project. Profiles and device fields are preserved. Memory IDs, timestamps and bounded payloads are validated; per-record merge rules and retained deletion tombstones prevent ordinary concurrent edits or offline devices from restoring deleted records. Changes made while a transaction is running remain pending for the next sync. Corrupt remote records are rejected without replacing phone data. Account switches fence pending results and require explicit import approval for the newly selected account. Local phone memories remain on the phone when signing out; separate sync records prevent automatic uploads into another account. Turning off sync preserves existing local and cloud data. Photos, face profiles, notifications, Home Assistant addresses, pairing data and credentials are excluded. See the [shared protocol](../../adam-desktop/docs/SYNC_PROTOCOL.md).

Photo sharing uses the Android share sheet. Temporary shared files remain readable after the chooser closes and expire on a later share after one day. Erase app data also removes this share cache. Memory backups intentionally exclude photos, face images and credentials. Restore validates the file and preserves the existing Home Assistant token/address pairing.

If Android recreates the app while its external camera is open, the restored result is recovered into Moments with a dismissible notice. Stable hashed photo IDs prevent duplicates and remain safe for sharing filenames. Recovery does not create a face profile automatically. The camera provider exposes only app-owned Pictures and cache files.

At the user's request, the ADAM hardware sections use an interactive simulation rather than unavailable screens. A clearly marked Demo experience walks through discovery, device naming, sample Wi-Fi selection and simulated pairing. The Your ADAM screen provides sample connection/battery/network status, responsive expressions, display brightness, voice-volume preferences, rest/wake and disconnect/re-pair. The laptop screen pairs a sample laptop and previews volume/media controls. Firmware status is explicitly a sample. Demo state lives under `adam.demo.v1`, separate from real memories, images, tokens and notification data, and is removed by Erase app data. No Bluetooth scan, real network provisioning, physical-device command, recognition or robot transfer happens in the demo. Voice/AI preferences are saved on the phone. Managed credits and payments are unavailable until a real billing service exists.

## Permissions and background behavior

No runtime permission or special-access request runs on first launch. Camera access is requested only from Take photo or a face capture button. Import uses Android's file picker without broad photo/storage access. Wi-Fi management opens system Settings. BLE, location and microphone permission requests are absent from this build.

Settings > Notifications & background explains the private-message access before the user taps Enable notification access. This opens Android's notification listener access screen and its system confirmation; Android does not offer an ordinary runtime popup for this special access. The app checks the actual grant on return and starts capture only following the enable action. Access can be revoked in Android at any time. Pause reading stops new storage without dismissing source notifications.

Android binds `AdamNotificationListener` after consent and maintains it while the app is in the background. No permanent foreground-service notification or unnecessary battery exemption is requested. Storage runs on a separate writer thread with a bounded queue, retains at most 100 notifications, deduplicates updates by key and limits each title/body to 200/1500 characters. Pause/Clear invalidates older queued callbacks. Ongoing notifications, group summaries and the app's own notifications are excluded. History stays in private app storage, is excluded from backups and can be cleared. Erase app data pauses capture and clears history. Nothing is forwarded to ADAM or the cloud.

Background operation remains subject to Android force-stop, low-memory device limitations and manufacturer battery policies. The screen reports access, capture and service connection separately and links to Android app settings for background battery management. Android or the source app may redact sensitive notification content. Sideloaded apps may require the user to allow restricted settings in Android's app info before notification access can be enabled.

## Build and signing

```powershell
pnpm install --frozen-lockfile
pnpm --filter @adam/web build
pnpm --filter @adam/api exec tsx --test ../../tests/local-data.test.ts ../../tests/home-assistant.test.ts ../../tests/mobile-sync.test.ts
./scripts/build-android.ps1 -Configuration Release -SkipWebBuild
```

Use Java 17 and Android SDK/platform/build-tools 35. Gradle 8.9 / Android Gradle Plugin 8.7.3 are pinned. Dependency module outputs go under Android's generated `build/modules` directory. The scripts repair read-only flags and remove Explorer metadata from generated Android files, which otherwise caused Next.js cleanup loops and Gradle resource generation failures. The web build clears only its validated generated `out` directory; the preview server releases file handles when requests close. Type and lint failures are not ignored.

The packaging script derives the output filename from `versionName` in `apps/mobile-shell/android/app/build.gradle`. Increment `versionCode` for each distributable update. Versioned APKs and checksum files remain side by side; the current build does not overwrite 0.2.0.

Release environment variables: `ADAM_KEYSTORE`, `ADAM_STORE_PASSWORD`, `ADAM_KEY_ALIAS`, optionally `ADAM_KEY_PASSWORD`. Keep these outside version control. The local script can alternatively load the private key created for this release:

- Key: `C:\Users\HP\.android\adam-release\adam-release.jks`
- Alias: `adam-release`
- Password: Windows DPAPI-protected `password.xml` in the same private directory, readable by this Windows account. It cannot simply be copied to another machine and decrypted there.
- SHA-1: `97:D9:7F:30:42:B7:C4:67:1A:56:71:FB:03:92:05:10:13:4D:76:79`
- SHA-256: `DE:BB:48:B6:DA:CB:E5:06:BB:9F:BC:38:C9:20:82:71:7C:DC:73:B2:08:9D:C2:31:E7:A7:C4:72:ED:7B:AB:09`

Back up this private signing identity and its recoverable password securely before distribution. Reuse it for updates. This identity was introduced in 0.2.0, and 0.2.1 uses the same key for an in-place update. It cannot update an older installation signed by a different key. Export existing local data before uninstalling a differently signed APK.

## External launch requirements

- Register the release SHA-1/SHA-256 in the existing Firebase Android app and configure the matching Android OAuth client for Google sign-in. The supplied Firebase file only had a web OAuth client. Refresh `google-services.json` and rebuild if configuration changes.
- Enable the chosen Firebase Auth providers and verify login, reset, account deletion and Firestore rules with a designated test account. No real user account was used during this implementation.
- Validate Home Assistant with the intended home server and token. Local HTTP protocol tests are not a live home-device test.
- Test Gemini key verification with an authorized key. No model invocation or billing transaction is needed for the verification endpoint.
- Perform physical-device, accessibility and store-policy testing before commercial publication. Android 6 is the manifest floor, not a claim that every manufacturer/model was tested. Store signing, Play registration, final legal review and billing services remain release-owner responsibilities.

## 0.2.1 verification

The production web export, lint/type validation, all 38 static pages, Capacitor sync and signed Gradle release build passed on 7 October 2026. All **27 automated mobile tests** passed: 16 cover shared sync (deterministic conflicts, Unicode parity, explicit consent, account isolation, offline deletions, in-flight edits, invalid data and capacity); 11 cover local persistence/recovery and Home Assistant HTTP behavior.

Artifact: `releases/ADAM-0.2.1-release.apk`, **7,956,810 bytes**, SHA-256:

```text
ea4636827ab316203d4aa0d0d9dfbdccb08838ec736bfcd851ba2bcb2f500dde
```

The companion `.sha256` matches. Android SDK 35 `apksigner` verifies v1/v2 signatures using the same RSA 3072-bit certificate as 0.2.0. `aapt` confirms version 0.2.1/code 8, API 23 minimum/API 35 target, disabled Android backups and protected notification-listener service. Release and WebView debugging are disabled, mixed content is disabled, APK ZIP integrity passes, and the packaged assets contain the new sync feature and version labels. No native ABI-specific library is bundled. Evidence: `output/android/release-verification-0.2.1.json`, `apksigner-0.2.1.txt`, `badging-0.2.1.txt` and `manifest-0.2.1.txt` in that directory.

The old 0.2.0 APK and checksum remain unchanged. This update changes shared sync, relevant privacy copy and version labels; native permissions/background/camera behavior is inherited. **No emulator or physical-phone test was repeated for 0.2.1.** No live Firebase account or desktop-to-phone cloud exchange was exercised. The automated sync tests use isolated data and mocked Firestore transactions; live Google consent, Firebase provider/fingerprint configuration and deployed owner rules still require a designated test-account check. The native results below belong to 0.2.0.

## Previous 0.2.0 verification (retained)

The 0.2.0 production web export and signed Gradle release build passed on 7 October 2026. Eleven automated tests cover empty state, concurrent writes, persistent edits/deletion, corrupt-data protection and confirmed recovery, invalid writes, Home Assistant URL validation, actual local HTTP state/control exchanges, URL-bound credentials and unavailable/read-only entities. The existing nine development API tests and web/API/shell type checks also passed for that release. Native checks below belong to 0.2.0 and do not establish live sync verification for 0.2.1.

Browser checks cover onboarding, memory creation/edit/search/reload, gallery import/reload/export/delete, light-theme persistence, backup/export/restore/erase, and local face-profile save/reload/delete. All 28 original product routes loaded at 360px and 800px widths (56 checks), with no uncaught page errors or horizontal overflow. Notifications and reorganized Settings also pass at both widths. Home's Save a memory opens the editor. A deliberately damaged local record reaches Recover app data and returns to Home after a confirmed valid restore. Demo pairing, cancellation, Back, device rename and long names, sliders, and laptop control persistence passed at both widths. The final compact welcome screen keeps its primary action visible at 360 × 592; Say hello also fits on the device screen.

Native verification used an Android 15/API 35 x86_64 emulator with WebView 124. The signed APK before the last layout polish was installed without automatic permission grants (`-g` was not used); its first launch showed no permission UI and no runtime grant. The final UI polish build was reinstalled with existing QA data and grants preserved. Notification consent/background capture, camera process recovery and force-stop persistence were repeated on that final artifact. The checks below passed.

- Notification access: Enable opened Android's real access screen, Allow enabled the listener, and the app reported Running in background. A synthetic notification posted while ADAM was in the background appeared in its inbox. Pause suppressed storage, Start reading resumed it, and Clear history returned the count to zero.
- Camera: Take photo opened the real Android camera permission prompt. Deny produced a helpful settings/retry message; Allow opened the camera and saved the resulting image. The native share chooser displayed the image preview; no image was sent to an external recipient.
- Camera recovery: while the external camera was open, the app's background process was deliberately killed with `am kill`. Returning the photo produced the Photo recovered notice and exactly one recovered Moments item, alongside the two existing images. No face profile was silently created.
- Persistence: after `am force-stop` and relaunch, the saved memory detail "Saved locally and survives restart." and all three Moments items remained. Camera recovery items also persisted across relaunch.
- Runtime health: the checked logs contained no AndroidRuntime crashes or Capacitor console errors. Exit history contained no crashes or ANRs; recorded exits were deliberate QA kills or installation updates. The temporary Don't keep activities setting was restored after testing.

The final artifact is `releases/ADAM-0.2.0-release.apk`, **7,927,481 bytes**, with SHA-256:

```text
8eee704907ad0a33af958e56936c4e46540d208bc46fd50d8ed72fb3659edbcd
```

The checksum matches its companion `.sha256` file. Android SDK 35 `apksigner` verifies v1 and v2 signatures with the documented RSA 3072-bit release certificate. `aapt` confirms package/version, API 23 minimum, API 35 target, no debuggable application flag, disabled backups, and the notification service's `BIND_NOTIFICATION_LISTENER_SERVICE` protection. Packaged Capacitor configuration disables WebView debugging and mixed content. Bundled assets contain the camera recovery and interactive ADAM demo code. Artifact evidence is recorded in `output/android/release-verification-0.2.0.json`; the native test record and screenshot references are in `output/android/native-qa-2026-10-07.md`.

The local preview server serves Next's exported navigation text with `text/plain`; incorrect MIME handling previously forced full reloads. Windows directory attributes are normalized with PowerShell because Node chmod alone did not clear read-only flags reliably. Browser screenshots include `output/playwright/settings-release.png` and `output/playwright/notifications-release.png`.

These checks do not replace testing on physical phones, older Android/WebView versions, or manufacturer battery policies. Live Firebase accounts, the intended Home Assistant server, Gemini keys and real ADAM hardware were not exercised; the external launch requirements above still apply.
