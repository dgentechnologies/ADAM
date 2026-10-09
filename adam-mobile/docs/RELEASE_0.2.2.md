# Android 0.2.2 and Windows 0.02 verification

These releases share [companion protocol version 2](../../shared/COMPANION_PROTOCOL.md).
Install both updates before syncing the same account. Existing memories migrate;
device renames and deletions retain stable identities. Physical BLE remains
simulated and shared clock entries are saved plans, not physical robot alarms.

## Automated and browser checks

- Mobile startup, shared sync and cross-language protocol: 24 tests passed.
  Coverage includes corrupt/stalled storage, all static route targets, version-1
  migration, Python/TypeScript parity, multiple devices, planner deletions,
  offline changes, in-flight edits and account switching.
- Desktop backend shared routes: 3 tests passed. Desktop identity: 18 passed;
  version-2 local records/BLE: 2 passed. Desktop UI: 6 passed, with the additional
  draft-preservation assertion passing after its fix.
- Full desktop suite: 109 passed initially; one unchanged connection timing
  test failed during heavy builds and passed on isolated rerun (2.18 seconds).
- Browser startup checks passed for damaged-data backup, resuming device naming,
  unavailable-storage Retry, and a rejected email login remaining signed out.
- Real local desktop backend/browser checks passed for plans, two saved devices,
  rename, simulated BLE and persistence after reload. The check exposed and
  fixed a navigation refresh clearing an unsaved device name.

Screenshots and executable browser check scripts are under the workspace's
`output/playwright/` directory, prefixed `android22-` and `desktop22-`.

## Live account acceptance still required

No live Google account or real two-client Firestore exchange has been verified.
The supplied Android configuration contains only a web OAuth client. Register
the release package/certificate in the existing project:

1. Open [Firebase project settings](https://console.firebase.google.com/project/adam-ai1/settings/general).
   Select the Android app `com.dgentechnologies.adam`.
2. Add SHA-1 `97:D9:7F:30:42:B7:C4:67:1A:56:71:FB:03:92:05:10:13:4D:76:79` and
   SHA-256 `DE:BB:48:B6:DA:CB:E5:06:BB:9F:BC:38:C9:20:82:71:7C:DC:73:B2:08:9D:C2:31:E7:A7:C4:72:ED:7B:AB:09`.
3. Enable Google in Firebase Authentication. Enable Email/Password for the email
   alternative. Confirm an Android OAuth client matches the package/SHA-1 and
   that the consent screen permits the intended account.
4. If Firebase supplies updated configuration, replace
   `apps/mobile-shell/android/app/google-services.json` and rebuild the APK.
5. Sign in on both apps with the same account. On Android, enable account sync,
   confirm import, and tap Sync now. Check create, edit and delete in both
   directions for memories, to-dos, clocks and named devices. Repeat after an
   offline edit and reconnection. Owner rules must permit only `users/{uid}` for
   that authenticated UID. Do not weaken those rules to make a test pass.

The implementation tests use isolated local data and mocked cloud transport;
they do not certify deployed Firebase configuration or successful consent.
