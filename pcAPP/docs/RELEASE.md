# Windows release 0.01

Expected artifact: **`pcAPP/releases/adamV0.01.exe`**. The release is a portable
Windows x64 app. It is not an MSI/setup installer and has no supplied Authenticode
signing certificate. The executable name requested for distribution is retained
independently of the internal PyInstaller output `dist/ADAM.exe`.

## Build prerequisites

- Windows x64 and Python3.11 x64.
- Dependencies from `requirements-dev.txt`; this includes the pinned runtime
  dependencies and PyInstaller6.16.0.
- Local assets in `static`, `assets`, and `logo.png`.
- The project's native Desktop OAuth client in `firebase-desktop-client.json`
  for the release's bundled Google configuration. Use an `installed` client
  for Firebase project `adam-ai1`, not `google-services.json` or a service-account
  key. Its contents must never be printed into a build log.
- Microsoft Edge WebView2 Runtime for launching/testing the desktop window.

From `pcAPP` in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe build_exe.py
```

`build_exe.py` reads the version from `config.py`, builds with PyInstaller, and
copies the output into `releases` with a matching `.exe.sha256` file. Verify the
final file rather than an older `dist` output:

```powershell
Get-FileHash -LiteralPath .\releases\adamV0.01.exe -Algorithm SHA256
Get-Content -LiteralPath .\releases\adamV0.01.exe.sha256
Get-AuthenticodeSignature -LiteralPath .\releases\adamV0.01.exe
```

A checksum establishes which bytes were delivered; it is not publisher signing.
Do not include `.env`, private data directories, Firebase sessions,
`touch_assignments.json`, or `laptop_pairing.json` in distribution/diagnostics.

## Installation, upgrade and removal

Copy the portable executable to a permanent folder and launch it as the normal
Windows user. There is no requirement to run as administrator. User data lives
under `%APPDATA%\ADAM`, separate from the executable. Enable launch at login in
Settings only after putting the file at its final location.

To update, Quit from the tray, replace the executable and reopen it. If its
path/name changed, toggle startup off and on to update the shortcut. Keep a
backup of the previous executable during rollout. Do not replace or discard the
user's data directory as an upgrade step. DPAPI credentials are tied to the
Windows user/machine; copying them to another PC is not a supported sign-in
migration.

To remove the portable app, disable launch at login, Quit, and remove the
executable. Account sign-out, robot revocation and local-data deletion are
separate choices; deleting the executable alone does not revoke ADAM's saved
authorization or erase `%APPDATA%\ADAM`.

## Final verification record

This document records the process and scope. The packaging owner must update
this section after verifying the **final** `adamV0.01.exe`; a previous build or
a source-browser check is not evidence for the final binary.

| Check | Status at documentation handoff |
| --- | --- |
| Desktop automated suite | Tests added; final aggregate run pending after integration fixes |
| Pi protocol and gesture suite | Isolated tests passed; no physical deployment |
| Mobile shared-schema tests/typecheck | Passed during source implementation; APK packaging separate |
| Portable EXE build/version/checksum | Packaging owner to record after final build |
| Packaged WebView2 UI smoke | Packaging owner to record against final EXE |
| Google consent, email and Firebase rules | Requires configured project and a live test account; not tested here |
| Real mobile↔desktop account exchange | Requires new mobile sync build and live test account; not tested here |
| Physical ADAM controls and touch pads | Pi source not deployed; hardware acceptance pending |
| Publisher signing | No certificate supplied; unsigned release |
| Broader Windows10/11 display/audio compatibility | Hardware matrix testing pending |

For packaged smoke testing, use an isolated `ADAM_DATA_DIR` and
`ADAM_DISABLE_HARDWARE=1`; confirm fresh setup, navigation, local persistence,
loopback session guards, empty/offline states, duplicate-launch activation and
tray lifecycle. Then test supported hardware controls with an explicit human
test session. Do not grant live account access merely to perform a UI smoke.

## External integration acceptance

1. In Google Cloud/Firebase `adam-ai1`, verify Desktop OAuth, Google provider,
   consent audience/test users, and the alternative email/password provider.
2. Confirm deployed Firestore rules allow a user only their `users/{uid}`
   document and owned device records. Never weaken owner rules to make a test
   pass. Querying devices requires `ownerUid` to match the signed-in UID.
3. With the mobile sync build installed, use the same test account on both
   clients. Exercise create/edit/delete, offline edits, reconnect merge,
   sign-out and account switching with non-sensitive test memories.
4. Deploy the reviewed Pi pairing/touch source through the normal robot release
   process, preserving its separate configuration. Check host/API identity,
   bad keys, authorization, revocation across restart and physical gesture
   priority. The desktop task has not performed that deployment.
5. Test WebView2 startup and controls on representative Windows10/11 PCs,
   including a missing/unsupported brightness device, multiple displays and
   network disconnection/reconnection.

For users, the concise setup guide is [README.md](../README.md). The behavior
and changes are described in [UPDATE_REPORT.md](UPDATE_REPORT.md); the exact
shared schema is [SYNC_PROTOCOL.md](SYNC_PROTOCOL.md).
