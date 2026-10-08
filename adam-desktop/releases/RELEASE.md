# Windows release 0.01

Artifact: **`adam-desktop/releases/adamV0.01.exe`**. The release is a portable
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

From `adam-desktop` in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:ADAM_DATA_DIR=Join-Path $PWD 'output\test-data'
$env:ADAM_DISABLE_HARDWARE='1'
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe build_exe.py
.\.venv\Scripts\python.exe test_exe.py
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
Profile & settings only after putting the file at its final location.

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

Verification date: **8 October 2026**. The following distinguishes automated
local checks from external account and physical-device acceptance.

| Check | Result |
| --- | --- |
| Desktop automated suite | 96 passed; `output/desktop-tests.log` |
| Pi touch suite | 16 passed; fixed single taps, double/triple/hold classification, priority, migration-compatible storage, async dispatch and API checks; no physical deployment |
| Responsive browser checks | All 8 views at 860×620, 1060×740 and 1440×960; no horizontal overflow; dashboard fits one viewport |
| Controls and navigation | 18 controls in 5 groups; 4 primary workspaces; permission save/reload, pause, search, response/error fixtures and workspace permission guidance passed |
| Memories | Local create/reload/delete and consolidated robot-memory section passed |
| Touch and face | Per-pad gesture choices, saved Touch 3 triple value, rear-shell anchor, enlarged eyes, full blink, gaze, no mouth and reduced-motion behavior passed |
| Glass menus | All 4 menus within all 3 window sizes; 28px blur and minimal content verified |
| Occupied port and native source lifecycle | Passed with another service occupying the requested port; exact process identity, saved fallback port, verified second launch, tray/restore/shutdown |
| Mobile package | `adam-mobile/releases/ADAM-0.2.1-release.apk` exists; mobile packaging has a separate record |
| Portable EXE build/version/checksum | Passed; build exited 0; Windows x64, version 0.01, 35,484,995 bytes; checksum verified against the sidecar; `output/windows-build.log` |
| Packaged WebView2 UI smoke | All 14 checks passed; process exited 0; `output/native-exe-x98gdxsg/report.json` and `native-window.png` |
| Google consent, email and Firebase rules | Requires configured project and a live test account; not tested here |
| Real mobile↔desktop account exchange | Requires both clients and a live test account; not tested here |
| Physical ADAM controls and touch pads | Pi source not deployed; hardware acceptance pending |
| Publisher signing | No certificate supplied; unsigned release |
| Windows hardware | Read-only volume/brightness check passed on this PC; a broader Windows 10/11 hardware matrix remains untested |

### Delivered artifact identity

- Filename: `adamV0.01.exe`
- File/product version: `0.01`
- Architecture: Windows x64 (PE machine `0x8664`)
- Size: **35,484,995 bytes**
- Authenticode status: **NotSigned**
- SHA256: `f2a35131d1036085b7db0523c6fd877d6602f1de01ddf8f90703889593c3bf62`
- Checksum sidecar: `adamV0.01.exe.sha256`

The final executable's smoke test verified that served UI assets match the
current source bytes, including the glass dropdown styling. It also checked
occupied-port fallback, runtime identity, protected endpoints, bundled Google
configuration, touch gesture validation/persistence, model and menu rendering,
grouped navigation, second-launch activation, close-to-tray, restore and clean
shutdown. All checks used isolated app data; no live account or physical robot
was required.

For packaged smoke testing, use an isolated `ADAM_DATA_DIR` and
`ADAM_DISABLE_HARDWARE=1`; confirm fresh setup, navigation, local persistence,
loopback session guards, empty/offline states, duplicate-launch activation and
tray lifecycle. Then test supported hardware controls with an explicit human
test session. Do not grant live account access merely to perform a UI smoke.

Browser evidence: `output/workspace-final-qa.log`, `output/glass-qa.log` and
screenshots under `output/playwright/release-*`. Control success/failure UI
checks use explicit browser response fixtures, not physical hardware changes.
Account and robot connectivity tests use isolated local services.

The current build environment reports Python **3.11.0rc2 x64** and
PyInstaller **6.16.0**. This records the actual toolchain used; a clean stable
Python toolchain and publisher signing are recommended for a wider commercial
distribution pipeline.

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

For users, the concise setup guide is [START_HERE.txt](START_HERE.txt). The behavior
and changes are described in [UPDATE_REPORT.md](UPDATE_REPORT.md); the exact
shared schema is [SYNC_PROTOCOL.md](SYNC_PROTOCOL.md).
