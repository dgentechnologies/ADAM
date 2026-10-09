# ADAM Companion for Windows

Version **0.01**. The desktop companion connects to an ADAM that has already
been set up with the Android app. It keeps the monochrome dashboard and 3D ADAM
view, provides Windows controls and configurable touch shortcuts, and shares
selected account data with mobile through Firebase.

The current distribution is a portable Windows executable named
`adamV0.01.exe`. Packaging and verification details are in
[RELEASE.md](docs/RELEASE.md); changes and reasons are in
[UPDATE_REPORT.md](docs/UPDATE_REPORT.md).

## Project layout

Application code lives in `src/`, shipped UI and branding in `resources/`,
build tools in `scripts/`, and deliverables in `releases/`. Design references
and historical plans are separated from runtime files. See
[PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for the full directory map,
migration details and commands. The Android sibling is `../adam-mobile/`.

## Start using the app

1. Use a Windows 10 or Windows 11 **x64** PC with Microsoft Edge WebView2 Runtime.
   WebView2 is normally present on current Windows installations. If the window
   cannot open, install or repair it from
   [Microsoft](https://developer.microsoft.com/microsoft-edge/webview2/).
2. Put `adamV0.01.exe` in a permanent user-writable folder and run it. Python,
   Node.js and a terminal are not needed to run the packaged companion.
3. Complete ADAM's Wi-Fi setup on the Android app. Desktop does not provision
   the robot over Bluetooth. Keep ADAM and the PC on a trusted network that
   allows devices to reach one another.
4. Open the **profile icon → Profile & settings** and sign in with the same Firebase account used on mobile, or
   explore the desktop locally first. Google opens the system browser.
5. Open **My ADAM → Connection**. Select an account device where available, or enter
   ADAM's hostname/IP address. The default hostname is `adam-pi.local`.
   **Check connection** verifies the device without saving a new connection;
   **Connect** saves a verified target.
6. In the advanced connection options, enter ADAM's connection key for write
   access. Data and live-status ports default to **8766** and **8765**.
   A connection without a key is read-only. Changing the address or data port
   does not carry the old device's key to the new target.
7. Authorize laptop control from Connection. ADAM verifies this PC's protected
   pairing endpoint before accepting its control key. This step requires the
   matching Pi companion service described below. Enable individual controls
   in **Controls → Permissions** according to what ADAM should be able to do.
8. Optionally enable **Launch at login** in Profile & settings. Closing the window keeps
   the companion in the system tray; use **Quit** in the tray to stop it.

Windows may show a firewall prompt when the local companion service starts.
Permit access on the trusted/private network used by ADAM. The desktop agent
tries the saved TCP port (initially **8642**) and reserves a free port if it is
occupied. It saves and advertises the actual port through `_adam-laptop._tcp.local.`
mDNS. The native window verifies the process identity before loading the UI;
second launches locate the same verified instance even after a port change.
Manual pairing details show the current port. Reauthorize laptop control if an
already paired PC's address or port changes. No router port forwarding is needed. Local robot traffic uses
HTTP/WebSocket; use a trusted LAN or an already configured encrypted overlay
network rather than exposing these ports to the Internet.

## What works in this source release

| Area | Behavior |
| --- | --- |
| Setup and connection | Guided first-run screen; account devices; manual address; identity verification; explicit connect/disconnect; separate data and live-status state |
| Dashboard | One-viewport monochrome scene, actual ADAM assembly model, four minimal touch callouts, real connection and system state |
| Touch shortcuts | Long press on every pad; Touch 3 also offers double/triple tap. Single taps keep built-in reactions. Save locally, test an enabled PC action, then apply to ADAM |
| Controls | Five permission categories, focused local test panel, pause/resume and a History subtab |
| My ADAM | Connection, Planner and Memories grouped together; robot memories appear within Memories |
| Windows controls | Volume, mute/unmute, supported display brightness, media controls, screen locking and clipboard operations with per-action enable switches |
| Clock and to-do | Reads and edits the connected Pi's supported scheduling/to-do data; writes require its connection key |
| Memories and preferences | Local memories/people; account-scoped cloud merge; shared voice, wake-word and brain selections |
| Workspace | Runs an installed Codex CLI or Claude Code in a selected project folder; displays real output and completion/failure; cancels the owned task process |
| Background operation | Tray, pause/resume, startup opt-in, single-instance handling and clean service shutdown |
| Profile and settings | Bottom-of-rail profile entry for account, sync and app preferences; settings stays out of the main navigation |
| Activity | Bounded activity history and rotating local diagnostic logs with clipboard/coding content redaction |

Brightness support depends on the display and driver. Media keys affect the
active Windows media session. The app reports hardware failures instead of
inventing a successful result. Clipboard, screen locking and coding-task
dispatch start disabled. The Windows desktop does not request microphone,
camera, notification-reading or accessibility access for these controls.

If voice controls cannot find the laptop or Planner reports read-only access,
follow [Voice controls and planner access](docs/CONTROL_TROUBLESHOOTING.md) to
configure the Pi key and authorize the return connection to this computer.

Coding tools are separate installations with their own login and permissions.
Select an existing absolute project path and enable coding tasks before use.
The runner is noninteractive and does not bypass the installed tool's approval
rules. A task that requires an interactive approval must be handled in that
tool. Cancellation stops the launched task; it does not undo file edits already
made by it.

## Touch sensors and robot compatibility

The dashboard uses the requested visual placements. The saved protocol IDs and
firmware wiring remain unchanged; visual placement does not remap GPIOs or
override protected stop/wake behavior:

| Dashboard label | Protocol ID | Firmware wiring |
| --- | --- | --- |
| Left side | `touch1` | Left cheek, GPIO12 |
| Right side | `touch2` | Right cheek, GPIO14 |
| Top of head | `touch3` | Stop / petting A, GPIO15 |
| Back of head | `touch4` | Petting B, GPIO2; optional on three-pad hardware |

Touch 1, 2 and 4 expose **long press only**. Touch 3 exposes **double tap,
triple tap and long press**. Single taps are reserved for built-in reactions
such as slap/petting and protected stop/wake behavior. The API rejects custom
single-tap assignments. Existing long-press choices and Touch 3 double-tap
choices survive migration; old reserved mappings are removed and need reapplying.
Touch 4 is anchored to the center of the rear shell. Its marker is hidden when
the shell faces away; use **View rear** in its menu to inspect it.

Saving preferences on the PC does not change the robot. **Apply to ADAM**
requires `touch_assignments`, `touch_gesture_policy: 2`, and an exact acknowledgement of
the normalized mapping. **Do nothing** is an explicit assignment that suppresses
the mapped gesture. Alarm stop/snooze and the firmware's protected stop/wake
interactions retain priority. A local test performs the selected PC action
immediately; test it only when that action is appropriate.

Pi source extensions are included under `MP-MC codes/pi/adam` for protected
laptop pairing and persistent touch assignments. They have isolated protocol
tests, but **have not been deployed to or exercised on a physical ADAM** during
this desktop update. Older Pi software remains usable for its existing
capabilities; it cannot acknowledge the new pairing/touch features until the
corresponding source is deployed. Refer to:

- [Pi laptop pairing](../MP-MC%20codes/pi/docs/laptop_pairing.md)
- [Pi touch controls](../MP-MC%20codes/pi/docs/touch_controls.md)
- [Pi/desktop integration](../MP-MC%20codes/pi/docs/pc_app_integration.md)

The current pairing extension supports **one active laptop**. Reauthorizing
replaces that target. If the PC's LAN address changes, authorize it again.
Revocation must reach ADAM to remove its saved target; **Pause** immediately
blocks local control while disconnected. Disconnecting the desktop's data
connection and revoking the robot's laptop authorization are separate actions.

## Google login and the shared mobile account

Use the existing Firebase project **`adam-ai1`**. A second Firebase project or a
second user database is unnecessary. Android's Firebase registration does not
replace a Windows native OAuth client.

This release bundles the supplied native Desktop OAuth client. No additional
credential file is needed to launch its Google sign-in flow. For a separate
build or project, create an OAuth client with application type
**Desktop app** in the Google Cloud project backing `adam-ai1`. Download its
`installed` JSON and use it as `config/local/firebase-desktop-client.json` for source
or `firebase-desktop-client.json` beside the executable, or set `ADAM_GOOGLE_CLIENT_FILE` to that file. A release can
bundle this native client configuration. Enable Google's provider in Firebase
Authentication and configure the Google consent screen; a testing consent
screen restricts login to its test-user list. Enable email/password if using
the alternative email login/create-account/reset flow.

Use **Sync now** to exchange account data and check the visible pending/error
status. Signing in does not automatically import guest memories: **Import local
memories** asks for the destination-account confirmation. Account switching
selects a different local data file. Cloud failures preserve unsent local edits.

Mobile's shared-data implementation targets **0.2.1**; an older APK does not
implement the new sync contract. On mobile, enable account sync in Account and
choose **Sync now**. An APK is available at
`adam-mobile/releases/ADAM-0.2.1-release.apk`. Mobile packaging has its own release
record; the Windows smoke test does not establish live two-client sync.

Only `users/{uid}.companion` is merged: memories/people plus voice, wake-word
and brain selections. Photos, face profiles, notifications, network addresses,
API keys and pairing tokens are excluded. Robot schedules are a separate Pi
data path. Cloud selection of a paid/managed mode does not create a subscription
or transfer API credentials. The complete schema and account-rule requirements
are in [SYNC_PROTOCOL.md](docs/SYNC_PROTOCOL.md).

## Local data and privacy

The default data directory is `%APPDATA%\ADAM`. `ADAM_DATA_DIR` can select an
isolated directory for tests or a separate profile. Settings are atomically
saved in `settings.json`; private connection and Firebase session tokens are
stored under `credentials` using current-user Windows DPAPI. Passwords are not
persisted, and tokens are not returned by routine settings/status endpoints.
The explicit local **Show connection details** action is the exception for the
PC's pairing key.

Memories and preferences are ordinary local data files, not DPAPI-encrypted
content. Logs rotate at roughly 2 MB with three backups. Clipboard contents and
coding prompts/results are omitted from persistent activity logs. Coding task
output remains in bounded process memory for the dashboard. No analytics is
introduced by the companion account/sync modules.

`desktop-instance.json` is a temporary process/port locator, containing no
account or control credential. It is removed on clean shutdown. A stale locator
cannot authorize opening a different service with the same product name.

## Run or build from source

Use Python **3.11 x64** on Windows. In PowerShell from `adam-desktop`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe src\app.py
```

Optional flags are `--tray` (or `--minimized`), `--port 8642`,
`--install-startup`, and `--uninstall-startup`. The startup flags modify only
the current user's Startup shortcut and exit.

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:ADAM_DATA_DIR=Join-Path $PWD 'artifacts\qa\test-data'
$env:ADAM_DISABLE_HARDWARE='1'
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts\build.py
.\.venv\Scripts\python.exe scripts\smoke.py
```

The build uses PyInstaller and includes local UI assets/native bridge modules.
It writes `artifacts/dist/ADAM.exe`, then `releases/adamV0.01.exe` and its SHA-256 file.
WebView2 remains an external runtime prerequisite. See
[RELEASE.md](docs/RELEASE.md) for release verification and known external checks.

The original [feature spec](docs/planning/product-specification.md) and
[phase plan](docs/planning/implementation-plan.md) are historical design references.
This README and the update report describe current behavior; a proposed item
in an older plan is not evidence that the current release implements it.
