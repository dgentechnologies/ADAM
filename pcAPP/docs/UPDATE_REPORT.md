# ADAM Windows companion 0.01 — update report

This update turns the desktop UI into a functioning Windows companion around
the existing action registry. The product flow is **mobile provisions ADAM →
desktop connects to the configured robot → user enables laptop controls**.
The black/white aesthetic and 3D ADAM dashboard remain the visual foundation.

## Changes and reasons

| Change | Why it was needed | Result |
| --- | --- | --- |
| Organized desktop navigation, setup guide and local icons/font | Disconnected panels and remote UI dependencies made setup and offline use unreliable | Clear Dashboard, Connection, Actions, Clock, Memories, Workspace, Activity and Settings views |
| Real connection service | Public discovery traffic and stale state must not imply a connected robot | Verifies ADAM identity/API, distinguishes data/live status, reconnects independently of the UI |
| Device-bound connection keys | Editing the address must not transmit the previous robot's key to a new endpoint | Key reuse requires the same hostname and data port; identity checks never include the key |
| Explicit authorize/revoke handshake | Data connectivity alone does not give ADAM permission to control the PC | Pi verifies the protected desktop identity; desktop checks exact acknowledgement before claiming pairing/revocation |
| Real touch assignment workflow | A dropdown selection alone cannot change firmware behavior | Save locally, test an enabled action, apply to the Pi, and accept only its exact returned mapping |
| Firmware-based sensor labels | Top/back labels were not supported by the board wiring | Left/right cheek and petting/stop pads match the ESP32 GPIO mapping |
| Windows hardware worker and error propagation | COM threading and unsupported displays can fail at runtime | Serialized hardware work, visible failures, and no fabricated success values |
| Action permissions and desktop/LAN separation | Account/settings data should not be exposed by the laptop action service | Local session for private UI endpoints; authenticated LAN action routes; sensitive actions disabled initially |
| Native Google OAuth and Firebase identity | Android credentials alone do not implement desktop Google sign-in | System-browser PKCE/state flow; same Firebase UID as mobile; DPAPI-protected session restoration |
| Shared companion data schema | Independent desktop/mobile local stores do not synchronize themselves | Deterministic per-record merge, tombstones, conflict retries, explicit guest import and account isolation |
| Owner-scoped device catalog | Account devices must come from the signed-in owner | Owner query, defensive result filtering, bounded list and account-switch checks |
| Real coding CLI runner | Simulated tasks and guessed completion states are misleading | Installed tool/workspace checks, structured output, explicit completion/error, bounded history and task-scoped cancellation |
| Managed Windows lifecycle | A background companion must survive closing its window and release resources on Quit | Tray operation, single instance, startup opt-in, server readiness checks and cleanup |
| Reproducible portable packaging | Running source is insufficient for a user-facing Windows deliverable | Version0.01 metadata, `adamV0.01.exe`, bundled local assets and checksum generation |

## Responsiveness and resource use

The Pi architecture remains asynchronous. Touch recognition uses bounded
queues; laptop dispatch runs in a separate worker with stale-event expiry.
Telemetry reports completed handling instead of causing the desktop to execute
the same shortcut a second time. Alarm dismissal and stop/wake gestures have
priority over user mappings.

Desktop connection monitoring and WebSocket reception use separate workers with
per-run cancellation. Stop/restart does not wait for a network timeout. Cloud
sync runs outside the UI request path and concurrent sync requests do not start
duplicate transfers. Settings changes use locked atomic file replacement.

Dashboard polling is sequential, view-aware and pauses while the document is
hidden. The display keeps its most recent result when a background refresh
fails; interactive actions show their error. Activity, log buffers, task history,
network response sizes and process output are bounded. These are engineering
limits to reduce runaway memory/work; no comparative CPU, battery or latency
benchmark is claimed.

## Account and mobile behavior

Both clients use Firebase project `adam-ai1`. Desktop requires a native
**Desktop app OAuth client** from that project's Google Cloud configuration.
The existing Android Firebase registration remains in use. No service-account
credential belongs in either client.

The shared field is `users/{uid}.companion`; it contains memories/people and
voice/wake-word/brain selections. Updating it preserves all other user-document
fields. Both clients preserve offline edits, use deletion tombstones and reject
malformed/unsupported schema data. Account changes are checked before accepting
network results. Local guest data requires explicit destination-account import.

Mobile now has an opt-in account-sync panel and manual **Sync now** workflow in
source, intended for version0.2.1. It does not upload photos, face data,
notifications, pairing keys or API credentials. This report does not certify a
new APK; inspect the mobile release record for packaging verification.

## Pi source additions

The new Pi pairing service validates a private/loopback/Tailscale address,
verifies `/pair/verify` with the PC's key, and persists its selected laptop
atomically. Revocation persists a marker so the old environment/mDNS fallback
cannot silently restore the revoked PC after restart. Only one laptop is active
at a time. Existing never-paired installations retain their legacy path.

The touch controller recognizes tap, double tap and hold independently of the
Gemini connection lifetime. Mappings persist atomically and dispatch through
the existing authenticated laptop action client. There are small call-site
extensions in Pi startup/session/sync code; the robot's main architecture is
preserved. These source changes have **not been deployed to a physical Pi**.

## Verification and its limits

The test suite includes isolated configuration/DPAPI checks, backend session and
LAN boundaries, identity and cloud merge tests, connection tests against a
loopback Pi, coding-runner fixtures, and pairing/touch endpoint integration.
The integration fixture invokes the actual protected `/pair/verify` route and
uses real local HTTP for Pi requests; outgoing traffic is restricted to that
fixture. It tests invalid keys, changed device identity, exact pair/unpair
acknowledgements, local touch persistence, malformed input and failed writes.
Catalog tests use fake Firebase responses to test owner filtering and account
switches. No test performs a physical laptop control or accesses a live account.

The Pi suite checks persistence/revocation, protected identity verification,
gesture priority, bounded asynchronous dispatch and protocol compatibility.
The shared mobile schema has transaction/concurrency and account-switch tests,
plus TypeScript checking. The release record is the authority for the final
test/build result and packaged UI smoke evidence.

External acceptance remains necessary for Google consent with the deployed
project, Firebase email delivery/rules, real two-device cloud exchange, physical
ADAM pairing/gestures and a broader Windows hardware matrix. There is no code
signing certificate supplied for this build. A successful source test or portable
build is not a signed installer or certification across all Windows PCs.

## How this release differs from the original plans

| Original plan item | Current scope |
| --- | --- |
| Windows/macOS/Linux packages | Windows x64 portable executable only |
| Signed installer and automatic updates | No signed installer or automatic updater; update the portable executable manually |
| Startup enabled by installer | Explicit user opt-in through Settings/current-user Startup shortcut |
| Six-digit voice/mobile pairing code | Verified address + connection key; guided local authorize handshake |
| Multiple paired laptops and mobile revocation list | One active laptop in the new Pi pairing store |
| Pi head servo angles mirrored in 3D | No claim of exact physical pan/tilt tracking |
| Screen capture, shutdown/restart/sleep, focus mode | Future features, not shipped actions |
| Interactive coding approval through ADAM | Noninteractive installed CLI tasks; no fabricated interactive prompt handling |
| All local content synchronizes automatically | Defined companion schema only; separate local/Pi data paths and mobile opt-in sync |

The historical plan documents are retained for intent and future work. Their
broader ambitions do not override the implemented behavior documented here.
