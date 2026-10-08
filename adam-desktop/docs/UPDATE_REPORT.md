# ADAM Windows companion 0.01 — update report

This update turns the desktop UI into a functioning Windows companion around
the existing action registry. The product flow is **mobile provisions ADAM →
desktop connects to the configured robot → user enables laptop controls**.
The black/white aesthetic and 3D ADAM dashboard remain the visual foundation.

## Project organization update

The workspace is now `adam-desktop`, alongside `adam-mobile`. Runtime code,
shipped resources, build scripts, packaging metadata, local configuration,
design references and generated artifacts have dedicated directories. An
ADAM folder icon identifies the desktop project in Explorer. Existing files
and previous build evidence were preserved. See
[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for the complete layout and reasons.

## Changes and reasons

| Change | Why it was needed | Result |
| --- | --- | --- |
| Single-viewport dashboard and slim navigation rail | The prior card layout diluted the product scene and required too much visual scanning | A large ADAM scene with four single-line touch dropdowns; account and settings live under the profile icon |
| Four primary workspaces | Seven peer tabs scattered related features | Dashboard; Controls with Permissions/History; My ADAM with Connection/Planner/Memories; Workspace; Profile remains a separate bottom entry |
| Categorized permission list and focused test panel | Eighteen equal cards repeated descriptions and inputs and hid the relationship between permissions and testing | Sound & media, Display, Clipboard, Security and Workspace groups; clear names, search, saved toggles, pause/resume and one selected control at a time |
| Quieter secondary layouts | Repeated panels and competing forms made the app harder to scan | Collapsible account device picker and schedule composer, staged connection, consolidated memories, numbered workspace fields, and a dedicated output pane |
| Exact local service ownership | A shared fixed port could collide with another desktop app | An exclusively reserved socket, free-port fallback, persisted actual pairing port, per-process identity verification and verified second-launch activation |
| Actual ADAM model and official branding | A procedural substitute and rough icons did not represent the physical product accurately | Bundled assembly GLB, official ADAM wordmark/A app icon, Lucide interface icons and Google's four-color sign-in mark |
| Guided setup and locally bundled UI assets | Disconnected panels and remote dependencies made setup and offline use unreliable | Connection, permissions, planner, memories and history organized within the four primary workspaces, with fonts, icons and model available offline |
| Real connection service | Public discovery traffic and stale state must not imply a connected robot | Verifies ADAM identity/API, distinguishes data/live status, reconnects independently of the UI |
| Device-bound connection keys | Editing the address must not transmit the previous robot's key to a new endpoint | Key reuse requires the same hostname and data port; identity checks never include the key |
| Explicit authorize/revoke handshake | Data connectivity alone does not give ADAM permission to control the PC | Pi verifies the protected desktop identity; desktop checks exact acknowledgement before claiming pairing/revocation |
| Real touch assignment workflow | A dropdown selection alone cannot change firmware behavior | Save locally, test an enabled action, apply to the Pi, and accept only its exact returned mapping |
| Rear-shell touch anchor and protected gestures | The old fourth anchor sat on the side, and single-tap customization could replace emotional reactions | Touch 4 sits on the actual rear shell; all pads offer hold, only Touch 3 offers double/triple tap, and single taps remain built in |
| Minimal glass touch dropdowns | Repeated titles, lock descriptions and instructions overwhelmed a small interaction | Translucent blurred panels, subtle edge highlights, gesture selectors, necessary values and local-test buttons; the existing footer handles save/apply |
| Enlarged capsule eyes | Small eyes and a smile did not match ADAM's reference face | Larger paired capsules, no mouth, natural irregular blinks and coordinated gaze shifts; reduced-motion support remains |
| Windows hardware worker and COM cleanup | Audio/display libraries initialize COM state and can fail during worker teardown | Initialize library lifetime state on the application thread; initialize/release worker COM separately; supported hardware reads and clean shutdown verified |
| Live telemetry and cached host metrics | Stale emotion state, duplicated touch execution and thread-local CPU sampling gave misleading behavior | Live emotion/speaking/touch status, no re-execution of handled Pi touches, and shared two-second CPU/network samples with correct Mbps units |
| Action permissions and desktop/LAN separation | Account/settings data should not be exposed by the laptop action service | Local session for private UI endpoints; authenticated LAN action routes; sensitive actions disabled initially |
| Native Google OAuth and Firebase identity | Android credentials alone do not implement desktop Google sign-in | System-browser PKCE/state flow; same Firebase UID as mobile; DPAPI-protected session restoration |
| Shared companion data schema | Independent desktop/mobile local stores do not synchronize themselves | Deterministic per-record merge, tombstones, conflict retries, explicit guest import and account isolation |
| Owner-scoped device catalog | Account devices must come from the signed-in owner | Owner query, defensive result filtering, bounded list and account-switch checks |
| Real coding CLI runner | Simulated tasks and guessed completion states are misleading | Installed tool/workspace checks, structured output, explicit completion/error, bounded history and task-scoped cancellation |
| Managed Windows lifecycle | A background companion must survive closing its window and release resources on Quit | Tray operation, single instance, startup opt-in, server readiness checks and cleanup |
| Reproducible portable packaging | Running source is insufficient for a user-facing Windows deliverable | Version 0.01 metadata, `adamV0.01.exe`, bundled local assets and checksum generation |

## Visual design and model provenance

The six supplied screenshots and `design/references/adam_dashboard_v2.html` informed the
monochrome palette, slim icon rail, product-centered scene and restrained
controls. Their layouts were combined into an original screen. Decorative
telemetry from the references was not presented as live device data.

The scene loads the existing `resources/static/models/adam-body.glb` assembly export
(Blender glTF exporter 3.5.30). Source design assets were located at
`design/ADAM_model.fbx` and `design/adam1.blend`; this update uses the existing
GLB rather than claiming a new conversion. Body, head, face screen, trim and
ADAM lettering use the actual assembly geometry. The face uses enlarged white
capsules (142×40 pixels on a 512-pixel face texture at rest) with coordinated
gaze, quick blink closure and softer reopening. Blinks occur at irregular
intervals, occasionally twice; live emotions subtly change the capsule shapes.
There is no mouth or speaking waveform. Physical servo pose tracking is not
claimed. A procedural fallback handles model-loading failure.

Four projected leader lines stay attached to the head as the view rotates.
Each callout opens only its configurable gestures: hold on Touch 1/2/4 and
double/triple/hold on Touch 3. Menus contain an action, optional value and local
test, remain inside the viewport, and close with Escape
or an outside click. Keyboard rotation and Reset view are available. Saving
keeps preferences on this PC; applying requires the device's acknowledgement.
Touch 4 uses a ray intersection with the center of the rear head shell. The
rear marker is not drawn through the front of the face; its menu has a View rear
button. The viewport layout, product framing and four labels remain intact.

The official ADAM wordmark is reused in the rail and the complete first A is
used for the Windows icon. Interface symbols come from Lucide 0.469.0, with
its license in `resources/static/licenses/LUCIDE.txt`; Google's own multicolor mark
identifies Google sign-in.

The touch panels use a translucent gradient, 28-pixel backdrop blur and light
inner/border highlights. Form fields share that restrained glass treatment;
native action selectors retain keyboard support and readable option surfaces.
An opaque fallback is provided for renderers without backdrop-filter support.

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

Port selection keeps the chosen socket bound while handing it to Waitress;
there is no check-free-then-reopen race. On Windows it requests exclusive socket
ownership. The app verifies a random per-launch identity and process ID before
opening WebView2. The instance locator lets a second launch find the actual port
without contacting a different app on the preferred port. A changed port is
saved for subsequent launches and used by pairing and mDNS.

Dashboard polling is sequential, view-aware and pauses while the document is
hidden. The display keeps its most recent result when a background refresh
fails; interactive actions show their error. Activity, log buffers, task history,
network response sizes and process output are bounded. These are engineering
limits to reduce runaway memory/work; no comparative CPU, battery or latency
benchmark is claimed.

The 3D renderer caps pixel ratio at 1.5 and animation at approximately 30 fps.
It stops scheduling frames while hidden, offscreen or after context loss;
reduced-motion mode renders on demand. In browser verification the actual
assembly used 46 draw calls and 127,366 triangles. Image assets were resized
for their display role and the duplicate model was removed from packaging.
The final bundle contains the model once under `static/models`.

## Account and mobile behavior

Both clients use Firebase project `adam-ai1`. Desktop requires a native
**Desktop app OAuth client** from that project's Google Cloud configuration.
The supplied Desktop OAuth client is bundled in the executable. The existing
Android Firebase registration remains in use. No service-account
credential belongs in either client.

The shared field is `users/{uid}.companion`; it contains memories/people and
voice/wake-word/brain selections. Updating it preserves all other user-document
fields. Both clients preserve offline edits, use deletion tombstones and reject
malformed/unsupported schema data. Account changes are checked before accepting
network results. Local guest data requires explicit destination-account import.

Mobile's shared-data source includes an opt-in account-sync panel and manual
**Sync now** workflow for version 0.2.1. The schema excludes photos, face data,
notifications, pairing keys and API credentials. The 0.2.1 APK is available in
`adam-mobile/releases`; its packaging record is separate from this Windows
release. Real account exchange between the two clients remains an acceptance
check with a live account.

## Pi source additions

The new Pi pairing service validates a private/loopback/Tailscale address,
verifies `/pair/verify` with the PC's key, and persists its selected laptop
atomically. Revocation persists a marker so the old environment/mDNS fallback
cannot silently restore the revoked PC after restart. Only one laptop is active
at a time. Existing never-paired installations retain their legacy path.

The touch controller recognizes tap, double tap and hold independently of the
Gemini connection lifetime, with an added triple-tap window on Touch 3. A double
tap waits for a possible third tap; a hold cancels pending tap sequences. Single
taps retain their built-in reactions, petting chords and protected stop/wake
behavior. Both desktop and Pi APIs reject disallowed gesture mappings. Old
stored mappings migrate by retaining only the now-configurable gestures.
Mappings persist atomically and dispatch through
the existing authenticated laptop action client. There are small call-site
extensions in Pi startup/session/sync code; the robot's main architecture is
preserved. These source changes have **not been deployed to a physical Pi**.

## Verification and its limits

The final desktop source suite passed **96 tests**. The revised Pi touch suite
passed **16 tests**, including single-tap protection, double/triple separation,
hold cancellation, petting chords, storage, async dispatch and API validation.

Browser QA covered all eight views at
**860×620, 1060×740 and 1440×960** with no horizontal overflow. The dashboard
fit one viewport at all three sizes. All four touch menus stayed inside the
viewport. Permission persistence, paused/disabled actions, cross-category
search, control success/error responses using fixtures, local memory
create/delete, and workspace permission guidance were checked. Profile
navigation, Google branding, actual model loading,
projected connectors and paused hidden rendering were checked. The packaged
WebView2 results and exact artifact checksum are recorded in
[RELEASE.md](RELEASE.md).

The enlarged face was sampled through a complete blink (open → closed → open),
with coordinated gaze movement and zero mouth pixels. Reduced-motion mode
stopped animated gaze/blinks. Touch 3's triple-tap value survived save/reload,
and Touch 4's point was verified against the rear shell intersection.

A separate read-only Windows hardware check returned volume and brightness
successfully and shut the hardware worker down without COM cleanup errors.
It did not change the user's volume or brightness.

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
