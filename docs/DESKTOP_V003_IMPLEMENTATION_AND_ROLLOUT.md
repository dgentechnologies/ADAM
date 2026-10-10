# Desktop 0.03 implementation and rollout

This change implements the desktop login and LAN onboarding flow against
`DESKTOP_LOGIN_AND_LAN_ONBOARDING_FLOW.md` and the final architecture/data
contract. It is a coordinated desktop, Pi and head-firmware update, not an
executable-only upgrade. The final contract remains the production target;
the rollout gates below are not claimed complete.

## Included behavior

- Mandatory Google or email/password Firebase sign-in, followed by cloud-owned
  physical-device selection. Simulators and unowned LAN advertisements cannot
  unlock the desktop. The existing dark visual theme continues through login,
  device discovery and authorization.
- Discovery matches the canonical device ID **and** durable hardware identity.
  A code displayed on ADAM binds first authorization to its TLS certificate.
  Authenticated HTTPS and WSS identity checks must both pass before controls open.
- Each account/device grant is kept in Windows DPAPI storage. Credentials are
  not stored in ordinary connection settings. Sign-out, ownership mismatch,
  expired authorization or lost verified connectivity closes access.
- Shared todos and schedules use `users/{uid}/todos` and `schedules`; facts and
  people use `devices/{deviceId}/memoryFacts` and `memoryPeople`, matching the mobile paths.
  The per-UID durable local outbox keeps offline edits and tombstones. Conditional
  Firestore writes preserve conflicting versions for explicit resolution.
- The Pi bridge exchanges full records with revision preconditions. A successful
  cloud write does not imply robot application: an exact persisted Pi revision
  acknowledgment is required. Biometric payloads and raw transcripts are excluded.
- The legacy desktop envelope is retained for inspection; new shared writes no
  longer target it. Migration inventory is available without destructive import.

## Installation and first authorization

1. Install the matching Pi Python changes and ESP32-S3 head firmware. Retain
   existing data backups and install the Pi's existing dependencies, including
   `cryptography`. Set its OS timezone to the owner's chosen IANA timezone.
2. A trusted administrator must provision the physical robot locally, from its
   `adam` Python directory:

   ```sh
   python desktop_pairing.py provision --uid FIREBASE_OWNER_UID --device-id ADAM-CANONICAL-ID --timezone Asia/Kolkata
   ```

   This creates private `.desktop-pairing` identity/certificate files. Preserve
   that directory across updates. Register its **public** `hardwareId` as
   `hardwareSerial` on the matching cloud device using trusted administration;
   confirm the device's cloud owner is the same Firebase UID. Never upload the
   directory, private key, grants or pairing codes to Firestore or Git.
3. Restart ADAM. Permit local TCP 8766 (HTTPS), TCP 8765 (WSS), desktop-agent
   connectivity and mDNS on the trusted LAN. Provisioned robots reject the old
   shared `SYNC_TOKEN` for companion access.
4. Run `adamV0.03.exe` on Windows with Microsoft Edge WebView2 Runtime installed.
   Sign in with the same owner account and select ADAM.
5. Ask ADAM to **pair my desktop**, or locally run
   `python desktop_pairing.py code`. Enter the code shown on its screen. It
   expires after five minutes, permits five guesses and authorizes one client.
   Future sessions reuse the protected grant until expiry or revocation.
6. In the Connection controls, authorize this computer for laptop control and
   resume controls. This registers its address and control credential with ADAM
   so voice brightness/audio requests can reach the desktop agent.
7. Use **Sync now** for cloud data and **Sync with ADAM** for the physical bridge.
   Resolve conflicts before expecting the corresponding record to converge.

`revoke-all` revokes desktop sessions locally. `transfer --uid NEW_UID` revokes
sessions and changes local ownership; cloud ownership must be updated separately
by trusted administration. A grant lasts 30 days and then requires authorization
again. The desktop never self-claims cloud ownership.

## Build and configuration

Build with Windows Python 3.11 and `adam-desktop/requirements.txt`, PyInstaller,
then `python scripts/build.py` in `adam-desktop`. Supply the installed-app Google
OAuth configuration through ignored `config/local/firebase-desktop-client.json`.
The native OAuth client configuration is bundled for login; it is not a Firebase
Admin credential. No service-account key or user refresh token belongs in a build.
The script emits the executable and a SHA-256 sidecar in `adam-desktop/releases`.
The cloud build uses Windows CPython/PyInstaller under Wine; this produces a
Windows PE executable but does not substitute for native Windows acceptance.

## Remaining production gates

- Deploy and emulator-test the final Firestore rules/indexes and trusted ownership
  provisioning/transfer workflow. This change does not deploy Firebase resources.
- Approve and execute the cross-client legacy migration before removing old data.
  Shared notes/preferences and installation metadata proposed by the contract
  are not yet enabled as production canonical collections.
- Add trusted cloud execution-state reporting, scheduled background reconciliation,
  retry/backoff policy and a UI for Pi-versus-cloud bridge conflicts. Current
  application evidence is local and manual reconciliation remains explicit.
- Validate Google consent/project settings and real Firebase authorization with
  deployed rules; local tests use controlled services, not a live owner account.
- Run native Windows acceptance for DPAPI, WebView2, Google browser return,
  brightness/audio controls and sleep/reconnect. Compile/flash the head firmware
  and verify physical code rendering, LAN discovery, revocation and voice-created
  alarms on the actual robot. These checks require those devices.

The executable is unsigned. Automated Python and browser checks cover the new
state gates, storage and conflict handling; they cannot certify physical monitor
brightness support or the installation's network/firewall configuration.

## Validation for this change

- 136 Python tests passed, including actual local pinned TLS and WSS exchanges,
  invalid credentials, account changes, durable outbox, conflict/tombstone
  handling, exact Pi acknowledgments and persistence failures.
- Nine UI tests passed for the login gate, device authorization, planner edits
  and brightness request values.
- Chromium smoke checks passed at 1440×950 and 650×700: no JavaScript errors,
  no dashboard flash before authorization and no horizontal overflow.
- Windows packaging, archive inspection and SHA-256 are separate from native
  Windows and physical-device acceptance described above.
