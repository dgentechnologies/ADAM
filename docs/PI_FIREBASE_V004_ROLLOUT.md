# Pi and Firebase contract rollout — desktop 0.04

This release implements the trusted cloud boundary and Pi record validation for
[the final data contract](ADAM_DESKTOP_FINAL_ARCHITECTURE_AND_DATA_CONTRACT.md).
It is source and a packaged desktop release, **not confirmation of a live Firebase
deployment or a physical robot upgrade**.

## Data path and technology

Windows uses Python/Flask/Waitress, pywebview/WebView2 and HTML/CSS/JavaScript.
Firebase Auth supplies the UID; Google login uses the system browser and PKCE.
Windows DPAPI protects credentials. A UID-scoped durable local outbox exchanges
canonical Firestore documents, then an authorized LAN bridge applies them to Pi.
The Android companion uses TypeScript and Firebase's JavaScript SDK; its canonical
adapter now preserves schedule timezones and respects backend-only ownership.
Android physical-device onboarding remains a separate integration requirement.

Pi runs Python, local JSON persistence and its independent scheduler. It stores no
Firebase administrator credentials. TLS identity and desktop grants protect LAN
access; the short code on ADAM's screen authorizes first-time desktop pairing.
Laptop-control permission remains separate from robot record synchronization.

Cloud Firestore is the shared account database. Firebase Authentication establishes
account identity. New Node.js 22 Cloud Functions (`claimDevice`, `transferDevice`,
`reportExecution`, region `us-central1`) perform trusted ownership/evidence writes
using Admin SDK transactions. Firebase Storage is not introduced for companion
records or biometrics in this release; existing storage configuration is unchanged.
Functions deployment requires the project's applicable billing/API configuration.

## Implemented boundaries

- Only `ownerUid` grants access to a physical device. Clients cannot create device
  ownership, replace hardware identities, or write execution acknowledgements.
- Canonical records have schema version 1, creation/update timestamps, bounded
  fields, at most eight target devices, and tombstones instead of hard deletion.
- Schedules carry an IANA timezone. Pi rejects incompatible records rather than
  silently interpreting them in another timezone. Timers retain second precision.
- Pi validates records before persistence. Provisioned robots reject legacy bulk
  schedule/to-do replacement and legacy release, which could bypass these checks.
- P-256 proofs bind the owner, physical identity, ownership epoch, expiry and nonce.
  The cloud checks exact applied schedule contents against current desired state.
  Only durable Pi scheduler evidence can report application/firing through this path.
- Desktop relays at most eight receipts per page. It advances its durable cursor
  only after cloud acceptance, and fences account/device changes. Retrying keeps
  pending evidence available; local application and cloud verification stay distinct.
- Simulated mobile devices never become trusted physical registrations. Client
  profile writes no longer maintain authoritative ownership mirrors.

The trust anchor is the administrator-verified Pi public key. A compromised robot
holding that private key can sign evidence; signatures do not attest uncompromised
hardware. Keep the private key on the robot and protect its filesystem.

## Prepare existing data before strict rules

Back up Firestore and Pi data before migration. From `firebase/`, with a securely
configured administrator Application Default Credential for `adam-ai1`, run:

```sh
npm ci
node scripts/audit-cloud.js
```

This is a read-only inventory, not a complete migration or an automatic repair.
It reports paths requiring review without printing record contents. Resolve legacy
`ownerId` mappings, schema versions, typed Firestore timestamps, target ownership,
legacy `users/{uid}.companion` data and schedule execution fields before rollout.
Preserve record IDs, original creation times and tombstones. Obtain the intended
IANA timezone for old schedules; do not guess it from the administrator's machine.
Canonical desired schedules must not carry client-authored firing acknowledgements.

The new rules intentionally deny incompatible legacy writes. Coordinate desktop,
mobile, Pi and rules upgrades; do not deploy strict rules to unmigrated production
data merely because local unit tests pass.

## Enroll physical trust

Install `MP-MC codes/pi/requirements-sync.txt` into the **existing ADAM runtime
virtual environment**, alongside its current audio/AI dependencies. Run pairing
commands from `MP-MC codes/pi/adam` using that environment's Python. Existing
provisioned identities must be retained; never regenerate a key to fix discovery.

```sh
python desktop_pairing.py public-bundle > /secure/path/adam-public-bundle.json
```

Verify the bundle against the physical robot. Transfer only this public bundle to
the administrator machine. From `firebase/`:

```sh
node scripts/register-hardware.js /secure/path/adam-public-bundle.json --dry-run
node scripts/register-hardware.js /secure/path/adam-public-bundle.json --apply
```

For an existing legacy device, supply its independently verified owner UID and
exact previous hardware serial as the final two arguments to both commands.
Registration refuses collisions or replacement of an existing registered key.
It is intentionally unavailable to normal client accounts.

A newly registered unclaimed device needs a fresh `python desktop_pairing.py
cloud-claim` proof relayed to `claimDevice` by its matching authenticated owner.
The callable payload is `{data: proof}` with the owner's Firebase ID token; proof
validity is five minutes. The SDK equivalent is `httpsCallable(functions,
'claimDevice')(proof)`. Treat proofs as short-lived authorization material, not
logs. This administrator-assisted enrollment is not a complete factory setup UI.

Restart Pi with the updated software, verify its configured timezone, sign into
desktop 0.04 with the owning account, select ADAM and enter the physical-screen
code. Sync a test record, then verify durable Pi application and cloud evidence.
Test firing, deletion, reconnect and account switching on the actual hardware.

## Ownership transfer

The trusted function checks current owner, signature and epoch, and rejects a
transfer while robot-private cloud subcollections contain data. It increments
the ownership epoch and updates backend ownership mirrors transactionally.

**Automatic end-to-end transfer/reset is not shipped.** The Pi `transfer` command
refuses in-place relabeling. An administrator must review and archive/purge old
owner data, revoke grants, reset private local state and coordinate the new cloud
owner/epoch with the physical identity. Do not call the cloud function alone and
expect the Pi or old private data to be safely reset.

## Validate and deploy

Use Node.js 22 and Java 21 for the Firebase project:

```sh
cd firebase
npm ci
npm test
npm run test:emulator
node scripts/audit-cloud.js
npm run deploy
```

`npm run deploy` repeats local and real emulator tests before deploying rules,
indexes and the three functions to `adam-ai1`. Administrator ADC is required for
audit/enrollment; the Firebase CLI also needs usable deployment authentication.
A desktop OAuth installed-client JSON is **not** an administrator credential.
Configure secrets securely outside the checkout; never put service-account keys
on desktop, mobile or Pi. This release does not change Firebase Auth providers.

The GitHub `Firebase contract` workflow runs Node 22 unit tests and real Firestore
emulator tests on relevant pushes/PRs. It does not deploy. Require its success and
review the migration inventory before production rollout. Nonce records contain
`expiresAt`; arrange administrator cleanup/TTL as appropriate for production.

## Validation and remaining gates

Local checks passed: 142 desktop/Pi-contract tests, 27 Pi tests, 10 mobile
canonical exchange tests, 4 cloud signature/transaction unit tests and 9 browser
interaction tests (192 total). Mobile TypeScript compilation also passed.
The cloud transaction unit adapter is not a substitute for Firestore's emulator.
In this workspace the official emulator download returned network-proxy 403, so
rules compilation and real emulator tests have **not run here**. Tests are included
in CI. Production deployment is also blocked by missing administrator login.

Desktop 0.04 was built as a Windows x64 PE using Windows Python under Wine; its
archive and checksum were inspected. Browser screenshots confirmed the coded oval
ADAM head, horizontal eyes, unclipped vector wordmark and shared dark gradient.
Native Windows execution and physical Pi/ESP32 acceptance remain hardware checks.
The executable does not flash firmware or deploy Firebase automatically.
