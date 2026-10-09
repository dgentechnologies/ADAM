# ADAM shared companion protocol, version 2

Android and Windows use Firebase project `adam-ai1`, Firebase Auth UID as the
account identity, and `users/{uid}.companion` as the shared envelope. Only the
owner can read or write this document. The same envelope is exchanged with a
per-account, per-device BLE simulator; physical BLE transport remains disabled.

The executable example is [companion-v2.fixture.json](companion-v2.fixture.json).
Android's `shared-protocol.test.ts` sends the example through both the TypeScript
and Python validators and compares their merge results.

| Map | Live-record fields, in addition to common fields |
| --- | --- |
| `memories` | `title` (1–80), `text` (1–2000), `kind` (`fact` or `person`) |
| `todos` | `text` (1–2000), `done` (boolean), `dueAt` (UTC timestamp or empty), `deviceId` (UUID or empty for all devices) |
| `clocks` | `kind` (`alarm`, `timer`, `reminder`), `label` (0–80), `when` (UTC timestamp), `enabled` (boolean), `deviceId` (UUID or empty) |
| `devices` | `name` (1–40), `serial` (1–80), `simulated` (boolean) |
| `preferences` | keys `voice`, `wakeWord`, `brain`; values retain the version-1 `{value, updatedAt}` format |

Common record fields are `id`, `createdAt`, `updatedAt`, and `deleted: false`.
IDs are lowercase UUIDs and must match their map key. Dates use UTC ISO 8601 with
exactly three decimal places and a `Z` suffix. Text bounds use UTF-16 code units;
both clients use ECMAScript trimming and reject unpaired surrogates. Deletions
retain only `{id, updatedAt, deleted: true}`. IDs remain stable across renames.
Deleting a saved device keeps plans and their device references for recovery;
it does not erase memories or authorize another user to access hardware.

Merge every record independently by `updatedAt`, then deletion, then canonical
JSON in Unicode code-point order. New local edits advance at least 1 ms past the
value they replace. Retain tombstones to prevent stale offline data from
restoring deleted items. Each record map permits at most 1000 records including
tombstones, except devices, which permits 100. The complete Firestore REST
encoded envelope is limited to 680,000 UTF-8 bytes. Invalid remote data is
rejected before local replacement. Cloud writes preserve other account fields
through a transaction (Android) or conditional PATCH (Windows).

Version-1 envelopes migrate in memory to version 2 with empty additional maps.
Version-2 writers always emit `schemaVersion: 2`; older released clients reject
this version instead of deleting the new maps. Update both apps together.
Unknown future versions are rejected. The local Android `LocalData.version`
remains 1 with default empty lists for compatible local upgrades.

Android imports local data only after account-sync confirmation and the first
Sync now action. Subsequent changes sync while the app is open and online.
Windows maintains separate guest and account files; importing guest records is
explicit. Account switches invalidate in-flight sync and clear desktop editors.
Offline changes remain local until a successful transfer. Shared clocks are
saved schedules; neither client claims that a disconnected physical ADAM has
scheduled an alarm. A simulator snapshot is local QA state, not a Bluetooth
connection or a hardware ownership credential.

Photos, face profiles, notification contents, network addresses, access tokens,
API keys, and physical pairing keys are excluded. Secrets remain in each
platform's secure storage. BLE simulation is scoped by UID and device ID and
uses the same validation and deletion rules as cloud sync.
