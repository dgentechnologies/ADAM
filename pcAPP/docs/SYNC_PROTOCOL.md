# ADAM companion account and sync protocol

Desktop and mobile use the existing Firebase project `adam-ai1` and the same
Firebase Auth user ID. A new Firebase project is not needed. Public web config
is not a service-account credential. No service-account key ships in the app.

## Desktop account setup

Email/password must be enabled in Firebase Authentication. Google uses the
system browser, not the embedded app browser. Create an OAuth client with
application type **Desktop app** in the Google Cloud project backing `adam-ai1`,
then save its downloaded `installed` JSON as `firebase-desktop-client.json`
beside `ADAM.exe` (beside `account.py` during development), or set
`ADAM_GOOGLE_CLIENT_FILE` to its absolute path. This config is distinct from the
Android client and the Firebase web config. Do not create a separate Firebase
project. Enable the Google provider and configure the OAuth consent screen.
An OAuth consent screen in testing permits only its configured test users.
The release can bundle the project's native OAuth configuration in the
PyInstaller data files; `_MEIPASS` is the final lookup fallback after the
environment override and executable sidecar. Native installed-app client
configuration is distributable and is not a server-side secret. Its contents
are still excluded from logs and source control.

The browser flow binds a random port on `127.0.0.1`, uses `/callback`, validates
state, exchanges a PKCE S256 authorization code, and times out after 3 minutes.
Sign-out or Cancel invalidates pending browser callbacks. Only the Google OAuth
access token is exchanged with Firebase; the desktop-audience Google ID token
is not submitted to Firebase, and no Google refresh token is requested. Firebase
refresh and ID tokens are encrypted with current-user Windows DPAPI. Passwords
are never persisted. No analytics is enabled by these modules.

Firebase documents this as the Google access-token credential path:
[Firebase Auth REST](https://firebase.google.com/docs/reference/rest/auth#section-sign-in-with-oauth-credential)
and [GoogleAuthProvider.credential](https://firebase.google.com/docs/reference/js/auth.googleauthprovider#googleauthprovidercredential).
Live consent and Firebase token acceptance still require verification with a
real test account; a mocked successful exchange is not proof of project setup.

## Shared document

The only cloud field patched by the companion is `users/{uid}.companion`.
Existing `profile`, device ownership, or any other fields remain untouched.
Owner rules must require `request.auth.uid == uid` for this document. The
existing mobile owner-document rules support this path; no new collection is
required. Photos, face profiles, notification contents, Home Assistant secrets,
local network addresses, and device pairing secrets do not enter this field.

```json
{
  "schemaVersion": 1,
  "memories": {
    "90f52987-d3d6-4f80-952c-bf15c66d37b8": {
      "id": "90f52987-d3d6-4f80-952c-bf15c66d37b8",
      "title": "Morning routine",
      "text": "I prefer a quiet start to the day.",
      "kind": "fact",
      "createdAt": "2026-10-07T08:00:00.000Z",
      "updatedAt": "2026-10-07T08:00:00.000Z",
      "deleted": false
    },
    "8b741b7b-00f5-48ae-9db5-76db7ed578bc": {
      "id": "8b741b7b-00f5-48ae-9db5-76db7ed578bc",
      "updatedAt": "2026-10-07T09:00:00.000Z",
      "deleted": true
    }
  },
  "preferences": {
    "voice": {"value": "Charon", "updatedAt": "2026-10-07T08:00:00.000Z"},
    "wakeWord": {"value": "Hey ADAM", "updatedAt": "2026-10-07T08:00:00.000Z"},
    "brain": {"value": "lite", "updatedAt": "2026-10-07T08:00:00.000Z"}
  }
}
```

Memory IDs are canonical lowercase UUIDs. Titles contain 1–80 UTF-16 code units;
text contains 1–2000 UTF-16 code units, matching the mobile form validation.
Titles and text trim exactly the ECMAScript `String.trim()` whitespace set on
both clients, including BOM but excluding NEL and control separators.
`kind` is `fact` or `person`. Dates use UTC ISO 8601
with exactly three fractional digits. Missing preference fields default to
`Charon`, `Hey ADAM`, and `lite` in the UI without uploading synthetic defaults.
Supported voices: Charon, Aoede, Kore, Puck, Fenrir. Wake words: Hey ADAM, ADAM.
Brain selections: lite, byok, managed. A preference is a selection only; access
to a paid service or a saved API credential is never implied by synchronizing it.

Merge independently by memory ID and preference name. The lexically greatest
`updatedAt` wins. For the same timestamp, a deletion wins over a live memory;
then the lexically greatest canonical JSON representation wins. Canonical JSON
sorts every object's keys, uses no optional whitespace, and preserves Unicode.
This last comparison is by Unicode scalar/code-point order (Python strings),
not UTF-16 code-unit order. Mobile should compare `Array.from(string)` code
points if it must resolve an exact timestamp tie involving supplementary
Unicode. Clients must advance their own edits by at least 1ms from the record
they replace. Device clocks should be synchronized; future clock skew can
otherwise make older-device edits lose to a newer timestamp.

Deleted entries are retained as tombstones; do not discard them during normal
sync, because offline devices may still hold the old memory. The entire map is
limited to 1000 entries including tombstones. Its Firestore REST-encoded value
is limited to 680,000 UTF-8 bytes, leaving room under Firestore's document limit
for the existing account fields. Capacity errors preserve all existing data.
Unknown schema versions or malformed records are rejected without overwriting
the local or cloud document.

Desktop GETs the document and merges it with current local data. PATCH uses
`updateMask.fieldPaths=companion` and `currentDocument.updateTime=<GET version>`.
For a missing user document, it uses `currentDocument.exists=false`. HTTP409/412
and HTTP400 with structured `FAILED_PRECONDITION` retry from GET up to 4 attempts.
Local edits made while a network request is in
flight merge back and remain pending for the next sync. Never replace the full
user document with a bare companion document.

## Local isolation and backend integration

Data directory: `ADAM_DATA_DIR`, otherwise `%APPDATA%/ADAM`. Account files use a
SHA-256 filename derived from Firebase UID; guest data is a separate file.
Changing accounts immediately changes the local scope. Sync verifies scope
before each remote mutation and before accepting results. Guest data is merged
only after the user explicitly chooses to import it; signing in alone never
copies it. Guest data remains locally available after an import.

`AccountService` exposes `status`, `start_google`, `cancel_google`,
`email_login(email,password,create=False,name='')`, `reset_password`, `signout`,
and `id_token`. `user` is a property containing only UID/email/name/photo URL.
Status includes `authenticated`, `user`, `google.configured`, `google.pending`,
and a safe `error`. Google completes on a daemon thread; poll status. Account
errors expose a safe message and `code`; no token is returned by a UI endpoint.

`CloudSync(account)` exposes `status`, `sync`, `list_memories`,
`save_memory({id?,title,text,kind})`, `delete_memory(id)`, `import_guest`,
`get_preferences`, and `save_preferences({voice?,wakeWord?,brain?})`. Mutations
save locally first. The desktop exposes explicit **Sync now** and runs the
transfer in a worker. Email sign-in and explicit guest import also start a
transfer; the dashboard starts one after saving signed-in account preferences.
Memory edits remain local until a subsequent sync. There is no periodic cloud
sync worker in this release. Concurrent sync calls return
the current status instead of issuing duplicate requests. Status includes
`enabled`, `signedIn`, `syncing`, `lastSynced`, `pending`, `guestMemories`, and
`error`. Expose explicit Import local data consent in the UI. Cloud errors never
discard pending local changes.

The corresponding mobile implementation is intended for Android version0.2.1.
It uses a Firebase SDK transaction with `mergeFields: ['companion']`, so account
profile fields are preserved. Mobile account sync is opt-in and **Sync now** is
manual. Enabling sync stages local content without uploading it immediately.
An owner marker and per-UID baseline prevent switching accounts from silently
importing another account's phone data; a different account requires explicit
import consent. UID/epoch checks fence old transaction callbacks, and local
edits made while a transfer is running remain pending for the next sync.

These modules are verified with isolated fake HTTP/cipher tests. Real Google
consent, email delivery, deployed rules, and mobile/desktop exchange require
the configured OAuth client and a test account; do not claim these were tested
against a user's live data.
