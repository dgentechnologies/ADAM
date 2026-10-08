# Local build configuration

Keep developer-specific files in `local/` (ignored by Git):

- `firebase-desktop-client.json`: the supplied native Google Desktop OAuth
  client. Source sign-in and `scripts/build.py` use this file by default.
  `ADAM_GOOGLE_CLIENT_FILE` can point source/runtime sign-in to an alternate
  file. Release builds use the canonical file above so its bundled name is stable.
- `.env`: the existing desktop-only legacy configuration, preserved during
  reorganization. It is never included in the executable.
- Original downloaded client JSON: retained locally for reference.

The packaged app still supports a `firebase-desktop-client.json` sidecar next
to the executable before falling back to its bundled client. Runtime settings,
memories and DPAPI sessions continue to live under `%APPDATA%\ADAM`.
