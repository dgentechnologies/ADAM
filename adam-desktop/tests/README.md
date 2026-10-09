Desktop tests run from `adam-desktop` with `python -m pytest` using the development dependencies. The full backend/account suite needs the project's `src/account.py` and Windows DPAPI. Connection and hardware-dispatch tests also run on Linux with simulated devices.

UI regression tests execute the shipped HTML and JavaScript in jsdom, without Windows hardware or external services:

```sh
npm install --prefix artifacts/ui-tests --no-audit --no-fund jsdom@26.1.0
NODE_PATH="$PWD/artifacts/ui-tests/node_modules" node --test tests/test_desktop_ui.cjs
```

In PowerShell, set `$env:NODE_PATH = "$PWD/artifacts/ui-tests/node_modules"` before the Node command. These tests cover read-only access, rejected credentials, alarm/timer/to-do submission and the connection-key recovery path. Native Windows/WebView2 checks remain in `scripts/smoke.py`.
