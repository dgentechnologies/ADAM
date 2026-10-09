# Project structure

The Windows project is named **ADAM Desktop**, with the repository folder
`adam-desktop`. Its Android sibling is **ADAM Mobile**, in `adam-mobile`.
These names identify development workspaces; installed product names, Android
application IDs, Firebase identities and user-data locations are unchanged.

```text
adam-desktop/
├── README.md                   Start here: setup, usage and development
├── requirements.txt            Pinned runtime dependencies
├── requirements-dev.txt        Build and test dependencies
├── pyproject.toml              Test discovery and tool configuration
├── src/                        Python application and backend modules
│   ├── app.py                  Native window, tray and lifecycle entry point
│   ├── backend.py              HTTP API and action registry
│   ├── desktop_runtime.py      Socket ownership and instance discovery
│   ├── config.py               Settings and runtime resource locations
│   ├── account.py              Desktop identity
│   ├── cloud_sync.py           Account data synchronization
│   ├── connection.py           Robot connection service
│   └── ...                     Hardware, storage and action services
├── resources/                  Assets shipped with the app
│   ├── static/                 HTML, CSS, JavaScript, fonts and 3D model
│   ├── icons/                  Application and Explorer folder icons
│   └── logo.png                Runtime branding fallback
├── scripts/                    Development and release entry points
│   ├── run.ps1                 Launch source from any working directory
│   ├── build.py                Build EXE and checksum
│   ├── smoke.py                Isolated source/EXE lifecycle verification
│   ├── set-folder-icon.ps1     Reapply Explorer branding after checkout
│   └── vendor-icons.cjs        Rebuild the local UI icon sprite
├── packaging/windows/          Windows executable version metadata
├── config/local/               Ignored developer configuration
├── tests/                      Automated desktop tests
├── docs/                       Current behavior, protocol and release notes
│   └── planning/               Historical product/design plans
├── design/                     Reference material, excluded from packaging
│   ├── branding/               Original logo and image concepts
│   └── references/             Supplied dashboard and screen references
├── releases/                   Deliverable EXE, checksum and user notes
└── artifacts/                  Ignored generated files and local history
    ├── build/                  Generated PyInstaller spec and work files
    ├── dist/                   Intermediate executable
    ├── qa/                     Test logs, screenshots and isolated app data
    ├── cache/                  Tool caches
    └── archive/                Preserved pre-migration outputs and files
```

## Development commands

From `adam-desktop` in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\scripts\run.ps1

$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:ADAM_DATA_DIR=Join-Path $PWD 'artifacts\qa\test-data'
$env:ADAM_DISABLE_HARDWARE='1'
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\smoke.py --source
.\.venv\Scripts\python.exe scripts\build.py
.\.venv\Scripts\python.exe scripts\smoke.py
```

The test variables are scoped to the current shell. Open a fresh shell for
normal hardware use. Source can also run directly with `python src/app.py`.
Application modules retain their existing imports and asynchronous architecture;
this repository is an application, not a published Python library.

The build script resolves all locations relative to itself. It bundles only
`resources/` and the explicitly selected native OAuth client. It does not bundle
the configuration directory, design references, tests or archives. In the
executable, UI URLs remain `/static/...`.

## What moved and why

| Previous location | Current location | Purpose |
| --- | --- | --- |
| `pcAPP/` | `adam-desktop/` | Consistent product/platform name |
| `mobileAPP/` | `adam-mobile/` | Matching mobile workspace name |
| Root Python modules | `src/` | Keep runtime code separate from tools and documents |
| `static/`, runtime icon/logo | `resources/` | Make shipped assets explicit |
| `build_exe.py`, `test_exe.py` | `scripts/build.py`, `scripts/smoke.py` | Separate release tooling from runtime code |
| `version_info.txt` | `packaging/windows/version-info.txt` | Platform packaging metadata |
| Root plans and design notes | `docs/planning/` | Distinguish historical intent from current documentation |
| `ref/`, root concept images | `design/` | Keep design inputs out of releases |
| `.env`, Desktop OAuth client JSON | `config/local/` | Keep local configuration together and ignored |
| `output/` | `artifacts/qa/` | Preserve verification evidence |
| Previous build/dist/cache directories | `artifacts/archive/` | Preserve existing files without cluttering the project root |

The repository's protocol parity check and cross-project links now use the new
names. The desktop virtual environment's activation scripts and console
launchers were refreshed after relocation. The mobile monorepo keeps its
existing `apps/`, `packages/`, dependency versions and Android package identity.

## Explorer icon

The folder uses ADAM's existing A mark on a dark folder icon, with multiple
resolutions for Explorer. `desktop.ini` stores a relative icon path. Run
`scripts/set-folder-icon.ps1` after a fresh checkout to apply the required
Windows hidden/system metadata and directory attribute. The script refreshes
the folder without restarting Explorer. Folder customization does not make
the project's files read-only.

Existing `%APPDATA%\ADAM` profiles and mobile installed data are not moved.
Use the executable in `releases/` for normal use; the archived intermediate
executables are retained only as build history.

## Verification after relocation

- Desktop suite: 96 passed.
- Source and packaged WebView2 smoke: all 14 checks passed in each mode.
- Shared desktop/Pi protocol file: byte-identical.
- Explorer: custom relative icon location verified through the Windows shell.
- Mobile: TypeScript and Capacitor commands resolve; web typecheck passed.
- Release checksums: rebuilt Windows EXE and preserved Android APK verified.

The final artifact identity and remaining external acceptance limits are in
[RELEASE.md](RELEASE.md).
