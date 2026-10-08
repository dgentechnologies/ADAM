# Windows packaging

`version-info.txt` contains the executable's Windows version/publisher metadata.
The build entry point is `../../scripts/build.py`; run it with the project's
virtual-environment Python from any working directory.

The script generates its PyInstaller spec and work files under
`artifacts/build/`, writes the intermediate executable to `artifacts/dist/`,
and publishes the portable executable and checksum to `releases/`.
Generated specs contain machine-specific paths and are not source files.

Only runtime resources and the native OAuth client are bundled. Design
references, local `.env`, test evidence and archived files are excluded.
