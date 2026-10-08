"""
ADAM Windows Companion App — Executable Build Script
==============================================================
DGEN Technologies Pvt. Ltd.
Compiles the app into a standalone Windows .exe using PyInstaller with all
local UI, native bridge libraries, 3D assets and icons embedded.
Microsoft Edge WebView2 Runtime must be installed on the user's Windows PC.
"""

import os
import sys
import shutil
import subprocess
import hashlib
import re
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
DIST_DIR = BASE_DIR / "dist"
BUILD_DIR = BASE_DIR / "build"
ICON_PATH = BASE_DIR / "assets" / "adam.ico"
VERSION = re.search(r'^APP_VERSION = "([0-9.]+)"',
                    (BASE_DIR / "config.py").read_text(encoding="utf-8"), re.M).group(1)


def build():
    print("=" * 60)
    print("Building ADAM Windows Companion App (.exe)...")
    print(f"Base Directory: {BASE_DIR}")
    print("=" * 60)

    # Ensure icon exists
    if not ICON_PATH.exists():
        print("Generating adam.ico from logo.png...")
        from PIL import Image
        img = Image.open(BASE_DIR / "logo.png")
        ICON_PATH.parent.mkdir(parents=True, exist_ok=True)
        img.save(ICON_PATH, format="ICO", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])

    # Collect webview data files
    import PyInstaller.utils.hooks as pi_hooks
    webview_data = pi_hooks.collect_data_files("webview")
    webview_data_args = []
    for src, dst in webview_data:
        webview_data_args.extend(["--add-data", f"{src};{dst}"])

    # Exclude unused heavy packages and conflicting Qt packages
    excludes = [
        "PyQt5", "PyQt6", "PySide2", "PySide6",
        "tkinter", "_tkinter", "matplotlib", "IPython", "jedi",
        "scipy", "numpy", "pandas", "torch", "tensorflow", "cv2",
        "zmq", "notebook", "tornado"
    ]
    exclude_args = []
    for exc in excludes:
        exclude_args.extend(["--exclude-module", exc])

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--name", "ADAM",
        "--onefile",
        "--windowed",
        "--icon", str(ICON_PATH),
        "--version-file", str(BASE_DIR / "version_info.txt"),
        "--add-data", f"{BASE_DIR / 'assets'};assets",
        "--add-data", f"{BASE_DIR / 'static'};static",
        "--add-data", f"{BASE_DIR / 'logo.png'};.",
        "--add-data", f"{BASE_DIR / 'firebase-desktop-client.json'};.",
        *webview_data_args,
        *exclude_args,
        "--hidden-import", "webview",
        "--hidden-import", "webview.platforms.winforms",
        "--hidden-import", "clr",
        "--hidden-import", "pystray",
        "--hidden-import", "pystray._win32",
        "--hidden-import", "PIL",
        "--hidden-import", "PIL.Image",
        "--hidden-import", "PIL.IcoImagePlugin",
        "--hidden-import", "pycaw",
        "--hidden-import", "pycaw.pycaw",
        "--hidden-import", "comtypes",
        "--hidden-import", "screen_brightness_control",
        "--hidden-import", "zeroconf",
        "--hidden-import", "zeroconf._utils.ipaddress",
        "--hidden-import", "flask",
        "--hidden-import", "werkzeug",
        "--hidden-import", "websockets",
        "--hidden-import", "websockets.sync.client",
        "--hidden-import", "requests",
        "--hidden-import", "waitress",
        "--hidden-import", "account",
        "--hidden-import", "cloud_sync",
        "--hidden-import", "secure_store",
        "--hidden-import", "connection",
        "--hidden-import", "hardware",
        "--hidden-import", "device_catalog",
        # The shared protocol module (v41). backend.py imports it inside a
        # try/except that degrades to untyped behaviour on failure, so a
        # packaged build that silently lost this module would still START —
        # it would just stop coercing values and stop redacting clipboard
        # text from the activity log. Naming it here means PyInstaller
        # cannot miss it, and a missing module becomes a build error rather
        # than a runtime downgrade nobody notices.
        "--hidden-import", "laptop_actions",
        "--hidden-import", "coding_agent",
        "--hidden-import", "pyperclip",
        str(BASE_DIR / "app.py")
    ]

    print("\nRunning PyInstaller command:")
    print(" ".join(cmd[:15]), "...")

    res = subprocess.run(cmd, cwd=str(BASE_DIR))
    if res.returncode != 0:
        print(f"\n[FAIL] PyInstaller failed with return code {res.returncode}")
        sys.exit(res.returncode)

    exe_path = DIST_DIR / "ADAM.exe"
    if exe_path.exists():
        release = BASE_DIR / "releases"
        release.mkdir(exist_ok=True)
        artifact = release / f"adamV{VERSION}.exe"
        shutil.copy2(exe_path, artifact)
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        artifact.with_suffix(".exe.sha256").write_text(f"{digest}  {artifact.name}\n", encoding="ascii")
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print("\n" + "=" * 60)
        print("[SUCCESS] BUILD SUCCESSFUL!")
        print(f"Executable Location: {exe_path}")
        print(f"File Size: {size_mb:.2f} MB")
        print("=" * 60)
    else:
        print(f"\n[FAIL] Expected output {exe_path} not found!")
        sys.exit(1)


if __name__ == "__main__":
    build()
