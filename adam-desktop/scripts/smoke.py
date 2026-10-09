"""Isolated Windows/WebView2 smoke check. All data stays under artifacts/qa/.

Run `python scripts/smoke.py --source` or pass the final executable. No clipboard,
media, workstation lock, cloud login or robot actions are invoked.
"""
import argparse
import base64
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
import psutil
import requests
import websocket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

BASE = Path(__file__).resolve().parents[1]
OUTPUT = BASE / "artifacts" / "qa"
VERSION = re.search(r'^APP_VERSION = "([0-9.]+)"', (BASE / "src/config.py").read_text(encoding="utf-8"), re.M).group(1)


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def eventually(probe, seconds=45):
    deadline = time.monotonic() + seconds
    last_error = None
    while time.monotonic() < deadline:
        try:
            result = probe()
            if result:
                return result
        except Exception as error:
            last_error = error
        time.sleep(0.25)
    raise AssertionError(f"Timed out waiting for {probe.__name__}: {last_error}")


def owned_windows(process_id):
    try:
        pids = {process_id, *(child.pid for child in psutil.Process(process_id).children(recursive=True))}
    except psutil.NoSuchProcess:
        return []
    windows = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @callback_type
    def visit(handle, _):
        process = wintypes.DWORD()
        thread = ctypes.windll.user32.GetWindowThreadProcessId(handle, ctypes.byref(process))
        title = ctypes.create_unicode_buffer(512)
        ctypes.windll.user32.GetWindowTextW(handle, title, len(title))
        if process.value in pids and title.value == "ADAM":
            windows.append((handle, thread, bool(ctypes.windll.user32.IsWindowVisible(handle))))
        return True

    ctypes.windll.user32.EnumWindows(visit, 0)
    return windows


class CDP:
    def __init__(self, url):
        self.socket = websocket.create_connection(url, timeout=15, suppress_origin=True)
        self.counter = 0

    def call(self, method, params=None):
        self.counter += 1
        self.socket.send(json.dumps({"id": self.counter, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self.socket.recv())
            if message.get("id") == self.counter:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message.get("result", {})

    def evaluate(self, expression):
        result = self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True,
                                                "awaitPromise": True})
        if result.get("exceptionDetails"):
            raise RuntimeError(result["exceptionDetails"])
        return result.get("result", {}).get("value")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", nargs="?", type=Path, default=BASE / "releases" / f"adamV{VERSION}.exe")
    parser.add_argument("--source", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="native-source-" if args.source else "native-exe-", dir=OUTPUT))
    command = [sys.executable, str(BASE / "src" / "app.py")] if args.source else [str(args.executable.resolve())]
    port, debug_port = free_port(), free_port()
    env = {**os.environ, "ADAM_DATA_DIR": str(root / "data"), "ADAM_STARTUP_DIR": str(root / "startup"),
           "ADAM_DISABLE_HARDWARE": "1", "WEBVIEW2_USER_DATA_FOLDER": str(root / "webview"),
           "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS": f"--remote-debugging-port={debug_port} --remote-debugging-address=127.0.0.1"}
    session = requests.Session()
    session.trust_env = False
    report = {"mode": "source" if args.source else "frozen", "checks": []}

    def passed(name):
        report["checks"].append(name)
        print("PASS", name, flush=True)

    def cli(*flags):
        result = subprocess.run([*command, *flags], env=env, cwd=BASE, capture_output=True,
                                timeout=40, creationflags=subprocess.CREATE_NO_WINDOW)
        assert result.returncode == 0, "Startup/activation command failed"

    cli("--install-startup")
    shortcut = root / "startup" / "ADAM Companion.lnk"
    assert shortcut.exists() and shortcut.stat().st_size > 100
    cli("--uninstall-startup")
    assert not shortcut.exists()
    passed("Startup shortcut creates and removes inside the isolated directory")
    foreign_requests = []
    class OtherApp(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_GET(self):
            foreign_requests.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"app":"ADAM Companion","instance_id":"other-application"}')
    other_app = ThreadingHTTPServer(("127.0.0.1", port), OtherApp)
    other_thread = threading.Thread(target=other_app.serve_forever, daemon=True)
    other_thread.start()
    preferred_port = port
    process = subprocess.Popen([*command, "--tray", "--port", str(port)], env=env, cwd=BASE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    client = None
    try:
        instance_file = root / "data" / "desktop-instance.json"
        instance = eventually(lambda: json.loads(instance_file.read_text(encoding="utf-8")))
        port = instance["port"]
        assert port != preferred_port
        base_url = f"http://127.0.0.1:{port}"
        ping = eventually(lambda: session.get(base_url + "/ping", timeout=2).json())
        assert ping["app"] == "ADAM Companion" and ping["version"] == VERSION
        assert ping["instance_id"] == instance["instance_id"] and ping["pid"] == instance["pid"]
        assert not foreign_requests
        passed("Occupied port falls back without loading or contacting the other application")
        passed("Local service reports the expected product and version")
        page = session.get(base_url + "/", timeout=5)
        assert page.ok and "default-src 'self'" in page.headers["Content-Security-Policy"]
        token = re.search(r'name="adam-session" content="([^\"]+)"', page.text).group(1)
        headers = {"X-ADAM-Session": token}
        assert session.get(base_url + "/settings", timeout=5).status_code == 401
        settings = session.get(base_url + "/settings", headers=headers, timeout=5).json()
        assert "agent_token" not in settings and "sync_token" not in settings
        assert settings["agent_port"] == port
        identity = session.get(base_url + "/account/status", headers=headers, timeout=5).json()
        assert identity["configured"] and identity["google_available"] and not identity["signed_in"]
        passed("Desktop session guard, credential redaction and Google configuration")
        rejected = session.post(base_url + "/touch/save", headers=headers, json={"assignments": {"touch1": {"tap": {"action": "volume_up"}}}}, timeout=5)
        assert rejected.status_code == 400
        saved = session.post(base_url + "/touch/save", headers=headers, json={"assignments": {"touch3": {"triple": {"action": "volume_set", "value": 37}}}}, timeout=5)
        assert saved.ok and saved.json()["assignments"]["touch3"]["triple"]["value"] == 37
        passed("Packaged touch API rejects fixed taps and persists Touch 3 triple-tap preferences")
        for asset in ("/static/js/dashboard.js", "/static/js/controls.js", "/static/js/companion.js", "/static/js/three.min.js", "/static/css/dashboard.css", "/static/css/dashboard-scene.css", "/static/css/workspace.css", "/static/css/companion.css", "/static/images/logo.png", "/static/images/icons.svg", "/static/models/adam-body.glb"):
            response = session.get(base_url + asset, timeout=5)
            assert response.ok and len(response.content) > 100
            assert response.content == (BASE / "resources" / asset.lstrip("/")).read_bytes(), "Packaged asset is stale: " + asset
        passed("Local interface, Three.js and ADAM logo assets load")
        eventually(lambda: owned_windows(process.pid))
        assert not any(visible for _, _, visible in owned_windows(process.pid))
        passed("Native window starts hidden in tray mode")
        targets = eventually(lambda: [target for target in session.get(f"http://127.0.0.1:{debug_port}/json", timeout=2).json()
                                      if target.get("type") == "page" and target.get("url", "").startswith(base_url)])
        client = CDP(targets[0]["webSocketDebuggerUrl"])
        eventually(lambda: client.evaluate("document.readyState === 'complete' && !!document.querySelector('#mainContent') && [...document.images].every(i=>i.complete && i.naturalWidth > 0)"))
        assert client.evaluate("typeof THREE === 'object'")
        passed("Real WebView2 renders the interface and loads all images")
        eventually(lambda: client.evaluate("document.querySelector('#threeCanvas').dataset.modelSource === 'adam-body.glb'"))
        client.evaluate("document.querySelector('#setupExploreBtn').click()")
        eventually(lambda: client.evaluate("!document.querySelector('#setupDialog').open"))
        assert client.evaluate("document.querySelectorAll('.sensor-pin').length === 4 && document.documentElement.scrollHeight <= innerHeight + 1 && document.documentElement.scrollWidth <= innerWidth + 1")
        assert client.evaluate("document.querySelectorAll('#sidebarNav [data-view=settings]').length === 0 && !!document.querySelector('[aria-label=\"Profile and settings\"]')")
        client.evaluate("document.querySelector('#pin-touch3 summary').click()")
        eventually(lambda: client.evaluate("document.querySelector('#pin-touch3').open && !document.querySelector('#menu-touch3').hidden"))
        assert client.evaluate("getComputedStyle(document.querySelector('#menu-touch3')).backdropFilter.includes('28px')")
        assert client.evaluate("document.querySelectorAll('[data-sensor=touch1]').length===1 && document.querySelectorAll('[data-sensor=touch3]').length===3 && !!document.querySelector('[data-sensor=touch3][data-event=triple]') && !document.querySelector('[data-sensor][data-event=tap]')")
        client.evaluate("document.querySelector('#pin-touch3').open=false")
        passed("Bundled ADAM model, four touch menus, profile access and one-screen layout")
        client.evaluate("document.querySelector('#sidebarNav [data-view=actions]').click()")
        eventually(lambda: client.evaluate("document.querySelectorAll('.control-row').length === 8 && document.querySelectorAll('#sidebarNav .nav-link').length === 4"))
        assert client.evaluate("document.querySelector('#sectionNav [data-view=activity]') !== null && document.querySelector('#actionInspector h3').textContent === 'Volume up'")
        client.evaluate("document.querySelector('#sidebarNav [data-view=devices]').click()")
        eventually(lambda: client.evaluate("document.querySelector('#sectionNav [data-view=clock]') !== null && document.querySelector('#sectionNav [data-view=memories]') !== null"))
        client.evaluate("document.querySelector('#sidebarNav [data-view=dashboard]').click()")
        passed("Grouped navigation and categorized permission inspector render in the package")
        assert client.evaluate("eyeMotion.width === 142 && eyeMotion.height > 35 && sensorLocations.touch4[2] < sensorLocations.touch3[2]")
        cli("--port", str(preferred_port))
        eventually(lambda: any(visible for _, _, visible in owned_windows(process.pid)))
        passed("Second launch activates the existing window")
        screenshot = client.call("Page.captureScreenshot", {"format": "png"})
        (root / "native-window.png").write_bytes(base64.b64decode(screenshot["data"]))
        handle, thread, _ = owned_windows(process.pid)[0]
        ctypes.windll.user32.PostMessageW(wintypes.HWND(handle), 0x0010, 0, 0)
        eventually(lambda: not any(visible for _, _, visible in owned_windows(process.pid)))
        assert session.get(base_url + "/ping", timeout=3).ok and process.poll() is None
        passed("Window close hides to tray while the backend stays available")
        assert session.post(base_url + "/show_window", json={}, timeout=3).ok
        eventually(lambda: any(visible for _, _, visible in owned_windows(process.pid)))
        passed("Dashboard restores after closing to tray")
        # The show endpoint schedules native restoration; let that UI callback
        # finish before deliberately ending this test window's message loop.
        time.sleep(1)
        client.socket.close()
        client = None
        ctypes.windll.user32.PostThreadMessageW(thread, 0x0012, 0, 0)
        process.wait(timeout=30)
        log = (root / "data" / "adam_agent.log").read_text(encoding="utf-8")
        assert "System tray icon active." in log and "Shutdown complete." in log
        assert "[ERROR]" not in log and "[CRITICAL]" not in log
        assert not instance_file.exists() and not foreign_requests
        passed("Native tray and graceful shutdown finish without application errors")
        report["result"] = "passed"
    finally:
        other_app.shutdown()
        other_app.server_close()
        other_thread.join(2)
        if client:
            client.socket.close()
        if process.poll() is None:
            try:
                owned = psutil.Process(process.pid)
                descendants = owned.children(recursive=True)
                for child in reversed(descendants):
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                owned.terminate()
                psutil.wait_procs([owned, *descendants], timeout=10)
            except psutil.NoSuchProcess:
                pass
        (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Evidence:", root, flush=True)


if __name__ == "__main__":
    main()
