"""Isolated Windows/WebView2 smoke check. All data stays under output/.

Run `python test_exe.py --source` or pass the final executable. No clipboard,
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

BASE = Path(__file__).resolve().parent


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
    parser.add_argument("executable", nargs="?", type=Path, default=BASE / "releases" / "adamV0.01.exe")
    parser.add_argument("--source", action="store_true")
    args = parser.parse_args()
    (BASE / "output").mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="native-source-" if args.source else "native-exe-", dir=BASE / "output"))
    command = [sys.executable, str(BASE / "app.py")] if args.source else [str(args.executable.resolve())]
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
    process = subprocess.Popen([*command, "--tray", "--port", str(port)], env=env, cwd=BASE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW)
    client = None
    try:
        base_url = f"http://127.0.0.1:{port}"
        ping = eventually(lambda: session.get(base_url + "/ping", timeout=2).json())
        assert ping["app"] == "ADAM Companion" and ping["version"] == "0.01"
        passed("Local service reports the expected product and version")
        page = session.get(base_url + "/", timeout=5)
        assert page.ok and "default-src 'self'" in page.headers["Content-Security-Policy"]
        token = re.search(r'name="adam-session" content="([^\"]+)"', page.text).group(1)
        headers = {"X-ADAM-Session": token}
        assert session.get(base_url + "/settings", timeout=5).status_code == 401
        settings = session.get(base_url + "/settings", headers=headers, timeout=5).json()
        assert "agent_token" not in settings and "sync_token" not in settings
        identity = session.get(base_url + "/account/status", headers=headers, timeout=5).json()
        assert identity["configured"] and identity["google_available"] and not identity["signed_in"]
        passed("Desktop session guard, credential redaction and Google configuration")
        for asset in ("/static/js/dashboard.js", "/static/js/three.min.js", "/static/css/dashboard.css", "/static/images/logo.png"):
            response = session.get(base_url + asset, timeout=5)
            assert response.ok and len(response.content) > 100
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
        cli("--port", str(port))
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
        passed("Native tray and graceful shutdown finish without application errors")
        report["result"] = "passed"
    finally:
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
