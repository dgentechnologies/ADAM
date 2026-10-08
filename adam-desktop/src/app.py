"""
ADAM Windows Companion App — Desktop Application Shell
======================================================
DGEN Technologies Pvt. Ltd. | Antigravity Phase
Real Windows desktop application wrapping the Flask action registry, mDNS,
system tray presence, startup folder integration, and Achromatic UI.
"""

import os
import sys

# If launched from a Windows terminal (cmd/PowerShell), attach to parent console so CLI flags output properly
if sys.platform == "win32":
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.windll.kernel32
        kernel32.AttachConsole.argtypes = [wintypes.DWORD]
        kernel32.AttachConsole.restype = wintypes.BOOL
        # ATTACH_PARENT_PROCESS is (DWORD)-1 = 0xFFFFFFFF
        if kernel32.AttachConsole(0xFFFFFFFF):
            sys.stdout = open("CONOUT$", "w", encoding="utf-8", errors="replace")
            sys.stderr = open("CONOUT$", "w", encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure standard streams exist even in windowed (no-console) mode
if sys.stdin is None:
    sys.stdin = open(os.devnull, "r", encoding="utf-8")
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import time
import argparse
import hashlib
import threading
import subprocess
from pathlib import Path
from typing import Optional

from PIL import Image
import requests
import pystray
import webview
from desktop_runtime import activate_instance, verify_instance, write_instance, remove_instance

from config import (
    APP_DIR, APP_NAME, APP_VERSION, LOG_FILE,
    load_settings, save_settings, is_startup_enabled, set_startup_enabled,
    logger, USER_DATA_DIR
)
backend = None

# Install global crash / uncaught exception handlers
def _handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("Uncaught application exception:", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = _handle_exception
if hasattr(threading, "excepthook"):
    threading.excepthook = lambda args: logger.critical(
        f"Uncaught thread exception in {args.thread.name}:",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
    )

# Global handles
main_window: Optional[webview.Window] = None
tray_icon: Optional[pystray.Icon] = None
is_quitting = False
_instance_handle = None
INSTANCE_FILE = USER_DATA_DIR / "desktop-instance.json"


def acquire_instance():
    global _instance_handle
    if sys.platform != "win32":
        return True
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateMutexW.restype = wintypes.HANDLE
    suffix = hashlib.sha256(str(USER_DATA_DIR.resolve()).lower().encode()).hexdigest()[:20]
    _instance_handle = kernel.CreateMutexW(None, False, "Local\\ADAM.Companion." + suffix)
    return bool(_instance_handle) and ctypes.get_last_error() != 183


def show_dashboard():
    """Bring the dashboard window to front."""
    global main_window
    if main_window and not is_quitting:
        try:
            main_window.show()
            main_window.restore()
        except Exception as e:
            logger.error(f"Error restoring window: {e}")


def hide_dashboard():
    """Hide the dashboard window to tray."""
    global main_window
    if main_window and not is_quitting:
        try:
            main_window.hide()
        except Exception as e:
            logger.error(f"Error hiding window: {e}")


def open_log_file():
    """Open log file in default Windows editor."""
    try:
        if LOG_FILE.exists():
            os.startfile(str(LOG_FILE))
        else:
            logger.warning("Log file does not exist yet.")
    except Exception as e:
        logger.error(f"Failed opening log file: {e}")


def toggle_pause_agent(icon, item):
    """Toggle agent pause state from tray menu."""
    settings = load_settings()
    new_paused = not settings.get("paused", False)
    settings["paused"] = new_paused
    save_settings(settings)
    if icon and hasattr(icon, "update_menu"):
        try:
            icon.update_menu()
        except Exception:
            pass
    logger.info(f"Tray: laptop control {'paused' if new_paused else 'resumed'}")


def get_pause_menu_label(item) -> str:
    settings = load_settings()
    return "Resume Agent" if settings.get("paused", False) else "Pause Agent"


def restart_agent(icon, item):
    """Restart local services cleanly."""
    logger.info("Tray: restarting agent services...")
    try:
        backend.restart_backend_services()
        logger.info("Tray: agent services restarted successfully.")
    except Exception as e:
        logger.error(f"Tray restart error: {e}")


def quit_application(icon=None, item=None):
    """Graceful application shutdown."""
    global is_quitting, tray_icon, main_window
    is_quitting = True
    logger.info("Shutting down ADAM Companion App...")

    # Stop mDNS and WS client cleanly
    if backend:
        try:
            backend.stop_backend()
        except Exception:
            logger.error("A background service could not finish shutting down")

    # Stop tray icon
    if tray_icon:
        try:
            tray_icon.stop()
        except Exception:
            pass

    # Destroy window
    if main_window:
        try:
            main_window.destroy()
        except Exception:
            pass

    if backend:
        remove_instance(INSTANCE_FILE, backend.DESKTOP_INSTANCE)

    logger.info("Shutdown complete.")
    sys.exit(0)


def setup_tray() -> pystray.Icon:
    """Setup pystray system tray icon and right-click menu."""
    icon_path = APP_DIR / "icons" / "adam.ico"
    if not icon_path.exists():
        icon_path = APP_DIR / "logo.png"

    try:
        icon_image = Image.open(str(icon_path)).resize((64, 64))
    except Exception as e:
        logger.warning(f"Could not load icon file ({e}), creating default image.")
        icon_image = Image.new("RGB", (64, 64), color=(0, 0, 0))

    menu = pystray.Menu(
        pystray.MenuItem("ADAM: Active", None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Open Dashboard", lambda icon, item: show_dashboard(), default=True),
        pystray.MenuItem(get_pause_menu_label, toggle_pause_agent),
        pystray.MenuItem("Restart Agent", restart_agent),
        pystray.MenuItem("View Logs", lambda icon, item: open_log_file()),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", quit_application)
    )

    icon = pystray.Icon(
        "ADAM_Companion",
        icon_image,
        "ADAM Companion",
        menu=menu
    )
    return icon


def on_window_closing():
    """Intercept window close (X) and minimize to tray instead."""
    global is_quitting
    if not is_quitting:
        threading.Timer(0.05, hide_dashboard).start()
        return False  # Cancels the close event, keeping app alive in tray
    return True


def main():
    global main_window, tray_icon, backend

    parser = argparse.ArgumentParser(description="ADAM Windows Companion App")
    parser.add_argument("--tray", "--minimized", dest="minimized", action="store_true",
                        help="Start minimized in the system tray")
    parser.add_argument("--install-startup", action="store_true",
                        help="Add application shortcut to Windows Startup folder and exit")
    parser.add_argument("--uninstall-startup", action="store_true",
                        help="Remove application shortcut from Windows Startup folder and exit")
    parser.add_argument("--port", type=int, default=None,
                        help="Port override for agent")

    args = parser.parse_args()

    # Startup folder CLI options
    if args.install_startup:
        success = set_startup_enabled(True)
        print(f"Startup shortcut installed: {'SUCCESS' if success else 'FAILED'}")
        sys.exit(0 if success else 1)

    if args.uninstall_startup:
        success = set_startup_enabled(False)
        print(f"Startup shortcut removed: {'SUCCESS' if success else 'FAILED'}")
        sys.exit(0 if success else 1)

    try:
        settings = load_settings()
    except Exception as error:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, str(error), "ADAM could not open", 0x10)
        logger.error("Settings initialization failed")
        return 1
    port = args.port or settings.get("agent_port", 8642)
    if not isinstance(port, int) or not 1024 <= port <= 65535:
        raise ValueError("Choose a local port between 1024 and 65535")

    # Single-instance lock: if already running, bring existing window forward and exit
    if not acquire_instance():
        if activate_instance(INSTANCE_FILE, APP_NAME):
            return 0
        logger.warning("The existing desktop instance could not be reached.")
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, "ADAM is already starting or running. Use its tray icon to open the dashboard.", "ADAM Companion", 0x40)
        return 1

    logger.info(f"Starting {APP_NAME} v{APP_VERSION} on port {port}...")

    # Initialize backend services (Coding Manager, Pi WS listener, window callback)
    import backend as backend_module
    backend = backend_module
    backend.init_backend(on_show_window=show_dashboard)

    # Start Flask server in background daemon thread
    flask_thread = threading.Thread(target=backend.run_flask_server, args=(port,), daemon=True)
    flask_thread.start()

    if not backend.server_ready.wait(15) or backend.server_error:
        backend.stop_backend()
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, backend.server_error or "The local service did not start in time.", "ADAM could not open", 0x10)
        return 1

    port = backend.server_port
    deadline = time.monotonic() + 12
    while not verify_instance(port, backend.DESKTOP_INSTANCE, os.getpid(), APP_NAME):
        if time.monotonic() >= deadline:
            backend.stop_backend()
            logger.error("Could not verify the owned local service; window was not opened.")
            if sys.platform == "win32":
                import ctypes
                ctypes.windll.user32.MessageBoxW(None, "ADAM could not verify its local service. Close ADAM and try again.", "ADAM could not open", 0x10)
            return 1
        time.sleep(.1)
    write_instance(INSTANCE_FILE, port, backend.DESKTOP_INSTANCE)

    # Start System Tray Icon in detached background thread
    try:
        tray_icon = setup_tray()
        tray_icon.run_detached()
        logger.info("System tray icon active.")
    except Exception as e:
        logger.error(f"Failed to start system tray icon: {e}")

    # Create pywebview Desktop Window
    start_hidden = bool(args.minimized)
    app_url = f"http://127.0.0.1:{port}/"

    main_window = webview.create_window(
        title="ADAM",
        url=app_url,
        width=1060,
        height=740,
        min_size=(860, 620),
        background_color="#000000",
        hidden=start_hidden,
        text_select=False,
    )

    # Attach window close handler to minimize to tray
    main_window.events.closing += on_window_closing

    try:
        webview.start(gui="edgechromium", debug=False)
    except Exception as e:
        logger.error(f"Webview error: {e}")
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None,
                "ADAM could not open its window. Install or repair Microsoft Edge WebView2 Runtime, then try again.\nhttps://developer.microsoft.com/microsoft-edge/webview2/",
                "ADAM display setup", 0x10)
    finally:
        quit_application()


if __name__ == "__main__":
    sys.exit(main())
