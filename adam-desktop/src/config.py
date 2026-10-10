"""Persistent Windows settings with protected credentials and atomic writes."""
import copy
import json
import logging
import os
import secrets
import sys
import threading
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path

APP_NAME = "ADAM Companion"
APP_VERSION = "0.04"
MDNS_SERVICE_TYPE = "_adam-laptop._tcp.local."
SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SOURCE_DIR.parent
APP_DIR = Path(getattr(sys, "_MEIPASS", PROJECT_DIR / "resources"))
EXE_PATH = sys.executable
BASE_DIR = Path(sys.executable).parent if getattr(sys, "frozen", False) else PROJECT_DIR
USER_DATA_DIR = Path(os.environ.get("ADAM_DATA_DIR") or Path(os.environ.get("APPDATA", Path.home())) / "ADAM")
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = USER_DATA_DIR / "settings.json"
LOCAL_ENV_FILE = (BASE_DIR if getattr(sys, "frozen", False) else BASE_DIR / "config" / "local") / ".env"
LOG_FILE = USER_DATA_DIR / "adam_agent.log"
LOG_BUFFER = deque(maxlen=200)
_lock = threading.RLock()
_cached = None
SECRET_KEYS = ("agent_token", "sync_token")

class UILogHandler(logging.Handler):
    def emit(self, record):
        LOG_BUFFER.append({"timestamp": record.created, "level": record.levelname,
                           "message": self.format(record)[:3000]})

def setup_logger():
    result = logging.getLogger("adam_agent")
    result.setLevel(logging.INFO)
    if not result.handlers:
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        for handler in (RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8"), UILogHandler()):
            handler.setFormatter(formatter)
            result.addHandler(handler)
    return result

logger = setup_logger()

def _defaults():
    return {"agent_port": 8642, "pi_ip": "", "pi_host": "adam-pi.local",
            "pi_ws_port": 8765, "pi_sync_port": 8766, "paired": False,
            "sync_token_host": "", "sync_token_port": 8766,
            "startup_on_login": False, "paused": False, "enabled_actions": {},
            "touch_assignments": {}, "touch_applied_hash": "", "touch_policy_version": 2, "setup_complete": False,
            "user_name": "", "coding_workspace": "", "version": APP_VERSION}

def _write_public(data):
    temporary = CONFIG_FILE.with_suffix(".tmp")
    public = {key: value for key, value in data.items() if key not in SECRET_KEYS}
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(public, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, CONFIG_FILE)

def load_settings():
    global _cached
    with _lock:
        if _cached is None:
            from secure_store import load_secret, save_secret
            data = _defaults()
            existed = CONFIG_FILE.exists()
            if existed:
                try:
                    saved = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                    if not isinstance(saved, dict):
                        raise ValueError("Settings must be an object")
                    data.update(saved)
                except (ValueError, OSError) as error:
                    raise RuntimeError("ADAM settings could not be read. Restore settings.json from a backup or rename it to recover.") from error
            touch_migration = existed and saved.get("touch_policy_version") != 2
            if touch_migration:
                previous = data.get("touch_assignments")
                previous = previous if isinstance(previous, dict) else {}
                data["touch_assignments"] = {
                    sensor: {event: binding for event, binding in events.items()
                             if event in (("double", "triple", "hold") if sensor == "touch3" else ("hold",))}
                    for sensor, events in previous.items()
                    if sensor in ("touch1", "touch2", "touch3", "touch4") and isinstance(events, dict)}
                data["touch_applied_hash"] = ""
                data["touch_policy_version"] = 2
            legacy = {}
            if not os.environ.get("ADAM_DATA_DIR") and LOCAL_ENV_FILE.exists():
                for line in LOCAL_ENV_FILE.read_text(encoding="utf-8-sig").splitlines():
                    if "=" in line and not line.lstrip().startswith("#"):
                        key, value = line.split("=", 1)
                        legacy[key.strip()] = value.strip().strip(chr(34)).strip(chr(39))
            migrate = not existed or touch_migration or any(key in data for key in SECRET_KEYS)
            for key in SECRET_KEYS:
                value = load_secret("settings_" + key)
                if value and key == "sync_token" and value.startswith("{"):
                    try:
                        bound = json.loads(value)
                        value = bound["value"] if (bound.get("host"), bound.get("port")) == (data.get("sync_token_host"), data.get("sync_token_port")) else ""
                    except (ValueError, KeyError, TypeError):
                        value = ""
                if not value:
                    value = str(data.get(key) or (legacy.get("AGENT_TOKEN", "") if key == "agent_token" else ""))
                    if key == "agent_token" and not value:
                        value = secrets.token_urlsafe(32)
                    if value:
                        save_secret("settings_" + key, value)
                data[key] = value or ""
            if migrate:
                _write_public(data)
            _cached = data
        return copy.deepcopy(_cached)

def save_settings(settings):
    global _cached
    from secure_store import load_secret, save_secret, delete_secret
    with _lock:
        next_data = copy.deepcopy(settings)
        originals = {}
        try:
            for key in SECRET_KEYS:
                if key not in next_data:
                    continue
                changed = next_data[key] != (_cached or {}).get(key)
                if key == "sync_token":
                    changed = changed or any(next_data.get(k) != (_cached or {}).get(k) for k in ("sync_token_host", "sync_token_port"))
                if changed:
                    name = "settings_" + key
                    originals[name] = load_secret(name)
                    if next_data[key]:
                        value = str(next_data[key])
                        if key == "sync_token":
                            value = json.dumps({"value": value, "host": next_data.get("sync_token_host"), "port": next_data.get("sync_token_port")})
                        save_secret(name, value)
                    else:
                        delete_secret(name)
            _write_public(next_data)
        except Exception:
            for name, value in originals.items():
                if value is not None:
                    save_secret(name, value)
                else:
                    delete_secret(name)
            raise
        _cached = next_data

def update_settings(patch):
    with _lock:
        data = load_settings()
        data.update(patch)
        save_settings(data)
        return copy.deepcopy(data)

def get_startup_shortcut_path() -> Path:
    """Get the path to the shortcut in Windows Startup folder."""
    startup_dir = Path(os.environ["ADAM_STARTUP_DIR"]) if os.environ.get("ADAM_STARTUP_DIR") else Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    return startup_dir / f"{APP_NAME}.lnk"


def is_startup_enabled() -> bool:
    """Check if startup shortcut exists in Startup folder."""
    return get_startup_shortcut_path().exists()


def set_startup_enabled(enable: bool) -> bool:
    """Create or delete Windows Startup folder shortcut."""
    shortcut_path = get_startup_shortcut_path()
    try:
        if not enable:
            if shortcut_path.exists():
                shortcut_path.unlink()
                logger.info(f"Removed startup shortcut: {shortcut_path}")
            return True

        shortcut_dir = shortcut_path.parent
        shortcut_dir.mkdir(parents=True, exist_ok=True)

        if getattr(sys, "frozen", False):
            target = str(EXE_PATH)
            args = "--tray"
            icon_location = target
            work_dir = str(Path(target).parent)
        else:
            target = sys.executable
            # Prefer pythonw.exe if available so no black terminal window pops up on login
            pythonw = Path(target).with_name("pythonw.exe")
            if pythonw.exists():
                target = str(pythonw)
            script_path = str(SOURCE_DIR / "app.py")
            args = f'"{script_path}" --tray'
            icon_location = str(APP_DIR / "icons" / "adam.ico")
            if not os.path.exists(icon_location):
                icon_location = target
            work_dir = str(PROJECT_DIR)

        # Use PowerShell WScript.Shell (zero external dependencies required)
        ps_quote = lambda value: str(value).replace("'", "''")
        ps_script = f"""
        $WshShell = New-Object -ComObject WScript.Shell
        $Shortcut = $WshShell.CreateShortcut('{ps_quote(str(shortcut_path))}')
        $Shortcut.TargetPath = '{ps_quote(target)}'
        $Shortcut.Arguments = '{ps_quote(args)}'
        $Shortcut.WorkingDirectory = '{ps_quote(work_dir)}'
        $Shortcut.Description = 'ADAM Companion Windows Agent'
        $Shortcut.IconLocation = '{ps_quote(icon_location)}'
        $Shortcut.Save()
        """
        import subprocess
        res = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
            capture_output=True,
            text=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW
        )
        if res.returncode == 0:
            logger.info("Updated Windows startup shortcut")
            return True
        else:
            logger.error(f"Failed creating startup shortcut: {res.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error updating startup shortcut: {e}")
        return False
