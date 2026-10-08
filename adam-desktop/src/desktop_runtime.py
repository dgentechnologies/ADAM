"""Reserve the local server socket and identify this desktop process exactly."""
import json
import os
from pathlib import Path
import socket
import time

import requests


def reserve_listener(preferred_port):
    """Keep the socket bound while handing it to Waitress; no free-port race."""
    for port in (preferred_port, 0):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if os.name == "nt":
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            listener.bind(("0.0.0.0", port))
            return listener
        except OSError:
            listener.close()
            if port == 0:
                raise


def write_instance(path, port, identity):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    data = {"port": port, "instance_id": identity, "pid": os.getpid()}
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_instance(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if (isinstance(data, dict) and type(data.get("port")) is int
                and 1024 <= data["port"] <= 65535
                and type(data.get("pid")) is int and data["pid"] > 0
                and isinstance(data.get("instance_id"), str)
                and len(data["instance_id"]) >= 32):
            return data
    except (OSError, ValueError):
        pass
    return None


def remove_instance(path, identity):
    data = read_instance(path)
    if data and data["instance_id"] == identity:
        Path(path).unlink(missing_ok=True)


def verify_instance(port, identity, pid, app_name, timeout=1):
    """Reject redirects and same-name services belonging to another process."""
    try:
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(f"http://127.0.0.1:{port}/ping", timeout=timeout,
                                   allow_redirects=False)
            if response.status_code != 200:
                return False
            data = response.json()
            return (data.get("app") == app_name and data.get("instance_id") == identity
                    and data.get("pid") == pid)
    except (requests.RequestException, ValueError, AttributeError):
        return False


def activate_instance(path, app_name, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        data = read_instance(path)
        if data and verify_instance(data["port"], data["instance_id"], data["pid"], app_name):
            try:
                with requests.Session() as session:
                    session.trust_env = False
                    response = session.post(f'http://127.0.0.1:{data["port"]}/show_window',
                                            json={"instance_id": data["instance_id"]},
                                            timeout=1, allow_redirects=False)
                    return response.status_code == 200
            except requests.RequestException:
                return False
        time.sleep(.2)
    return False
