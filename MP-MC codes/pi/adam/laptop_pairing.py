"""Authenticated LAN laptop pairing without editing the Pi's environment."""
from __future__ import annotations

import asyncio
import ipaddress
import json
import os
from pathlib import Path
import tempfile
import time

import requests


def validate_pairing(value):
    if not isinstance(value, dict):
        raise ValueError("Provide this laptop's address, port and key")
    try:
        host = ipaddress.ip_address(value.get("host", ""))
    except (ValueError, TypeError):
        raise ValueError("Use this laptop's local IP address") from None
    tailscale = host.version == 4 and host in ipaddress.ip_network("100.64.0.0/10")
    if host.is_unspecified or host.is_multicast or not (host.is_private or host.is_loopback or tailscale):
        raise ValueError("Laptop pairing is available only on a local or private network")
    port = value.get("port")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Use a valid laptop port")
    token = value.get("token")
    if not isinstance(token, str) or not 16 <= len(token) <= 512 or any(ord(c) < 33 or ord(c) > 126 for c in token):
        raise ValueError("Use a valid laptop connection key")
    return {"version": 1, "enabled": True, "host": host.compressed, "port": port, "token": token}


def verify_laptop(record):
    host = f"[{record['host']}]" if ":" in record["host"] else record["host"]
    try:
        with requests.Session() as session:
            session.trust_env = False
            with session.get(f"http://{host}:{record['port']}/pair/verify",
                             headers={"X-ADAM-Token": record["token"]},
                             timeout=(2, 3), allow_redirects=False, stream=True) as response:
                if response.status_code in (401, 403):
                    raise ValueError("The laptop did not accept its connection key")
                if response.status_code != 200:
                    raise ValueError("The laptop did not confirm pairing")
                content = bytearray()
                deadline = time.monotonic() + 7
                for chunk in response.iter_content(1024):
                    content.extend(chunk)
                    if len(content) > 4096 or time.monotonic() > deadline:
                        raise ValueError("The laptop returned an invalid pairing response")
                payload = json.loads(content)
            if (not isinstance(payload, dict) or payload.get("status") != "ok"
                    or payload.get("app") != "ADAM Companion" or payload.get("kind") != "laptop"
                    or type(payload.get("api")) is not int or payload["api"] != 1):
                raise ValueError("This address did not identify itself as the ADAM desktop app")
    except requests.RequestException:
        raise ValueError("ADAM could not reach this laptop. Check the network and Windows Firewall") from None
    except json.JSONDecodeError:
        raise ValueError("The laptop returned an invalid pairing response") from None


class LaptopPairing:
    def __init__(self, path, configure, verify=verify_laptop):
        self.path, self.configure, self.verify = Path(path), configure, verify
        self.record = None
        self._lock = asyncio.Lock()

    async def load(self):
        def read():
            if not self.path.exists():
                return None
            if self.path.stat().st_size > 4096:
                raise ValueError("Saved laptop pairing is too large")
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or value.get("version") != 1 or type(value.get("enabled")) is not bool:
                raise ValueError("Saved laptop pairing could not be read")
            return validate_pairing(value) if value["enabled"] else {"version": 1, "enabled": False}
        self.record = await asyncio.to_thread(read)
        if self.record is not None:
            self.configure(self.record)

    def status(self):
        return {"paired": bool(self.record and self.record["enabled"]),
                "host": self.record.get("host", "") if self.record else "",
                "port": self.record.get("port") if self.record else None}

    async def _save(self, record):
        def write():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                                 prefix=".laptop-", delete=False) as file:
                    temporary = Path(file.name)
                    json.dump(record, file, separators=(",", ":"))
                    file.flush()
                    os.fsync(file.fileno())
                os.chmod(temporary, 0o600)
                os.replace(temporary, self.path)
            finally:
                if temporary:
                    temporary.unlink(missing_ok=True)
        await asyncio.to_thread(write)
        self.configure(record)
        self.record = record

    async def pair(self, body):
        record = validate_pairing(body)
        async with self._lock:
            await asyncio.to_thread(self.verify, record)
            await self._save(record)
        return {"ok": True, "paired": True, "host": record["host"], "port": record["port"]}

    async def unpair(self):
        async with self._lock:
            # A tombstone disables the old environment fallback after restart.
            await self._save({"version": 1, "enabled": False})
        return {"ok": True, "paired": False}


_service = None


async def initialize_laptop_pairing():
    global _service
    if _service is not None:
        return _service
    from config import BASE_DIR
    from laptop_agent_client import configure_laptop_pairing
    _service = LaptopPairing(Path(BASE_DIR) / "laptop_pairing.json", configure_laptop_pairing)
    try:
        await _service.load()
    except Exception:
        # A corrupt/revoked pairing must never silently revert to an older key.
        configure_laptop_pairing({"version": 1, "enabled": False})
        print("  Laptop pairing could not be loaded; reconnect from the desktop app.")
    return _service


def available():
    return _service is not None


async def pair(body):
    if _service is None:
        raise RuntimeError("Laptop pairing has not started")
    return await _service.pair(body)


async def unpair():
    if _service is None:
        raise RuntimeError("Laptop pairing has not started")
    return await _service.unpair()
