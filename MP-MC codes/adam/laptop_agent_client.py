"""
laptop_agent_client.py — ADAM v40 laptop remote-control client
==============================================================================
Talks to laptop_agent.py (a small HTTP server running on the user's laptop)
so ADAM can change the laptop's volume / screen brightness by voice.

Discovery is production-grade: the laptop is found on the LAN via mDNS/Zeroconf
(service '_adam-laptop._tcp.local.'), with an optional static LAPTOP_AGENT_IP
fallback for networks where mDNS is blocked. The discovered IP is cached
briefly (LAPTOP_DISCOVERY_TTL_S) so we don't re-run discovery on every call,
and is invalidated on a connection failure so a laptop that changed networks
gets re-discovered automatically.

The available actions are fetched from the agent's /actions manifest (so the
laptop decides what it can do); a hard-coded fallback keeps the tool usable if
the manifest can't be fetched. All config constants come from config.py.
"""

import time

import requests

from config import (
    LAPTOP_AGENT_PORT,
    LAPTOP_AGENT_TOKEN,
    LAPTOP_AGENT_TIMEOUT_S,
    LAPTOP_AGENT_STATIC_IP,
    LAPTOP_MDNS_SERVICE,
    LAPTOP_DISCOVERY_TIMEOUT_S,
    LAPTOP_DISCOVERY_TTL_S,
    LAPTOP_ACTIONS_TTL_S,
)

_laptop_agent_ip_cache: dict = {"ip": LAPTOP_AGENT_STATIC_IP or None, "ts": 0.0}

ZEROCONF_AVAILABLE = False
try:
    from zeroconf import Zeroconf, ServiceBrowser
    ZEROCONF_AVAILABLE = True
except ImportError:
    pass

if not LAPTOP_AGENT_STATIC_IP and not ZEROCONF_AVAILABLE:
    print("  ⚠️  Neither LAPTOP_AGENT_IP nor zeroconf package are available — "
          "laptop_control tool will not work. Run: "
          "pip install zeroconf --break-system-packages")
elif not LAPTOP_AGENT_STATIC_IP:
    print("  ℹ️  LAPTOP_AGENT_IP not set — will auto-discover via mDNS "
          f"('{LAPTOP_MDNS_SERVICE}')")


def _discover_laptop_agent_ip(timeout: float = LAPTOP_DISCOVERY_TIMEOUT_S) -> str | None:
    """Find the laptop agent's current IP via mDNS. Cached briefly to avoid
    repeated network discovery on every tool call. Falls back to a static
    LAPTOP_AGENT_IP if mDNS is unavailable or fails."""
    now = time.time()
    if (_laptop_agent_ip_cache["ip"]
            and now - _laptop_agent_ip_cache["ts"] < LAPTOP_DISCOVERY_TTL_S):
        return _laptop_agent_ip_cache["ip"]

    if ZEROCONF_AVAILABLE:
        try:
            import socket as _socket
            found: dict = {}

            class _Listener:
                def add_service(self, zc, service_type, name):
                    info = zc.get_service_info(service_type, name,
                                               timeout=int(timeout * 1000))
                    if info and info.addresses:
                        found["ip"] = _socket.inet_ntoa(info.addresses[0])

                def update_service(self, *a, **k):
                    pass

                def remove_service(self, *a, **k):
                    pass

            zc = Zeroconf()
            try:
                ServiceBrowser(zc, LAPTOP_MDNS_SERVICE, _Listener())
                deadline = time.time() + timeout
                while time.time() < deadline and "ip" not in found:
                    time.sleep(0.1)
            finally:
                zc.close()

            if "ip" in found:
                _laptop_agent_ip_cache["ip"] = found["ip"]
                _laptop_agent_ip_cache["ts"] = now
                print(f"  📡 Discovered laptop agent via mDNS: {found['ip']}")
                return found["ip"]
            else:
                print(f"  ⚠️  mDNS discovery found no '{LAPTOP_MDNS_SERVICE}' "
                      f"service within {timeout}s")
        except Exception as e:
            print(f"  ⚠️  mDNS discovery error: {e}")

    if LAPTOP_AGENT_STATIC_IP:
        return LAPTOP_AGENT_STATIC_IP
    return None


def _laptop_agent_url() -> str | None:
    ip = _discover_laptop_agent_ip()
    if not ip:
        return None
    return f"http://{ip}:{LAPTOP_AGENT_PORT}/control"


_LAPTOP_ACTIONS_FALLBACK = {
    "volume_up":       {"description": "Increase system volume by 10%.", "needs_value": False, "value_hint": ""},
    "volume_down":     {"description": "Decrease system volume by 10%.", "needs_value": False, "value_hint": ""},
    "volume_set":      {"description": "Set system volume to an exact percentage.", "needs_value": True, "value_hint": "0-100"},
    "volume_mute":     {"description": "Mute system audio.", "needs_value": False, "value_hint": ""},
    "volume_unmute":   {"description": "Unmute system audio.", "needs_value": False, "value_hint": ""},
    "brightness_up":   {"description": "Increase screen brightness by 10%.", "needs_value": False, "value_hint": ""},
    "brightness_down": {"description": "Decrease screen brightness by 10%.", "needs_value": False, "value_hint": ""},
    "brightness_set":  {"description": "Set screen brightness to an exact percentage.", "needs_value": True, "value_hint": "0-100"},
}

_laptop_actions_cache: dict = {"actions": None, "ts": 0.0}


def refresh_laptop_actions(force: bool = False) -> dict:
    now = time.time()
    if (not force and _laptop_actions_cache["actions"] is not None
            and now - _laptop_actions_cache["ts"] < LAPTOP_ACTIONS_TTL_S):
        return _laptop_actions_cache["actions"]

    ip = _discover_laptop_agent_ip()
    if ip is None:
        return _laptop_actions_cache["actions"] or _LAPTOP_ACTIONS_FALLBACK

    try:
        resp = requests.get(f"http://{ip}:{LAPTOP_AGENT_PORT}/actions",
                             timeout=LAPTOP_AGENT_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
        actions = data.get("actions", {})
        if actions:
            _laptop_actions_cache["actions"] = actions
            _laptop_actions_cache["ts"] = now
            print(f"  🔧 Laptop actions ({data.get('platform','?')}): "
                  f"{', '.join(actions.keys())}")
            return actions
    except Exception as e:
        print(f"  ⚠️  Could not fetch laptop /actions manifest: {e}")

    return _laptop_actions_cache["actions"] or _LAPTOP_ACTIONS_FALLBACK


def get_laptop_actions() -> dict:
    return refresh_laptop_actions(force=False)


def laptop_control_sync(action: str, value: int | None = None) -> dict:
    url = _laptop_agent_url()
    if url is None:
        return {"status": "error",
                "reason": "Laptop agent not found on network. Make sure "
                          "laptop_agent.py is running on the laptop, both "
                          "devices are on the same LAN, and either mDNS is "
                          "allowed on your router or LAPTOP_AGENT_IP is set "
                          "in .env as a fallback."}

    payload = {"action": action, "token": LAPTOP_AGENT_TOKEN}
    if value is not None:
        payload["value"] = value

    try:
        resp = requests.post(url, json=payload, timeout=LAPTOP_AGENT_TIMEOUT_S)
        try:
            data = resp.json()
        except Exception:
            data = {"raw": resp.text}
        if resp.status_code != 200:
            return {"status": "error",
                    "reason": data.get("reason", f"HTTP {resp.status_code}"),
                    "http_status": resp.status_code}
        return data
    except requests.exceptions.ConnectTimeout:
        _laptop_agent_ip_cache["ip"] = None
        return {"status": "error",
                "reason": "Connection timed out — laptop may have changed "
                          "networks or gone to sleep. Will re-discover on "
                          "next attempt."}
    except requests.exceptions.ConnectionError as e:
        _laptop_agent_ip_cache["ip"] = None
        return {"status": "error", "reason": f"could not connect to laptop agent: {e}"}
    except Exception as e:
        return {"status": "error", "reason": f"{type(e).__name__}: {e}"}
