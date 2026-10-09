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
laptop decides what it can do). As of v41 both the fallback manifest and the
value-type rules come from laptop_actions.py — the one file that is deployed
byte-identically to the Pi and the laptop, so the two sides cannot disagree
about what an action is called or what kind of value it takes. All config
constants come from config.py.
"""

import time
import threading

import requests

import laptop_actions
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

_laptop_agent_ip_cache: dict = {"ip": None, "port": LAPTOP_AGENT_PORT, "ts": 0.0}
_pairing_lock = threading.RLock()
_discovery_lock = threading.Lock()
_paired_endpoint = None
_pairing_revoked = False


def configure_laptop_pairing(record: dict) -> None:
    """Activate an already-validated, durably saved pairing atomically."""
    global _paired_endpoint, _pairing_revoked
    with _pairing_lock:
        _paired_endpoint = ((record["host"], record["port"], record["token"])
                            if record.get("enabled") else None)
        _pairing_revoked = not bool(record.get("enabled"))
        _laptop_agent_ip_cache.update(ip=None, port=LAPTOP_AGENT_PORT, ts=0.0)
        _laptop_actions_cache.update(actions=None, ts=0.0)


def get_laptop_endpoint():
    """Snapshot host, port and key together; never mix two concurrent pairings."""
    with _pairing_lock:
        if _paired_endpoint is not None:
            return _paired_endpoint
        if _pairing_revoked:
            return None
    ip = _discover_laptop_agent_ip()
    with _pairing_lock:
        if _paired_endpoint is not None:
            return _paired_endpoint
        if _pairing_revoked:
            return None
        port = (_laptop_agent_ip_cache["port"]
                if _laptop_agent_ip_cache["ip"] == ip else LAPTOP_AGENT_PORT)
        return (ip, port, LAPTOP_AGENT_TOKEN) if ip else None


def _authority(host):
    return f"[{host}]" if ":" in host else host

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
    # Manifest refresh and voice dispatch can ask concurrently. Share one
    # bounded discovery, including a short cache for a missing laptop.
    with _discovery_lock:
        return _discover_laptop_agent_ip_locked(timeout)


def _discover_laptop_agent_ip_locked(timeout: float) -> str | None:
    """Find the laptop agent's current IP via mDNS. Cached briefly to avoid
    repeated network discovery on every tool call. Falls back to a static
    LAPTOP_AGENT_IP if mDNS is unavailable or fails."""
    with _pairing_lock:
        if _paired_endpoint is not None:
            return _paired_endpoint[0]
        if _pairing_revoked:
            return None
        now = time.monotonic()
        ttl = LAPTOP_DISCOVERY_TTL_S if _laptop_agent_ip_cache["ip"] else min(5, LAPTOP_DISCOVERY_TTL_S)
        if _laptop_agent_ip_cache["ts"] and now - _laptop_agent_ip_cache["ts"] < ttl:
            return _laptop_agent_ip_cache["ip"]

    if ZEROCONF_AVAILABLE:
        try:
            import socket as _socket
            found: dict = {}

            class _Listener:
                def add_service(self, zc, service_type, name):
                    info = zc.get_service_info(service_type, name,
                                               timeout=int(timeout * 1000))
                    if info and info.addresses and 1 <= info.port <= 65535:
                        for address in info.addresses:
                            if len(address) == 4:
                                found.update(ip=_socket.inet_ntoa(address), port=info.port)
                                break

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
                with _pairing_lock:
                    if _paired_endpoint is not None:
                        return _paired_endpoint[0]
                    if _pairing_revoked:
                        return None
                    _laptop_agent_ip_cache.update(**found, ts=time.monotonic())
                print(f"  📡 Discovered laptop agent via mDNS: {found['ip']}:{found['port']}")
                return found["ip"]
            else:
                print(f"  ⚠️  mDNS discovery found no '{LAPTOP_MDNS_SERVICE}' "
                      f"service within {timeout}s")
        except Exception as e:
            print(f"  ⚠️  mDNS discovery error: {e}")

    with _pairing_lock:
        if _paired_endpoint is not None:
            return _paired_endpoint[0]
        if _pairing_revoked:
            return None
        _laptop_agent_ip_cache.update(ip=LAPTOP_AGENT_STATIC_IP or None,
                                     port=LAPTOP_AGENT_PORT, ts=time.monotonic())
        return _laptop_agent_ip_cache["ip"]


def _laptop_agent_url() -> str | None:
    endpoint = get_laptop_endpoint()
    if not endpoint:
        return None
    return f"http://{_authority(endpoint[0])}:{endpoint[1]}/control"


_LAPTOP_ACTIONS_FALLBACK = laptop_actions.fallback_manifest()

_laptop_actions_cache: dict = {"actions": None, "ts": 0.0}


def _annotate(actions: dict) -> dict:
    """Give every entry of a live manifest a usable value_type.

    An agent running pre-v41 code answers /actions with only
    (description, needs_value, value_hint) — no type. Without this, ADAM would
    be back to guessing, and the guess that used to be made (int) destroyed
    every string argument. infer_value_type() reconstructs the type from the
    canonical table first and the hint second, so a stale laptop still works.
    """
    out = {}
    for name, spec in actions.items():
        if not isinstance(spec, dict):
            continue
        spec = dict(spec)
        if spec.get("value_type") not in ("none", "int", "str", "enum"):
            spec["value_type"] = laptop_actions.infer_value_type(
                spec.get("needs_value", False), spec.get("value_hint", ""), name)
        out[name] = spec
    return out


def refresh_laptop_actions(force: bool = False) -> dict:
    now = time.time()
    if (not force and _laptop_actions_cache["actions"] is not None
            and now - _laptop_actions_cache["ts"] < LAPTOP_ACTIONS_TTL_S):
        return _laptop_actions_cache["actions"]

    endpoint = get_laptop_endpoint()
    if endpoint is None:
        return _laptop_actions_cache["actions"] or _LAPTOP_ACTIONS_FALLBACK
    ip, port, _ = endpoint

    try:
        with requests.Session() as http:
            http.trust_env = False
            resp = http.get(f"http://{_authority(ip)}:{port}/actions",
                            timeout=LAPTOP_AGENT_TIMEOUT_S, allow_redirects=False)
        resp.raise_for_status()
        data = resp.json()
        actions = data.get("actions", {})
        if actions:
            actions = _annotate(actions)
            _laptop_actions_cache["actions"] = actions
            _laptop_actions_cache["ts"] = now
            print(f"  🔧 Laptop actions ({data.get('platform','?')}): "
                  f"{len(actions)} available")

            # Tell the operator when the two sides have drifted. A silent
            # mismatch here is what made ten capabilities invisible in v40.
            rep = laptop_actions.parity_report(actions)
            if rep["missing"]:
                print(f"  ⚠️  Laptop agent is missing expected actions: "
                      f"{', '.join(rep['missing'])} — is it running older code?")
            if rep["untyped"]:
                print(f"  ℹ️  Laptop agent sent an untyped manifest "
                      f"({len(rep['untyped'])} actions); types inferred locally.")
            return actions
    except Exception as e:
        print(f"  ⚠️  Could not fetch laptop /actions manifest: {e}")

    return _laptop_actions_cache["actions"] or _LAPTOP_ACTIONS_FALLBACK


def get_laptop_actions() -> dict:
    return refresh_laptop_actions(force=False)


def laptop_control_sync(action: str, value=None) -> dict:
    """Run one action on the laptop. `value` may be an int or a str — its type
    is decided by the manifest, not by this function, and has already been
    coerced by the caller (tool_handler) via laptop_actions.coerce()."""
    # The live agent only knows its own action names, so an alias ADAM may have
    # been handed (clipboard_get) is translated before it goes on the wire.
    action = laptop_actions.resolve(action) or action

    endpoint = get_laptop_endpoint()
    if endpoint is None:
        return {"status": "error",
                "reason": "ADAM cannot reach a paired laptop. Open the ADAM PC app, "
                          "go to My ADAM > Connection, enter the Pi's connection key, "
                          "connect, then choose Allow laptop control. This saves a direct "
                          "connection even when mDNS is blocked. Keep both devices on "
                          "the same network and allow the app through Windows Firewall."}

    ip, port, token = endpoint
    url = f"http://{_authority(ip)}:{port}/control"
    payload = {"action": action, "token": token}
    if value is not None:
        payload["value"] = value

    try:
        with requests.Session() as http:
            http.trust_env = False
            resp = http.post(url, json=payload, timeout=LAPTOP_AGENT_TIMEOUT_S,
                             allow_redirects=False)
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
        with _pairing_lock:
            _laptop_agent_ip_cache.update(ip=None, ts=0.0)
        return {"status": "error",
                "reason": "Connection timed out — laptop may have changed "
                          "networks or gone to sleep. Will re-discover on "
                          "next attempt."}
    except requests.exceptions.ConnectionError as e:
        with _pairing_lock:
            _laptop_agent_ip_cache.update(ip=None, ts=0.0)
        return {"status": "error", "reason": f"could not connect to laptop agent: {e}"}
    except Exception as e:
        return {"status": "error", "reason": f"{type(e).__name__}: {e}"}
