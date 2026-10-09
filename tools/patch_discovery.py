import io
import re
from pathlib import Path

p = Path(r"D:\Dgen Technologies Pvt. Ltd\ADAM\adam-desktop\src\backend.py")
s = p.read_text(encoding="utf-8")
orig = s

lines = s.splitlines(keepends=True)
idx = next((i for i, l in enumerate(lines) if l.strip() == "# mDNS BROADCAST"), None)
if idx is None:
    raise SystemExit("could not find the '# mDNS BROADCAST' comment line")

# Step back over the banner line above it, if there is one, so the new block
# lands before the whole comment header rather than splitting it.
start = idx
if idx > 0:
    above = lines[idx - 1].strip()
    if above.startswith("#") and len(set(above)) <= 3 and len(above) > 3:
        start = idx - 1
insert_at = sum(len(l) for l in lines[:start])

BLOCK = r'''# =========================================================================
# ADAM DISCOVERY - "Find ADAM", then one click to pair
#
# This is the other half of mDNS. Below, the app ADVERTISES itself as
# `_adam-laptop._tcp` so the Pi can find the laptop it controls. Here we BROWSE
# for `_adam._tcp`, which the Pi publishes (pi/adam/discovery.py).
#
# That direction did not exist before, which is why connecting ADAM meant
# typing an IP into Settings and pasting a token out of the Pi's .env by hand.
# The address then went stale every time DHCP moved the Pi, and the Clock tab
# just reported "not connected".
#
# The ServiceBrowser is kept running for the life of the app rather than
# started per request: mDNS answers arrive asynchronously, so a browser created
# inside a request handler would have to block for seconds and would still miss
# units that answer late. "Find ADAM" reads an already-warm cache instead.
# =========================================================================

_adam_browser = None
_adam_zc = None
_found_adams = {}          # deviceId -> record

ADAM_SERVICE_TYPE = "_adam._tcp.local."
ADAM_STALE_AFTER_S = 120


def _decode_txt(props):
    out = {}
    for k, v in (props or {}).items():
        try:
            key = k.decode() if isinstance(k, bytes) else str(k)
            if v is None:
                continue
            out[key] = v.decode() if isinstance(v, bytes) else str(v)
        except Exception:
            continue
    return out


def start_adam_discovery():
    """Begin browsing for ADAM units. Never raises - discovery is a
    convenience, and the manual IP override in Settings still works."""
    global _adam_browser, _adam_zc
    if _adam_browser is not None:
        return
    try:
        from zeroconf import ServiceBrowser, Zeroconf, ServiceListener
    except ImportError:
        logger.warning("zeroconf not installed - 'Find ADAM' will not work.")
        return

    class _Listener(ServiceListener):
        def _record(self, zc, type_, name):
            try:
                info = zc.get_service_info(type_, name, timeout=3000)
                if not info or not info.addresses:
                    return
                txt = _decode_txt(info.properties)
                ip = socket.inet_ntoa(info.addresses[0])
                dev_id = txt.get("id") or name.split(".")[0]
                _found_adams[dev_id] = {
                    "id": dev_id,
                    "name": txt.get("name", "ADAM"),
                    "host": ip,
                    "port": int(info.port or 8766),
                    "version": txt.get("ver", ""),
                    "api": txt.get("api", ""),
                    # From the TXT record, so an already-claimed unit can be
                    # shown greyed out rather than letting the user pick it and
                    # then fail with a 409.
                    "paired": txt.get("paired") == "1",
                    "seen_at": time.time(),
                }
                logger.info("[discovery] found %s at %s:%s", dev_id, ip, info.port)
            except Exception as e:
                logger.warning("[discovery] could not read %s: %s", name, e)

        def add_service(self, zc, type_, name):
            self._record(zc, type_, name)

        def update_service(self, zc, type_, name):
            self._record(zc, type_, name)

        def remove_service(self, zc, type_, name):
            for k in list(_found_adams):
                if name.startswith(k):
                    _found_adams.pop(k, None)

    try:
        _adam_zc = Zeroconf()
        _adam_browser = ServiceBrowser(_adam_zc, ADAM_SERVICE_TYPE, _Listener())
        logger.info("[discovery] browsing for '%s'", ADAM_SERVICE_TYPE)
    except Exception as e:
        logger.warning("[discovery] browse failed: %s", e)
        _adam_browser = None


def stop_adam_discovery():
    global _adam_browser, _adam_zc
    try:
        if _adam_zc is not None:
            _adam_zc.close()
    except Exception:
        pass
    finally:
        _adam_browser, _adam_zc = None, None


@app.route("/discover/adam", methods=["GET"])
def discover_adam_endpoint():
    """What "Find ADAM" renders. Starts the browser on first call, so a user
    who never opens this screen pays nothing for it."""
    start_adam_discovery()
    # Drop units not seen recently: an ADAM that was switched off should leave
    # the list rather than sit there failing to connect.
    cutoff = time.time() - ADAM_STALE_AFTER_S
    units = [dict(u) for u in _found_adams.values() if u["seen_at"] > cutoff]
    units.sort(key=lambda u: u["id"])
    settings = load_settings()
    current = str(settings.get("pi_ip", "")).strip()
    have_token = bool(str(settings.get("sync_token", "")).strip())
    for u in units:
        u["connected"] = bool(current) and u["host"] == current and have_token
    return jsonify({"status": "ok", "units": units, "count": len(units)})


@app.route("/pair/adam", methods=["POST"])
def pair_adam_endpoint():
    """Claim an ADAM and store what it hands back.

    This is the step that removes the manual token: the Pi gives up its sync
    token while it is unclaimed, and the app saves it alongside the address.
    The user picks a name from a list; nothing is typed.
    """
    data = request.get_json(silent=True) or {}
    host = str(data.get("host", "")).strip()
    try:
        port = int(data.get("port", 8766) or 8766)
    except (TypeError, ValueError):
        port = 8766
    if not host:
        return jsonify({"status": "error", "reason": "no address given"}), 200

    try:
        response = requests.post("http://%s:%d/api/pair/claim" % (host, port),
                                 json={}, timeout=(3.05, 8))
        payload = response.json()
    except requests.RequestException:
        return jsonify({"status": "error",
                        "reason": "Could not reach ADAM at %s:%d" % (host, port)}), 200
    except ValueError:
        return jsonify({"status": "error",
                        "reason": "ADAM returned an unreadable reply"}), 200

    if response.status_code == 409:
        return jsonify({"status": "error",
                        "reason": payload.get("hint") or "That ADAM is already paired."}), 200
    if response.status_code != 200 or not payload.get("token"):
        return jsonify({"status": "error",
                        "reason": payload.get("error")
                        or "Pairing refused (HTTP %d)" % response.status_code}), 200

    settings = load_settings()
    settings["pi_ip"] = host
    settings["pi_sync_port"] = port
    settings["sync_token"] = str(payload["token"])
    settings["pi_device_id"] = str(payload.get("id", ""))
    save_settings(settings)

    CURRENT_ROBOT_STATE["pi_ip"] = host
    logger.info("[pair] paired with %s at %s:%d", payload.get("id"), host, port)
    log_activity("adam_paired", str(payload.get("id", "")), "ok")

    return jsonify({"status": "ok", "id": payload.get("id", ""),
                    "name": payload.get("name", "ADAM"),
                    "host": host, "port": port})


@app.route("/pair/adam/forget", methods=["POST"])
def unpair_adam_endpoint():
    """Release the claim so another device can pair, and forget it locally.

    Order matters: tell the Pi FIRST, while we still hold the token that
    authorises the release. Clearing our settings first would leave the Pi
    permanently claimed by a device that can no longer prove it owns it, and
    the user would have to SSH in to recover.
    """
    payload, err = _pi_call("POST", "/api/pair/release", {})
    settings = load_settings()
    settings["sync_token"] = ""
    settings["pi_device_id"] = ""
    save_settings(settings)
    log_activity("adam_unpaired", "", "ok" if not err else "warn")
    if err:
        return jsonify({"status": "ok",
                        "warning": "Forgotten locally, but ADAM did not confirm: %s" % err})
    return jsonify({"status": "ok"})


'''

s = s[:insert_at] + BLOCK + s[insert_at:]
assert s != orig
p.write_text(s, encoding="utf-8")
print("backend.py patched")
