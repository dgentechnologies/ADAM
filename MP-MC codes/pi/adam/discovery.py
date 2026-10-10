"""
discovery.py — make ADAM findable on the LAN, so nobody has to type an IP
==============================================================================
Publishes `_adam._tcp.local.` over mDNS. The companion app browses for that
service, shows whatever it finds, and the user picks one. No IP address, no
port, no token typed by hand.

WHY THIS DID NOT EXIST BEFORE
-----------------------------
Discovery was one-directional. The desktop app advertised ITSELF as
`_adam-laptop._tcp.local.` so the Pi could find the laptop to control it, and
`laptop_agent_client.py` browses for exactly that. But nothing ever advertised
the Pi, and nothing browsed for it — so the app could only reach ADAM through
an address a human typed into Settings. That address then went stale every
time DHCP moved the Pi (it has gone .9 -> .11 during this project), and the
Clock tab just said "not connected".

This module is the missing half.

WHAT IS PUBLISHED
-----------------
Service : _adam._tcp.local.
Port    : SYNC_PORT (the sync API, 8766)
TXT     : id      short device id, e.g. "ADAM-47F8"   (stable, from the MAC)
          name    friendly name shown in the app
          api     sync API version
          ver     firmware/software version string
          paired  "1" once this unit has been claimed, else "0"

`paired` is in the TXT record so the app can grey out a unit that already
belongs to somebody, instead of letting the user pick it and then fail.

NOTHING SECRET IS ADVERTISED. A TXT record is readable by every device on the
network, so the sync token is never put here — the app has to ask for it over
HTTP, and the Pi only answers that while it is unclaimed. See sync_api's
pairing endpoints.
"""

import socket
import desktop_pairing

from config import SYNC_PORT, APP_VERSION

SERVICE_TYPE = "_adam._tcp.local."

_zc = None
_info = None
_short_id = ""


def short_id() -> str:
    """Stable per-unit id derived from the MAC: 'ADAM-47F8'.

    Stable matters: it is what the user recognises in the app's list, and it
    must survive reboots and DHCP changes. The MAC is the only identifier on
    this device that does.
    """
    global _short_id
    if _short_id:
        return _short_id
    try:
        mac = open("/sys/class/net/wlan0/address").read().strip()
        tail = mac.replace(":", "")[-4:].upper()
    except Exception:
        try:
            import uuid
            tail = f"{uuid.getnode() & 0xFFFF:04X}"
        except Exception:
            tail = "0000"
    _short_id = f"ADAM-{tail}"
    return _short_id


def _local_ip() -> str:
    """The address other devices on this LAN can actually reach.

    A UDP connect to a public address picks the right interface without
    sending a packet; socket.gethostbyname(hostname) returns 127.0.1.1 on
    Debian and would advertise a loopback address to the whole network.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def _txt(paired: bool) -> dict:
    return {
        b"id": desktop_pairing.public_info().get("id", short_id()).encode(),
        b"secure": b"2" if desktop_pairing.identity() else b"0",
        b"name": b"ADAM",
        b"api": b"1",
        b"ver": str(APP_VERSION).encode(),
        b"paired": b"1" if paired else b"0",
    }


async def start_discovery(paired: bool = False) -> bool:
    """Begin advertising. Never raises — discovery is a convenience, not a
    dependency: if it fails ADAM still works, the user just has to type an
    address like before."""
    global _zc, _info
    if _zc is not None:
        return True
    try:
        from zeroconf import ServiceInfo
        from zeroconf.asyncio import AsyncZeroconf
    except ImportError:
        print("  ℹ️  zeroconf not installed — the app will not be able to "
              "find ADAM automatically (pip install zeroconf)")
        return False

    try:
        ip = _local_ip()
        # NOTE: no `server=` kwarg on purpose.
        #
        # Passing server="adam-47f8.local." here claims a NEW hostname, and
        # avahi-daemon already owns this Pi's hostname (adam-pi.local).
        # python-zeroconf sees the conflicting claim and raises
        # NonUniqueNameException — whose str() is EMPTY, so it surfaced as the
        # uninformative "mDNS advertise failed: " with nothing after the colon.
        # Letting zeroconf derive the server from the real hostname registers
        # cleanly and is what we want anyway: the app resolves the service, not
        # a hostname we invented. Verified on the device.
        _info = ServiceInfo(
            SERVICE_TYPE,
            f"{short_id()}.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(ip)],
            port=int(SYNC_PORT),
            properties=_txt(paired),
        )
        # AsyncZeroconf, not Zeroconf.
        #
        # zeroconf's sync API runs its own internal event loop and
        # register_service() blocks on it. Called from inside OUR running
        # asyncio loop that deadlocks and the library raises EventLoopBlocked.
        # The async API shares this loop instead. Verified on the device: the
        # sync call failed every boot, this one registers.
        _zc = AsyncZeroconf()
        await _zc.async_register_service(_info)
        print(f"✅ Discoverable as '{short_id()}' → {ip}:{SYNC_PORT} "
              f"({'paired' if paired else 'open to pair'})")
        return True
    except Exception as e:
        # Print the exception TYPE, not just str(e). zeroconf's
        # NonUniqueNameException has an empty message, so "advertise failed: "
        # with nothing after it was all the log said the first time this broke.
        print(f"  ⚠️  mDNS advertise failed: {type(e).__name__}: {e or '(no message)'}")
        _zc = None
        return False


async def update_paired(paired: bool) -> None:
    """Re-publish with a new `paired` flag.

    Called the moment a claim succeeds or is released, so the app's list stops
    offering a unit that is already taken without needing a restart.
    """
    global _info
    if _zc is None or _info is None:
        return
    try:
        from zeroconf import ServiceInfo
        # Build a REPLACEMENT ServiceInfo rather than mutating this one.
        # ServiceInfo.properties is read-only in python-zeroconf (the class is
        # Cython-compiled): assigning to it raises "attribute 'properties' ...
        # is not writable", which silently left the TXT record advertising
        # paired=0 after a successful claim, so the app would keep offering a
        # unit that was already taken. Verified on the device.
        fresh = ServiceInfo(
            SERVICE_TYPE,
            f"{short_id()}.{SERVICE_TYPE}",
            addresses=list(_info.addresses),
            port=int(SYNC_PORT),
            properties=_txt(paired),
            # Carry over the server zeroconf RESOLVED for us at registration.
            # async_update_service asserts "ServiceInfo must have a server",
            # and inventing one here is what triggered NonUniqueNameException
            # at startup — avahi already owns this host's name. Reusing the
            # resolved value satisfies both constraints.
            server=_info.server,
        )
        await _zc.async_update_service(fresh)
        _info = fresh
    except Exception as e:
        print(f"  ⚠️  mDNS update failed: {type(e).__name__}: {e or '(no message)'}")


async def stop_discovery() -> None:
    global _zc, _info
    if _zc is None:
        return
    try:
        if _info is not None:
            await _zc.async_unregister_service(_info)
        await _zc.async_close()
    except Exception:
        pass
    finally:
        _zc, _info = None, None
