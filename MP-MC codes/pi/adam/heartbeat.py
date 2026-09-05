"""
heartbeat.py — ADAM Health & Liveliness Monitoring
==================================================
Shared module for publishing and inspecting ADAM's runtime health.
Writes atomic JSON snapshots to /dev/shm (RAM tmpfs) for zero SD card wear
and microsecond access time.
"""

import json
import os
import time

HEARTBEAT_PATH = "/dev/shm/adam_heartbeat.json" if os.path.exists("/dev/shm") else "/tmp/adam_heartbeat.json"


def record_heartbeat(**kwargs) -> None:
    """Atomic write of runtime health metadata."""
    try:
        data = {
            "timestamp": time.time(),
            "pid": os.getpid(),
            **kwargs,
        }
        tmp = f"{HEARTBEAT_PATH}.tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, HEARTBEAT_PATH)
    except Exception:
        pass


def read_heartbeat() -> dict | None:
    """Read the latest heartbeat metadata, or None if unavailable/corrupt."""
    try:
        if not os.path.exists(HEARTBEAT_PATH):
            return None
        with open(HEARTBEAT_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return None


def clear_heartbeat() -> None:
    """Clear heartbeat file on shutdown/restart."""
    try:
        if os.path.exists(HEARTBEAT_PATH):
            os.remove(HEARTBEAT_PATH)
    except Exception:
        pass
