"""
ADAM Laptop Agent — MODULAR ACTION REGISTRY & WINDOWS DESKTOP APP
==================================================================
DGEN Technologies Pvt. Ltd.

Runs on YOUR LAPTOP (not the Pi). Exposes a local HTTP endpoint that
ADAM's Pi calls over the LAN to control volume, brightness, coding agent,
and mirrors robot emotions in 3D.

When run directly (`python laptop_agent.py`), it launches the desktop
application with system tray and dashboard.
"""

from backend import (
    ACTIONS, action, app,
    start_mdns_broadcast, stop_mdns_broadcast,
    get_system_volume, set_system_volume,
    get_system_brightness, set_system_brightness,
    CURRENT_ROBOT_STATE
)
from config import load_settings

if __name__ == "__main__":
    from app import main
    main()
