"""
ws_server.py — ADAM v40 WebSocket face-broadcast server
==============================================================================
Tiny WebSocket server (ws://WS_HOST:WS_PORT, default localhost:8765) that
pushes emotion/head-gesture events to any connected face UI. handle_tool_call
calls ws_broadcast() on set_emotion; run_session broadcasts speaking-state and
emotion changes too. Purely a UI mirror — nothing here is required for the
robot's own TFT face (that goes over UART), so it degrades to a no-op if the
websockets package or the port isn't available.

WS_HOST/WS_PORT come from config.py.
"""

import json

from config import WS_HOST, WS_PORT

ws_clients: set = set()

async def ws_broadcast(payload: dict) -> None:
    if not ws_clients:
        return
    msg  = json.dumps(payload)
    dead = set()
    for ws in list(ws_clients):
        try:
            await ws.send(msg)
        except Exception:
            dead.add(ws)
    ws_clients.difference_update(dead)

async def ws_handler(websocket) -> None:
    ws_clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        ws_clients.discard(websocket)

async def start_ws_server() -> None:
    try:
        import websockets.server
        srv = await websockets.server.serve(ws_handler, WS_HOST, WS_PORT)
        print(f"✅ WebSocket face server → ws://{WS_HOST}:{WS_PORT}")
        return srv
    except Exception as e:
        print(f"⚠️  WebSocket server unavailable: {e}")
        return None
