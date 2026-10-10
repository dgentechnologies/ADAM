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
import asyncio
import desktop_pairing

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
    headers = getattr(websocket, 'request_headers', {})
    token = headers.get('X-ADAM-Token', '')
    if desktop_pairing.identity() and not desktop_pairing.authenticate(token):
        await websocket.close(code=1008, reason='Authorization required')
        return
    grant = desktop_pairing.authenticate(token)
    if grant:
        await websocket.send(json.dumps({'type': 'authorized', **grant}))
    ws_clients.add(websocket)
    try:
        while not websocket.closed:
            if desktop_pairing.identity() and not desktop_pairing.authenticate(token):
                await websocket.close(code=1008, reason='Authorization expired')
                break
            try:
                await asyncio.wait_for(websocket.wait_closed(), timeout=5)
            except asyncio.TimeoutError:
                pass
    finally:
        ws_clients.discard(websocket)

async def start_ws_server() -> None:
    try:
        import websockets.server
        srv = await websockets.server.serve(ws_handler, '0.0.0.0' if desktop_pairing.identity() else WS_HOST, WS_PORT, ssl=desktop_pairing.ssl_context())
        scheme = "wss" if desktop_pairing.identity() else "ws"
        print(f"✅ WebSocket face server → {scheme} on port {WS_PORT}")
        return srv
    except Exception as e:
        print(f"⚠️  WebSocket server unavailable: {e}")
        return None
