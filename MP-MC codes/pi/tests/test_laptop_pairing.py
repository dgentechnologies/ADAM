"""Consumer pairing tests use only temporary files and loopback fake laptops."""
import asyncio
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

PI_DIR = Path(__file__).resolve().parents[1] / "adam"
sys.path.insert(0, str(PI_DIR))
import laptop_pairing as pairing


class FakeLaptop:
    def __init__(self):
        self.requests = []
        self.valid = True
        self.redirect = None
        self.accept_key = True
        parent = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                parent.requests.append(("GET", self.path, self.headers.get("X-ADAM-Token")))
                if parent.redirect:
                    self.send_response(302)
                    self.send_header("Location", parent.redirect)
                    self.end_headers()
                    return
                payload = {"status": "ok", "app": "ADAM Companion", "kind": "laptop", "api": 1}
                if not parent.valid: payload["kind"] = "different-app"
                self.respond(payload, 200 if parent.accept_key else 401)
            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                parent.requests.append(("POST", self.path, data))
                self.respond({"status": "ok", "action": data.get("action")})
            def respond(self, payload, status=200):
                encoded = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
    def body(self):
        return {"host": "127.0.0.1", "port": self.server.server_port, "token": "isolated-laptop-key-123456"}
    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


def isolated_client():
    config = types.ModuleType("config")
    values = {"LAPTOP_AGENT_PORT": 8642, "LAPTOP_AGENT_TOKEN": "legacy-test-key-12345",
              "LAPTOP_AGENT_TIMEOUT_S": .5, "LAPTOP_AGENT_STATIC_IP": "127.0.0.9",
              "LAPTOP_MDNS_SERVICE": "_adam-laptop._tcp.local.",
              "LAPTOP_DISCOVERY_TIMEOUT_S": .1, "LAPTOP_DISCOVERY_TTL_S": 1,
              "LAPTOP_ACTIONS_TTL_S": 1}
    for key, value in values.items(): setattr(config, key, value)
    spec = importlib.util.spec_from_file_location("isolated_laptop_client", PI_DIR / "laptop_agent_client.py")
    client = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"config": config}): spec.loader.exec_module(client)
    return client


class PairingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.laptop = FakeLaptop()
        self.client = isolated_client()
        self.service = pairing.LaptopPairing(Path(self.directory.name) / "pairing.json",
                                            self.client.configure_laptop_pairing)
    async def asyncTearDown(self):
        await asyncio.to_thread(self.laptop.close)
        self.directory.cleanup()

    async def test_verified_pair_is_persisted_and_commands_use_its_snapshot(self):
        result = await self.service.pair(self.laptop.body())
        self.assertTrue(result["paired"])
        self.assertEqual(result["port"], self.laptop.server.server_port)
        self.assertNotIn("token", result)
        self.assertNotIn("token", self.service.status())
        self.client._laptop_agent_ip_cache.update(ip="192.0.2.1", ts=0)
        self.assertEqual(self.client._discover_laptop_agent_ip(), "127.0.0.1")
        reply = await asyncio.to_thread(self.client.laptop_control_sync, "clipboard_get")
        self.assertEqual(reply["status"], "ok")
        command = [row for row in self.laptop.requests if row[0] == "POST"][0][2]
        self.assertEqual(command["token"], self.laptop.body()["token"])
        self.assertEqual(command["action"], "read_clipboard")
        replacement = isolated_client()
        restored = pairing.LaptopPairing(self.service.path, replacement.configure_laptop_pairing)
        await restored.load()
        self.assertEqual(replacement.get_laptop_endpoint(), self.client.get_laptop_endpoint())

    async def test_bad_key_or_wrong_application_never_saves_a_pairing(self):
        self.laptop.accept_key = False
        with self.assertRaisesRegex(ValueError, "connection key"):
            await self.service.pair(self.laptop.body())
        self.assertFalse(self.service.path.exists())
        self.laptop.accept_key = True
        self.laptop.valid = False
        with self.assertRaisesRegex(ValueError, "identify"):
            await self.service.pair(self.laptop.body())
        self.assertFalse(self.service.path.exists())

    async def test_redirect_does_not_forward_the_private_key(self):
        other = FakeLaptop()
        try:
            self.laptop.redirect = f"http://127.0.0.1:{other.server.server_port}/pair/verify"
            with self.assertRaises(ValueError):
                await self.service.pair(self.laptop.body())
            self.assertEqual(other.requests, [])
        finally:
            await asyncio.to_thread(other.close)

    async def test_unpair_revokes_environment_fallback_now_and_after_restart(self):
        await self.service.pair(self.laptop.body())
        self.assertEqual(await self.service.unpair(), {"ok": True, "paired": False})
        self.assertIsNone(self.client.get_laptop_endpoint())
        before = len(self.laptop.requests)
        response = await asyncio.to_thread(self.client.laptop_control_sync, "volume_up")
        self.assertEqual(response["status"], "error")
        self.assertEqual(len(self.laptop.requests), before)
        replacement = isolated_client()
        restored = pairing.LaptopPairing(self.service.path, replacement.configure_laptop_pairing)
        await restored.load()
        self.assertIsNone(replacement.get_laptop_endpoint())
        self.assertNotIn("token", json.loads(self.service.path.read_text()))

    async def test_invalid_targets_and_values_are_rejected_before_network_access(self):
        for values in ({"host": "https://example.com"}, {"host": "8.8.8.8"}, {"host": "0.0.0.0"},
                       {"port": True}, {"port": 65536}, {"token": "short"}, {"token": "a" * 20 + "\r\n"}):
            with self.assertRaises(ValueError):
                await self.service.pair({**self.laptop.body(), **values})
        self.assertEqual(self.laptop.requests, [])

    async def test_failed_disk_write_preserves_the_previous_pair(self):
        await self.service.pair(self.laptop.body())
        previous = self.service.path.read_bytes()
        endpoint = self.client.get_laptop_endpoint()
        with patch.object(pairing.os, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                await self.service.unpair()
        self.assertEqual(self.service.path.read_bytes(), previous)
        self.assertEqual(self.client.get_laptop_endpoint(), endpoint)

    async def test_pi_http_pair_and_revoke_require_its_sync_key(self):
        config = types.ModuleType("config")
        config.SYNC_HOST, config.SYNC_PORT, config.SYNC_TOKEN = "127.0.0.1", 0, "isolated-pi-key"
        memories = types.ModuleType("memory_store")
        memories.memory, memories.conv_log = {}, []
        scheduler = types.ModuleType("scheduler")
        scheduler.snapshot = lambda: {"schedules": [], "todos": []}
        spec = importlib.util.spec_from_file_location("pairing_sync_api", PI_DIR / "sync_api.py")
        api = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"config": config, "memory_store": memories, "scheduler": scheduler}):
            spec.loader.exec_module(api)
        with patch.object(pairing, "_service", self.service):
            server = await asyncio.start_server(api._handle, "127.0.0.1", 0)
            port = server.sockets[0].getsockname()[1]
            async def call(path, body, token=""):
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                content = json.dumps(body).encode()
                writer.write((f"POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: {len(content)}\r\n"
                              f"X-ADAM-Token: {token}\r\n\r\n").encode() + content)
                await writer.drain()
                response = await asyncio.wait_for(reader.read(), 3)
                writer.close()
                await writer.wait_closed()
                header, content = response.split(b"\r\n\r\n", 1)
                return int(header.split()[1]), json.loads(content)
            try:
                self.assertTrue((await api._r_ping())["capabilities"]["laptop_pairing"])
                status, _ = await call("/api/laptops/pair", self.laptop.body())
                self.assertEqual(status, 403)
                self.assertEqual(self.laptop.requests, [])
                status, ack = await call("/api/laptops/pair", self.laptop.body(), "isolated-pi-key")
                self.assertEqual(status, 200)
                self.assertTrue(ack["paired"])
                self.assertNotIn("token", ack)
                status, _ = await call("/api/laptops/unpair", {})
                self.assertEqual(status, 403)
                self.assertIsNotNone(self.client.get_laptop_endpoint())
                status, ack = await call("/api/laptops/unpair", {}, "isolated-pi-key")
                self.assertEqual(status, 200)
                self.assertFalse(ack["paired"])
                self.assertIsNone(self.client.get_laptop_endpoint())
            finally:
                server.close()
                await server.wait_closed()


if __name__ == "__main__": unittest.main()
