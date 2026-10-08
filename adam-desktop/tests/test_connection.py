"""Connection contract checks against an isolated simulated Pi on loopback."""

import copy
import json
import sys
import threading
import time
import unittest
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from connection import ConnectionService


class SimulatedPi:
    def __init__(self):
        self.identity = {"ok": True, "app": "adam", "kind": "pi", "api": 1,
                         "readonly": False, "device_name": "Test ADAM"}
        self.records = []
        self.redirect = ""
        self.write_status = 200
        self.ping_gate = None
        self.entered_ping = threading.Event()
        self.lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def respond(self):
                size = int(self.headers.get("Content-Length", 0))
                data = self.rfile.read(size) if size else b""
                with owner.lock:
                    owner.records.append((self.command, self.path,
                                          self.headers.get("X-ADAM-Token", ""), data))
                status = 200
                if self.path == "/api/ping":
                    owner.entered_ping.set()
                    if owner.ping_gate:
                        owner.ping_gate.wait(2)
                    response = owner.identity
                elif self.command == "GET":
                    response = {"ok": True, "api": 1, "data": {"todos": []}}
                else:
                    status = owner.write_status
                    response = {"ok": status == 200, "api": 1, "data": {"saved": True}}
                if owner.redirect:
                    self.send_response(302)
                    self.send_header("Location", owner.redirect)
                    self.end_headers()
                    return
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                payload = json.dumps(response).encode()
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = respond
            do_POST = respond
            do_PUT = respond

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def config(self):
        return {"host": "127.0.0.1", "sync_port": self.server.server_port,
                "ws_port": self.server.server_port, "sync_token": "test-key"}

    def close(self):
        if self.ping_gate:
            self.ping_gate.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


@contextmanager
def simulated_pi():
    device = SimulatedPi()
    try:
        yield device
    finally:
        device.close()


def eventually(condition, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


class ConnectionTests(unittest.TestCase):
    def setUp(self):
        self.settings = {}
        self.saved_patches = []
        self.events = []
        self.service = ConnectionService(self.read_settings, self.save_settings, self.events.append)

    def tearDown(self):
        self.service.stop()

    def read_settings(self):
        return copy.deepcopy(self.settings)

    def save_settings(self, patch):
        self.saved_patches.append(copy.deepcopy(patch))
        self.settings.update(patch)

    def test_probe_identifies_pi_without_saving_or_sending_private_key(self):
        with simulated_pi() as pi:
            result = self.service.probe(pi.config)
            self.assertTrue(result["ok"])
            self.assertTrue(result["data_connected"])
            self.assertFalse(result["telemetry_connected"])
            self.assertEqual(result["device_name"], "Test ADAM")
            self.assertFalse(self.service.status()["connected"])
            self.assertEqual(self.saved_patches, [])
            self.assertTrue(all(not row[2] for row in pi.records))
            self.assertNotIn("test-key", json.dumps(result))

    def test_arbitrary_server_and_unsupported_version_never_pair(self):
        with simulated_pi() as pi:
            for bad_identity in ({"ok": True}, {"ok": True, "app": "adam", "kind": "pc", "api": 1},
                                 {"ok": True, "app": "adam", "kind": "pi", "api": True},
                                 {"ok": True, "app": "adam", "kind": "pi", "api": 999}):
                pi.identity = bad_identity
                result = self.service.connect(pi.config)
                self.assertFalse(result["ok"])
                self.assertFalse(self.service.status()["connected"])
            self.assertEqual(self.saved_patches, [])

    def test_invalid_addresses_and_ports_make_no_network_requests(self):
        with simulated_pi() as pi:
            for bad in ("http://127.0.0.1", "127.0.0.1:8642", "user@localhost", "../localhost",
                        "localhost/path", "localhost?token=x", "localhost#x", "a b", "a..b", "0.0.0.0"):
                self.assertFalse(self.service.probe({**pi.config, "host": bad})["ok"], bad)
            for bad in (0, 65536, -1, True, "eight", 2.5):
                self.assertFalse(self.service.probe({**pi.config, "sync_port": bad})["ok"])
            self.assertEqual(pi.records, [])

    def test_redirect_cannot_leak_key_or_pair_with_a_different_endpoint(self):
        with simulated_pi() as first, simulated_pi() as second:
            first.redirect = f"http://127.0.0.1:{second.server.server_port}/api/ping"
            result = self.service.connect(first.config)
            self.assertFalse(result["ok"])
            self.assertEqual(second.records, [])
            self.assertEqual(self.saved_patches, [])

    def test_connect_persists_only_verified_target_and_bound_key(self):
        with simulated_pi() as pi:
            result = self.service.connect(pi.config)
            self.assertTrue(result["ok"])
            self.assertTrue(self.settings["paired"])
            self.assertEqual(self.settings["pi_host"], "127.0.0.1")
            self.assertEqual(self.settings["sync_token_port"], pi.server.server_port)
            self.assertEqual(self.settings["sync_token_host"], "127.0.0.1")
            self.assertNotIn("test-key", json.dumps(self.service.status()))

    def test_changing_port_never_inherits_previous_private_key(self):
        with simulated_pi() as first, simulated_pi() as second:
            self.assertTrue(self.service.connect(first.config)["ok"])
            config = dict(second.config)
            del config["sync_token"]
            result = self.service.connect(config)
            self.assertTrue(result["ok"])
            self.assertTrue(result["read_only"])
            self.assertEqual(self.settings["sync_token"], "")
            payload, error = self.service.call("GET", "/api/snapshot")
            self.assertIsNone(error)
            self.assertTrue(payload["ok"])
            self.assertTrue(all(not row[2] for row in second.records))

    def test_call_preserves_payload_and_sends_exactly_one_write(self):
        with simulated_pi() as pi:
            self.assertTrue(self.service.connect(pi.config)["ok"])
            payload, error = self.service.call("POST", "/api/todos", {"text": "A real task"})
            self.assertIsNone(error)
            self.assertTrue(payload["data"]["saved"])
            writes = [row for row in pi.records if row[0] == "POST"]
            self.assertEqual(len(writes), 1)
            self.assertEqual(writes[0][2], "test-key")
            self.assertEqual(json.loads(writes[0][3]), {"text": "A real task"})
            self.assertTrue(all(not row[2] for row in pi.records if row[1] == "/api/ping"))

    def test_read_only_connection_reads_but_never_attempts_a_write(self):
        with simulated_pi() as pi:
            pi.identity["readonly"] = True
            self.assertTrue(self.service.connect(pi.config)["read_only"])
            payload, error = self.service.call("GET", "/api/snapshot")
            self.assertIsNone(error)
            self.assertTrue(payload["ok"])
            payload, error = self.service.call("POST", "/api/todos", {"text": "blocked"})
            self.assertIsNone(payload)
            self.assertIn("read-only", error)
            self.assertFalse(any(row[0] == "POST" for row in pi.records))

    def test_rejected_key_returns_authorization_error_without_retry(self):
        with simulated_pi() as pi:
            self.assertTrue(self.service.connect(pi.config)["ok"])
            pi.write_status = 403
            payload, error = self.service.call("POST", "/api/todos", {"text": "blocked"})
            self.assertIsNone(payload)
            self.assertIn("authorize", error)
            self.assertTrue(self.service.status()["read_only"])
            self.assertEqual(len([row for row in pi.records if row[0] == "POST"]), 1)
            payload, error = self.service.call("POST", "/api/todos", {"text": "still blocked"})
            self.assertIsNone(payload)
            self.assertIn("read-only", error)
            self.assertEqual(len([row for row in pi.records if row[0] == "POST"]), 1)

    def test_identity_is_checked_again_before_a_private_key_is_sent(self):
        with simulated_pi() as pi:
            self.assertTrue(self.service.connect(pi.config)["ok"])
            pi.identity = {"ok": True, "app": "different-server"}
            payload, error = self.service.call("POST", "/api/todos", {"text": "blocked"})
            self.assertIsNone(payload)
            self.assertIn("identify", error)
            self.assertFalse(self.service.status()["data_connected"])
            self.assertFalse(any(row[0] == "POST" for row in pi.records))

    def test_disconnect_cancels_inflight_pairing_and_clears_key(self):
        with simulated_pi() as pi:
            pi.ping_gate = threading.Event()
            result = []
            worker = threading.Thread(target=lambda: result.append(self.service.connect(pi.config)))
            worker.start()
            self.assertTrue(pi.entered_ping.wait(1))
            self.assertTrue(self.service.disconnect()["ok"])
            pi.ping_gate.set()
            worker.join(timeout=2)
            self.assertFalse(result[0]["ok"])
            self.assertFalse(self.settings["paired"])
            self.assertEqual(self.settings["sync_token"], "")
            self.assertFalse(self.service.status()["connected"])

    def test_start_returns_immediately_and_discovers_saved_connection_in_background(self):
        with simulated_pi() as pi:
            pi.ping_gate = threading.Event()
            self.settings = {"paired": True, "pi_host": "127.0.0.1",
                             "pi_sync_port": pi.server.server_port, "pi_ws_port": pi.server.server_port}
            before = time.monotonic()
            self.service.start()
            self.assertLess(time.monotonic() - before, 0.1)
            self.assertTrue(pi.entered_ping.wait(1))
            self.assertFalse(self.service.status()["connected"])
            self.service.stop()
            pi.ping_gate.set()
            time.sleep(0.1)
            self.assertFalse(self.service.status()["connected"])
            self.service.start()
            self.assertTrue(eventually(lambda: self.service.status()["data_connected"]))

    def test_settings_save_failure_does_not_claim_pairing(self):
        with simulated_pi() as pi:
            service = ConnectionService(self.read_settings, lambda patch: (_ for _ in ()).throw(OSError("full")))
            self.assertFalse(service.connect(pi.config)["ok"])
            self.assertFalse(service.status()["connected"])
            service.stop()

    def test_status_returns_defensive_snapshot(self):
        snapshot = self.service.status()
        snapshot["connected"] = True
        snapshot["capabilities"]["touch_events"] = True
        self.assertFalse(self.service.status()["connected"])
        self.assertEqual(self.service.status()["capabilities"], {})

    def test_websocket_touch_events_require_advertised_capability(self):
        from websockets.sync.server import serve
        for capability in (False, True):
            self.events.clear()
            sent = threading.Event()
            release = threading.Event()

            def handler(socket):
                socket.send(json.dumps({"type": "emotion", "emotion": "happy"}))
                socket.send(json.dumps({"type": "touch", "id": "fixture-1", "sensor": "touch3",
                    "event": "hold", "action": "volume_set", "status": "ok", "handled_by": "pi"}))
                sent.set()
                release.wait(2)

            with serve(handler, "127.0.0.1", 0, close_timeout=0.1) as server, simulated_pi() as pi:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                pi.identity["capabilities"] = {"touch_events": capability}
                config = {**pi.config, "ws_port": server.socket.getsockname()[1]}
                self.assertTrue(self.service.connect(config)["ok"])
                self.assertTrue(sent.wait(2))
                self.assertTrue(eventually(lambda: any(e["type"] == "emotion" for e in self.events)))
                if capability:
                    self.assertTrue(eventually(lambda: self.service.status()["last_touch"] is not None))
                    self.assertEqual(self.service.status()["last_touch"]["sensor"], "touch3")
                else:
                    time.sleep(0.1)
                    self.assertIsNone(self.service.status()["last_touch"])
                self.assertFalse(any(e["type"] == "touch" for e in self.events), "Already handled touch events must not redispatch")
                self.assertEqual(self.service.status()["emotion"], "happy")
                self.assertTrue(self.service.status()["telemetry_connected"])
                self.service.disconnect()
                release.set()
                server.shutdown()
                thread.join(timeout=2)

    def test_verified_live_state_survives_http_refresh_and_resets_on_disconnect(self):
        from websockets.sync.server import serve
        release = threading.Event()
        def handler(socket):
            socket.send(json.dumps({"type": "emotion", "emotion": "thinking", "head": "nod"}))
            socket.send(json.dumps({"type": "speaking", "speaking": True}))
            socket.send(json.dumps({"type": "speaking", "speaking": "false"}))
            release.wait(6)
        with serve(handler, "127.0.0.1", 0, close_timeout=0.1) as server, simulated_pi() as pi:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                self.service.connect({**pi.config, "ws_port": server.socket.getsockname()[1]})
                self.assertTrue(eventually(lambda: self.service.status()["speaking"]))
                payload, error = self.service.call("GET", "/api/snapshot")
                self.assertIsNone(error)
                self.assertEqual(self.service.status()["emotion"], "thinking")
                self.assertTrue(self.service.status()["speaking"])
                self.service.disconnect()
                self.assertFalse(self.service.status()["speaking"])
                self.assertEqual(self.service.status()["emotion"], "")
            finally:
                release.set()
                server.shutdown()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
