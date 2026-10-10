"""Desktop setup contracts against a loopback Pi, without Windows side effects.

The simulator checks both connection keys and invokes the real protected Flask
``/pair/verify`` route before acknowledging pairing. All outgoing HTTP is limited
to the fixture's loopback socket. No Firebase, mDNS or physical controls run.
"""

from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit
import uuid

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_config import APP, isolated_config


class PairingPi:
    def __init__(self):
        self.key = "isolated-pi-connection-key"
        self.identity = {"ok": True, "app": "adam", "kind": "pi", "api": 1,
                         "readonly": False, "capabilities": {
                             "laptop_pairing": True, "touch_assignments": True, "touch_gesture_policy": 2}}
        self.records = []
        self.paired = False
        self.assignments = {}
        self.verify_desktop = lambda token: False
        self.pair_reply = None
        self.revoke_reply = None
        self.touch_reply = None
        self.pair_status = 200
        self.revoke_status = 200
        self.touch_status = 200
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def respond(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                data = json.loads(body) if body else {}
                token = self.headers.get("X-ADAM-Token", "")
                owner.records.append((self.command, self.path, token, data))
                status = 200
                if self.path == "/api/ping" and self.command == "GET":
                    result = deepcopy(owner.identity)
                elif token != owner.key:
                    status, result = 401, {"ok": False, "error": "Wrong fixture connection key"}
                elif self.path == "/api/laptops/pair" and self.command == "POST":
                    status = owner.pair_status
                    verified = owner.verify_desktop(data.get("token", ""))
                    if status == 200 and verified:
                        owner.paired = True
                        result = {"ok": True, "api": 1, "paired": True,
                                  "host": data["host"], "port": data["port"]}
                        if owner.pair_reply is not None:
                            result = deepcopy(owner.pair_reply)
                    else:
                        status, result = 403, {"ok": False, "error": "Laptop verification failed"}
                elif self.path == "/api/laptops/unpair" and self.command == "POST":
                    status = owner.revoke_status
                    result = {"ok": status == 200, "api": 1, "paired": False}
                    if status == 200:
                        owner.paired = False
                    if owner.revoke_reply is not None:
                        result = deepcopy(owner.revoke_reply)
                elif self.path == "/api/touch/assignments" and self.command == "POST":
                    status = owner.touch_status
                    if status == 200:
                        owner.assignments = deepcopy(data["assignments"])
                    result = {"ok": status == 200, "api": 1,
                              "assignments": deepcopy(owner.assignments)}
                    if owner.touch_reply is not None:
                        result = deepcopy(owner.touch_reply)
                else:
                    status, result = 404, {"ok": False, "error": "Unknown fixture endpoint"}
                encoded = json.dumps(result).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            do_GET = respond
            do_POST = respond

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def config(self):
        return {"host": "127.0.0.1", "sync_port": self.server.server_port,
                "ws_port": self.server.server_port, "sync_token": self.key}

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class PairingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.scope = isolated_config()
        self.config, self.root = self.scope.__enter__()
        self.addCleanup(lambda: self.scope.__exit__(None, None, None))
        self.pi = PairingPi()
        self.addCleanup(self.pi.close)
        original_request = requests.sessions.Session.request

        def only_fixture(session, method, url, **kwargs):
            address = urlsplit(url)
            if (address.scheme, address.hostname, address.port) != (
                    "http", "127.0.0.1", self.pi.server.server_port):
                raise AssertionError("The fixture attempted external HTTP")
            return original_request(session, method, url, **kwargs)

        network = patch("requests.sessions.Session.request", new=only_fixture)
        network.start()
        self.addCleanup(network.stop)
        spec = importlib.util.spec_from_file_location("_adam_pairing_" + uuid.uuid4().hex, APP / "backend.py")
        self.backend = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.backend)
        # Existing endpoint tests run inside an already verified session.
        # Mandatory-gate denial and lifecycle tests live in test_onboarding.py.
        gate = patch.object(self.backend.onboarding, 'status', return_value={'ready': True})
        gate.start()
        self.addCleanup(gate.stop)
        self.backend.app.config.update(TESTING=False, PROPAGATE_EXCEPTIONS=False)
        self.client = self.backend.app.test_client()
        self.headers = {"X-ADAM-Session": self.backend.DESKTOP_SESSION}
        self.addCleanup(self.backend.connection.stop)
        # Keep the real connection lifecycle/identity/write path, without live
        # telemetry or polling competing with deterministic HTTP assertions.
        for worker in ("_monitor", "_telemetry"):
            patched = patch.object(self.backend.connection, worker,
                                   side_effect=lambda run: run.stop.wait(30))
            patched.start()
            self.addCleanup(patched.stop)
        self.pi.verify_desktop = self.verify_desktop

    def local(self, path, body=None, headers=None):
        return self.client.open(path, method="POST" if body is not None else "GET",
            json=body, base_url="http://127.0.0.1:8642",
            headers=self.headers if headers is None else headers,
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"})

    def remote(self, path, token="", body=None):
        # A separate test client permits the simulated Pi's server thread to
        # verify the desktop while an authorize request waits for its response.
        return self.backend.app.test_client().open(path,
            method="POST" if body is not None else "GET", json=body,
            base_url="http://192.168.1.4:8642", headers={"X-ADAM-Token": token},
            environ_overrides={"REMOTE_ADDR": "192.168.1.8"})

    def verify_desktop(self, token):
        result = self.remote("/pair/verify", token)
        return result.status_code == 200 and result.get_json() == {
            "status": "ok", "app": "ADAM Companion", "kind": "laptop", "api": 1}

    def connect(self, **overrides):
        result = self.local("/connection/connect", {**self.pi.config, **overrides})
        self.assertEqual(result.get_json()["status"], "ok", result.get_json())
        return result.get_json()

    def save_touch(self):
        result = self.local("/touch/save", {"assignments": {
            "touch1": {"hold": {"action": "volume_set", "value": "40%"}},
            "touch2": {"hold": {"action": "media_play_pause"}}}})
        self.assertEqual(result.status_code, 200, result.get_json())
        return result.get_json()["assignments"]

    def test_probe_does_not_pair_persist_or_disclose_key(self):
        before = self.config.load_settings()
        result = self.local("/connection/probe", self.pi.config).get_json()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(self.config.load_settings(), before)
        self.assertFalse(self.backend.connection.status()["connected"])
        self.assertEqual(self.pi.records, [("GET", "/api/ping", "", {})])
        self.assertNotIn(self.pi.key, json.dumps(result))

    def test_connect_then_authorize_verifies_desktop_without_running_actions(self):
        result = self.connect()
        self.assertTrue(result["connected"])
        self.assertFalse(self.pi.paired)
        activity = deepcopy(list(self.backend.ACTIVITY_LOG))
        with patch.object(self.backend, "_get_local_ip", return_value="127.0.0.1"):
            result = self.local("/connection/authorize", {}).get_json()
        self.assertEqual(result["status"], "ok")
        self.assertTrue(self.pi.paired)
        pairs = [item for item in self.pi.records if item[1] == "/api/laptops/pair"]
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][2], self.pi.key)
        self.assertEqual(pairs[0][3]["token"], self.config.load_settings()["agent_token"])
        self.assertEqual(pairs[0][3]["host"], result["host"])
        self.assertEqual(pairs[0][3]["port"], result["port"])
        self.assertEqual(len(self.backend.ACTIVITY_LOG), len(activity) + 1)
        self.assertEqual(self.backend.ACTIVITY_LOG[0]["action"], "laptop_paired")
        self.assertNotIn(self.config.load_settings()["agent_token"], json.dumps(result))

    def test_pair_verification_is_protected_read_only_and_separate_from_connection(self):
        before = self.config.load_settings()
        for token in ("", "incorrect"):
            self.assertEqual(self.remote("/pair/verify", token).status_code, 401)
        self.assertTrue(self.verify_desktop(before["agent_token"]))
        self.assertEqual(self.config.load_settings(), before)
        self.assertFalse(self.backend.connection.status()["connected"])
        self.assertFalse(self.backend.ACTIVITY_LOG)
        self.assertEqual(self.remote("/pair/verify", before["agent_token"], {}).status_code, 405)

    def test_lan_agent_key_does_not_authorize_desktop_setup_routes(self):
        token = self.config.load_settings()["agent_token"]
        for path in ("/connection/probe", "/connection/connect", "/connection/authorize",
                     "/connection/revoke", "/touch/save", "/touch/apply"):
            with self.subTest(path=path):
                self.assertEqual(self.remote(path, token, {}).status_code, 401)
                self.assertEqual(self.local(path, {}, headers={}).status_code, 401)
        self.assertFalse(self.pi.records)

    def test_read_only_connection_blocks_pairing_and_touch_without_writes(self):
        self.connect(sync_token="")
        self.save_touch()
        for path in ("/connection/authorize", "/connection/revoke", "/touch/apply"):
            with self.subTest(path=path):
                result = self.local(path, {})
                self.assertEqual(result.get_json()["status"], "error")
                self.assertGreaterEqual(result.status_code, 400)
        self.assertTrue(all(record[0] == "GET" for record in self.pi.records))
        self.assertFalse(self.config.load_settings()["touch_applied_hash"])

    def test_wrong_key_does_not_claim_pairing_and_blocks_subsequent_writes(self):
        self.connect(sync_token="incorrect-fixture-key")
        self.assertEqual(self.local("/connection/authorize", {}).get_json()["status"], "error")
        self.assertFalse(self.pi.paired)
        self.assertTrue(self.backend.connection.status()["read_only"])
        writes = len([r for r in self.pi.records if r[0] == "POST"])
        self.local("/connection/authorize", {})
        self.assertEqual(len([r for r in self.pi.records if r[0] == "POST"]), writes)
        self.assertFalse(self.backend.ACTIVITY_LOG)

    def test_authorize_requires_exact_paired_acknowledgement(self):
        self.connect()
        replies = [
            {"ok": True},
            {"ok": True, "api": 1, "paired": False, "host": "127.0.0.1", "port": 8642},
            {"ok": True, "api": 1, "paired": True, "host": "192.168.1.200", "port": 8642},
            {"ok": True, "api": 1, "paired": True, "host": "127.0.0.1", "port": 9999},
            {"ok": "true", "api": 1, "paired": True, "host": "127.0.0.1", "port": 8642},
        ]
        for reply in replies:
            with self.subTest(reply=reply):
                self.pi.pair_reply = reply
                before = deepcopy(list(self.backend.ACTIVITY_LOG))
                result = self.local("/connection/authorize", {})
                self.assertEqual(result.get_json()["status"], "error")
                self.assertGreaterEqual(result.status_code, 400)
                self.assertEqual(list(self.backend.ACTIVITY_LOG), before)

    def test_revoke_pauses_only_after_explicit_revocation_acknowledgement(self):
        self.connect()
        for reply in ({"ok": True}, {"ok": True, "api": 1, "paired": True},
                      {"ok": "true", "api": 1, "paired": False}):
            with self.subTest(reply=reply):
                self.config.update_settings({"paused": False})
                self.pi.revoke_reply = reply
                result = self.local("/connection/revoke", {})
                self.assertEqual(result.get_json()["status"], "error")
                self.assertGreaterEqual(result.status_code, 400)
                self.assertFalse(self.config.load_settings()["paused"])
        self.pi.revoke_reply = None
        result = self.local("/connection/revoke", {})
        self.assertEqual(result.get_json(), {"status": "ok", "paired": False})
        self.assertTrue(self.config.load_settings()["paused"])

    def test_touch_save_is_local_and_apply_requires_exact_returned_mapping(self):
        assignments = self.save_touch()
        self.assertEqual(assignments["touch1"]["hold"], {"action": "volume_set", "value": 40})
        self.assertEqual(len(assignments), 4)
        self.assertEqual(assignments["touch4"]["hold"], {"action": "none", "value": None})
        self.assertFalse(self.pi.records)
        self.connect()
        result = self.local("/touch/apply", {})
        self.assertEqual(result.get_json()["status"], "ok")
        self.assertEqual(self.pi.assignments, assignments)
        self.assertTrue(self.config.load_settings()["touch_applied_hash"])
        self.save_touch()
        self.assertFalse(self.config.load_settings()["touch_applied_hash"])
        self.pi.touch_reply = {"ok": True, "api": 1, "assignments": {}}
        result = self.local("/touch/apply", {})
        self.assertEqual(result.get_json()["status"], "error")
        self.assertFalse(self.config.load_settings()["touch_applied_hash"])

    def test_touch_validation_preserves_saved_mapping_and_rejects_sensitive_actions(self):
        saved = self.save_touch()
        for name, value in (("volume_set", "not a number"), ("read_clipboard", None),
                            ("dispatch_coding_task", "run a task"), ("unknown", None)):
            with self.subTest(action=name):
                result = self.local("/touch/save", {"assignments": {
                    "touch1": {"hold": {"action": name, "value": value}}}})
                self.assertEqual(result.status_code, 400)
                self.assertEqual(self.config.load_settings()["touch_assignments"], saved)
        self.assertFalse(self.pi.records)

    def test_touch_rejection_or_missing_capability_never_claims_applied(self):
        self.save_touch()
        self.pi.identity["capabilities"].pop("touch_assignments")
        self.connect()
        self.assertEqual(self.local("/touch/apply", {}).get_json()["status"], "error")
        self.assertTrue(all(r[0] == "GET" for r in self.pi.records))
        self.pi.identity["capabilities"]["touch_assignments"] = True
        self.connect()
        self.pi.touch_status = 503
        self.assertEqual(self.local("/touch/apply", {}).get_json()["status"], "error")
        self.assertFalse(self.config.load_settings()["touch_applied_hash"])

    def test_identity_change_prevents_private_key_or_writes_to_replacement_device(self):
        self.connect()
        self.pi.identity["kind"] = "unrelated-device"
        self.pi.records.clear()
        result = self.local("/connection/authorize", {})
        self.assertEqual(result.get_json()["status"], "error")
        self.assertEqual(self.pi.records, [("GET", "/api/ping", "", {})])
        self.assertFalse(self.backend.connection.status()["connected"])


if __name__ == "__main__":
    unittest.main()
