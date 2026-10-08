"""Local/LAN boundary and validation tests with no real hardware or network calls."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_config import isolated_config, APP


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.scope = isolated_config()
        self.config, self.root = self.scope.__enter__()
        self.addCleanup(lambda: self.scope.__exit__(None, None, None))
        self.network = patch("requests.sessions.Session.request", side_effect=AssertionError("Unexpected external HTTP in fixture"))
        self.network.start()
        self.addCleanup(self.network.stop)
        name = "_adam_backend_fixture_" + uuid.uuid4().hex
        spec = importlib.util.spec_from_file_location(name, APP / "backend.py")
        self.backend = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.backend)
        self.backend.app.config.update(TESTING=False, PROPAGATE_EXCEPTIONS=False)
        self.client = self.backend.app.test_client()
        self.headers = {"X-ADAM-Session": self.backend.DESKTOP_SESSION}
        self.base = "http://127.0.0.1:8642"
        self.backend._control_call_times.clear()

    def local(self, path, data=None, method=None, headers=None):
        kwargs = {"base_url": self.base, "headers": headers if headers is not None else self.headers,
                  "environ_overrides": {"REMOTE_ADDR": "127.0.0.1"}}
        if data is not None:
            kwargs["json"] = data
        return self.client.open(path, method=method or ("POST" if data is not None else "GET"), **kwargs)

    def remote(self, path, data=None, token=None, method=None):
        kwargs = {"base_url": "http://192.168.1.4:8642", "environ_overrides": {"REMOTE_ADDR": "192.168.1.8"},
                  "headers": {"X-ADAM-Token": token} if token is not None else {}}
        if data is not None:
            kwargs["json"] = data
        return self.client.open(path, method=method or ("POST" if data is not None else "GET"), **kwargs)

    def test_public_discovery_does_not_mark_robot_connected(self):
        before = deepcopy(self.backend.CURRENT_ROBOT_STATE)
        for path in ("/ping", "/actions"):
            response = self.remote(path)
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("agent_token", response.get_data(as_text=True))
        self.assertEqual(before, self.backend.CURRENT_ROBOT_STATE)
        self.assertFalse(self.backend.connection.status()["connected"])

    def test_empty_or_wrong_lan_token_cannot_execute(self):
        settings = self.config.load_settings()
        settings["agent_token"] = ""
        with patch.object(self.backend, "load_settings", return_value=settings):
            self.assertEqual(self.remote("/control", {"action": "volume_up"}).status_code, 401)
        self.assertEqual(self.remote("/control", {"action": "volume_up"}, "wrong").status_code, 401)
        self.assertEqual(self.backend.ACTIVITY_LOG, self.backend.ACTIVITY_LOG.__class__(maxlen=100))

    def test_valid_lan_token_runs_enabled_action_without_claiming_robot_connection(self):
        self.backend._hw_worker = Mock()
        self.backend._hw_worker.call_sync.return_value = 35
        token = self.config.load_settings()["agent_token"]
        response = self.remote("/control", {"action": "volume_set", "value": 35}, token)
        self.assertEqual(response.get_json()["status"], "ok")
        self.backend._hw_worker.call_sync.assert_called_once_with("set_vol", (35,))
        self.assertFalse(self.backend.connection.status()["connected"])
        self.assertFalse(self.backend.CURRENT_ROBOT_STATE["connected"])

    def test_private_routes_require_local_session_even_with_agent_token(self):
        token = self.config.load_settings()["agent_token"]
        for path in ("/settings", "/status", "/memories", "/account/status", "/logs"):
            with self.subTest(path=path):
                self.assertEqual(self.remote(path, token=token).status_code, 401)
                self.assertEqual(self.local(path, headers={}).status_code, 401)
                self.assertEqual(self.local(path).status_code, 200)
        self.assertEqual(self.remote("/connection/credentials", {}, token).status_code, 401)

    def test_origin_and_rebinding_are_rejected(self):
        for origin in ("https://attacker.example", "null", "http://localhost:9999"):
            with self.subTest(origin=origin):
                response = self.local("/settings", headers={**self.headers, "Origin": origin})
                self.assertEqual(response.status_code, 403)
        response = self.client.get("/settings", base_url="http://attacker.example:8642",
            headers=self.headers, environ_overrides={"REMOTE_ADDR": "127.0.0.1"})
        self.assertIn(response.status_code, (401, 403))
        root = self.client.get("/", base_url="http://attacker.example:8642",
            environ_overrides={"REMOTE_ADDR": "127.0.0.1"})
        self.assertEqual(root.status_code, 403)

    def test_secrets_only_revealed_by_explicit_local_credentials_endpoint(self):
        self.config.update_settings({"sync_token": "fixture-private-sync-key"})
        token = self.config.load_settings()["agent_token"]
        for path in ("/settings", "/status", "/account/status", "/logs"):
            text = self.local(path).get_data(as_text=True)
            self.assertNotIn(token, text)
            self.assertNotIn("fixture-private-sync-key", text)
        with patch.object(self.backend, "_get_local_ip", return_value="192.168.1.4"):
            result = self.local("/connection/credentials", {}).get_json()
        self.assertEqual(result["token"], token)
        self.assertEqual(result["port"], 8642)

    def test_payload_shape_content_type_and_unicode_tokens_fail_cleanly(self):
        for body in ([], None, "text", 42):
            response = self.client.post("/settings", base_url=self.base, headers=self.headers,
                data=json.dumps(body), content_type="application/json")
            self.assertEqual(response.status_code, 400)
        response = self.client.post("/settings", base_url=self.base, headers=self.headers,
            data="paused=true", content_type="application/x-www-form-urlencoded")
        self.assertEqual(response.status_code, 415)
        for header in ("X-ADAM-Session", "X-ADAM-Token"):
            response = self.local("/control", {"action": "volume_up"}, headers={header: "é"})
            self.assertEqual(response.status_code, 401)

    def test_malformed_actions_and_touch_controls_return_validation_errors(self):
        cases = [("/control", {"action": []}),
                 ("/control", {"action": {"bad": "value"}}),
                 ("/touch/save", {"assignments": {"touch1": {"tap": {"action": []}}}}),
                 ("/touch/save", {"assignments": {"unknown": {}}}),
                 ("/touch/test", {"sensor": [], "event": "tap"}),
                 ("/touch/test", {"sensor": "touch1", "event": []})]
        for path, body in cases:
            with self.subTest(path=path, body=body):
                response = self.local(path, body)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json()["status"], "error")

    def test_memory_validation_preserves_existing_records(self):
        response = self.local("/memories", {"title": "Fixture", "text": "Keep this", "kind": "fact"})
        self.assertEqual(response.status_code, 200)
        original = response.get_json()["memories"]
        for body in ({"title": [], "text": "No", "kind": "fact"},
                     {"title": "No", "text": "No", "kind": []},
                     {"title": "No", "text": "x" * 2001, "kind": "fact"}):
            response = self.local("/memories", body)
            self.assertNotEqual(response.status_code, 200)
            self.assertEqual(response.get_json()["status"], "error")
        self.assertEqual(self.local("/memories").get_json()["memories"], original)

    def test_hardware_failure_and_disabled_action_never_report_success(self):
        self.backend._hw_worker = Mock()
        self.backend._hw_worker.call_sync.side_effect = RuntimeError("No supported audio output")
        response = self.local("/control", {"action": "volume_set", "value": 40})
        self.assertGreaterEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["status"], "error")
        self.assertEqual(self.backend.ACTIVITY_LOG[0]["status"], "error")
        self.backend.ENABLED_ACTIONS["volume_set"] = False
        response = self.local("/control", {"action": "volume_set", "value": 40})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.backend._hw_worker.call_sync.call_count, 1)

    def test_action_type_coercion_and_sensitive_log_redaction(self):
        action = self.backend.ACTIONS["volume_set"]
        with patch.dict(action, {"fn": Mock(return_value={"volume": 40})}):
            result = self.local("/control", {"action": "volume_set", "value": "40%"})
            self.assertEqual(result.get_json()["status"], "ok")
            action["fn"].assert_called_once_with(40)
        self.backend.log_activity("read_clipboard", None, "ok", "fixture-private-content")
        self.backend.log_activity("dispatch_coding_task", "fixture-private-prompt", "ok", "fixture-private-response")
        text = self.local("/logs").get_data(as_text=True)
        self.assertNotIn("fixture-private-content", text)
        self.assertNotIn("fixture-private-prompt", text)
        self.assertNotIn("fixture-private-response", text)

    def test_clipboard_backend_error_never_looks_like_success(self):
        self.backend.ENABLED_ACTIONS["read_clipboard"] = True
        with patch.dict(sys.modules, {"pyperclip": SimpleNamespace(paste=Mock(side_effect=RuntimeError("Clipboard is busy")))}):
            response = self.local("/control", {"action": "read_clipboard"})
        self.assertEqual(response.get_json()["status"], "error")
        self.assertEqual(self.backend.ACTIVITY_LOG[0]["status"], "error")

    def test_touch_test_preserves_action_failure(self):
        self.local("/touch/save", {"assignments": {"touch1": {"hold": {"action": "media_play_pause"}}}})
        with patch.dict(self.backend.ACTIONS["media_play_pause"], {"fn": Mock(return_value={"error": "Media unavailable"})}):
            response = self.local("/touch/test", {"sensor": "touch1", "event": "hold"})
        self.assertEqual(response.get_json()["status"], "error")
        self.assertFalse(any(item["status"] == "ok" for item in self.backend.ACTIVITY_LOG))

    def test_touch_gesture_policy_protects_fixed_reactions(self):
        for sensor in ("touch1", "touch2", "touch3", "touch4"):
            self.assertEqual(self.local("/touch/save", {"assignments": {sensor: {"tap": {"action": "none"}}}}).status_code, 400)
            self.assertEqual(self.local("/touch/test", {"sensor": sensor, "event": "tap"}).status_code, 400)
        for sensor in ("touch1", "touch2", "touch4"):
            for gesture in ("double", "triple"):
                self.assertEqual(self.local("/touch/save", {"assignments": {sensor: {gesture: {"action": "volume_up"}}}}).status_code, 400)
        response = self.local("/touch/save", {"assignments": {"touch3": {"triple": {"action": "volume_set", "value": 37}}}})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["assignments"]["touch3"]["triple"], {"action": "volume_set", "value": 37})

    def test_instance_identity_is_local_and_activation_rejects_another_instance(self):
        self.assertEqual(self.local("/ping").json["instance_id"], self.backend.DESKTOP_INSTANCE)
        self.assertNotIn("instance_id", self.remote("/ping").json)
        self.assertEqual(self.local("/show_window", {"instance_id": "other"}).status_code, 409)

    def test_coding_action_response_is_private_in_activity_history(self):
        self.backend.ENABLED_ACTIONS["dispatch_coding_task"] = True
        self.backend.coding_manager = Mock()
        self.backend.coding_manager.dispatch.return_value = {"status": "dispatched", "task": {
            "id": "fixture-task", "prompt": "private-coding-prompt", "result": "private-coding-result"}}
        response = self.local("/control", {"action": "dispatch_coding_task", "value": "private-coding-prompt"})
        self.assertEqual(response.status_code, 200)
        text = self.local("/logs").get_data(as_text=True)
        self.assertNotIn("private-coding-prompt", text)
        self.assertNotIn("private-coding-result", text)

    def test_settings_and_account_types_checked_before_side_effects(self):
        for body in ({"paused": "false"}, {"startup_on_login": 1}, {"enabled_actions": []},
                     {"agent_token": "new"}, {"user_name": []}):
            self.assertEqual(self.local("/settings", body).status_code, 400)
        with patch.object(self.backend.account, "email_login") as login:
            response = self.local("/account/email", {"email": "a@example.test", "password": "secret", "create": "false"})
            self.assertEqual(response.status_code, 400)
            login.assert_not_called()

    def test_coding_manager_not_ready_and_bad_task_ids_fail_cleanly(self):
        for path in ("/coding/input", "/coding/cancel"):
            response = self.local(path, {})
            self.assertEqual(response.status_code, 503)
        from coding_agent import CodingAgentManager
        self.backend.coding_manager = CodingAgentManager(finder=lambda _: None)
        for path in ("/coding/input", "/coding/cancel"):
            response = self.local(path, {"task_id": []})
            self.assertEqual(response.status_code, 400)

    def test_metrics_cache_uses_shared_cpu_deltas_and_megabits(self):
        from collections import namedtuple
        counters = namedtuple("CPU", "user system idle")
        clock = [10.0]
        probe = Mock()
        probe.cpu_times.side_effect = [counters(10, 10, 80), counters(20, 20, 100)]
        probe.net_io_counters.side_effect = [SimpleNamespace(bytes_sent=0, bytes_recv=0),
                                           SimpleNamespace(bytes_sent=2000000, bytes_recv=4000000)]
        probe.virtual_memory.return_value.percent = 45
        probe.disk_usage.return_value.percent = 60
        probe.net_if_stats.return_value = {"Ethernet": SimpleNamespace(isup=True)}
        sampler = self.backend._SystemStatsSampler(probe, clock=lambda: clock[0])
        self.assertIsNone(sampler.snapshot()["cpu_pct"])
        sampler.snapshot()["ram_pct"] = -1
        self.assertEqual(sampler.snapshot()["ram_pct"], 45)
        self.assertEqual(probe.cpu_times.call_count, 1)
        clock[0] = 12.0
        sample = sampler.snapshot()
        self.assertEqual(sample["cpu_pct"], 50)
        self.assertEqual(sample["net_up_mbps"], 8)
        self.assertEqual(sample["net_down_mbps"], 16)

    def test_metrics_returns_cached_result_instead_of_waiting_for_other_sampler(self):
        probe = Mock(side_effect=AssertionError("Must not sample"))
        sampler = self.backend._SystemStatsSampler(probe)
        sampler._sample_lock.acquire()
        try:
            self.assertIsNone(sampler.snapshot()["cpu_pct"])
            probe.cpu_times.assert_not_called()
        finally:
            sampler._sample_lock.release()


if __name__ == "__main__":
    unittest.main()
