"""Settings tests use isolated data/startup paths and never open the user's .env."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

APP = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(APP))


@contextmanager
def isolated_config():
    with tempfile.TemporaryDirectory() as root:
        root_path = Path(root)
        fixture_logger = logging.Logger("adam-config-fixture-" + uuid.uuid4().hex)
        module_name = "_adam_config_fixture_" + uuid.uuid4().hex
        spec = importlib.util.spec_from_file_location(module_name, APP / "config.py")
        config = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, {"ADAM_DATA_DIR": str(root_path / "data"),
                                    "ADAM_STARTUP_DIR": str(root_path / "startup"),
                                    "ADAM_DISABLE_HARDWARE": "1"}):
            with patch("logging.getLogger", return_value=fixture_logger):
                spec.loader.exec_module(config)
            # Inject a cipher only in the isolated fixture; production keeps
            # Windows DPAPI mandatory. This makes backend tests portable.
            from test_identity_sync import FakeCipher
            import secure_store
            store = secure_store.SecretStore(root_path / 'data' / 'credentials', cipher=FakeCipher())
            with patch.dict(sys.modules, {"config": config}), patch.object(secure_store, '_store', return_value=store):
                try:
                    yield config, root_path
                finally:
                    for handler in list(fixture_logger.handlers):
                        handler.close()
                        fixture_logger.removeHandler(handler)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.scope = isolated_config()
        self.config, self.root = self.scope.__enter__()
        self.addCleanup(lambda: self.scope.__exit__(None, None, None))

    def test_generated_tokens_are_protected_and_not_in_public_settings(self):
        from secure_store import load_secret
        settings = self.config.load_settings()
        self.assertGreaterEqual(len(settings["agent_token"]), 32)
        self.assertEqual(load_secret("settings_agent_token"), settings["agent_token"])
        public_bytes = self.config.CONFIG_FILE.read_bytes()
        self.assertNotIn(settings["agent_token"].encode(), public_bytes)
        self.assertNotIn(b'"agent_token"', public_bytes)
        self.config.update_settings({"sync_token": "isolated-sync-key"})
        ciphertext = (self.config.USER_DATA_DIR / "credentials/settings_sync_token.dpapi").read_bytes()
        self.assertNotIn(b"isolated-sync-key", ciphertext)
        self.config._cached = None
        self.assertEqual(self.config.load_settings()["sync_token"], "isolated-sync-key")

    def test_concurrent_patch_updates_preserve_unrelated_keys(self):
        self.config.load_settings()
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda number: self.config.update_settings({"fixture_" + str(number): number}), range(40)))
        settings = self.config.load_settings()
        for number in range(40):
            self.assertEqual(settings["fixture_" + str(number)], number)
        self.config._cached = None
        reloaded = self.config.load_settings()
        self.assertEqual(settings, reloaded)
        self.assertFalse(self.config.CONFIG_FILE.with_suffix(".tmp").exists())

    def test_load_returns_copy_and_environment_file_remains_untouched(self):
        env_file = self.root / "fixture.env"
        original = b"AGENT_TOKEN=legacy-fixture-token\nPRIVATE_VALUE=fixture\n"
        env_file.write_bytes(original)
        self.config.LOCAL_ENV_FILE = env_file
        initial = self.config.load_settings()
        self.assertNotEqual(initial["agent_token"], "legacy-fixture-token")
        initial["enabled_actions"]["fixture"] = True
        self.assertNotIn("fixture", self.config.load_settings()["enabled_actions"])
        self.config.update_settings({"user_name": "Fixture", "paused": True})
        self.assertEqual(env_file.read_bytes(), original)

    def test_corrupt_settings_preserved_for_recovery(self):
        self.config.CONFIG_FILE.write_text("broken data", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "could not be read"):
            self.config.load_settings()
        self.assertEqual(self.config.CONFIG_FILE.read_text(), "broken data")

    def test_touch_migration_preserves_holds_and_removes_reserved_bindings(self):
        binding = {"action": "volume_up", "value": None}
        self.config.CONFIG_FILE.write_text(json.dumps({"touch_assignments": {
            "touch1": {"tap": binding, "double": binding, "hold": binding},
            "touch3": {"tap": binding, "double": binding, "hold": binding}},
            "touch_applied_hash": "previous-policy", "user_name": "Keep me"}), encoding="utf-8")
        settings = self.config.load_settings()
        self.assertEqual(set(settings["touch_assignments"]["touch1"]), {"hold"})
        self.assertEqual(set(settings["touch_assignments"]["touch3"]), {"double", "hold"})
        self.assertEqual(settings["touch_applied_hash"], "")
        self.assertEqual(settings["user_name"], "Keep me")
        self.config._cached = None
        self.assertEqual(self.config.load_settings()["touch_policy_version"], 2)

    def test_legacy_plaintext_tokens_migrate_into_dpapi(self):
        from secure_store import load_secret
        self.config.CONFIG_FILE.write_text(json.dumps({"agent_token": "legacy-agent-token",
            "sync_token": "legacy-sync-token", "user_name": "Keep Name"}), encoding="utf-8")
        settings = self.config.load_settings()
        self.assertEqual(settings["user_name"], "Keep Name")
        self.assertEqual(load_secret("settings_agent_token"), "legacy-agent-token")
        self.config._cached = None
        self.assertEqual(self.config.load_settings()["sync_token"], "legacy-sync-token")
        self.assertNotIn("legacy-agent-token", self.config.CONFIG_FILE.read_text())
        self.assertNotIn("legacy-sync-token", self.config.CONFIG_FILE.read_text())

    def test_failed_public_write_does_not_rebind_saved_secret(self):
        from secure_store import load_secret
        self.config.load_settings()
        self.config.update_settings({"sync_token": "old-host-key", "sync_token_host": "old.local"})
        old_public = self.config.CONFIG_FILE.read_bytes()
        with patch.object(self.config, "_write_public", side_effect=OSError("fixture disk failure")):
            with self.assertRaises(OSError):
                self.config.update_settings({"sync_token": "new-host-key", "sync_token_host": "new.local"})
        self.assertEqual(self.config.CONFIG_FILE.read_bytes(), old_public)
        self.config._cached = None
        reloaded = self.config.load_settings()
        self.assertEqual(reloaded["sync_token"], "old-host-key")
        self.assertEqual(reloaded["sync_token_host"], "old.local")


if __name__ == "__main__":
    unittest.main()
