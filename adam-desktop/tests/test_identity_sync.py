"""No external accounts or cloud writes: HTTP, cipher and files are isolated."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlencode, urlsplit

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from account import AccountService, AccountError, _client_path
from cloud_sync import (CloudSync, CloudSyncError, empty_companion, merge_companions,
                        validate_companion, encode_value, decode_value)
from secure_store import SecretStore, SecretStoreError


class FakeCipher:
    def encrypt(self, value):
        return b"protected:" + bytes(byte ^ 0xA7 for byte in value)

    def decrypt(self, value):
        if not value.startswith(b"protected:"):
            raise SecretStoreError("Invalid ciphertext")
        return bytes(byte ^ 0xA7 for byte in value[10:])


class Response:
    def __init__(self, status=200, data=None):
        self.status_code = status
        self.ok = 200 <= status < 300
        self.data = data or {}

    def json(self):
        return deepcopy(self.data)


class Http:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        response = self.responses.pop(0)
        if callable(response):
            return response(method, url, kwargs)
        if isinstance(response, Exception):
            raise response
        return response

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)


def session_response(uid="alice", **extra):
    return Response(data={"localId": uid, "email": uid + "@example.test", "displayName": uid,
                          "idToken": "test-id-token", "refreshToken": "test-refresh-token",
                          "expiresIn": "3600", **extra})


MEMORY_ID = "90f52987-d3d6-4f80-952c-bf15c66d37b8"
OTHER_ID = "8b741b7b-00f5-48ae-9db5-76db7ed578bc"
STAMP = "2026-10-07T08:00:00.000Z"


def memory(key=MEMORY_ID, title="Routine", stamp=STAMP):
    return {"id": key, "title": title, "text": "A calm morning", "kind": "fact",
            "createdAt": STAMP, "updatedAt": stamp, "deleted": False}


class FakeAccount:
    def __init__(self, uid="alice"):
        self.uid = uid

    @property
    def user(self):
        return {"uid": self.uid} if self.uid else None

    def id_token(self):
        return "fake-firebase-token"


class SecureStoreTests(unittest.TestCase):
    def test_atomic_encryption_roundtrip_concurrency_and_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SecretStore(Path(directory), FakeCipher())
            self.assertIsNone(store.load_secret("auth"))
            store.save_secret("auth", "never-plaintext")
            self.assertNotIn(b"never-plaintext", (Path(directory) / "auth.dpapi").read_bytes())
            self.assertEqual(store.load_secret("auth"), "never-plaintext")
            with ThreadPoolExecutor(max_workers=6) as pool:
                list(pool.map(lambda i: store.save_secret("auth", "token-" + str(i)), range(30)))
            self.assertTrue(store.load_secret("auth").startswith("token-"))
            self.assertEqual(len(list(Path(directory).iterdir())), 1)
            store.delete_secret("auth")
            self.assertIsNone(store.load_secret("auth"))
            with self.assertRaises(ValueError):
                store.save_secret("../escape", "value")

    def test_encrypt_failure_never_creates_plaintext(self):
        class FailingCipher:
            def encrypt(self, _):
                raise SecretStoreError("fail")
        with tempfile.TemporaryDirectory() as directory:
            store = SecretStore(Path(directory), FailingCipher())
            with self.assertRaises(SecretStoreError):
                store.save_secret("auth", "never-plaintext")
            self.assertEqual(list(Path(directory).iterdir()), [])


class AccountTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = SecretStore(Path(self.temp.name), FakeCipher())

    def test_login_secure_restore_status_and_signout(self):
        http = Http(session_response())
        account = AccountService(store=self.store, session=http)
        result = account.email_login("alice@example.test", "a-password")
        self.assertTrue(result["authenticated"])
        self.assertNotIn("test-id-token", json.dumps(result))
        self.assertNotIn("a-password", self.store.load_secret("firebase-session"))
        restored = AccountService(store=self.store, session=Http())
        self.assertEqual(restored.user["uid"], "alice")
        self.assertEqual(restored.id_token(), "test-id-token")
        restored.signout()
        self.assertIsNone(self.store.load_secret("firebase-session"))
        self.assertFalse(restored.status()["authenticated"])

    def test_errors_are_safe_and_reset_does_not_reveal_account(self):
        http = Http(Response(400, {"error": {"message": "INVALID_LOGIN_CREDENTIALS"}}),
                    Response(400, {"error": {"message": "EMAIL_NOT_FOUND"}}))
        account = AccountService(store=self.store, session=http)
        with self.assertRaisesRegex(AccountError, "email or password"):
            account.email_login("alice@example.test", "wrong")
        self.assertIn("If an account", account.reset_password("alice@example.test")["message"])

    def test_refresh_is_serialized_and_persisted(self):
        http = Http(session_response(expiresIn="1"), Response(data={"user_id": "alice", "id_token": "new-id",
                      "refresh_token": "new-refresh", "expires_in": "3600"}))
        account = AccountService(store=self.store, session=http)
        account.email_login("alice@example.test", "password")
        with ThreadPoolExecutor(max_workers=5) as pool:
            self.assertEqual(list(pool.map(lambda _: account.id_token(), range(10))), ["new-id"] * 10)
        self.assertEqual(len(http.calls), 2)
        self.assertEqual(json.loads(self.store.load_secret("firebase-session"))["refreshToken"], "new-refresh")

    def test_created_account_survives_optional_profile_network_failure(self):
        http = Http(session_response(), requests.ConnectionError("offline"))
        account = AccountService(store=self.store, session=http)
        status = account.email_login("alice@example.test", "password", create=True, name="Alice")
        self.assertTrue(status["authenticated"])
        self.assertIn("profile name", status["error"])
        self.assertEqual(account.id_token(), "test-id-token")

    def test_signout_invalidates_inflight_login(self):
        entered, release = threading.Event(), threading.Event()
        def slow(*_):
            entered.set()
            release.wait(3)
            return session_response()
        account = AccountService(store=self.store, session=Http(slow))
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(account.email_login, "alice@example.test", "password")
            self.assertTrue(entered.wait(2))
            account.signout()
            release.set()
            with self.assertRaisesRegex(AccountError, "cancelled"):
                future.result(timeout=3)
        self.assertFalse(account.status()["authenticated"])
        self.assertIsNone(self.store.load_secret("firebase-session"))

    def test_google_loopback_pkce_state_and_firebase_exchange(self):
        client = Path(self.temp.name) / "client.json"
        client.write_text(json.dumps({"installed": {"client_id": "759320300226-desktop.apps.googleusercontent.com",
            "client_secret": "desktop-client-secret", "project_id": "adam-ai1"}}))
        observed = {}
        callback_threads = []
        def browser(url):
            query = parse_qs(urlsplit(url).query)
            observed.update(query)
            def callback():
                redirect = query["redirect_uri"][0]
                wrong = requests.get(redirect, params={"state": "wrong", "code": "bad"}, timeout=3)
                observed["bad_status"] = wrong.status_code
                good = requests.get(redirect, params={"state": query["state"][0], "code": "test-code"}, timeout=3)
                observed["good_status"] = good.status_code
            thread = threading.Thread(target=callback)
            callback_threads.append(thread)
            thread.start()
            return True
        http = Http(Response(data={"id_token": "desktop-audience-google-id", "access_token": "google-access-token"}), session_response())
        account = AccountService(store=self.store, session=http, browser_open=browser)
        with patch.dict(os.environ, {"ADAM_GOOGLE_CLIENT_FILE": str(client)}):
            account.start_google()
            deadline = time.monotonic() + 5
            while account.status()["google"]["pending"] and time.monotonic() < deadline:
                time.sleep(0.02)
            for thread in callback_threads:
                thread.join(timeout=3)
            self.assertTrue(account.status()["authenticated"], account.status())
        self.assertEqual(observed["bad_status"], 400)
        self.assertEqual(observed["good_status"], 200)
        self.assertEqual(observed["code_challenge_method"], ["S256"])
        import base64
        verifier = http.calls[0][2]["data"]["code_verifier"]
        expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        self.assertEqual(observed["code_challenge"], [expected])
        self.assertIn("127.0.0.1", observed["redirect_uri"][0])
        body = http.calls[1][2]["json"]
        credential = parse_qs(body["postBody"])
        self.assertEqual(credential["providerId"], ["google.com"])
        self.assertEqual(credential["access_token"], ["google-access-token"])
        self.assertNotIn("id_token", credential)
        self.assertFalse(body["returnIdpCredential"])

    def test_oauth_config_lookup_env_sidecar_and_bundle_precedence(self):
        import account as account_module
        root = Path(self.temp.name)
        exe = root / "installed" / "ADAM.exe"
        exe.parent.mkdir()
        bundle = root / "unpacked"
        bundle.mkdir()
        bundled_config = bundle / "firebase-desktop-client.json"
        bundled_config.write_text("{}")
        sidecar = exe.parent / "firebase-desktop-client.json"
        with patch.dict(os.environ, {"ADAM_GOOGLE_CLIENT_FILE": ""}), \
                patch.object(account_module.sys, "frozen", True, create=True), \
                patch.object(account_module.sys, "executable", str(exe)), \
                patch.object(account_module.sys, "_MEIPASS", str(bundle), create=True):
            self.assertEqual(_client_path(), bundled_config)
            sidecar.write_text("{}")
            self.assertEqual(_client_path(), sidecar)
            override = root / "override.json"
            with patch.dict(os.environ, {"ADAM_GOOGLE_CLIENT_FILE": str(override)}):
                self.assertEqual(_client_path(), override)


class SyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.account = FakeAccount()

    def test_merge_deterministic_delete_and_independent_preferences(self):
        left, right = empty_companion(), empty_companion()
        left["memories"][MEMORY_ID] = memory()
        right["memories"][MEMORY_ID] = {"id": MEMORY_ID, "updatedAt": STAMP, "deleted": True}
        right["memories"][OTHER_ID] = memory(OTHER_ID)
        left["preferences"]["voice"] = {"value": "Kore", "updatedAt": STAMP}
        right["preferences"]["wakeWord"] = {"value": "ADAM", "updatedAt": STAMP}
        merged = merge_companions(left, right)
        self.assertEqual(merged, merge_companions(right, left))
        self.assertTrue(merged["memories"][MEMORY_ID]["deleted"])
        self.assertIn(OTHER_ID, merged["memories"])
        self.assertEqual(len(merged["preferences"]), 2)
        self.assertEqual(merged, decode_value(encode_value(merged)))

    def test_validation_rejects_corruption_and_oversize(self):
        doc = empty_companion()
        doc["memories"][MEMORY_ID] = memory()
        for value in ("bad-date", "2026-02-30T08:00:00.000Z"):
            doc["memories"][MEMORY_ID]["updatedAt"] = value
            with self.assertRaises(CloudSyncError):
                validate_companion(doc)
        doc["memories"][MEMORY_ID] = memory()
        doc["memories"][MEMORY_ID]["kind"] = []
        with self.assertRaises(CloudSyncError):
            validate_companion(doc)
        doc["memories"][MEMORY_ID] = {**memory(), "title": "\U0001f916" * 41}
        with self.assertRaises(CloudSyncError):
            validate_companion(doc)
        doc = empty_companion()
        import uuid
        for _ in range(400):
            key = str(uuid.uuid4())
            doc["memories"][key] = {**memory(key), "text": "x" * 2000}
        with self.assertRaisesRegex(CloudSyncError, "full"):
            validate_companion(doc)

    def test_text_trim_exactly_matches_mobile_javascript(self):
        doc = empty_companion()
        doc["memories"][MEMORY_ID] = {**memory(), "title": "\ufeff Title \ufeff", "text": "\u0085Keep NEL\u0085"}
        checked = validate_companion(doc)["memories"][MEMORY_ID]
        self.assertEqual(checked["title"], "Title")
        self.assertEqual(checked["text"], "\u0085Keep NEL\u0085")

    def test_scope_isolation_explicit_guest_import_and_reload(self):
        sync = CloudSync(self.account, self.directory)
        self.account.uid = None
        guest = sync.save_memory({"title": "Guest", "text": "Only local", "kind": "fact"})
        self.account.uid = "alice"
        self.assertEqual(sync.list_memories(), [])
        self.assertEqual(sync.status()["guestMemories"], 1)
        sync.import_guest()
        self.assertEqual(sync.list_memories()[0]["id"], guest["id"])
        self.account.uid = "bob"
        self.assertEqual(sync.list_memories(), [])
        self.account.uid = "alice"
        self.assertEqual(CloudSync(self.account, self.directory).list_memories()[0]["title"], "Guest")
        sync.delete_memory(guest["id"])
        self.assertEqual(sync.list_memories(), [])
        self.account.uid = None
        self.assertEqual(len(sync.list_memories()), 1)

    def test_conditional_patch_preserves_other_user_fields_and_retries_conflict(self):
        remote = empty_companion()
        remote["memories"][OTHER_ID] = memory(OTHER_ID)
        http = Http(Response(data={"fields": {"name": {"stringValue": "Alice"}}, "updateTime": "version1"}),
                    Response(400, {"error": {"status": "FAILED_PRECONDITION"}}),
                    Response(data={"fields": {"name": {"stringValue": "Alice"}, "companion": encode_value(remote)}, "updateTime": "version2"}),
                    Response())
        sync = CloudSync(self.account, self.directory, http)
        saved = sync.save_memory({"title": "PC", "text": "Local note", "kind": "fact"})
        sync.sync()
        patch_call = http.calls[3][2]
        self.assertEqual(patch_call["params"], {"updateMask.fieldPaths": "companion", "currentDocument.updateTime": "version2"})
        self.assertEqual(set(patch_call["json"]["fields"]), {"companion"})
        merged = decode_value(patch_call["json"]["fields"]["companion"])
        self.assertIn(saved["id"], merged["memories"])
        self.assertIn(OTHER_ID, merged["memories"])
        self.assertFalse(sync.status()["pending"])

    def test_creates_only_companion_and_offline_keeps_local_data(self):
        http = Http(Response(404), Response())
        sync = CloudSync(self.account, self.directory, http)
        sync.save_preferences({"voice": "Kore"})
        sync.sync()
        self.assertEqual(http.calls[1][2]["params"]["currentDocument.exists"], "false")
        sync._http = Http(requests.ConnectionError("sensitive internal network detail"))
        sync.save_memory({"title": "Keep", "text": "Offline", "kind": "fact"})
        with self.assertRaisesRegex(CloudSyncError, "offline"):
            sync.sync()
        self.assertEqual(len(sync.list_memories()), 1)
        self.assertTrue(sync.status()["pending"])
        self.assertNotIn("sensitive", sync.status()["error"])

    def test_account_switch_inflight_prevents_patch_and_cross_account_leak(self):
        def switch(*_):
            self.account.uid = "bob"
            return Response(data={"fields": {}, "updateTime": "v1"})
        http = Http(switch)
        sync = CloudSync(self.account, self.directory, http)
        sync.save_memory({"title": "Private", "text": "Alice", "kind": "fact"})
        with self.assertRaisesRegex(CloudSyncError, "account changed"):
            sync.sync()
        self.assertEqual(len(http.calls), 1)
        self.assertEqual(sync.list_memories(), [])
        self.account.uid = "alice"
        self.assertEqual(sync.list_memories()[0]["title"], "Private")

    def test_edit_during_patch_remains_pending(self):
        sync = CloudSync(self.account, self.directory)
        item = sync.save_memory({"title": "Before", "text": "Note", "kind": "fact"})
        def concurrent_edit(*_):
            sync.save_memory({**item, "title": "After"})
            return Response()
        sync._http = Http(Response(data={"fields": {}, "updateTime": "v1"}), concurrent_edit)
        sync.sync()
        self.assertEqual(sync.list_memories()[0]["title"], "After")
        self.assertTrue(sync.status()["pending"])

    def test_corrupt_local_record_is_preserved(self):
        sync = CloudSync(self.account, self.directory)
        path = sync._path("alice")
        path.write_text("broken data")
        with self.assertRaisesRegex(CloudSyncError, "preserved"):
            sync.save_memory({"title": "No", "text": "Overwrite", "kind": "fact"})
        self.assertEqual(path.read_text(), "broken data")


if __name__ == "__main__":
    unittest.main()
