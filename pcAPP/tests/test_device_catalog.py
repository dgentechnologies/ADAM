"""Account-owned device discovery with isolated fake Firebase responses."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from device_catalog import list_devices


def device(uid="alice", **fields):
    values = {"ownerUid": uid, "deviceId": "fixture-adam", "name": "Office ADAM",
              "tailscaleIp": "100.64.0.20", "status": "online", **fields}
    return {"document": {"name": "projects/fixture/databases/(default)/documents/devices/doc-id",
                         "fields": {key: {"stringValue": value} for key, value in values.items()}}}


class DeviceCatalogTests(unittest.TestCase):
    def setUp(self):
        self.account = SimpleNamespace(user={"uid": "alice"}, id_token=Mock(return_value="fixture-token"))
        self.response = Mock(ok=True)
        self.response.json.return_value = [device()]
        self.http = Mock()
        self.http.post.return_value = self.response

    def test_signed_out_does_not_request_cloud_or_token(self):
        self.account.user = None
        self.assertEqual(list_devices(self.account, self.http), [])
        self.http.post.assert_not_called()
        self.account.id_token.assert_not_called()

    def test_query_is_owner_scoped_and_response_excludes_foreign_and_private_fields(self):
        self.response.json.return_value = [device(syncToken="private-token", wifiPassword="private-password"),
                                           device("bob"), {"readTime": "fixture-time"}]
        result = list_devices(self.account, self.http)
        self.assertEqual(len(result), 1)
        self.assertEqual(set(result[0]), {"deviceId", "name", "hardwareSerial", "tailscaleIp", "osVersion", "status"})
        self.assertNotIn("private", str(result))
        args, kwargs = self.http.post.call_args
        self.assertIn("/documents:runQuery", args[0])
        self.assertEqual(kwargs["json"]["structuredQuery"]["where"]["fieldFilter"], {
            "field": {"fieldPath": "ownerUid"}, "op": "EQUAL", "value": {"stringValue": "alice"}})
        self.assertEqual(kwargs["headers"], {"Authorization": "Bearer fixture-token"})
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(kwargs["timeout"], (5, 15))

    def test_response_limits_and_document_id_fallback(self):
        rows = [device(name="A" * 500, deviceId="") for _ in range(70)]
        self.response.json.return_value = rows
        result = list_devices(self.account, self.http)
        self.assertEqual(len(result), 50)
        self.assertEqual(result[0]["name"], "A" * 256)
        self.assertEqual(result[0]["deviceId"], "doc-id")

    def test_signout_or_account_change_in_flight_discards_old_devices(self):
        for replacement in (None, {"uid": "bob"}):
            with self.subTest(replacement=replacement):
                self.account.user = {"uid": "alice"}
                def arrive():
                    self.account.user = deepcopy(replacement)
                    return [device()]
                self.response.json.side_effect = arrive
                with self.assertRaisesRegex(RuntimeError, "account changed"):
                    list_devices(self.account, self.http)

    def test_account_change_during_token_refresh_never_sends_previous_users_query(self):
        def refresh():
            self.account.user = {"uid": "bob"}
            return "bob-fixture-token"
        self.account.id_token.side_effect = refresh
        with self.assertRaisesRegex(RuntimeError, "account changed"):
            list_devices(self.account, self.http)
        self.http.post.assert_not_called()

    def test_malformed_rows_are_ignored_without_losing_valid_owned_devices(self):
        self.response.json.return_value = [None, [], {"document": None},
            {"document": {"fields": None}}, {"document": {"fields": []}},
            {"document": {"fields": {"ownerUid": None}}}, device()]
        result = list_devices(self.account, self.http)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["deviceId"], "fixture-adam")

    def test_network_http_and_malformed_response_errors_stay_safe(self):
        self.http.post.side_effect = requests.ConnectionError("contains private transport context")
        with self.assertRaisesRegex(RuntimeError, "could not be reached"):
            list_devices(self.account, self.http)
        self.http.post.side_effect = None
        self.response.ok = False
        with self.assertRaisesRegex(RuntimeError, "could not be loaded"):
            list_devices(self.account, self.http)
        self.response.ok = True
        self.response.json.return_value = {"unexpected": "shape"}
        with self.assertRaisesRegex(RuntimeError, "could not be reached"):
            list_devices(self.account, self.http)


if __name__ == "__main__":
    unittest.main()
