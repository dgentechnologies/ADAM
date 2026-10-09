"""Bounded, account-scoped memories and preferences shared with ADAM mobile.

Only users/{uid}.companion is patched. Existing profile and device fields remain
untouched. Tombstones and conditional writes prevent deleted notes reappearing
or simultaneous devices replacing each other's changes.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
import os
import threading
from urllib.parse import quote
import uuid

import requests

from account import AccountError, FIREBASE_PROJECT_ID
from secure_store import data_directory


MAX_ENTRIES = 1000
MAX_SYNC_BYTES = 680000
PREFERENCES = {"voice": {"Charon", "Aoede", "Kore", "Puck", "Fenrir"},
               "wakeWord": {"Hey ADAM", "ADAM"}, "brain": {"lite", "byok", "managed"}}
DEFAULT_PREFERENCES = {"voice": "Charon", "wakeWord": "Hey ADAM", "brain": "lite"}
_ECMASCRIPT_WHITESPACE = "\u0009\u000a\u000b\u000c\u000d\u0020\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"


class CloudSyncError(RuntimeError):
    def __init__(self, message: str, code: str = "sync_error"):
        super().__init__(message)
        self.code = code


def _canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _text_length(value: str) -> int:
    # Mobile validates JavaScript strings in UTF-16 code units. Match its bounds
    # for emoji too, and reject malformed lone surrogates before serializing.
    try:
        return len(value.encode("utf-16-le")) // 2
    except UnicodeError as exc:
        raise CloudSyncError("A shared memory contains invalid text.", "invalid_data") from exc


def _trim_text(value: str) -> str:
    # Python strip() differs for BOM, NEL and several control separators.
    # Normalize identically to JavaScript String.trim() on the mobile client.
    return value.strip(_ECMASCRIPT_WHITESPACE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _timestamp(value) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", value):
        raise CloudSyncError("Shared data contains an invalid timestamp.", "invalid_data")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CloudSyncError("Shared data contains an invalid timestamp.", "invalid_data") from exc
    return value


def _memory_id(value) -> str:
    try:
        if not isinstance(value, str) or str(uuid.UUID(value)) != value:
            raise ValueError()
    except (ValueError, AttributeError) as exc:
        raise CloudSyncError("This memory has an invalid identifier.", "invalid_data") from exc
    return value


def encode_value(value) -> dict:
    if isinstance(value, bool):
        return {"booleanValue": value}
    if isinstance(value, int):
        return {"integerValue": str(value)}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, dict):
        return {"mapValue": {"fields": {key: encode_value(item) for key, item in value.items()}}}
    raise CloudSyncError("Shared data contains an unsupported value.", "invalid_data")


def decode_value(value: dict):
    if not isinstance(value, dict):
        raise CloudSyncError("Shared data could not be read.", "invalid_data")
    if "mapValue" in value:
        if not isinstance(value["mapValue"], dict):
            raise CloudSyncError("Shared data could not be read.", "invalid_data")
        fields = value["mapValue"].get("fields", {})
        if not isinstance(fields, dict):
            raise CloudSyncError("Shared data could not be read.", "invalid_data")
        return {key: decode_value(item) for key, item in fields.items()}
    if "stringValue" in value:
        return value["stringValue"]
    if "booleanValue" in value:
        return value["booleanValue"]
    if "integerValue" in value:
        try:
            return int(value["integerValue"])
        except (TypeError, ValueError):
            pass
    raise CloudSyncError("Shared data contains an unsupported value.", "invalid_data")


def empty_companion() -> dict:
    return {"schemaVersion": 2, "memories": {}, "preferences": {}, "todos": {}, "clocks": {}, "devices": {}}


def _record_text(value, maximum, empty=False):
    if not isinstance(value, str) or not (0 if empty else 1) <= _text_length(_trim_text(value)) <= maximum:
        raise CloudSyncError("A shared record contains invalid text.", "invalid_data")
    return _trim_text(value)


def validate_records(kind, values):
    if not isinstance(values, dict) or len(values) > (100 if kind == "devices" else MAX_ENTRIES):
        raise CloudSyncError("Shared data exceeds the supported limit.", "capacity")
    result = {}
    for key, item in values.items():
        _memory_id(key)
        if not isinstance(item, dict) or item.get("id") != key or type(item.get("deleted")) is not bool:
            raise CloudSyncError("A shared record could not be read.", "invalid_data")
        record = {"id": key, "updatedAt": _timestamp(item.get("updatedAt")), "deleted": item["deleted"]}
        if not item["deleted"]:
            record["createdAt"] = _timestamp(item.get("createdAt"))
            if record["createdAt"] > record["updatedAt"]:
                raise CloudSyncError("A shared record has inconsistent dates.", "invalid_data")
            if kind == "devices":
                record.update(name=_record_text(item.get("name"), 40), serial=_record_text(item.get("serial"), 80))
                boolean = "simulated"
            else:
                device = item.get("deviceId")
                record["deviceId"] = "" if device == "" else _memory_id(device)
                if kind == "todos":
                    record.update(text=_record_text(item.get("text"), 2000), dueAt="" if item.get("dueAt") == "" else _timestamp(item.get("dueAt")))
                    boolean = "done"
                else:
                    if item.get("kind") not in ("alarm", "timer", "reminder"):
                        raise CloudSyncError("Choose an alarm, timer or reminder.", "invalid_data")
                    record.update(kind=item["kind"], label=_record_text(item.get("label"), 80, empty=True), when=_timestamp(item.get("when")))
                    boolean = "enabled"
            if type(item.get(boolean)) is not bool:
                raise CloudSyncError("A shared record could not be read.", "invalid_data")
            record[boolean] = item[boolean]
        result[key] = record
    return result


def validate_companion(value: dict) -> dict:
    if not isinstance(value, dict) or type(value.get("schemaVersion")) is not int or value["schemaVersion"] not in (1, 2):
        raise CloudSyncError("This shared-data version needs an app update.", "invalid_data")
    memories = value.get("memories", {})
    preferences = value.get("preferences", {})
    if not isinstance(memories, dict) or len(memories) > MAX_ENTRIES or not isinstance(preferences, dict):
        raise CloudSyncError("Shared data exceeds the supported limit.", "capacity")
    result = empty_companion()
    for kind in ("todos", "clocks", "devices"):
        result[kind] = validate_records(kind, value.get(kind, {}))
    for key, item in memories.items():
        _memory_id(key)
        if not isinstance(item, dict) or item.get("id") != key or type(item.get("deleted")) is not bool:
            raise CloudSyncError("A shared memory could not be read.", "invalid_data")
        updated = _timestamp(item.get("updatedAt"))
        if item["deleted"]:
            result["memories"][key] = {"id": key, "updatedAt": updated, "deleted": True}
            continue
        if (not isinstance(item.get("title"), str) or not 1 <= _text_length(_trim_text(item["title"])) <= 80
                or not isinstance(item.get("text"), str) or not 1 <= _text_length(_trim_text(item["text"])) <= 2000
                or not isinstance(item.get("kind"), str) or item.get("kind") not in {"fact", "person"}):
            raise CloudSyncError("A shared memory could not be read.", "invalid_data")
        created = _timestamp(item.get("createdAt"))
        if created > updated:
            raise CloudSyncError("A shared memory has inconsistent dates.", "invalid_data")
        result["memories"][key] = {"id": key, "title": _trim_text(item["title"]), "text": _trim_text(item["text"]),
            "kind": item["kind"], "createdAt": created, "updatedAt": updated, "deleted": False}
    for key, item in preferences.items():
        if (key not in PREFERENCES or not isinstance(item, dict) or not isinstance(item.get("value"), str)
                or item.get("value") not in PREFERENCES[key]):
            raise CloudSyncError("A shared preference could not be read.", "invalid_data")
        result["preferences"][key] = {"value": item["value"], "updatedAt": _timestamp(item.get("updatedAt"))}
    if len(_canonical(encode_value(result)).encode("utf-8")) > MAX_SYNC_BYTES:
        raise CloudSyncError("Shared memories are full. Shorten a memory before saving more.", "capacity")
    return result


def merge_companions(left: dict, right: dict) -> dict:
    left, right = validate_companion(left), validate_companion(right)
    result = empty_companion()
    for field in ("memories", "preferences", "todos", "clocks", "devices"):
        for key in sorted(set(left[field]) | set(right[field])):
            options = [source[field][key] for source in (left, right) if key in source[field]]
            result[field][key] = max(options, key=lambda item: (item["updatedAt"],
                bool(item.get("deleted", False)), _canonical(item)))
    return validate_companion(result)


class CloudSync:
    def __init__(self, account, directory: Path | None = None, session=None):
        self.account = account
        self.directory = Path(directory) if directory is not None else data_directory() / "companion"
        self._http = session if session is not None else requests.Session()
        self._lock = threading.RLock()
        self._sync_lock = threading.Lock()
        self._syncing_uid: str | None = None
        self._errors: dict[str, str] = {}

    def _uid(self) -> str:
        user = self.account.user
        return user["uid"] if user else "guest"

    def _path(self, uid: str) -> Path:
        return self.directory / ("guest.json" if uid == "guest" else hashlib.sha256(uid.encode("utf-8")).hexdigest() + ".json")

    def _read(self, uid: str) -> dict:
        path = self._path(uid)
        if not path.exists():
            return {"companion": empty_companion(), "dirty": False, "lastSynced": None}
        try:
            if path.stat().st_size > MAX_SYNC_BYTES * 2:
                raise ValueError()
            record = json.loads(path.read_text(encoding="utf-8"))
            record["companion"] = validate_companion(record["companion"])
            if type(record.get("dirty")) is not bool:
                raise ValueError()
            if record.get("lastSynced") is not None:
                _timestamp(record["lastSynced"])
            return record
        except (OSError, ValueError, TypeError, KeyError, CloudSyncError) as exc:
            raise CloudSyncError("Saved memories could not be read. Your file has been preserved for recovery.", "local_data") from exc

    def _write(self, uid: str, record: dict) -> None:
        record = deepcopy(record)
        record["companion"] = validate_companion(record["companion"])
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory,
                                             prefix=".companion-", delete=False) as file:
                temporary = Path(file.name)
                file.write(_canonical(record))
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self._path(uid))
        except OSError as exc:
            raise CloudSyncError("Memories could not be saved. Check available disk space.", "local_storage") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def status(self) -> dict:
        with self._lock:
            uid = self._uid()
            result = {"enabled": uid != "guest", "signedIn": uid != "guest", "syncing": self._syncing_uid == uid,
                      "lastSynced": None, "pending": False, "guestMemories": 0, "error": self._errors.get(uid)}
            try:
                record = self._read(uid)
                result.update(lastSynced=record.get("lastSynced"), pending=record["dirty"])
                if uid != "guest":
                    guest = self._read("guest")["companion"]
                    result["guestMemories"] = sum(not item["deleted"] for kind in ("memories", "todos", "clocks", "devices") for item in guest[kind].values())
            except CloudSyncError as exc:
                result["error"] = str(exc)
            return result

    def list_memories(self) -> list[dict]:
        with self._lock:
            items = self._read(self._uid())["companion"]["memories"].values()
            return sorted((deepcopy(item) for item in items if not item["deleted"]),
                          key=lambda item: (item["updatedAt"], item["id"]), reverse=True)

    def list_records(self, kind: str) -> list[dict]:
        if kind not in ("todos", "clocks", "devices"):
            raise CloudSyncError("Unsupported shared collection.", "invalid_data")
        with self._lock:
            return [deepcopy(item) for item in self._read(self._uid())["companion"][kind].values() if not item["deleted"]]

    def sync_simulated_ble(self, device_id: str) -> dict:
        """Exchange the shared envelope with one account-scoped local robot simulator."""
        _memory_id(device_id)
        with self._lock:
            uid = self._uid()
            saved = self._read(uid)
            device = saved["companion"]["devices"].get(device_id)
            if not device or device["deleted"] or not device.get("simulated"):
                raise CloudSyncError("Select a simulated ADAM for BLE sync.", "invalid_device")
            robot_uid = "ble-simulation:" + uid + ":" + device_id
            robot = self._read(robot_uid)
            merged = merge_companions(saved["companion"], robot["companion"])
            self._write(robot_uid, {"companion": merged, "dirty": False, "lastSynced": _now()})
            saved.update(companion=merged, dirty=True)
            self._write(uid, saved)
            return {"simulated": True, "deviceId": device_id, "companion": merged}

    def save_record(self, kind: str, data: dict) -> dict:
        if kind not in ("todos", "clocks", "devices") or not isinstance(data, dict):
            raise CloudSyncError("Unsupported shared collection.", "invalid_data")
        with self._lock:
            uid = self._uid()
            saved = self._read(uid)
            key = _memory_id(data["id"]) if data.get("id") else str(uuid.uuid4())
            old = saved["companion"][kind].get(key, {})
            stamp = self._after(old.get("updatedAt"))
            record = {**data, "id": key, "createdAt": old.get("createdAt", stamp), "updatedAt": stamp, "deleted": False}
            record = validate_records(kind, {key: record})[key]
            saved["companion"][kind][key] = record
            saved["dirty"] = True
            self._write(uid, saved)
            return deepcopy(record)

    def delete_record(self, kind: str, key: str) -> None:
        if kind not in ("todos", "clocks", "devices"):
            raise CloudSyncError("Unsupported shared collection.", "invalid_data")
        _memory_id(key)
        with self._lock:
            uid = self._uid()
            saved = self._read(uid)
            old = saved["companion"][kind].get(key)
            if not old:
                raise CloudSyncError("This record no longer exists.", "not_found")
            saved["companion"][kind][key] = {"id": key, "updatedAt": self._after(old["updatedAt"]), "deleted": True}
            saved["dirty"] = True
            self._write(uid, saved)

    def save_memory(self, data: dict) -> dict:
        if not isinstance(data, dict):
            raise CloudSyncError("Enter a title and memory.", "invalid_data")
        with self._lock:
            uid = self._uid()
            record = self._read(uid)
            key = _memory_id(data["id"]) if data.get("id") else str(uuid.uuid4())
            previous = record["companion"]["memories"].get(key)
            # Advance a millisecond when the clock has not moved, so an edit is
            # newer than a deletion or earlier action in the same millisecond.
            stamp = self._after(previous.get("updatedAt") if previous else None)
            item = {"id": key, "title": data.get("title", ""), "text": data.get("text", ""),
                    "kind": data.get("kind", "fact"), "createdAt": previous.get("createdAt", stamp) if previous else stamp,
                    "updatedAt": stamp, "deleted": False}
            record["companion"]["memories"][key] = item
            record["dirty"] = True
            self._write(uid, record)
            return validate_companion(record["companion"])["memories"][key]

    @staticmethod
    def _after(previous: str | None) -> str:
        stamp = _now()
        if previous and previous >= stamp:
            from datetime import timedelta
            return (datetime.fromisoformat(previous.replace("Z", "+00:00")) + timedelta(milliseconds=1)).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        return stamp

    def delete_memory(self, memory_id: str) -> dict:
        key = _memory_id(memory_id)
        with self._lock:
            uid = self._uid()
            record = self._read(uid)
            previous = record["companion"]["memories"].get(key)
            if not previous:
                raise CloudSyncError("This memory no longer exists.", "not_found")
            record["companion"]["memories"][key] = {"id": key, "deleted": True,
                "updatedAt": self._after(previous["updatedAt"])}
            record["dirty"] = True
            self._write(uid, record)
            return {"deleted": True, "id": key}

    def get_preferences(self) -> dict:
        with self._lock:
            values = self._read(self._uid())["companion"]["preferences"]
            return {**DEFAULT_PREFERENCES, **{key: item["value"] for key, item in values.items()}}

    def save_preferences(self, data: dict) -> dict:
        if not isinstance(data, dict) or not data or any(key not in PREFERENCES for key in data):
            raise CloudSyncError("Choose a supported companion preference.", "invalid_data")
        with self._lock:
            uid = self._uid()
            record = self._read(uid)
            values = record["companion"]["preferences"]
            for key, value in data.items():
                if not isinstance(value, str) or value not in PREFERENCES[key]:
                    raise CloudSyncError("Choose a supported companion preference.", "invalid_data")
                if values.get(key, {}).get("value") != value:
                    values[key] = {"value": value, "updatedAt": self._after(values.get(key, {}).get("updatedAt"))}
                    record["dirty"] = True
            self._write(uid, record)
            return {**DEFAULT_PREFERENCES, **{key: item["value"] for key, item in values.items()}}

    def import_guest(self) -> dict:
        with self._lock:
            uid = self._uid()
            if uid == "guest":
                raise CloudSyncError("Sign in before adding this PC's local memories to your account.", "signed_out")
            record = self._read(uid)
            record["companion"] = merge_companions(record["companion"], self._read("guest")["companion"])
            record["dirty"] = True
            self._write(uid, record)
            return self.status()

    def _request(self, method: str, url: str, **kwargs):
        try:
            return self._http.request(method, url, timeout=(5, 20), allow_redirects=False, **kwargs)
        except requests.RequestException as exc:
            raise CloudSyncError("You're offline. Your changes are saved on this PC and will sync when you reconnect.", "network") from exc

    @staticmethod
    def _response(response) -> dict:
        if not response.ok:
            if response.status_code in (401, 403):
                raise CloudSyncError("Cloud access was denied. Check your sign-in and the Firebase owner rules.", "permission_denied")
            if response.status_code == 429:
                raise CloudSyncError("Sync is busy. Your changes are saved; please try again shortly.", "rate_limit")
            raise CloudSyncError("Cloud sync could not finish. Your changes are saved on this PC.", "server")
        try:
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError()
            return data
        except ValueError as exc:
            raise CloudSyncError("Cloud sync returned an unreadable response. Your changes are saved.", "invalid_data") from exc

    def _ensure_uid(self, uid: str) -> None:
        if self._uid() != uid:
            raise CloudSyncError("The account changed while syncing. Please sync the current account.", "account_changed")

    def sync(self) -> dict:
        if not self._sync_lock.acquire(blocking=False):
            return self.status()
        uid = self._uid()
        try:
            if uid == "guest":
                raise CloudSyncError("Sign in to sync memories with your mobile app.", "signed_out")
            with self._lock:
                self._syncing_uid = uid
                self._errors.pop(uid, None)
            token = self.account.id_token()
            self._ensure_uid(uid)
            url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/users/{quote(uid, safe='')}"
            headers = {"Authorization": "Bearer " + token}
            for _attempt in range(4):
                self._ensure_uid(uid)
                response = self._request("GET", url, headers=headers)
                if response.status_code == 404:
                    remote, precondition = empty_companion(), {"currentDocument.exists": "false"}
                else:
                    document = self._response(response)
                    fields = document.get("fields", {})
                    if not isinstance(fields, dict):
                        raise CloudSyncError("Cloud sync returned an unreadable document.", "invalid_data")
                    field = fields.get("companion")
                    remote = validate_companion(decode_value(field)) if field is not None else empty_companion()
                    if not document.get("updateTime"):
                        raise CloudSyncError("Cloud sync could not verify the document version.", "invalid_data")
                    precondition = {"currentDocument.updateTime": document["updateTime"]}
                with self._lock:
                    self._ensure_uid(uid)
                    local = self._read(uid)
                    merged = merge_companions(local["companion"], remote)
                # No mutation when the cloud already has the desired record.
                if merged != remote or response.status_code == 404:
                    self._ensure_uid(uid)
                    saved = self._request("PATCH", url, headers=headers,
                        params={"updateMask.fieldPaths": "companion", **precondition},
                        json={"fields": {"companion": encode_value(merged)}})
                    conflict = saved.status_code in (409, 412)
                    if saved.status_code == 400:
                        try:
                            conflict = saved.json().get("error", {}).get("status") in {"FAILED_PRECONDITION", "ABORTED"}
                        except (ValueError, AttributeError, TypeError):
                            pass
                    if conflict:
                        continue
                    self._response(saved)
                with self._lock:
                    self._ensure_uid(uid)
                    latest = self._read(uid)
                    latest["companion"] = merge_companions(latest["companion"], merged)
                    latest["dirty"] = latest["companion"] != merged
                    latest["lastSynced"] = _now()
                    self._write(uid, latest)
                    self._errors.pop(uid, None)
                return self.status()
            raise CloudSyncError("Another device is updating your account. Please sync again shortly.", "conflict")
        except (CloudSyncError, AccountError) as exc:
            with self._lock:
                self._errors[uid] = str(exc)
            raise
        finally:
            with self._lock:
                self._syncing_uid = None
            self._sync_lock.release()
