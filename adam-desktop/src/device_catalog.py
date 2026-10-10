"""Read the signed-in user's provisioned devices using Firebase owner rules."""
from urllib.parse import quote
import requests
from account import FIREBASE_PROJECT_ID


def list_devices(account, session=None):
    user = account.user
    if not user:
        return []
    uid = user["uid"]
    client = session or requests.Session()
    query = {"structuredQuery": {"from": [{"collectionId": "devices"}],
        "where": {"fieldFilter": {"field": {"fieldPath": "ownerUid"}, "op": "EQUAL", "value": {"stringValue": uid}}},
        "limit": 50}}
    try:
        token = account.id_token()
        current_user = account.user
        if not current_user or current_user["uid"] != uid:
            raise RuntimeError("The account changed. Refresh your devices.")
        response = client.post(f"https://firestore.googleapis.com/v1/projects/{quote(FIREBASE_PROJECT_ID)}/databases/(default)/documents:runQuery",
                               headers={"Authorization": "Bearer " + token}, json=query,
                               timeout=(5, 15), allow_redirects=False)
        if not response.ok:
            raise RuntimeError("Your account's devices could not be loaded. Check Firebase access and try again.")
        rows = response.json()
        if not isinstance(rows, list):
            raise ValueError()
    except (requests.RequestException, ValueError) as error:
        raise RuntimeError("Your account's devices could not be reached. Check your internet connection and try again.") from error
    current_user = account.user
    if not current_user or current_user["uid"] != uid:
        raise RuntimeError("The account changed. Refresh your devices.")
    result = []
    for row in rows[:50]:
        document = row.get("document", {}) if isinstance(row, dict) else {}
        if not isinstance(document, dict):
            continue
        fields = document.get("fields", {})
        if not isinstance(fields, dict):
            continue
        def string_field(key):
            field = fields.get(key)
            value = field.get("stringValue") if isinstance(field, dict) else None
            return value[:256] if isinstance(value, str) else ""
        if (string_field("ownerUid") != uid or fields.get('simulated', {}).get('booleanValue') is True
                or fields.get('deleted', {}).get('booleanValue') is True
                or string_field('deviceId').startswith('ADAM-SIM-')):
            continue
        item = {key: string_field(key) for key in
                ("deviceId", "name", "hardwareSerial", "tailscaleIp", "osVersion", "status")}
        document_name = document.get("name", "")
        item["deviceId"] = item["deviceId"] or (document_name.rsplit("/", 1)[-1][:256]
            if isinstance(document_name, str) else "")
        result.append(item)
    return result
