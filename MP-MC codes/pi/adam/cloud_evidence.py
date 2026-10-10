"""Signed offline robot evidence relayed by an authenticated owner client.

The robot never stores Firebase credentials or contacts Firestore. Its public
verification key must first be registered through trusted cloud administration.
"""
import base64
import hashlib
import json
import secrets
import time
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
import desktop_pairing


def public_bundle():
    identity = desktop_pairing.identity()
    if not identity:
        raise ValueError('Provision this robot first.')
    certificate = x509.load_pem_x509_certificate((desktop_pairing.ROOT / 'certificate.pem').read_bytes())
    return {'version': 1, 'deviceId': identity['deviceId'], 'hardwareId': identity['hardwareId'],
            'ownershipEpoch': identity.get('ownershipEpoch', 0),
            'publicKeyPem': certificate.public_key().public_bytes(serialization.Encoding.PEM,
                            serialization.PublicFormat.SubjectPublicKeyInfo).decode(),
            'certificateSha256': certificate.fingerprint(hashes.SHA256()).hex()}


def sign(purpose, **fields):
    identity = desktop_pairing.identity()
    if not identity:
        raise ValueError('Provision this robot first.')
    now = int(time.time() * 1000)
    payload = {'version': 1, 'purpose': purpose, 'uid': identity['uid'],
               'deviceId': identity['deviceId'], 'hardwareId': identity['hardwareId'],
               'ownershipEpoch': identity.get('ownershipEpoch', 0), 'nonce': secrets.token_hex(16),
               'issuedAt': now, 'expiresAt': now + 300000, **fields}
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(encoded) > 32768:
        raise ValueError('Evidence is too large.')
    key = serialization.load_pem_private_key((desktop_pairing.ROOT / 'private.pem').read_bytes(), password=None)
    signature = key.sign(encoded, ec.ECDSA(hashes.SHA256()))
    return {'payload': base64.b64encode(encoded).decode(), 'signature': base64.b64encode(signature).decode()}


def execution_receipts(after=""):
    if not isinstance(after,str) or len(after)>256: raise ValueError("Invalid receipt cursor.")
    import record_sync
    import scheduler
    state = record_sync.snapshot()
    from config import SCHEDULE_FILE
    durable = json.loads(SCHEDULE_FILE.read_text())
    rows = {r['id']: r for r in durable['schedules']}
    result = []
    candidates = sorted((path,record) for path,record in state['documents'].items() if path > after and '/schedules/' in path)
    page = candidates[:8]
    for path, record in page:
        if not path.startswith(f'users/{state["uid"]}/schedules/'):
            continue
        # Only records actually received from the canonical bridge qualify.
        row = rows.get(record['scheduleId'])
        accepted = row.get('sync_canonical') if row else next((r.get('sync_canonical') for r in durable.get('tombstones', []) if r['id'] == record['scheduleId'] and r['kind'] == 'schedule'), None)
        if accepted != record:
            continue
        fired = row.get('last_fired') if row else None
        occurrence = fired
        if fired:
            fired = record_sync.stamp(fired.split('#', 1)[0], desktop_pairing.identity()['timeZone'])
        record_json = json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)
        rev = hashlib.sha256(record_json.encode()).hexdigest()
        result.append(sign('execution', path=path, scheduleId=record['scheduleId'], recordJson=record_json,
                           appliedRevision=rev, lastFiredAt=fired,
                           lastOccurrenceId=occurrence if fired else None,
                           status='deleted' if record['deleted'] else 'fired' if fired else 'applied'))
    return {'ok': True, 'deviceId': state['deviceId'], 'receipts': result, 'nextCursor': page[-1][0] if len(candidates)>8 else ''}
