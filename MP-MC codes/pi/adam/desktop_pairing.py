"""Locally provisioned identity and per-desktop TLS grants. No Firebase credentials.

Run on the robot: python desktop_pairing.py provision --uid FIREBASE_UID --device-id ADAM-XXXX
Then: python desktop_pairing.py code
The code is displayed only on the physical robot screen, expires in five minutes,
and is consumed once. Never open a pairing window just because LAN traffic asks.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import re
import secrets
import ssl
import tempfile
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parent / '.desktop-pairing'
LOCK = threading.RLock()


def read(name):
    path = ROOT / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write(name, value):
    ROOT.mkdir(mode=0o700, exist_ok=True)
    fd, path = tempfile.mkstemp(dir=ROOT)
    try:
        with os.fdopen(fd, 'w') as file:
            json.dump(value, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(path, ROOT / name)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def identity():
    return read('identity.json')


def ssl_context():
    if not identity():
        return None
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(ROOT / 'certificate.pem', ROOT / 'private.pem')
    return ctx


def public_info():
    record = identity()
    if not record:
        return {}
    return {'pairingVersion': 2, 'id': record['deviceId'], 'hardwareId': record['hardwareId']}


def authenticate(token):
    if not isinstance(token, str) or not token:
        return None
    record = identity()
    if not record:
        return None
    digest = hashlib.sha256(token.encode()).hexdigest()
    with LOCK:
        grants = read('grants.json') or {}
        grant = grants.get(digest)
    if (grant and grant['expiresAt'] > time.time() and grant['uid'] == record['uid']):
        return {**grant, 'id': record['deviceId'], 'hardwareId': record['hardwareId']}
    return None


def claim(body):
    with LOCK:
        record, window = identity(), read('window.json')
        if not record or not window or window['expiresAt'] <= time.time() or window['attempts'] >= 5:
            raise ValueError('Open a new pairing window on your ADAM.')
        window['attempts'] += 1
        write('window.json', window)
        if (body.get('uid') != record['uid'] or not isinstance(body.get('code'), str)
                or not secrets.compare_digest(hashlib.sha256(body['code'].encode()).hexdigest(), window['hash'])
                or not re.fullmatch(r'[a-f0-9]{32}', str(body.get('clientId', '')))):
            raise ValueError('Authorization refused.')
        grants = read('grants.json') or {}
        grants = {k: v for k, v in grants.items() if v['expiresAt'] > time.time()}
        if len(grants) >= 20:
            raise ValueError('Remove an old desktop authorization on ADAM first.')
        token = secrets.token_urlsafe(32)
        grant = {'uid': record['uid'], 'clientId': body['clientId'], 'scope': ['data', 'telemetry', 'laptop-pairing'], 'expiresAt': time.time() + 30 * 86400}
        grants[hashlib.sha256(token.encode()).hexdigest()] = grant
        # Consume before persisting a grant: a failed write must never leave a
        # reusable code after credentials could have been issued.
        write('window.json', {**window, 'expiresAt': 0})
        write('display.json', {'code': '', 'expiresAt': 0})
        write('grants.json', grants)
        return {'ok': True, **grant, 'token': token, 'hardwareId': record['hardwareId']}


def provision(uid, device_id):
    if not uid or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', uid) or not re.fullmatch(r'ADAM-[A-Za-z0-9-]+', device_id):
        raise ValueError('Supply the owner Firebase UID and canonical physical ADAM ID.')
    if identity():
        raise ValueError('Already provisioned. Use transfer to change owner; identity must remain stable.')
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    ROOT.mkdir(mode=0o700, exist_ok=True)
    key = ec.generate_private_key(ec.SECP256R1())
    hardware = str(uuid.uuid4())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'ADAM-' + hardware)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=3650))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .sign(key, hashes.SHA256()))
    for filename, data in [('private.pem', key.private_bytes(serialization.Encoding.PEM,
                           serialization.PrivateFormat.PKCS8, serialization.NoEncryption())),
                           ('certificate.pem', cert.public_bytes(serialization.Encoding.PEM))]:
        fd = os.open(ROOT / filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as file:
            file.write(data)
    write('identity.json', {'uid': uid, 'deviceId': device_id, 'hardwareId': hardware, 'ownershipEpoch': 0})


async def display_codes():
    """Forward only locally requested codes to the physical head, never WS."""
    import asyncio
    from esp32_link import esp_link
    last = None
    while True:
        try:
            display = read('display.json') or {}
            expires = display.get('expiresAt', 0)
            code = display.get('code', '') if expires > time.time() else ''
            if code != last:
                if re.fullmatch(r'[a-f0-9]{12}[0-9]{6}', code):
                    esp_link.send_line(f'PAIR:{code}:{max(1, int(expires-time.time()))}')
                else:
                    esp_link.send_line('PAIR:CLEAR')
                    if display.get('code'):
                        write('display.json', {'code': '', 'expiresAt': 0})
                last = code
        except (OSError, ValueError):
            pass
        await asyncio.sleep(1)


def open_window():
    if not identity():
        raise ValueError('Provision this ADAM first.')
    code = f'{secrets.randbelow(1000000):06d}'
    pem = (ROOT / 'certificate.pem').read_text()
    pin = hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem)).hexdigest()[:12]
    write('window.json', {'hash': hashlib.sha256(code.encode()).hexdigest(),
                         'expiresAt': time.time() + 300, 'attempts': 0})
    write('display.json', {'code': pin + code, 'expiresAt': time.time() + 300})
    return {'status': 'ok', 'message': 'Enter the code shown on ADAM’s screen in the desktop app within five minutes.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['provision', 'code', 'revoke-all', 'transfer', 'public-bundle', 'cloud-claim'])
    parser.add_argument('--uid')
    parser.add_argument('--device-id')
    parser.add_argument('--timezone')
    args = parser.parse_args()
    if args.action == 'provision':
        from zoneinfo import ZoneInfo
        if not args.timezone:
            raise ValueError('Supply --timezone matching the robot OS timezone, for example Asia/Kolkata.')
        ZoneInfo(args.timezone)
        provision(args.uid, args.device_id)
        write('identity.json', {**identity(), 'timeZone': args.timezone})
        print('Provisioned. Restart ADAM to enable authenticated HTTPS/WSS, then run code.')
    elif args.action in ('public-bundle', 'cloud-claim'):
        import cloud_evidence
        print(json.dumps(cloud_evidence.public_bundle() if args.action == 'public-bundle' else cloud_evidence.sign('claim')))
    elif args.action == 'code':
        if not identity():
            raise ValueError('Provision this ADAM first.')
        print(open_window()['message'])
    else:
        if args.action == 'transfer':
            raise ValueError('Ownership transfer requires an administrator-reviewed data archive and reset. Use the documented cloud transfer workflow; do not relabel private robot data.')
        record = identity()
        if not record:
            raise ValueError('Provision this ADAM first.')
        write('grants.json', {})
        write('display.json', {'code': '', 'expiresAt': 0})
        write('window.json', {'expiresAt': 0, 'attempts': 5})
        print('Desktop grants revoked. Update cloud ownership separately when transferring.')


if __name__ == '__main__':
    main()
