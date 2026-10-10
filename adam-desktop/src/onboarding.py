"""Account-bound onboarding. Discovery is a hint; the TLS pin and Pi grant are proof."""
from __future__ import annotations
import hashlib
import http.client
import json
import re
import secrets
import socket
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import secure_store
from device_catalog import list_devices


def certificate_at(host, port):
    # Untrusted discovery only. Never send credentials on this connection.
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    with socket.create_connection((host, port), timeout=3) as raw:
        with context.wrap_socket(raw, server_hostname=host) as stream:
            return ssl.DER_cert_to_PEM_cert(stream.getpeercert(binary_form=True))


def fingerprint(pem):
    return hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem)).hexdigest()


def tls_context(pem):
    context = ssl.create_default_context(cadata=pem)
    # The exact self-signed robot certificate is the trust anchor. DHCP changes
    # its address; the certificate pin, rather than the address, is identity.
    context.check_hostname = False
    return context


def robot_request(host, port, pem, path, body=None, token=''):
    client = http.client.HTTPSConnection(host, port, timeout=5, context=tls_context(pem))
    try:
        headers = {'Accept': 'application/json'}
        if token:
            headers['X-ADAM-Token'] = token
        if body is not None:
            headers['Content-Type'] = 'application/json'
        client.request('GET' if body is None else 'POST', path,
                       json.dumps(body) if body is not None else None, headers)
        response = client.getresponse()
        data = response.read(1_000_001)
        if len(data) > 1_000_000:
            raise ValueError('ADAM returned too much data.')
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise ValueError('ADAM returned an invalid response.')
        if response.status != 200 or payload.get('ok') is False:
            raise ValueError('ADAM refused authorization. Check the pairing code or authorize this computer again.')
        return payload
    except (OSError, ValueError, http.client.HTTPException) as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError('ADAM is unavailable. Check that both devices are on the same network.') from None
    finally:
        client.close()


class OnboardingService:
    def __init__(self, account, connection, discover, catalog=list_devices, store=secure_store):
        self.account, self.connection, self.discover = account, connection, discover
        self.catalog, self.store = catalog, store
        self.lock = threading.RLock()
        self.epoch = 0
        self.uid = ''
        self.auth_epoch = getattr(account, '_epoch', 0)
        self.devices = []
        self.targets = {}
        self.active = None
        self.catalog_at = 0
        self.error = ''

    def _user(self):
        return (self.account.user or {}).get('uid', '')

    def invalidate(self):
        with self.lock:
            self.epoch += 1
            self.uid = self._user()
            self.auth_epoch = getattr(self.account, '_epoch', 0)
            self.devices, self.targets, self.active = [], {}, None
            self.catalog_at = 0
            self.connection.stop()

    def _fence(self, uid, epoch):
        if not uid or self._user() != uid or self.epoch != epoch or self.auth_epoch != getattr(self.account, '_epoch', 0):
            raise ValueError('The account or connection changed. Please try again.')

    def _key(self, uid, device):
        return 'robot_' + hashlib.sha256((uid + '\0' + device).encode()).hexdigest()

    def _grant(self, uid, device):
        raw = self.store.load_secret(self._key(uid, device))
        if not raw:
            return None
        try:
            grant = json.loads(raw)
            if grant['uid'] == uid and grant['deviceId'] == device and grant['expiresAt'] > time.time():
                return grant
        except (ValueError, KeyError, TypeError):
            pass
        return None

    def status(self):
        with self.lock:
            uid = self._user()
            if uid != self.uid or self.auth_epoch != getattr(self.account, '_epoch', 0):
                self.invalidate()
            state = self.connection.status()
            ready = bool(uid and self.active and time.monotonic() - self.catalog_at < 60
                         and state.get('data_connected') and state.get('telemetry_connected')
                         and not state.get('read_only') and state.get('secure_authorized')
                         and state.get('authorized_uid') == uid and state.get('authorized_device') == self.active)
            return {'stage': 'ready' if ready else 'devices' if uid else 'login',
                    'ready': ready, 'devices': self.devices, 'error': self.error,
                    'selected': self.active, 'connection': state}

    def refresh(self):
        with self.lock:
            uid = self._user()
            if uid != self.uid or self.auth_epoch != getattr(self.account, '_epoch', 0):
                self.invalidate()
            epoch = self.epoch
        if not uid:
            return self.status()
        try:
            self.account.id_token()  # Restored sessions must refresh successfully.
            owned = self.catalog(self.account)
            units = self.discover()
            targets, visible = {}, []
            def inspect(unit):
                try:
                    # Only literal LAN addresses supplied by zeroconf are used.
                    import ipaddress
                    address = ipaddress.ip_address(unit['host'])
                    if not address.is_private or address.is_loopback or address.is_unspecified or address.is_multicast:
                        return None
                    port = int(unit.get('port', 8766))
                    pem = certificate_at(str(address), port)
                    info = robot_request(str(address), port, pem, '/api/pair/info')
                    return {**unit, 'pem': pem, 'info': info}
                except (OSError, ValueError, KeyError, ssl.SSLError):
                    return None
            eligible = [u for u in units[:50] if any(d['deviceId'] == u.get('id') for d in owned)]
            with ThreadPoolExecutor(max_workers=8) as pool:
                found = [item for item in pool.map(inspect, eligible) if item]
            for device in owned:
                device_id = device['deviceId']
                if device_id.startswith('ADAM-SIM-') or device.get('simulated'):
                    continue
                matches = [u for u in found if u['info'].get('id') == device_id
                           and u['info'].get('pairingVersion') == 2
                           and device.get('hardwareSerial') == u['info'].get('hardwareId')]
                grant = self._grant(uid, device_id)
                if grant:
                    matches = [u for u in matches if fingerprint(u['pem']) == grant.get('fingerprint')
                               and u['info'].get('hardwareId') == grant.get('hardwareId')]
                state = 'registration_required' if any(u['info'].get('id') == device_id for u in found) and not matches else 'offline'
                if len(matches) > 1:
                    state = 'ambiguous'
                elif len(matches) == 1:
                    targets[device_id] = matches[0]
                    state = 'available' if grant else 'authorization_required'
                visible.append({'deviceId': device_id, 'name': device.get('name') or 'ADAM', 'state': state,
                                'savedPairing': bool(grant)})
            with self.lock:
                self._fence(uid, epoch)
                if self.active and self.active not in targets:
                    self.active = None
                    self.connection.stop()
                self.devices, self.targets, self.error = visible, targets, ''
                self.catalog_at = time.monotonic()
        except Exception:
            with self.lock:
                if self._user() == uid and self.epoch == epoch:
                    self.error = 'Your devices could not be verified. Check your internet and local network, then retry.'
                    self.catalog_at = 0
                    self.active = None
                    self.targets = {}
                    self.connection.stop()
            raise
        return self.status()

    def connect(self, device_id, code=''):
        # Always re-check cloud ownership and rediscover before authorization.
        self.refresh()
        with self.lock:
            uid, epoch = self._user(), self.epoch
            self._fence(uid, epoch)
            target = self.targets.get(device_id)
            if not target:
                raise ValueError('This ADAM is not uniquely available in your account on this network.')
            grant = self._grant(uid, device_id)
        pem, info = target['pem'], target['info']
        if grant is None:
            # First half authenticates the robot's TLS certificate out of band;
            # second half is a one-time possession secret, sent only after pinning.
            code = re.sub(r'[\s-]', '', str(code)).lower()
            if not re.fullmatch(r'[a-f0-9]{12}[0-9]{6}', code):
                raise ValueError('Enter the complete pairing code shown on your ADAM.')
            if not secrets.compare_digest(code[:12], fingerprint(pem)[:12]):
                raise ValueError('The pairing code belongs to a different ADAM. Connection stopped.')
            client_id = secrets.token_hex(16)
            result = robot_request(target['host'], target['port'], pem, '/api/pair/desktop',
                                   {'uid': uid, 'clientId': client_id, 'code': code[12:]})
            grant = {**result, 'uid': uid, 'deviceId': device_id, 'certificate': pem,
                     'fingerprint': fingerprint(pem), 'clientId': client_id}
        try:
            verified = robot_request(target['host'], target['port'], pem, '/api/pair/session', token=grant['token'])
        except ValueError:
            self.store.delete_secret(self._key(uid, device_id))
            raise ValueError('Authorization expired or could not be verified. Refresh and authorize this computer again.') from None
        if (verified.get('uid') != uid or verified.get('id') != device_id
                or verified.get('hardwareId') != grant.get('hardwareId')
                or verified.get('hardwareId') != info.get('hardwareId')
                or verified.get('clientId') != grant.get('clientId')):
            raise ValueError('ADAM identity or account authorization did not match. Connection stopped.')
        with self.lock:
            self._fence(uid, epoch)
            self.store.save_secret(self._key(uid, device_id), json.dumps(grant))
            result = self.connection.connect({'host': target['host'], 'sync_port': target['port'],
                'ws_port': info.get('wsPort', 8765), 'sync_token': grant['token'],
                'certificate': pem, 'expected_uid': uid, 'expected_device': device_id,
                'expected_hardware': grant['hardwareId'], 'expected_client': grant['clientId']})
            if not result.get('ok'):
                raise ValueError(result.get('reason') or 'The connection could not be verified.')
            self.active = device_id
        return self.status()

    def cancel(self):
        with self.lock:
            self.epoch += 1
            self.active = None
            self.connection.stop()
        return self.status()
