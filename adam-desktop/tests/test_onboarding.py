"""Account, possession, TLS-pin and cancellation boundaries for onboarding."""
import importlib.util
import json
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest
from onboarding import OnboardingService, certificate_at, fingerprint, robot_request
from secure_store import SecretStore
from test_identity_sync import FakeCipher

@pytest.fixture
def robot(tmp_path):
    path = Path(__file__).parents[2] / 'MP-MC codes/pi/adam/desktop_pairing.py'
    spec = importlib.util.spec_from_file_location('pairing_fixture', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.ROOT = tmp_path / 'robot'; module.provision('alice', 'ADAM-TEST')
    return module

def window(robot):
    import hashlib
    robot.write('window.json', {'expiresAt': time.time() + 300, 'attempts': 0,
                               'hash': hashlib.sha256(b'123456').hexdigest()})

def test_pairing_requires_window_owner_single_use_code(robot):
    body = {'uid': 'alice', 'clientId': 'a' * 32, 'code': '123456'}
    with pytest.raises(ValueError): robot.claim(body)
    window(robot)
    with pytest.raises(ValueError): robot.claim({**body, 'uid': 'bob'})
    with pytest.raises(ValueError): robot.claim({**body, 'code': '000000'})
    grant = robot.claim(body)
    assert robot.authenticate(grant['token'])['uid'] == 'alice'
    assert grant['token'] not in (robot.ROOT / 'grants.json').read_text()
    with pytest.raises(ValueError): robot.claim(body)
    robot.write('grants.json', {})
    assert robot.authenticate(grant['token']) is None

def test_guess_limit_transfer_and_distinct_desktops(robot):
    window(robot)
    for _ in range(5):
        with pytest.raises(ValueError): robot.claim({'uid': 'alice', 'code': 'bad', 'clientId': 'a' * 32})
    with pytest.raises(ValueError): robot.claim({'uid': 'alice', 'code': '123456', 'clientId': 'a' * 32})
    window(robot); first = robot.claim({'uid': 'alice', 'code': '123456', 'clientId': 'a' * 32})
    window(robot); second = robot.claim({'uid': 'alice', 'code': '123456', 'clientId': 'b' * 32})
    assert first['token'] != second['token']
    robot.write('identity.json', {**robot.identity(), 'uid': 'bob'})
    assert robot.authenticate(first['token']) is None
    assert robot.authenticate(second['token']) is None

def test_real_tls_pinning_and_authorized_request(robot, tmp_path):
    window(robot); grant = robot.claim({'uid': 'alice', 'code': '123456', 'clientId': 'a' * 32})
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            auth = robot.authenticate(self.headers.get('X-ADAM-Token', ''))
            payload = json.dumps({'ok': bool(auth), **(auth or {})}).encode()
            self.send_response(200 if auth else 403)
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers(); self.wfile.write(payload)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.socket = robot.ssl_context().wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        pem = certificate_at('127.0.0.1', server.server_port)
        assert fingerprint(pem) == fingerprint((robot.ROOT / 'certificate.pem').read_text())
        assert robot_request('127.0.0.1', server.server_port, pem, '/api/pair/session', token=grant['token'])['uid'] == 'alice'
        with pytest.raises(ValueError): robot_request('127.0.0.1', server.server_port, pem, '/api/pair/session', token='bad')
        robot.ROOT = tmp_path / 'impostor'; robot.provision('alice', 'ADAM-TEST')
        with pytest.raises(ValueError): robot_request('127.0.0.1', server.server_port, (robot.ROOT / 'certificate.pem').read_text(), '/api/pair/session', token=grant['token'])
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)

@pytest.fixture
def service(tmp_path):
    account = SimpleNamespace(user={'uid': 'alice'}, id_token=Mock(return_value='firebase-fixture'))
    connection = Mock()
    connection.status.return_value = {'data_connected': True, 'telemetry_connected': True, 'read_only': False,
        'secure_authorized': True, 'authorized_uid': 'alice', 'authorized_device': 'ADAM-TEST'}
    return OnboardingService(account, connection, Mock(return_value=[]),
        catalog=Mock(return_value=[{'deviceId': 'ADAM-TEST', 'name': 'Desk'}]),
        store=SecretStore(tmp_path / 'vault', cipher=FakeCipher()))

def test_dashboard_never_unlocks_from_connection_alone(service):
    assert not service.status()['ready']
    service.active = 'ADAM-TEST'; service.catalog_at = time.monotonic()
    assert service.status()['ready']
    service.connection.status.return_value['telemetry_connected'] = False
    assert not service.status()['ready']
    service.connection.status.return_value['telemetry_connected'] = True
    service.catalog_at -= 61
    assert not service.status()['ready']
    service.account.user = None
    assert service.status()['stage'] == 'login'
    service.connection.stop.assert_called()

def test_unowned_offline_or_simulated_devices_cannot_connect(service):
    assert service.refresh()['devices'][0]['state'] == 'offline'
    with pytest.raises(ValueError): service.connect('ADAM-UNOWNED')
    service.catalog.return_value = [{'deviceId': 'ADAM-SIM-fixture'}]
    assert service.refresh()['devices'] == []

def test_catalog_failure_and_account_change_fence_connection(service):
    service.refresh();service.active = 'ADAM-TEST'
    service.catalog.side_effect = RuntimeError('offline')
    with pytest.raises(RuntimeError): service.refresh()
    assert not service.status()['ready']
    service.catalog.side_effect = lambda account: setattr(account, 'user', {'uid': 'bob'}) or []
    with pytest.raises(ValueError): service.refresh()
    assert service.status()['stage'] == 'devices'

def test_cancel_fences_inflight_result_and_vault_is_account_bound(service):
    service.status();epoch=service.epoch;service.cancel()
    with pytest.raises(ValueError): service._fence('alice', epoch)
    assert service._key('alice', 'ADAM-TEST') != service._key('bob', 'ADAM-TEST')

def test_first_pair_pin_mismatch_never_sends_code(service, robot):
    service.status();pem = (robot.ROOT / 'certificate.pem').read_text()
    service.targets = {'ADAM-TEST': {'host': '192.168.1.2', 'port': 8766, 'pem': pem, 'info': robot.public_info()}}
    with patch.object(service, 'refresh'), patch('onboarding.robot_request') as request:
        with pytest.raises(ValueError, match='different ADAM'): service.connect('ADAM-TEST', '0' * 18)
        request.assert_not_called()

def test_backend_requires_login_even_with_local_session():
    from test_config import isolated_config, APP
    with isolated_config():
        spec = importlib.util.spec_from_file_location('onboarding_backend_fixture', APP / 'backend.py')
        backend = importlib.util.module_from_spec(spec); spec.loader.exec_module(backend)
        client = backend.app.test_client();headers = {'X-ADAM-Session': backend.DESKTOP_SESSION}
        for path in ['/settings', '/status', '/pi/snapshot', '/memories']:
            assert client.get(path, base_url='http://127.0.0.1:8642', headers=headers).status_code == 403
        assert client.get('/onboarding/status', base_url='http://127.0.0.1:8642', headers=headers).json['stage'] == 'login'

def test_real_wss_requires_grant_and_emits_bound_identity(robot, monkeypatch):
    import asyncio
    import ssl
    import sys
    from websockets.legacy.server import serve
    from websockets.legacy.client import connect
    from websockets.exceptions import ConnectionClosedError
    window(robot)
    grant=robot.claim({'uid':'alice','clientId':'a'*32,'code':'123456'})
    monkeypatch.setitem(sys.modules,'desktop_pairing',robot)
    monkeypatch.setitem(sys.modules,'config',SimpleNamespace(WS_HOST='127.0.0.1',WS_PORT=0))
    spec=importlib.util.spec_from_file_location('wss_fixture',Path(__file__).parents[2]/'MP-MC codes/pi/adam/ws_server.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    async def check():
        context=ssl.create_default_context(cadata=(robot.ROOT/'certificate.pem').read_text())
        context.check_hostname=False
        async with serve(module.ws_handler,'127.0.0.1',0,ssl=robot.ssl_context()) as server:
            url=f'wss://127.0.0.1:{server.sockets[0].getsockname()[1]}'
            async with connect(url,ssl=context,extra_headers={'X-ADAM-Token':'invalid'}) as ws:
                with pytest.raises(ConnectionClosedError):await ws.recv()
            assert not module.ws_clients
            async with connect(url,ssl=context,extra_headers={'X-ADAM-Token':grant['token']}) as ws:
                hello=json.loads(await ws.recv())
                assert hello['type']=='authorized' and hello['uid']=='alice'
                assert hello['clientId']=='a'*32
                assert hello['id']=='ADAM-TEST'
                assert hello['hardwareId']==robot.identity()['hardwareId']
                assert grant['token'] not in json.dumps(hello)
    asyncio.run(check())

def test_voice_pairing_response_never_contains_display_secret(robot):
    response=robot.open_window()
    display=robot.read('display.json')
    assert len(display['code'])==18
    assert display['expiresAt']>time.time()
    assert display['code'] not in json.dumps(response)
    assert display['code'][12:] not in json.dumps(response)
    grant=robot.claim({'uid':'alice','clientId':'a'*32,'code':display['code'][12:]})
    assert robot.authenticate(grant['token'])
    assert not robot.read('display.json')['code']
