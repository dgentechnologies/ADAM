import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import test from 'node:test';
import { readHome, savedHomeToken, setHomePower } from '../apps/web/src/lib/home-assistant';
import { clearSecrets, setSecret } from '../apps/web/src/lib/native/secure-storage';

test('Home Assistant reads real HTTP state and posts the correct service command', async () => {
  let state = 'off';
  const requests: string[] = [];
  const server = createServer(async (request, response) => {
    assert.equal(request.headers.authorization, 'Bearer test-token');
    response.setHeader('Content-Type', 'application/json');
    requests.push(`${request.method} ${request.url}`);
    if (request.method === 'POST') {
      let body = '';
      for await (const chunk of request) body += chunk;
      assert.deepEqual(JSON.parse(body), { entity_id: 'light.desk' });
      state = 'on';
    }
    response.end(
      JSON.stringify([{ entity_id: 'light.desk', state, attributes: { friendly_name: 'Desk' } }]),
    );
  });
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address() as { port: number };
  const url = `http://127.0.0.1:${address.port}`;
  try {
    await setSecret('home-assistant', JSON.stringify({ url, token: 'test-token' }));
    const [light] = await readHome(url);
    assert.equal(light?.state, 'off');
    await setHomePower(url, light!);
    assert.equal((await readHome(url))[0]?.state, 'on');
    assert.deepEqual(requests, [
      'GET /api/states',
      'POST /api/services/light/turn_on',
      'GET /api/states',
    ]);
  } finally {
    await clearSecrets();
    server.closeAllConnections();
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});

test('saved Home Assistant credentials cannot be reused for a different server', async () => {
  await setSecret(
    'home-assistant',
    JSON.stringify({ url: 'https://home.example', token: 'test-token' }),
  );
  try {
    assert.equal(await savedHomeToken('https://home.example/'), 'test-token');
    assert.equal(await savedHomeToken('https://other.example'), null);
    await assert.rejects(readHome('https://other.example'), /access token/);
  } finally {
    await clearSecrets();
  }
});

test('unavailable and read-only entities never dispatch commands', async () => {
  await assert.rejects(
    setHomePower('http://localhost:8123', {
      entity_id: 'light.desk',
      state: 'unavailable',
      attributes: {},
    }),
    /unavailable/,
  );
  await assert.rejects(
    setHomePower('http://localhost:8123', {
      entity_id: 'sensor.temperature',
      state: '24',
      attributes: {},
    }),
    /read-only/,
  );
});
