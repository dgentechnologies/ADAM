import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  readLocalData,
  updateLocalData,
  saveFact,
  deleteFact,
  restoreLocalData,
  LocalData,
} from '../apps/web/src/lib/local-data';
import { validateHomeUrl } from '../apps/web/src/lib/home-assistant';
const values = new Map<string, string>();
const events = new EventTarget();
Object.defineProperty(globalThis, 'window', {
  value: {
    localStorage: {
      getItem: (k: string) => values.get(k) ?? null,
      setItem: (k: string, v: string) => values.set(k, v),
      removeItem: (k: string) => values.delete(k),
    },
    dispatchEvent: events.dispatchEvent.bind(events),
  },
  configurable: true,
});
test('first run starts without invented memories or device identity', async () => {
  values.clear();
  const data = await readLocalData();
  assert.equal(data.facts.length, 0);
  assert.equal(data.onboardingComplete, false);
  assert.equal(data.name, '');
});
test('concurrent memory writes persist both updates', async () => {
  values.clear();
  await Promise.all([
    saveFact({ title: 'Coffee', text: 'No sugar', kind: 'fact' }),
    saveFact({ title: 'A friend', text: 'Enjoys painting', kind: 'person' }),
  ]);
  const data = await readLocalData();
  assert.equal(data.facts.length, 2);
  assert.deepEqual(new Set(data.facts.map((f) => f.title)), new Set(['Coffee', 'A friend']));
});
test('editing preserves identity and creation time; deletion persists', async () => {
  const item = (await readLocalData()).facts[0]!;
  await saveFact({ ...item, title: 'Updated', text: 'Updated details' });
  let data = await readLocalData();
  const edited = data.facts.find((f) => f.id === item.id)!;
  assert.equal(edited.createdAt, item.createdAt);
  assert.equal(edited.text, 'Updated details');
  await deleteFact(item.id);
  data = await readLocalData();
  assert.equal(
    data.facts.some((f) => f.id === item.id),
    false,
  );
});
test('corrupt storage is reported and never silently overwritten', async () => {
  values.set('adam.companion.v1', '{broken');
  await assert.rejects(() => readLocalData());
  await assert.rejects(() => updateLocalData((d) => ({ ...d, name: 'New' })));
  assert.equal(values.get('adam.companion.v1'), '{broken');
  values.clear();
  await updateLocalData((d) => ({ ...d, name: 'Recovered' }));
  assert.equal((await readLocalData()).name, 'Recovered');
});
test('invalid user data cannot replace a valid saved record', async () => {
  await assert.rejects(() => saveFact({ title: ' ', text: 'details', kind: 'fact' }));
  assert.equal((await readLocalData()).name, 'Recovered');
});
test('Home Assistant permits explicit local networks and public HTTPS', () => {
  for (const url of [
    'http://homeassistant.local:8123',
    'http://192.168.1.2:8123',
    'http://10.0.0.2:8123',
    'http://172.20.1.2:8123',
    'https://home.example.com',
  ])
    assert.ok(validateHomeUrl(url));
});
test('tokens cannot be sent to public cleartext or credential-bearing URLs', () => {
  for (const url of [
    'http://example.com',
    'http://10.evil.example',
    'http://192.168.evil.example',
    'http://172.32.1.2',
    'https://user:pass@example.com',
    'https://example.com?token=x',
    'javascript:alert(1)',
  ])
    assert.throws(() => validateHomeUrl(url));
});

test('confirmed backup restore recovers damaged storage without importing a token destination', async () => {
  values.set('adam.companion.v1', '{broken');
  await restoreLocalData(
    LocalData.parse({
      version: 1,
      name: 'Restored',
      homeAssistantUrl: 'https://different.example',
    }),
  );
  assert.equal((await readLocalData()).name, 'Restored');
  assert.equal((await readLocalData()).homeAssistantUrl, '');
  await updateLocalData((d) => ({ ...d, homeAssistantUrl: 'https://current.example' }));
  await restoreLocalData(
    LocalData.parse({ version: 1, name: 'Again', homeAssistantUrl: 'https://different.example' }),
  );
  assert.equal((await readLocalData()).homeAssistantUrl, 'https://current.example');
});
