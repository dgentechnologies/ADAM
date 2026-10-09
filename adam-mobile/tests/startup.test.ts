import assert from 'node:assert/strict';
import { test } from 'node:test';
import { setupStorage } from '../apps/web/src/lib/setup-storage';
import { readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';

test('corrupt setup JSON is backed up before recovering startup', async () => {
  const files = new Map([['setup', '{broken']]);
  const storage = setupStorage({ getItem: (key) => files.get(key) ?? null, setItem: (key, value) => { files.set(key, value); }, removeItem: () => {} });
  assert.equal(await storage.getItem('setup'), null);
  assert.equal(files.get('setup'), '{broken');
  assert.ok([...files].some(([key, value]) => key.startsWith('setup.recovery.') && value === '{broken'));
});
test('failed backup or unavailable storage never discards setup', async () => {
  const storage = setupStorage({ getItem: () => '{broken', setItem: () => { throw new Error('full'); }, removeItem: () => {} });
  await assert.rejects(() => Promise.resolve(storage.getItem('setup')), /full/);
});
test('a stalled native setup read returns an actionable failure and can be retried', async () => {
  let slow = true;
  const storage = setupStorage({ getItem: () => slow ? new Promise<string>(() => {}) : '{"saved":true}', setItem: () => {}, removeItem: () => {} }, 10);
  await assert.rejects(() => Promise.resolve(storage.getItem('setup')), /too long/);
  slow = false;
  assert.equal(await storage.getItem('setup'), '{"saved":true}');
});
test('every static screen link points to an existing exported route', () => {
  const root = path.resolve(__dirname, '../apps/web/src');
  const files = (dir: string): string[] => readdirSync(dir, { withFileTypes: true }).flatMap((entry) => entry.isDirectory() ? files(path.join(dir, entry.name)) : [path.join(dir, entry.name)]);
  const routes = new Set(files(path.join(root, 'app')).filter((file) => file.endsWith('page.tsx')).map((file) => '/' + path.relative(path.join(root, 'app'), path.dirname(file)).split(path.sep).filter((part) => part && !part.startsWith('(')).join('/')));
  const missing: string[] = [];
  for (const file of files(root).filter((file) => file.endsWith('.tsx'))) {
    for (const match of readFileSync(file, 'utf8').matchAll(/(?:href=["']|(?:push|replace)\(["'])(\/[a-z0-9/-]*)/g)) {
      const route = match[1]!.replace(/\/+$/, '') || '/';
      if (!routes.has(route)) missing.push(`${path.relative(root, file)}: ${route}`);
    }
  }
  assert.deepEqual(missing, []);
});
