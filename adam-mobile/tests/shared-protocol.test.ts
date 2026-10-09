import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { validateCompanion, mergeCompanions, capturePhoneChanges, emptyCompanion, MobileCompanionSync } from '../apps/web/src/lib/firebase/companion-sync';
import { LocalData } from '../apps/web/src/lib/local-data';
const root = path.resolve(__dirname, '../..');
const fixture = validateCompanion(JSON.parse(readFileSync(path.join(root, 'shared/companion-v2.fixture.json'), 'utf8')));
const ids = { todo: Object.keys(fixture.todos)[0]!, clock: Object.keys(fixture.clocks)[0]!, device: Object.keys(fixture.devices)[0]! };
const later = '2026-10-09T09:00:00.000Z';
test('desktop and Android normalize and merge every shared record identically', () => {
  const changed = structuredClone(fixture);
  changed.todos[ids.todo] = { id: ids.todo, updatedAt: later, deleted: true };
  changed.devices[ids.device] = { ...changed.devices[ids.device]!, name: 'Renamed ADAM 🦾', updatedAt: later } as any;
  const script = 'import json,sys; from cloud_sync import validate_companion,merge_companions; a,b=json.load(sys.stdin); print(json.dumps([validate_companion(a),merge_companions(a,b)],ensure_ascii=False))';
  const process = spawnSync('python', ['-c', script], { input: JSON.stringify([fixture, changed]), encoding: 'utf8', env: { ...globalThis.process.env, PYTHONPATH: path.join(root, 'adam-desktop/src'), PYTHONIOENCODING: 'utf-8' } });
  assert.equal(process.status, 0, process.stderr);
  assert.deepEqual(JSON.parse(process.stdout), [fixture, mergeCompanions(fixture, changed)]);
});
test('legacy memories migrate without loss and old clients cannot silently drop new collections', () => {
  const migrated = validateCompanion({ schemaVersion: 1, memories: fixture.memories, preferences: fixture.preferences });
  assert.equal(migrated.schemaVersion, 2);assert.deepEqual(migrated.memories, fixture.memories);assert.deepEqual(migrated.devices, {});
  assert.throws(() => validateCompanion({ schemaVersion: 3 }));
});
test('phone edits and removals across planner and devices keep tombstones', () => {
  const live = (items: any) => Object.values(items).map(({ deleted, ...item }: any) => item);
  const before = LocalData.parse({ version: 1, todos: live(fixture.todos), clocks: live(fixture.clocks), devices: live(fixture.devices) });
  const after = LocalData.parse({ ...before, todos: [], clocks: before.clocks.map((clock) => ({ ...clock, enabled: false })), devices: before.devices.map((device) => device.id === ids.device ? { ...device, name: 'Kitchen ADAM' } : device) });
  const changes = capturePhoneChanges(fixture, before, after, later);
  assert.equal(changes.todos[ids.todo]?.deleted, true);
  assert.equal((changes.clocks[ids.clock] as any).enabled, false);
  assert.equal((changes.devices[ids.device] as any).name, 'Kitchen ADAM');
  assert.equal(Object.keys(changes.devices).length, 2);
  assert.equal(mergeCompanions(changes, fixture).todos[ids.todo]?.deleted, true);
});
test('cloud sync downloads multiple devices and planner, uploads renames and deletions, and survives restart', async () => {
  const files = new Map<string, string>();let phone = LocalData.parse({ version: 1 });let remote = structuredClone(fixture);
  const deps = { getUid: () => 'account-one', getItem: async (key: string) => files.get(key) ?? null, setItem: async (key: string, value: string) => { files.set(key, value); }, readLocal: async () => phone, updateLocal: async (update: any) => phone = LocalData.parse(update(phone)), transact: async (_uid: string, merge: any) => remote = merge(remote), now: () => later };
  const service = new MobileCompanionSync(deps);await service.enable();await service.sync();
  assert.equal(phone.devices.length, 2);assert.equal(phone.todos.length, 1);assert.equal(phone.clocks.length, 1);
  phone = LocalData.parse({ ...phone, todos: [], devices: phone.devices.map((device) => device.id === ids.device ? { ...device, name: 'Library ADAM' } : device) });
  await new MobileCompanionSync(deps).sync();
  assert.equal(remote.todos[ids.todo]?.deleted, true);assert.equal((remote.devices[ids.device] as any).name, 'Library ADAM');
  await service.disable();await service.enable();await service.sync();assert.equal(remote.todos[ids.todo]?.deleted, true);
});
