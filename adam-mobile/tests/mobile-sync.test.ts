import assert from 'node:assert/strict';
import { test } from 'node:test';
import { LocalData, type Fact } from '../apps/web/src/lib/local-data';
import {
  MobileCompanionSync, canonical, capturePhoneChanges, emptyCompanion, mergeCompanions,
  validateCompanion, type Companion, type SyncDependencies,
} from '../apps/web/src/lib/firebase/companion-sync';

const FIRST = '00000000-0000-0000-0000-000000000001';
const SECOND = '00000000-0000-0000-0000-000000000002';
const THIRD = '00000000-0000-0000-0000-000000000003';
const TIME = '2026-10-07T08:00:00.000Z';
const LATER = '2026-10-07T09:00:00.000Z';
const NOW = '2026-10-07T10:00:00.000Z';
const clone = <T>(value: T): T => structuredClone(value);
function fact(id = FIRST, text = 'A local memory', updatedAt = TIME): Fact {
  return { id, title: 'A memory', text, kind: 'fact', createdAt: TIME, updatedAt };
}
function shared(...facts: Fact[]): Companion {
  const value = emptyCompanion();
  for (const memory of facts) value.memories[memory.id] = { ...memory, deleted: false };
  return value;
}
function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => { resolve = done; });
  return { promise, resolve };
}
function harness() {
  let uid: string | null = 'account-a';
  let phone = LocalData.parse({ version: 1, facts: [fact()], name: 'Private name',
    homeAssistantUrl: 'http://private.local:8123', onboardingComplete: true });
  const storage = new Map<string, string>();
  const documents = new Map<string, { displayName: string; companion?: Companion }>();
  documents.set('account-a', { displayName: 'Account profile' });
  let calls = 0;
  let failed = false;
  let beforeRead: Promise<void> | undefined;
  let afterCommit: Promise<void> | undefined;
  const entered = deferred();
  const committed = deferred();
  const deps: SyncDependencies = {
    getUid: () => uid,
    getItem: async (key) => storage.get(key) ?? null,
    setItem: async (key, value) => { storage.set(key, value); },
    readLocal: async () => clone(phone),
    updateLocal: async (update) => { phone = LocalData.parse(update(clone(phone))); return clone(phone); },
    now: () => NOW,
    transact: async (account, merge) => {
      calls++;
      entered.resolve();
      if (beforeRead) await beforeRead;
      if (failed) throw new Error('network offline');
      const document = documents.get(account);
      const merged = merge(clone(document?.companion));
      documents.set(account, { ...document, displayName: document?.displayName ?? '', companion: clone(merged) });
      committed.resolve();
      if (afterCommit) await afterCommit;
      return clone(merged);
    },
  };
  const service = new MobileCompanionSync(deps);
  return {
    service, documents, storage, entered, committed,
    phone: () => clone(phone),
    edit: (update: (data: LocalData) => LocalData) => { phone = LocalData.parse(update(phone)); },
    account: (next: string | null) => { uid = next; service.accountChanged(next); },
    fail: (value: boolean) => { failed = value; },
    pauseRead: (wait: Promise<void>) => { beforeRead = wait; },
    pauseCommitResponse: (wait: Promise<void>) => { afterCommit = wait; },
    calls: () => calls,
  };
}

test('equal timestamps retain the local document and separate IDs are preserved', () => {
  const left = shared(fact(FIRST, 'left'), fact(SECOND));
  const right = shared(fact(FIRST, 'right'), fact(THIRD));
  right.memories[SECOND] = { id: SECOND, updatedAt: TIME, deleted: true };
  assert.equal((mergeCompanions(left,right).memories[FIRST] as Fact).text,'left');
  assert.equal((mergeCompanions(right,left).memories[FIRST] as Fact).text,'right');
  assert.equal(Object.keys(mergeCompanions(left, right).memories).length, 3);
  assert.equal(mergeCompanions(left, right).memories[SECOND]?.deleted, false);
  assert.equal(mergeCompanions(right, left).memories[SECOND]?.deleted, true);
});

test('equal timestamps retain local Unicode text without a lexical tiebreak', () => {
  const left = shared(fact(FIRST, '\u{10000}'));
  const right = shared(fact(FIRST, '\uE000'));
  const winner = mergeCompanions(left, right).memories[FIRST]!;
  assert.equal(winner.deleted, false);
  assert.equal((winner as Fact).text, '\u{10000}');
  assert.equal((mergeCompanions(right,left).memories[FIRST] as Fact).text,'\uE000');
});

test('unsupported, malformed, oversized and invalid Unicode cloud data are rejected', () => {
  assert.throws(() => validateCompanion({ schemaVersion: 3 }));
  assert.throws(() => validateCompanion({ schemaVersion: 1, memories: null }));
  assert.throws(() => validateCompanion(shared({ ...fact(), updatedAt: '2026-02-30T00:00:00.000Z' })));
  assert.throws(() => validateCompanion(shared({ ...fact(), text: '\ud800' })));
  assert.throws(() => validateCompanion(shared({ ...fact(), id: 'invalid-id' })));
  const tooLarge = emptyCompanion();
  for (let i = 0; i < 400; i++) {
    const id = `00000000-0000-0000-0000-${String(i).padStart(12, '0')}`;
    tooLarge.memories[id] = { ...fact(id, 'x'.repeat(2000)), deleted: false };
  }
  assert.throws(() => validateCompanion(tooLarge), /full/);
});

test('capture advances local edits past future records and excludes private phone data', () => {
  const future = '2027-10-07T10:00:00.000Z';
  const base = shared(fact(FIRST, 'old', future));
  const baseline = LocalData.parse({ version: 1, facts: [fact(FIRST, 'old', future)] });
  const current = { ...baseline, facts: [fact(FIRST, 'edited', TIME)] };
  const next = capturePhoneChanges(base, baseline, current, NOW);
  assert.equal(next.memories[FIRST]?.updatedAt, '2027-10-07T10:00:00.001Z');
  assert.deepEqual(Object.keys(next).sort(), ['clocks', 'devices', 'memories', 'preferences', 'schemaVersion', 'todos']);
  assert.deepEqual(next.preferences, {});
});

test('sync stays opt-in and confirmation alone never sends phone data', async () => {
  const app = harness();
  assert.equal((await app.service.status()).enabled, false);
  await assert.rejects(app.service.sync(), /Enable sync/);
  assert.equal(app.calls(), 0);
  await app.service.enable('account-a');
  assert.equal((await app.service.status()).enabled, true);
  assert.equal(app.calls(), 0);
  await app.service.sync('account-a');
  assert.equal(app.calls(), 1);
  const document = app.documents.get('account-a')!;
  assert.equal(document.displayName, 'Account profile');
  assert.equal(document.companion?.memories[FIRST]?.deleted, false);
  assert.equal(canonical(document.companion).includes('private.local'), false);
  assert.equal(canonical(document.companion).includes('Private name'), false);
  assert.equal((await app.service.status()).pending, false);
});

test('remote additions and preferences appear while private phone settings stay intact', async () => {
  const app = harness();
  const remote = shared(fact(SECOND, 'From desktop', LATER));
  remote.preferences = { voice: { value: 'Kore', updatedAt: LATER } };
  app.documents.set('account-a', { displayName: 'Account profile', companion: remote });
  await app.service.enable();
  await app.service.sync();
  assert.deepEqual(new Set(app.phone().facts.map((f) => f.id)), new Set([FIRST, SECOND]));
  assert.equal(app.phone().voice, 'Kore');
  assert.equal(app.phone().name, 'Private name');
  assert.equal(app.phone().homeAssistantUrl, 'http://private.local:8123');
});

test('offline deletions persist as tombstones and do not resurrect on retry', async () => {
  const app = harness();
  await app.service.enable();
  await app.service.sync();
  app.edit((data) => ({ ...data, facts: [] }));
  app.fail(true);
  await assert.rejects(app.service.sync(), /could not finish/);
  assert.equal(app.phone().facts.length, 0);
  assert.equal((await app.service.status()).pending, true);
  assert.equal(app.documents.get('account-a')?.companion?.memories[FIRST]?.deleted, false);
  app.fail(false);
  await app.service.sync();
  assert.equal(app.documents.get('account-a')?.companion?.memories[FIRST]?.deleted, true);
  assert.equal(app.phone().facts.length, 0);
});

test('edits during an in-flight transaction remain local and pending for the next sync', async () => {
  const app = harness();
  const gate = deferred();
  app.pauseCommitResponse(gate.promise);
  await app.service.enable();
  const syncing = app.service.sync();
  await app.committed.promise;
  app.edit((data) => ({ ...data, facts: [fact(FIRST, 'Edited while syncing', LATER)], voice: 'Aoede' }));
  gate.resolve();
  await syncing;
  assert.equal(app.phone().facts[0]?.text, 'Edited while syncing');
  assert.equal(app.phone().voice, 'Aoede');
  assert.equal((app.documents.get('account-a')?.companion?.memories[FIRST] as Fact).text, 'A local memory');
  assert.equal((await app.service.status()).pending, true);
  await app.service.sync();
  assert.equal((app.documents.get('account-a')?.companion?.memories[FIRST] as Fact).text, 'Edited while syncing');
  assert.equal((await app.service.status()).pending, false);
});

test('deletion during an in-flight transaction survives downloaded copies', async () => {
  const app = harness();
  const gate = deferred();
  app.pauseCommitResponse(gate.promise);
  await app.service.enable();
  const syncing = app.service.sync();
  await app.committed.promise;
  app.edit((data) => ({ ...data, facts: [] }));
  gate.resolve();
  await syncing;
  assert.equal(app.phone().facts.length, 0);
  assert.equal((await app.service.status()).pending, true);
  await app.service.sync();
  assert.equal(app.documents.get('account-a')?.companion?.memories[FIRST]?.deleted, true);
});

test('account switch before the transaction merge prevents remote mutation', async () => {
  const app = harness();
  const gate = deferred();
  app.pauseRead(gate.promise);
  await app.service.enable();
  const syncing = app.service.sync();
  await app.entered.promise;
  app.account('account-b');
  gate.resolve();
  await assert.rejects(syncing, /account changed/);
  assert.equal(app.documents.get('account-a')?.companion, undefined);
  assert.equal(app.documents.has('account-b'), false);
  assert.equal(app.phone().facts[0]?.text, 'A local memory');
  assert.equal((await app.service.status()).enabled, false);
});

test('account switch after remote commit prevents applying the old account to this phone', async () => {
  const app = harness();
  const gate = deferred();
  app.documents.set('account-a', { displayName: 'Account profile', companion: shared(fact(SECOND, 'A-only memory')) });
  app.pauseCommitResponse(gate.promise);
  await app.service.enable();
  const syncing = app.service.sync();
  await app.committed.promise;
  app.account('account-b');
  gate.resolve();
  await assert.rejects(syncing, /account changed/);
  assert.deepEqual(app.phone().facts.map((f) => f.id), [FIRST]);
  await assert.rejects(app.service.sync(), /Enable sync/);
});

test('each account must explicitly import the current phone before syncing', async () => {
  const app = harness();
  await app.service.enable('account-a');
  app.account('account-b');
  await assert.rejects(app.service.enable('account-a'), /account changed/);
  assert.equal((await app.service.status()).enabled, false);
  await app.service.enable('account-b');
  await app.service.sync('account-b');
  app.account('account-a');
  assert.equal((await app.service.status()).enabled, false);
  await assert.rejects(app.service.sync('account-a'), /Enable sync/);
});

test('turning off sync invalidates a pending response and retains cloud and phone data', async () => {
  const app = harness();
  const gate = deferred();
  app.pauseCommitResponse(gate.promise);
  await app.service.enable();
  const syncing = app.service.sync();
  await app.committed.promise;
  await app.service.disable();
  gate.resolve();
  await assert.rejects(syncing, /account changed/);
  assert.equal((await app.service.status()).enabled, false);
  assert.equal(app.phone().facts.length, 1);
  assert.equal(app.documents.get('account-a')?.companion?.memories[FIRST]?.deleted, false);
});

test('malformed remote data leaves both the phone and existing cloud fields untouched', async () => {
  const app = harness();
  const bad = { schemaVersion: 999, memories: {}, preferences: {} } as unknown as Companion;
  app.documents.set('account-a', { displayName: 'Account profile', companion: bad });
  await app.service.enable();
  const before = app.phone();
  await assert.rejects(app.service.sync(), /app update/);
  assert.deepEqual(app.phone(), before);
  assert.equal(app.documents.get('account-a')?.companion?.schemaVersion, 999);
});

test('simultaneous Sync now calls share one network operation', async () => {
  const app = harness();
  await app.service.enable();
  await Promise.all([app.service.sync(), app.service.sync()]);
  assert.equal(app.calls(), 1);
});

test('a damaged deletion baseline is preserved and never used to rewrite cloud data', async () => {
  const app = harness();
  await app.service.enable();
  const key = 'adam.account-sync.v1.account-a';
  const record = JSON.parse(app.storage.get(key)!);
  record.baseline = {};
  const damaged = JSON.stringify(record);
  app.storage.set(key, damaged);
  await assert.rejects(app.service.sync(), /Saved sync data could not be read/);
  assert.equal(app.storage.get(key), damaged);
  assert.equal(app.calls(), 0);
  assert.equal(app.phone().facts.length, 1);
});
