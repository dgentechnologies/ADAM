import type { Fact, LocalData } from '../local-data';

export type SharedMemory =
  | (Fact & { deleted: false })
  | { id: string; updatedAt: string; deleted: true };
type PreferenceName = 'voice' | 'wakeWord' | 'brain';
type Preferences = Partial<Record<PreferenceName, { value: string; updatedAt: string }>>;
export type Companion = {
  schemaVersion: 1;
  memories: Record<string, SharedMemory>;
  preferences: Preferences;
};
type PhoneSnapshot = Pick<LocalData, 'facts' | 'voice' | 'wakeWord' | 'brain'>;
type SyncRecord = {
  version: 1;
  uid: string;
  enabled: boolean;
  companion: Companion;
  baseline: PhoneSnapshot | null;
  pending: boolean;
  lastSynced: string | null;
};
export type CompanionSyncStatus = {
  enabled: boolean;
  syncing: boolean;
  pending: boolean;
  lastSynced: string | null;
  error: string;
};
export interface SyncDependencies {
  getUid: () => string | null;
  getItem: (key: string) => Promise<string | null>;
  setItem: (key: string, value: string) => Promise<void>;
  readLocal: () => Promise<LocalData>;
  updateLocal: (update: (data: LocalData) => LocalData) => Promise<LocalData>;
  transact: (uid: string, merge: (remote: unknown) => Companion) => Promise<Companion>;
  now?: () => string;
  onChange?: () => void;
}

const OWNER_KEY = 'adam.account-sync.owner.v1';
const RECORD_KEY = 'adam.account-sync.v1.';
const DEFAULTS = { voice: 'Charon', wakeWord: 'Hey ADAM', brain: 'lite' } as const;
const OPTIONS: Record<PreferenceName, readonly string[]> = {
  voice: ['Charon', 'Aoede', 'Kore', 'Puck', 'Fenrir'],
  wakeWord: ['Hey ADAM', 'ADAM'],
  brain: ['lite', 'byok', 'managed'],
};
const NAMES = Object.keys(OPTIONS) as PreferenceName[];
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
class SyncError extends Error {}

export function emptyCompanion(): Companion {
  return { schemaVersion: 1, memories: {}, preferences: {} };
}
function object(value: unknown): value is Record<string, any> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}
function timestamp(value: unknown): string {
  if (
    typeof value !== 'string' ||
    !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(value) ||
    value.startsWith('0000') ||
    !Number.isFinite(Date.parse(value)) ||
    new Date(value).toISOString() !== value
  )
    throw new SyncError('Shared data contains an invalid date. Your phone data has been kept.');
  return value;
}
function validText(value: unknown, max: number): string {
  if (typeof value !== 'string') throw new SyncError('A shared memory could not be read.');
  const text = value.trim();
  if (!text.length || text.length > max)
    throw new SyncError('A shared memory is longer than this app supports.');
  for (const character of text) {
    const point = character.codePointAt(0)!;
    if (point >= 0xd800 && point <= 0xdfff)
      throw new SyncError('A shared memory contains invalid text.');
  }
  return text;
}
export function compareCodePoints(left: string, right: string): number {
  const a = Array.from(left, (c) => c.codePointAt(0)!);
  const b = Array.from(right, (c) => c.codePointAt(0)!);
  for (let index = 0; index < Math.min(a.length, b.length); index++) {
    if (a[index] !== b[index]) return a[index]! > b[index]! ? 1 : -1;
  }
  return a.length === b.length ? 0 : a.length > b.length ? 1 : -1;
}
export function canonical(value: unknown): string {
  if (object(value))
    return `{${Object.keys(value)
      .sort(compareCodePoints)
      .map((key) => `${JSON.stringify(key)}:${canonical(value[key])}`)
      .join(',')}}`;
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  return JSON.stringify(value);
}
function encodeValue(value: unknown): unknown {
  if (typeof value === 'boolean') return { booleanValue: value };
  if (typeof value === 'number') return { integerValue: String(value) };
  if (typeof value === 'string') return { stringValue: value };
  if (object(value))
    return { mapValue: { fields: Object.fromEntries(Object.entries(value).map(([k, v]) => [k, encodeValue(v)])) } };
  throw new SyncError('Shared data contains an unsupported value.');
}
export function validateCompanion(value: unknown): Companion {
  if (!object(value) || value.schemaVersion !== 1)
    throw new SyncError('This shared-data version needs an app update. Your phone data has been kept.');
  const memories = 'memories' in value ? value.memories : {};
  const preferences = 'preferences' in value ? value.preferences : {};
  if (!object(memories) || !object(preferences) || Object.keys(memories).length > 1000)
    throw new SyncError('Shared memories have reached the supported limit.');
  const result = emptyCompanion();
  for (const [id, item] of Object.entries(memories)) {
    if (!UUID.test(id) || !object(item) || item.id !== id || typeof item.deleted !== 'boolean')
      throw new SyncError('A shared memory could not be read. Your phone data has been kept.');
    const updatedAt = timestamp(item.updatedAt);
    if (item.deleted) {
      result.memories[id] = { id, updatedAt, deleted: true };
      continue;
    }
    if (item.kind !== 'fact' && item.kind !== 'person')
      throw new SyncError('A shared memory has an unsupported type.');
    const createdAt = timestamp(item.createdAt);
    if (createdAt > updatedAt) throw new SyncError('A shared memory has inconsistent dates.');
    result.memories[id] = {
      id, title: validText(item.title, 80), text: validText(item.text, 2000), kind: item.kind,
      createdAt, updatedAt, deleted: false,
    };
  }
  for (const [key, item] of Object.entries(preferences)) {
    if (!NAMES.includes(key as PreferenceName) || !object(item) || !OPTIONS[key as PreferenceName].includes(item.value))
      throw new SyncError('A shared preference could not be read.');
    result.preferences[key as PreferenceName] = { value: item.value, updatedAt: timestamp(item.updatedAt) };
  }
  if (new TextEncoder().encode(canonical(encodeValue(result))).length > 680000)
    throw new SyncError('Shared memories are full. Shorten a memory before syncing more.');
  return result;
}
function winner<T extends { updatedAt: string; deleted?: boolean }>(a: T, b: T): T {
  if (a.updatedAt !== b.updatedAt) return a.updatedAt > b.updatedAt ? a : b;
  if (Boolean(a.deleted) !== Boolean(b.deleted)) return a.deleted ? a : b;
  return compareCodePoints(canonical(a), canonical(b)) >= 0 ? a : b;
}
export function mergeCompanions(left: Companion, right: Companion): Companion {
  const a = validateCompanion(left);
  const b = validateCompanion(right);
  const result = emptyCompanion();
  for (const id of new Set([...Object.keys(a.memories), ...Object.keys(b.memories)])) {
    result.memories[id] = a.memories[id] && b.memories[id]
      ? winner(a.memories[id]!, b.memories[id]!) : (a.memories[id] ?? b.memories[id])!;
  }
  for (const key of NAMES) {
    const first = a.preferences[key], second = b.preferences[key];
    if (first || second) result.preferences[key] = first && second ? winner(first, second) : (first ?? second)!;
  }
  return validateCompanion(result);
}
function snapshot(data: LocalData): PhoneSnapshot {
  return { facts: data.facts.map((fact) => ({ ...fact })), voice: data.voice, wakeWord: data.wakeWord, brain: data.brain };
}
function after(now: string, ...previous: (string | undefined)[]): string {
  let stamp = timestamp(now);
  for (const value of previous) {
    if (value && value >= stamp) stamp = new Date(Date.parse(value) + 1).toISOString();
  }
  return timestamp(stamp);
}
/** Detect edits/deletions against the last observed phone state, retaining tombstones. */
export function capturePhoneChanges(base: Companion, baseline: PhoneSnapshot | null,
                                    current: PhoneSnapshot, now: string): Companion {
  const next = validateCompanion(base);
  const before = new Map((baseline?.facts ?? []).map((fact) => [fact.id, fact]));
  const present = new Set(current.facts.map((fact) => fact.id));
  if (present.size !== current.facts.length) throw new SyncError('Two memories have the same identifier.');
  for (const fact of current.facts) {
    const old = before.get(fact.id);
    if (old && canonical(old) === canonical(fact)) continue;
    const existing = next.memories[fact.id];
    // Local form timestamps can be equal or behind a received record. An edit
    // must advance beyond the value it replaces, including a future clock.
    const updatedAt = existing
      ? after(now, existing.updatedAt, fact.updatedAt)
      : timestamp(fact.updatedAt);
    next.memories[fact.id] = { ...fact, updatedAt, deleted: false };
  }
  for (const [id, old] of before) {
    if (!present.has(id))
      next.memories[id] = { id, deleted: true, updatedAt: after(now, old.updatedAt, next.memories[id]?.updatedAt) };
  }
  for (const key of NAMES) {
    if (baseline ? current[key] === baseline[key]
      : current[key] === DEFAULTS[key] && !next.preferences[key]) continue;
    next.preferences[key] = { value: current[key], updatedAt: after(now, next.preferences[key]?.updatedAt) };
  }
  return validateCompanion(next);
}
function applyShared(current: LocalData, companion: Companion): LocalData {
  const next = { ...current };
  next.facts = Object.values(companion.memories)
    .filter((item): item is Fact & { deleted: false } => !item.deleted)
    .map(({ deleted: _deleted, ...fact }) => fact)
    .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt) || b.id.localeCompare(a.id));
  for (const key of NAMES) {
    const preference = companion.preferences[key];
    if (preference) (next[key] as string) = preference.value;
  }
  return next;
}
function syncError(error: unknown): string {
  if (error instanceof SyncError) return error.message;
  const code = String((error as { code?: string })?.code ?? '');
  if (/permission-denied|unauthenticated/.test(code))
    return 'Sync was not authorized. Sign in again and check account access.';
  return 'Sync could not finish. Your phone data is safe. Check your connection and try again.';
}

export class MobileCompanionSync {
  private epoch = 0;
  private account: string | null;
  private running = new Map<string, Promise<void>>();
  private errors = new Map<string, string>();
  private pendingMutation: Promise<unknown> = Promise.resolve();
  constructor(private deps: SyncDependencies) { this.account = deps.getUid(); }
  accountChanged(uid: string | null) {
    if (uid !== this.account) { this.account = uid; this.epoch++; this.deps.onChange?.(); }
  }
  private scope(expectedUid?: string) {
    this.accountChanged(this.deps.getUid());
    if (!this.account) throw new SyncError('Sign in to sync with your desktop.');
    if (expectedUid && expectedUid !== this.account)
      throw new SyncError('The account changed. Review sync for the current account.');
    return { uid: this.account, epoch: this.epoch };
  }
  private check(scope: { uid: string; epoch: number }) {
    if (this.deps.getUid() !== scope.uid || this.epoch !== scope.epoch)
      throw new SyncError('The account changed. Sync again from the current account.');
  }
  private serialized<T>(operation: () => Promise<T>): Promise<T> {
    const next = this.pendingMutation.catch(() => undefined).then(operation);
    this.pendingMutation = next;
    return next;
  }
  private async read(uid: string): Promise<SyncRecord> {
    const raw = await this.deps.getItem(RECORD_KEY + encodeURIComponent(uid));
    if (!raw)
      return { version: 1, uid, enabled: false, companion: emptyCompanion(), baseline: null, pending: false, lastSynced: null };
    try {
      const record = JSON.parse(raw);
      if (!object(record) || record.version !== 1 || record.uid !== uid || typeof record.enabled !== 'boolean'
          || typeof record.pending !== 'boolean') throw new Error();
      record.companion = validateCompanion(record.companion);
      if (record.lastSynced !== null) timestamp(record.lastSynced);
      if (record.baseline !== null) {
        if (!object(record.baseline) || !Array.isArray(record.baseline.facts)
            || NAMES.some((key) => !OPTIONS[key].includes(record.baseline[key]))) throw new Error();
        const { LocalData: validator } = await import('../local-data');
        record.baseline = snapshot(validator.parse({ version: 1, ...record.baseline }));
        const ids = new Set(record.baseline.facts.map((item: Fact) => item.id));
        if (ids.size !== record.baseline.facts.length) throw new Error();
        const observed = emptyCompanion();
        for (const item of record.baseline.facts) observed.memories[item.id] = { ...item, deleted: false };
        validateCompanion(observed);
      }
      return record as SyncRecord;
    } catch {
      throw new SyncError('Saved sync data could not be read. Your memories and saved records have been kept.');
    }
  }
  private write(record: SyncRecord) {
    validateCompanion(record.companion);
    return this.deps.setItem(RECORD_KEY + encodeURIComponent(record.uid), JSON.stringify(record));
  }
  async status(): Promise<CompanionSyncStatus> {
    const uid = this.deps.getUid();
    if (!uid) return { enabled: false, syncing: false, pending: false, lastSynced: null, error: '' };
    try {
      const record = await this.read(uid);
      const owner = await this.deps.getItem(OWNER_KEY);
      const enabled = record.enabled && owner === uid;
      const local = enabled ? snapshot(await this.deps.readLocal()) : null;
      if (this.deps.getUid() !== uid)
        return { enabled: false, syncing: false, pending: false, lastSynced: null, error: '' };
      return { enabled, syncing: this.running.has(uid),
        pending: enabled && (record.pending || canonical(local) !== canonical(record.baseline)),
        lastSynced: record.lastSynced, error: this.errors.get(uid) ?? '' };
    } catch (error) {
      return { enabled: false, syncing: this.running.has(uid), pending: false, lastSynced: null, error: syncError(error) };
    }
  }
  /** Call only after confirming import of this phone's data into this account. */
  async enable(expectedUid?: string): Promise<void> {
    this.epoch++;
    const scope = this.scope(expectedUid);
    await this.serialized(async () => {
      this.check(scope);
      const record = await this.read(scope.uid);
      const phone = snapshot(await this.deps.readLocal());
      this.check(scope);
      record.companion = capturePhoneChanges(record.companion, null, phone, this.now());
      record.baseline = phone;
      record.pending = true;
      record.enabled = true;
      await this.write(record);
      this.check(scope);
      await this.deps.setItem(OWNER_KEY, scope.uid);
      this.errors.delete(scope.uid);
    });
    this.deps.onChange?.();
  }
  async disable(expectedUid?: string): Promise<void> {
    this.epoch++;
    const scope = this.scope(expectedUid);
    await this.serialized(async () => {
      this.check(scope);
      const record = await this.read(scope.uid);
      this.check(scope);
      record.enabled = false;
      await this.write(record);
    });
    this.deps.onChange?.();
  }
  private now() { return this.deps.now?.() ?? new Date().toISOString(); }
  sync(expectedUid?: string): Promise<void> {
    const scope = this.scope(expectedUid);
    const existing = this.running.get(scope.uid);
    if (existing) return existing;
    const operation = this.performSync(scope).catch((error) => {
      this.errors.set(scope.uid, syncError(error));
      throw new SyncError(syncError(error));
    }).finally(() => { this.running.delete(scope.uid); this.deps.onChange?.(); });
    this.running.set(scope.uid, operation);
    this.deps.onChange?.();
    return operation;
  }
  private async performSync(scope: { uid: string; epoch: number }): Promise<void> {
    const captured = await this.serialized(async () => {
      this.check(scope);
      const record = await this.read(scope.uid);
      const owner = await this.deps.getItem(OWNER_KEY);
      if (!record.enabled || owner !== scope.uid)
        throw new SyncError('Enable sync for this account before uploading phone data.');
      const phone = snapshot(await this.deps.readLocal());
      this.check(scope);
      record.companion = capturePhoneChanges(record.companion, record.baseline, phone, this.now());
      record.baseline = phone;
      record.pending = true;
      await this.write(record); // Offline edits and deletions survive a failed transaction.
      return record;
    });
    this.check(scope);
    const merged = await this.deps.transact(scope.uid, (remote) => {
      this.check(scope);
      return mergeCompanions(captured.companion, remote === undefined ? emptyCompanion() : validateCompanion(remote));
    });
    this.check(scope);
    validateCompanion(merged);
    await this.serialized(async () => {
      this.check(scope);
      let final = merged;
      const applied = await this.deps.updateLocal((current) => {
        this.check(scope);
        // The local write queue supplies the newest phone state. Any edit made
        // while Firestore was in flight wins locally and remains pending.
        final = capturePhoneChanges(merged, captured.baseline, snapshot(current), this.now());
        return applyShared(current, final);
      });
      this.check(scope);
      await this.write({ ...captured, companion: final, baseline: snapshot(applied),
        pending: canonical(final) !== canonical(merged), lastSynced: this.now() });
      this.errors.delete(scope.uid);
    });
  }
}

let instance: Promise<MobileCompanionSync> | undefined;
/** Lazy: loading the app does not start a transaction or import phone data. */
export function getCompanionSync(): Promise<MobileCompanionSync> {
  if (!instance) instance = (async () => {
    const [{ getFirebaseAuth, getFirebaseFirestore }, { doc, runTransaction }, { onAuthStateChanged },
      local, preferences] = await Promise.all([
      import('./config'), import('firebase/firestore'), import('firebase/auth'),
      import('../local-data'), import('../native/preferences'),
    ]);
    const auth = getFirebaseAuth();
    const service = new MobileCompanionSync({
      getUid: () => auth.currentUser?.uid ?? null,
      getItem: preferences.getItem, setItem: preferences.setItem,
      readLocal: local.readLocalData, updateLocal: local.updateLocalData,
      onChange: () => { if (typeof window !== 'undefined') window.dispatchEvent(new Event('adam:account-sync')); },
      transact: (uid, merge) => runTransaction(getFirebaseFirestore(), async (transaction) => {
        const ref = doc(getFirebaseFirestore(), 'users', uid);
        const saved = await transaction.get(ref);
        const next = merge(saved.exists() ? saved.data().companion : undefined);
        // Replace only this bounded field; profile and device ownership survive.
        transaction.set(ref, { companion: next }, { mergeFields: ['companion'] });
        return next;
      }),
    });
    onAuthStateChanged(auth, (user) => service.accountChanged(user?.uid ?? null));
    return service;
  })();
  return instance;
}
