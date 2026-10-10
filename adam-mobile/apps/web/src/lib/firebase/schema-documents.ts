/** Cloud boundary for docs/CLOUD_DATA_SCHEMA.md. The companion envelope is a
 * local checkpoint only; Firestore stores individual canonical documents. */
import { z } from 'zod';
import { documentId, wallTime } from '../companion-records';
import { validateCompanion, type Companion } from './companion-sync';

export type CloudDocument = Record<string, any>;
export type CloudDocuments = Record<string, CloudDocument>;
export type DocumentEntry = { path: string; kind: 'devices' | 'memories' | 'todos' | 'clocks'; id: string; data: CloudDocument };
const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;
const iso = z.string().datetime({ precision: 3 });
const weekdays = z.array(z.enum(['mon','tue','wed','thu','fri','sat','sun'])).max(7);
const text = (value: unknown, max: number, empty = false) => z.string().trim().min(empty ? 0 : 1).max(max).parse(value);

export function utcTime(value: unknown): string {
  if (value == null) throw new Error('A cloud document is missing its timestamp.');
  const date = value && typeof (value as any).toDate === 'function' ? (value as any).toDate() : new Date(value as string);
  if (!(date instanceof Date) || !Number.isFinite(date.getTime())) throw new Error('A cloud document has an invalid timestamp.');
  return iso.parse(date.toISOString());
}
export function localWallTime(utc: string): string {
  const d = new Date(utc);
  if (!Number.isFinite(d.getTime())) throw new Error('Invalid schedule time.');
  return wallTime.parse(`${d.getFullYear().toString().padStart(4,'0')}-${(d.getMonth()+1).toString().padStart(2,'0')}-${d.getDate().toString().padStart(2,'0')}T${d.getHours().toString().padStart(2,'0')}:${d.getMinutes().toString().padStart(2,'0')}`);
}
export function stableDeviceId(device: any): string {
  if (device.deviceId) return documentId.parse(device.deviceId);
  if (!device.simulated && !device.deleted) throw new Error('This ADAM needs its advertised device ID before cloud sync.');
  // A simulated unit has no MAC. Its saved identity survives app/cloud transfers.
  return documentId.parse(`ADAM-SIM-${device.id.toUpperCase()}`);
}
function targetIds(record: any, devices: Companion['devices']): string[] {
  if (record.deviceIds) return [...new Set(z.array(documentId).max(8).parse(record.deviceIds))];
  if (!record.deviceId) return [];
  const device = devices[record.deviceId];
  if (!device) throw new Error('Choose a saved ADAM for this plan before syncing.');
  return [stableDeviceId(device)];
}
export function normalizeCloudMetadata(input: Companion): Companion {
  const next = validateCompanion(input);
  for (const device of Object.values(next.devices)) (device as any).deviceId = stableDeviceId(device);
  for (const kind of ['todos','clocks'] as const) for (const record of Object.values(next[kind])) {
    if (record.deleted) continue;
    const item = record as any;
    item.deviceIds = targetIds(item, next.devices);
    if (kind === 'clocks') item.at ??= localWallTime(item.when);
    else item.due ??= item.dueAt ? localWallTime(item.dueAt) : null;
  }
  return validateCompanion(next);
}
export function toCloudDocuments(input: Companion, uid: string): DocumentEntry[] {
  documentId.parse(uid);
  const local = normalizeCloudMetadata(input);
  const entries: DocumentEntry[] = [];
  for (const kind of ['devices','memories','todos','clocks'] as const) for (const [id, raw] of Object.entries(local[kind])) {
    const item = raw as any;
    const cloudId = documentId.parse(item.cloudId || id);
    const bookkeeping = { schemaVersion:1, createdAt:item.createdAt ?? item.updatedAt, updatedAt: item.updatedAt, deleted: item.deleted, deletedAt: item.deleted ? item.updatedAt : null, origin: 'mobile' };
    let path: string, data: CloudDocument;
    if (kind === 'devices') {
      if (item.simulated || item.deleted) continue; // Local simulations/unlinking never claim or delete physical cloud identities.
      const deviceId = stableDeviceId(item);
      path = `devices/${deviceId}`;
      data = { deviceId, ownerUid: uid, ...bookkeeping };
      if (!item.deleted) Object.assign(data, { name:item.name, hardwareSerial:item.serial, simulated:item.simulated, status:'setup', bleAddress:null, wifiSsid:null, tailscaleIp:null, osVersion:'', lastSeen:null, createdAt:item.createdAt });
    } else if (kind === 'memories') {
      // Unassigned phone memories stay local; never copy room-specific memories
      // to every robot merely because those robots share an account.
      if (!item.deviceId) continue;
      const deviceId = documentId.parse(item.deviceId);
      if (!Object.values(local.devices).some(d => stableDeviceId(d) === deviceId && !d.deleted)) continue;
      const person = item.kind === 'person';
      path = `devices/${deviceId}/${person ? 'memoryPeople' : 'memoryFacts'}/${cloudId}`;
      data = { [person ? 'personId' : 'factId']:cloudId, ...bookkeeping };
      if (!item.deleted) Object.assign(data, person
        ? { name:item.title, relationship:item.relationship ?? '', notes:item.text, faceEncodingId:item.faceEncodingId ?? null, firstSeen:item.firstSeen ?? item.createdAt, lastSeen:item.lastSeen ?? item.updatedAt }
        : { category:item.title, content:item.text, confidence:item.confidence ?? 1, learnedAt:item.createdAt, source:item.source ?? 'manual' });
    } else {
      path = `users/${uid}/${kind === 'clocks' ? 'schedules' : 'todos'}/${cloudId}`;
      data = { [kind === 'clocks' ? 'scheduleId' : 'todoId']:cloudId, ...bookkeeping };
      if (!item.deleted) {
        Object.assign(data, { createdAt:item.createdAt, deviceIds:targetIds(item,local.devices) });
        Object.assign(data, kind === 'todos'
          ? { text:item.text, done:item.done, due:item.due ?? null, doneAt:item.done ? item.doneAt ?? item.updatedAt : null }
          : { kind:item.kind, label:item.label, at:wallTime.parse(item.at), timeOfDay:item.timeOfDay ?? null, repeat:item.repeat ?? null, enabled:item.enabled, timeZone:item.timeZone ?? Intl.DateTimeFormat().resolvedOptions().timeZone, lastFired:null, snoozes:0, ...(item.kind === 'timer' ? {durationSeconds:item.durationSeconds ?? Math.round((Date.parse(item.when)-Date.parse(item.createdAt))/1000),deadline:item.deadline ?? item.when} : {}) });
      }
    }
    entries.push({ path, kind, id, data });
  }
  if (entries.length > 3100 || new TextEncoder().encode(JSON.stringify(entries)).length > 1500000) throw new Error('Cloud data exceeds the supported transfer limit.');
  return entries;
}
async function localId(path: string): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(path)));
  digest[6] = (digest[6]! & 15) | 80; digest[8] = (digest[8]! & 63) | 128;
  const h = Array.from(digest.slice(0,16),b=>b.toString(16).padStart(2,'0')).join('');
  return `${h.slice(0,8)}-${h.slice(8,12)}-${h.slice(12,16)}-${h.slice(16,20)}-${h.slice(20)}`;
}
export function documentUpdatedAt(data: CloudDocument): string {
  // Existing device/memory documents predate updatedAt. Their deployed creation
  // fields are the migration baseline; never substitute the current time.
  return utcTime(data.updatedAt ?? data.learnedAt ?? data.lastSeen ?? data.firstSeen ?? data.createdAt);
}
export async function applyCloudDocuments(input: Companion, uid: string, documents: CloudDocuments): Promise<Companion> {
  const next = normalizeCloudMetadata(input);
  const known = new Map(toCloudDocuments(next,uid).map(e=>[e.path,e]));
  const ordered = Object.entries(documents).sort(([a],[b]) => (a.split('/').length - b.split('/').length) || a.localeCompare(b));
  for (const [path,data] of ordered) {
    const parts = path.split('/'), cloudId = documentId.parse(parts.at(-1));
    const device = parts[0] === 'devices' && parts.length === 2;
    const memory = parts[0] === 'devices' && parts.length === 4 && ['memoryFacts','memoryPeople'].includes(parts[2]!);
    const plan = parts[0] === 'users' && parts[1] === uid && parts.length === 4 && ['schedules','todos'].includes(parts[2]!);
    if (!device && !memory && !plan) throw new Error('Unexpected cloud document path.');
    const kind = device ? 'devices' : memory ? 'memories' : parts[2] === 'schedules' ? 'clocks' : 'todos';
    const old = known.get(path);
    const id = old?.id ?? (plan && uuid.test(cloudId) ? cloudId : await localId(path));
    const current = next[kind][id];
    const updatedAt = documentUpdatedAt(data);
    if (current && current.updatedAt >= updatedAt) continue;
    const deleted = data.deleted === undefined ? false : z.boolean().parse(data.deleted);
    let item: any = { id, cloudId, updatedAt, deleted };
    if (device) {
      const owner = data.ownerUid;
      if (owner !== uid || (data.deviceId !== undefined && data.deviceId !== cloudId)) throw new Error('ADAM ownership does not match this account.');
      item.deviceId = cloudId;
      if (!deleted) Object.assign(item,{ createdAt:utcTime(data.createdAt), name:text(data.name,40), serial:text(data.hardwareSerial ?? data.serial ?? cloudId,80), simulated:data.simulated === true });
    } else if (memory) {
      const deviceId = documentId.parse(parts[1]);
      if (!Object.values(next.devices).some(d=>!d.deleted && (d as any).deviceId === deviceId)) continue;
      const person = parts[2] === 'memoryPeople';
      if ((data[person ? 'personId' : 'factId'] ?? cloudId) !== cloudId) throw new Error('Memory identity does not match its document.');
      Object.assign(item,{ deviceId,kind:person ? 'person' : 'fact' });
      if (!deleted) Object.assign(item,person
        ? { title:text(data.name,80), text:text(data.notes || data.relationship || data.name,2000), createdAt:utcTime(data.firstSeen), relationship:text(data.relationship ?? '',2000,true), faceEncodingId:data.faceEncodingId == null ? null : text(data.faceEncodingId,128), firstSeen:utcTime(data.firstSeen), lastSeen:utcTime(data.lastSeen ?? data.lastSeenAt ?? data.firstSeen) }
        : { title:text(data.category ?? data.key,80),text:text(data.content ?? data.value,2000),createdAt:utcTime(data.learnedAt ?? data.savedAt),confidence:z.number().min(0).max(1).parse(data.confidence ?? 1),source:z.enum(['conversation','vision','manual']).parse(data.source ?? 'manual') });
    } else {
      if (data[kind === 'clocks' ? 'scheduleId' : 'todoId'] !== cloudId) throw new Error('Plan identity does not match its document.');
      if (!deleted) {
        const deviceIds = [...new Set(z.array(documentId).max(8).parse(data.deviceIds))];
        const first = Object.values(next.devices).find(d=>!d.deleted && (d as any).deviceId === deviceIds[0]);
        Object.assign(item,{createdAt:utcTime(data.createdAt),deviceIds,deviceId:first?.id ?? ''});
        if (kind === 'todos') Object.assign(item,{text:text(data.text,2000),done:z.boolean().parse(data.done),due:data.due == null ? null : text(data.due,32),dueAt:data.due && /^\d{4}-\d{2}-\d{2}T/.test(data.due) ? utcTime(data.due) : '',doneAt:data.doneAt == null ? null : utcTime(data.doneAt)});
        else {
          const at = wallTime.parse(data.at);
          Object.assign(item,{kind:z.enum(['alarm','timer','reminder']).parse(data.kind),...(data.timeZone ? {timeZone:data.timeZone} : {}),...(data.durationSeconds != null ? {durationSeconds:data.durationSeconds} : {}),...(data.deadline ? {deadline:data.deadline} : {}),label:text(data.label,80,true),at,when:utcTime(at),enabled:z.boolean().parse(data.enabled),timeOfDay:data.timeOfDay == null ? null : z.string().regex(/^([01]\d|2[0-3]):[0-5]\d$/).parse(data.timeOfDay),repeat:data.repeat == null ? null : weekdays.parse(data.repeat),lastFired:data.lastFired == null ? null : text(data.lastFired,32),snoozes:z.number().int().min(0).parse(data.snoozes ?? 0)});
        }
      }
    }
    (next[kind] as any)[id] = item;
  }
  return validateCompanion(next);
}

/** One document wins only when strictly newer. The Pi owns lastFired. */
export function chooseCloudWrite(local: CloudDocument, remote: CloudDocument | undefined, device = false): CloudDocument | null {
  if (remote && documentUpdatedAt(remote) >= documentUpdatedAt(local)) return null;
  let update = { ...local };
  if (device && remote) {
    if (remote.ownerUid !== local.ownerUid) throw new Error('This ADAM belongs to another account.');
    update = Object.fromEntries(['name','updatedAt'].filter(k=>local[k] !== undefined).map(k=>[k,local[k]]));
  }
  if (!device && remote?.createdAt) update.createdAt = remote.createdAt;
  if ('scheduleId' in local) { update.lastFired=null; update.snoozes=0; }
  return update;
}
