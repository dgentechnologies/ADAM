import { createHash, createPublicKey, verify } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
export class Rejected extends Error {}
export const requireThat = (condition, message) => { if (!condition) throw new Rejected(message); };
const id = /^[A-Za-z0-9_-]{1,128}$/;
export function parseEnvelope(envelope) {
  requireThat(envelope && Object.keys(envelope).every(k => ['payload','signature'].includes(k)), 'Invalid proof envelope.');
  requireThat(typeof envelope.payload === 'string' && envelope.payload.length <= 48000 && /^[A-Za-z0-9+/]+={0,2}$/.test(envelope.payload), 'Invalid proof payload.');
  requireThat(typeof envelope.signature === 'string' && envelope.signature.length <= 200, 'Invalid proof signature.');
  let payload;
  try { payload = JSON.parse(Buffer.from(envelope.payload,'base64').toString('utf8')); } catch { throw new Rejected('Invalid proof JSON.'); }
  requireThat(payload && typeof payload === 'object' && !Array.isArray(payload), 'Invalid proof.');
  return payload;
}
export function verifyProof(envelope, hardware, purpose, uid, now = Date.now()) {
  const p = parseEnvelope(envelope);
  requireThat(hardware && hardware.enabled === true, 'Hardware is not registered or has been revoked.');
  requireThat(p.version === 1 && p.purpose === purpose && p.uid === uid && id.test(p.uid) && id.test(p.deviceId) && id.test(p.hardwareId), 'Proof scope mismatch.');
  requireThat(p.hardwareId === hardware.hardwareId && p.deviceId === hardware.deviceId && p.ownershipEpoch === hardware.ownershipEpoch, 'Stale hardware identity or ownership epoch.');
  requireThat(typeof p.nonce === 'string' && /^[a-f0-9]{32}$/.test(p.nonce), 'Invalid nonce.');
  requireThat(Number.isSafeInteger(p.issuedAt) && Number.isSafeInteger(p.expiresAt) && p.expiresAt > now && p.issuedAt <= now + 30000 && p.expiresAt - p.issuedAt > 0 && p.expiresAt - p.issuedAt <= 300000, 'Proof expired or clock invalid.');
  const key = createPublicKey(hardware.publicKeyPem);
  requireThat(key.asymmetricKeyType === 'ec' && key.asymmetricKeyDetails?.namedCurve === 'prime256v1', 'Unsupported hardware key.');
  requireThat(verify('sha256',Buffer.from(envelope.payload,'base64'),key,Buffer.from(envelope.signature,'base64')), 'Hardware signature rejected.');
  return p;
}
export function normalize(value) {
  if (value && typeof value.toDate === 'function') return value.toDate().toISOString();
  if (Array.isArray(value)) return value.map(normalize);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([k,v]) => [k,normalize(v)]));
  return value;
}
export function validateReceipt(p, cloudRecord) {
  requireThat(typeof p.recordJson === 'string' && Buffer.byteLength(p.recordJson) <= 16384, 'Record evidence is too large.');
  let record; try { record = JSON.parse(p.recordJson); } catch { throw new Rejected('Invalid record evidence.'); }
  requireThat(p.path === `users/${p.uid}/schedules/${p.scheduleId}` && id.test(p.scheduleId), 'Invalid receipt path.');
  requireThat(p.appliedRevision === createHash('sha256').update(p.recordJson).digest('hex'), 'Revision does not match evidence.');
  requireThat(isDeepStrictEqual(record, normalize(cloudRecord)), 'Cloud desired state changed; deliver its current revision first.');
  requireThat(Array.isArray(record.deviceIds) && (!record.deviceIds.length || record.deviceIds.includes(p.deviceId)), 'This schedule does not target this robot.');
  requireThat(['applied','deleted','fired'].includes(p.status) && (record.deleted ? p.status === 'deleted' : p.status !== 'deleted'), 'Invalid execution state.');
  requireThat(p.lastFiredAt === null || (typeof p.lastFiredAt === 'string' && Number.isFinite(Date.parse(p.lastFiredAt)) && Date.parse(p.lastFiredAt) <= p.issuedAt + 30000), 'Invalid firing timestamp.');
  requireThat(p.lastOccurrenceId === null || (typeof p.lastOccurrenceId === 'string' && p.lastOccurrenceId.length <= 256), 'Invalid occurrence.');
  requireThat(p.status !== 'fired' || (p.lastFiredAt && p.lastOccurrenceId), 'Missing firing evidence.');
  return record;
}
