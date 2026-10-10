/** Run only by a trusted administrator after verifying the public bundle at
 * the physical unit. No client claim can register or replace a hardware key. */
import { readFileSync } from 'node:fs';
import { createPublicKey } from 'node:crypto';
import { initializeApp, applicationDefault } from 'firebase-admin/app';
import { getFirestore, Timestamp } from 'firebase-admin/firestore';
const [bundlePath, mode, expectedOwnerUid, expectedLegacySerial] = process.argv.slice(2);
if (!bundlePath || !['--dry-run','--apply'].includes(mode)) throw new Error('Usage: node scripts/register-hardware.js PUBLIC_BUNDLE.json --dry-run|--apply [expectedOwnerUid expectedLegacySerial]');
const b=JSON.parse(readFileSync(bundlePath,'utf8'));
if (b.version !== 1 || !/^ADAM-[A-Za-z0-9-]+$/.test(b.deviceId) || !/^[A-Za-z0-9_-]{1,128}$/.test(b.hardwareId) || b.ownershipEpoch !== 0 || !/^[a-f0-9]{64}$/.test(b.certificateSha256)) throw new Error('Invalid public bundle.');
const key=createPublicKey(b.publicKeyPem);
if (key.asymmetricKeyType !== 'ec' || key.asymmetricKeyDetails?.namedCurve !== 'prime256v1') throw new Error('Expected P-256 public key.');
initializeApp({credential:applicationDefault(),projectId:'adam-ai1'});
const db=getFirestore();
await db.runTransaction(async tx => {
 const hardware=db.doc(`hardwareRegistry/${b.hardwareId}`), identity=db.doc(`hardwareDeviceIds/${b.deviceId}`), device=db.doc(`devices/${b.deviceId}`);
 const snapshots=await Promise.all([tx.get(hardware),tx.get(identity),tx.get(device)]);
 if (snapshots[0].exists || snapshots[1].exists) throw new Error('Hardware identity already exists; registration never replaces keys.');
 const existing=snapshots[2].data();
 if (existing && (!expectedOwnerUid || !expectedLegacySerial || (existing.ownerUid ?? existing.ownerId)!==expectedOwnerUid || (existing.ownerUid && existing.ownerId && existing.ownerUid!==existing.ownerId) || existing.hardwareSerial!==expectedLegacySerial)) throw new Error('Existing device requires an explicitly verified owner and exact previous serial; conflicting identities cannot be enrolled.');
 if (!existing && (expectedOwnerUid || expectedLegacySerial)) throw new Error('Expected existing device was not found.');
 if (mode === '--apply') {
  tx.create(hardware,{...b,enabled:true,ownerUid:expectedOwnerUid || null,createdAt:Timestamp.now()});
  tx.create(identity,{hardwareId:b.hardwareId});
  if (existing) tx.update(device,{ownerUid:expectedOwnerUid,hardwareSerial:b.hardwareId,kind:'physical',ownershipEpoch:0,schemaVersion:1,updatedAt:Timestamp.now()});
 }
});
console.log(mode === '--apply' ? 'Public hardware identity registered.' : 'Dry run passed; no writes performed.');
