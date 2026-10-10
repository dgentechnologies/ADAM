import { Timestamp, FieldValue } from 'firebase-admin/firestore';
import { parseEnvelope, verifyProof, validateReceipt, requireThat } from './protocol.js';
export function workflows(db) {
  async function execute(uid, envelope, purpose) {
    requireThat(typeof uid === 'string' && uid.length > 0, 'Sign in first.');
    const hints = parseEnvelope(envelope);
    requireThat(/^[A-Za-z0-9_-]{1,128}$/.test(hints.hardwareId), 'Invalid hardware identity.');
    const hardwareRef = db.doc(`hardwareRegistry/${hints.hardwareId}`);
    return db.runTransaction(async tx => {
      const hardwareSnapshot = await tx.get(hardwareRef);
      const hardware = hardwareSnapshot.data();
      const p = verifyProof(envelope, hardware, purpose, uid);
      const deviceRef = db.doc(`devices/${p.deviceId}`), nonceRef = db.doc(`hardwareProofs/${p.hardwareId}_${p.nonce}`);
      const [deviceSnapshot, used] = await Promise.all([tx.get(deviceRef),tx.get(nonceRef)]);
      const device = deviceSnapshot.data();
      requireThat(!device || (device.hardwareSerial === p.hardwareId && device.ownershipEpoch === p.ownershipEpoch), 'Device identity collision or stale ownership.');
      if (used.exists) {
        requireThat(used.data().uid === uid && used.data().purpose === purpose, 'Proof replay rejected.');
        return {ok:true,replayed:true,deviceId:p.deviceId};
      }
      const ownerRef=db.doc(`users/${uid}`), ownerProfile=await tx.get(ownerRef);
      const now = Timestamp.now();
      if (purpose === 'claim') {
        requireThat(!device || device.ownerUid === uid, 'Device already belongs to another account.');
        requireThat(!hardware.ownerUid || hardware.ownerUid === uid, 'Hardware already claimed.');
        tx.set(deviceRef,{deviceId:p.deviceId,ownerUid:uid,hardwareSerial:p.hardwareId,kind:'physical',name:device?.name || 'ADAM',ownershipEpoch:p.ownershipEpoch,schemaVersion:1,createdAt:device?.createdAt || now,updatedAt:now,lastSeen:now});
        tx.update(hardwareRef,{ownerUid:uid});
        if (ownerProfile.exists) tx.update(ownerRef,{linkedDeviceIds:FieldValue.arrayUnion(p.deviceId)});
      } else if (purpose === 'transfer') {
        requireThat(device?.ownerUid === uid && hardware.ownerUid === uid, 'Only the current owner can transfer.');
        requireThat(typeof p.newUid === 'string' && /^[A-Za-z0-9_-]{1,128}$/.test(p.newUid) && p.newUid !== uid, 'Invalid new owner.');
        // Transfer is deliberately blocked until private device data has been
        // archived/purged by a reviewed administrative migration. Never expose
        // the old owner's memories to the new owner through inherited paths.
        for (const collection of ['memoryFacts','memoryPeople','laptopPairings','executionState']) {
          const rows = await tx.get(deviceRef.collection(collection).limit(1));
          requireThat(rows.empty, 'Archive and purge device-private cloud data before transfer.');
        }
        const newOwnerRef=db.doc(`users/${p.newUid}`), newProfile=await tx.get(newOwnerRef);
        tx.update(deviceRef,{ownerUid:p.newUid,ownershipEpoch:p.ownershipEpoch+1,updatedAt:now});
        tx.update(hardwareRef,{ownerUid:p.newUid,ownershipEpoch:p.ownershipEpoch+1});
        if (ownerProfile.exists) tx.update(ownerRef,{linkedDeviceIds:FieldValue.arrayRemove(p.deviceId)});
        if (newProfile.exists) tx.update(newOwnerRef,{linkedDeviceIds:FieldValue.arrayUnion(p.deviceId)});
      } else {
        requireThat(device?.ownerUid === uid && hardware.ownerUid === uid, 'Receipt owner no longer owns this ADAM.');
        requireThat(typeof p.scheduleId === 'string' && /^[A-Za-z0-9_-]{1,128}$/.test(p.scheduleId), 'Invalid schedule.');
        const desired = await tx.get(db.doc(`users/${uid}/schedules/${p.scheduleId}`));
        requireThat(desired.exists, 'Desired schedule is missing.');
        validateReceipt(p,desired.data());
        const stateRef=deviceRef.collection('executionState').doc(p.scheduleId), previous=await tx.get(stateRef);
        requireThat(!previous.exists || previous.data().reportedAt.toMillis() <= p.issuedAt, 'Stale execution evidence.');
        tx.set(stateRef,{scheduleId:p.scheduleId,receivedRevision:p.appliedRevision,appliedRevision:p.appliedRevision,lastOccurrenceId:p.lastOccurrenceId,lastFiredAt:p.lastFiredAt ? Timestamp.fromDate(new Date(p.lastFiredAt)) : null,status:p.status,error:null,reportedAt:Timestamp.fromMillis(p.issuedAt),origin:`pi:${p.deviceId}`,ownershipEpoch:p.ownershipEpoch,proofNonce:p.nonce});
      }
      tx.create(nonceRef,{uid,purpose,deviceId:p.deviceId,createdAt:now,expiresAt:Timestamp.fromMillis(p.expiresAt)});
      return {ok:true,deviceId:p.deviceId};
    });
  }
  return {claim:(uid,p)=>execute(uid,p,'claim'),transfer:(uid,p)=>execute(uid,p,'transfer'),report:(uid,p)=>execute(uid,p,'execution')};
}
