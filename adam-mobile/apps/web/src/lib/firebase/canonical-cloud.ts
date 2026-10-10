import { collection, doc, getDocsFromServer, limit, query, runTransaction, Timestamp, where } from 'firebase/firestore';
import { getFirebaseAuth, getFirebaseFirestore } from './config';
import { applyCloudDocuments, chooseCloudWrite, normalizeCloudMetadata, toCloudDocuments, type CloudDocument, type CloudDocuments, type DocumentEntry } from './schema-documents';
import type { Companion } from './companion-sync';

const timestampFields = new Set(['createdAt','updatedAt','deletedAt','doneAt','learnedAt','firstSeen','lastSeen']);
function firestoreFields(data: CloudDocument) {
  return Object.fromEntries(Object.entries(data).map(([key,value]) => [key,
    timestampFields.has(key) && typeof value === 'string' ? Timestamp.fromDate(new Date(value)) : value]));
}

/** Per-document transactions keep concurrent app changes and retry safely after
 * partial network failures. No client removes a cloud document. */
export async function exchangeCanonicalCloud(uid: string, input: Companion, assertActive: () => void): Promise<Companion> {
  const db = getFirebaseFirestore(), auth = getFirebaseAuth();
  const check = () => { assertActive(); if (auth.currentUser?.uid !== uid) throw new Error('The signed-in account changed.'); };
  check();
  const local = normalizeCloudMetadata(input), entries = toCloudDocuments(local,uid);
  const documents: CloudDocuments = {};
  async function readCollection(path: string) {
    check();
    const snapshot = await getDocsFromServer(query(collection(db,path),limit(1001)));
    if (snapshot.size > 1000) throw new Error('This cloud collection exceeds the supported limit.');
    check();
    for (const item of snapshot.docs) documents[item.ref.path] = item.data();
  }
  const owned = await getDocsFromServer(query(collection(db,'devices'),where('ownerUid','==',uid),limit(101)));
  check();
  if (owned.size > 100) throw new Error('This account has too many saved ADAM devices.');
  for (const item of owned.docs) {
    const data=item.data();
    if (data.ownerUid === uid && data.kind === 'physical') documents[item.ref.path]=data;
  }
  async function exchange(entry: DocumentEntry) {
    check();
    const ref = doc(db,entry.path);
    const result = await runTransaction(db,async transaction => {
      check();
      const snapshot = await transaction.get(ref), remote = snapshot.exists() ? snapshot.data() : undefined;
      check();
      if (entry.kind === 'devices' && !remote)
        throw new Error('Pair this physical ADAM before claiming its cloud record.');
      const update = chooseCloudWrite(entry.data,remote,entry.kind === 'devices');
      if (update) transaction.set(ref,firestoreFields(update),{ merge:true });
      return update ? { ...remote,...update } : remote!;
    });
    check();
    documents[entry.path] = result;
  }
  await Promise.all([readCollection(`users/${uid}/todos`),readCollection(`users/${uid}/schedules`)]);
  const existingDevices = Object.entries(documents).filter(([path,data])=>path.startsWith('devices/') && path.split('/').length === 2 && !data.deleted);
  for (let offset=0; offset<existingDevices.length; offset+=5) {
    await Promise.all(existingDevices.slice(offset,offset+5).flatMap(([path])=>[
      readCollection(`${path}/memoryFacts`),readCollection(`${path}/memoryPeople`),
    ]));
  }
  // Validate incoming changes before writing any part of this transfer.
  await applyCloudDocuments(local,uid,documents);
  for (const entry of entries.filter(e=>e.kind === 'devices')) await exchange(entry);
  const deviceDocs = Object.entries(documents).filter(([path])=>path.startsWith('devices/') && path.split('/').length === 2);
  // linkedDeviceIds is maintained by trusted claim/transfer, never a client claim.
  const liveDevices = deviceDocs.filter(([,data])=>!data.deleted);
  for (const entry of entries.filter(e=>e.kind !== 'devices')) {
    if (entry.kind === 'memories' && !liveDevices.some(([path])=>entry.path.startsWith(path+'/'))) continue;
    await exchange(entry);
  }
  check();
  return applyCloudDocuments(local,uid,documents);
}
