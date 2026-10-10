/** Read-only preflight. Produces counts and document paths, never record values. */
import {initializeApp,applicationDefault} from 'firebase-admin/app';
import {getFirestore} from 'firebase-admin/firestore';
initializeApp({credential:applicationDefault(),projectId:'adam-ai1'});
const db=getFirestore(), blockers=[],counts={users:0,devices:0,records:0,legacyEnvelopes:0};
async function* scan(collection) {
 let cursor;
 while(true){let q=collection.orderBy('__name__').limit(200);if(cursor)q=q.startAfter(cursor);
  const page=await q.get();for(const doc of page.docs)yield doc;if(page.size<200)return;cursor=page.docs.at(-1);}
}
async function records(ref, kinds) {
 for(const kind of kinds)for await(const doc of scan(ref.collection(kind))){counts.records++;const d=doc.data();
  const errors=[];
  if(d.schemaVersion!==1)errors.push('missing schemaVersion 1');
  for(const field of ['createdAt','updatedAt'])if(typeof d[field]?.toDate!=='function')errors.push(`missing Timestamp ${field}`);
  if(typeof d.deleted!=='boolean' || (d.deleted ? typeof d.deletedAt?.toDate!=='function' : d.deletedAt!==null))errors.push('invalid deletion metadata');
  if(!d.origin)errors.push('missing origin');
  if(kind==='schedules'&&!d.deleted&&(!d.timeZone||d.lastFired||d.snoozes))errors.push('timezone / desired-state migration required');
  if(d.deviceIds?.length>8)errors.push('more than eight explicit targets');
  if(errors.length)blockers.push({path:doc.ref.path,reasons:errors});
 }
}
for await(const user of scan(db.collection('users'))){counts.users++;if(user.data().companion)counts.legacyEnvelopes++;await records(user.ref,['todos','schedules','notes']);}
for await(const device of scan(db.collection('devices'))){counts.devices++;const d=device.data();
 if(!d.ownerUid || (d.ownerId && d.ownerId!==d.ownerUid))blockers.push({path:device.ref.path,reasons:['unresolved owner mapping']});
 const hardware=typeof d.hardwareSerial==='string'&&/^[A-Za-z0-9_-]{1,128}$/.test(d.hardwareSerial)?await db.doc(`hardwareRegistry/${d.hardwareSerial}`).get():null;
 if(!hardware?.exists||hardware.data().deviceId!==device.id||hardware.data().ownerUid!==d.ownerUid||d.kind!=='physical')blockers.push({path:device.ref.path,reasons:['trusted physical enrollment required']});
 await records(device.ref,['memoryFacts','memoryPeople']);
}
console.log(JSON.stringify({readOnly:true,counts,blockers},null,2));
if(blockers.length)process.exitCode=2;
