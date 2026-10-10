import {test,before,after,beforeEach} from 'node:test';
import assert from 'node:assert/strict';
import {randomBytes,createHash} from 'node:crypto';
import {initializeApp,deleteApp} from 'firebase-admin/app';
import {getFirestore,Timestamp} from 'firebase-admin/firestore';
import {workflows} from '../src/workflows.js';
import {fixture} from './proof-fixture.js';
let app,db,service;
before(()=>{app=initializeApp({projectId:'demo-adam'},'workflows');db=getFirestore(app);service=workflows(db);});
after(async()=>{await db.terminate();await deleteApp(app);});
beforeEach(async()=>{for(const c of ['devices','hardwareRegistry','hardwareProofs','users'])await db.recursiveDelete(db.collection(c));});
async function setup(){const f=fixture();await db.doc('hardwareRegistry/hardware-1').set(f.hardware);return f;}
test('registered signed claim is atomic and idempotent; arbitrary client payload cannot claim',async()=>{
 const f=await setup();await service.claim('alice',f.envelope());await service.claim('alice',f.envelope());
 assert.equal((await db.doc('devices/ADAM-TEST').get()).data().ownerUid,'alice');
 await assert.rejects(service.claim('bob',f.envelope()));
 const changed={...f.payload,uid:'bob',nonce:randomBytes(16).toString('hex')};await assert.rejects(service.claim('bob',f.envelope(changed)));
});
test('execution requires registered signature, current desired revision, owner and monotonic evidence',async()=>{
 const f=await setup();await service.claim('alice',f.envelope());
 const record={schemaVersion:1,scheduleId:'one',deviceIds:[],deleted:false,label:'Morning',updatedAt:'2026-10-10T00:00:00.000Z'};
 await db.doc('users/alice/schedules/one').set({...record,updatedAt:Timestamp.fromDate(new Date(record.updatedAt))});
 const recordJson=JSON.stringify(record), proof={...f.payload,purpose:'execution',nonce:'b'.repeat(32),path:'users/alice/schedules/one',scheduleId:'one',recordJson,appliedRevision:createHash('sha256').update(recordJson).digest('hex'),status:'applied',lastFiredAt:null,lastOccurrenceId:null};
 await service.report('alice',f.envelope(proof));
 assert.equal((await db.doc('devices/ADAM-TEST/executionState/one').get()).data().appliedRevision,proof.appliedRevision);
 await db.doc('users/alice/schedules/one').update({label:'Changed'});
 await assert.rejects(service.report('alice',f.envelope({...proof,nonce:'c'.repeat(32)})));
 await db.doc('devices/ADAM-TEST').update({ownerUid:'bob'});
 await assert.rejects(service.report('alice',f.envelope({...proof,nonce:'d'.repeat(32)})));
});
test('transfer blocks private-data leakage, advances epoch and fences old owner evidence',async()=>{
 const f=await setup();await service.claim('alice',f.envelope());
 const p={...f.payload,purpose:'transfer',newUid:'bob',nonce:'c'.repeat(32)};
 await db.doc('devices/ADAM-TEST/memoryFacts/private').set({content:'Private'});
 await assert.rejects(service.transfer('alice',f.envelope(p)));
 await db.doc('devices/ADAM-TEST/memoryFacts/private').delete();
 await service.transfer('alice',f.envelope(p));
 const d=(await db.doc('devices/ADAM-TEST').get()).data();assert.equal(d.ownerUid,'bob');assert.equal(d.ownershipEpoch,1);
 await assert.rejects(service.claim('alice',f.envelope()));
});
