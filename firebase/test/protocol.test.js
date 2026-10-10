import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {verifyProof,validateReceipt} from '../src/protocol.js';
import {fixture} from './proof-fixture.js';
test('proof binds signature, owner, purpose, identity epoch and expiry',()=>{
 const f=fixture();assert.equal(verifyProof(f.envelope(),f.hardware,'claim','alice').uid,'alice');
 for(const [hardware,purpose,uid] of [[f.hardware,'claim','bob'],[f.hardware,'execution','alice'],[{...f.hardware,enabled:false},'claim','alice'],[{...f.hardware,ownershipEpoch:1},'claim','alice']]) assert.throws(()=>verifyProof(f.envelope(),hardware,purpose,uid));
 assert.throws(()=>verifyProof(f.envelope({...f.payload,expiresAt:Date.now()-1}),f.hardware,'claim','alice'));
 const e=f.envelope();e.payload=Buffer.from(JSON.stringify({...f.payload,uid:'bob'})).toString('base64');assert.throws(()=>verifyProof(e,f.hardware,'claim','bob'));
});
test('receipt must describe the exact current targeted desired record',()=>{
 const row={scheduleId:'one',deviceIds:['ADAM-TEST'],deleted:false,label:'सुबह',durationSeconds:30};const recordJson=JSON.stringify(row);
 const p={uid:'alice',deviceId:'ADAM-TEST',scheduleId:'one',path:'users/alice/schedules/one',recordJson,appliedRevision:createHash('sha256').update(recordJson).digest('hex'),status:'applied',lastFiredAt:null,lastOccurrenceId:null,issuedAt:Date.now()};
 assert.deepEqual(validateReceipt(p,row),row);
 assert.throws(()=>validateReceipt(p,{...row,label:'Changed'}));
 assert.throws(()=>validateReceipt({...p,status:'fired'},row));
 assert.throws(()=>validateReceipt({...p,appliedRevision:'fake'},row));
});
