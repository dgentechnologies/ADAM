import test from 'node:test';
import assert from 'node:assert/strict';
import {fixture} from './proof-fixture.js';
import {memoryDB} from './memory-db.js';
import {workflows} from '../src/workflows.js';
test('trusted claim works without a profile stub and blocks a second owner',async()=>{
 const f=fixture(),db=memoryDB(),service=workflows(db);db.rows.set('hardwareRegistry/hardware-1',f.hardware);
 await service.claim('alice',f.envelope());assert.equal(db.rows.get('devices/ADAM-TEST').ownerUid,'alice');assert.equal(db.rows.has('users/alice'),false);
 await service.claim('alice',f.envelope());
 await assert.rejects(service.claim('bob',f.envelope({...f.payload,uid:'bob',nonce:'b'.repeat(32)})));
});
test('transfer refuses private collections without changing cloud ownership',async()=>{
 const f=fixture(),db=memoryDB(),service=workflows(db);db.rows.set('hardwareRegistry/hardware-1',f.hardware);await service.claim('alice',f.envelope());
 db.rows.set('devices/ADAM-TEST/memoryFacts/private',{content:'Private'});
 const proof=f.envelope({...f.payload,purpose:'transfer',newUid:'bob',nonce:'c'.repeat(32)});
 await assert.rejects(service.transfer('alice',proof));assert.equal(db.rows.get('devices/ADAM-TEST').ownerUid,'alice');
 db.rows.delete('devices/ADAM-TEST/memoryFacts/private');await service.transfer('alice',proof);
 assert.equal(db.rows.get('devices/ADAM-TEST').ownerUid,'bob');assert.equal(db.rows.get('devices/ADAM-TEST').ownershipEpoch,1);
 await assert.rejects(service.claim('alice',f.envelope()));
});
