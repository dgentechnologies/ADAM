import assert from 'node:assert/strict';
import { test } from 'node:test';
import { emptyCompanion, capturePhoneChanges, mergeCompanions } from '../apps/web/src/lib/firebase/companion-sync';
import { applyCloudDocuments, chooseCloudWrite, documentUpdatedAt, normalizeCloudMetadata, toCloudDocuments, type CloudDocuments } from '../apps/web/src/lib/firebase/schema-documents';
import { exchangeSimulatedDocuments } from '../apps/web/src/lib/ble-simulation';
import { LocalData } from '../apps/web/src/lib/local-data';

const uid='schema-test-account', time='2026-10-09T10:00:00.000Z', later='2026-10-09T11:00:00.000Z';
const d1='devices/ADAM-1111',d2='devices/ADAM-2222';
function documents(): CloudDocuments { return {
  [d1]:{deviceId:'ADAM-1111',ownerUid:uid,name:'Desk',hardwareSerial:'DGEN-1',createdAt:time,updatedAt:time,status:'offline',simulated:false,wifiSsid:'Keep my network'},
  [d2]:{deviceId:'ADAM-2222',ownerUid:uid,name:'Kitchen',hardwareSerial:'DGEN-2',createdAt:time,updatedAt:time,status:'offline',simulated:false},
  [`${d1}/memoryFacts/chair`]:{factId:'chair',category:'Chair',content:'Left of the desk',confidence:0.8,source:'conversation',learnedAt:time,updatedAt:time},
  [`${d2}/memoryFacts/chair`]:{factId:'chair',category:'Chair',content:'By the kitchen window',confidence:1,source:'manual',learnedAt:time,updatedAt:time},
  [`${d1}/memoryPeople/person_1`]:{personId:'person_1',name:'Alex',relationship:'Friend',notes:'Likes tea',faceEncodingId:'local-face-reference',firstSeen:time,lastSeen:time,updatedAt:time},
  [`users/${uid}/schedules/al_c8e761a3`]:{scheduleId:'al_c8e761a3',kind:'alarm',label:'Morning',timeZone:'Asia/Kolkata',at:'2026-11-01T07:00',timeOfDay:'07:00',repeat:['mon','tue'],enabled:true,deviceIds:[],lastFired:'2026-10-09T07:00',snoozes:1,createdAt:time,updatedAt:time,deleted:false,deletedAt:null,origin:'desktop'},
  [`users/${uid}/todos/td_plan`]:{todoId:'td_plan',text:'Review notes',done:false,due:null,deviceIds:['ADAM-1111'],createdAt:time,updatedAt:time,doneAt:null,deleted:false,deletedAt:null,origin:'desktop'},
}; }
test('canonical documents round-trip stable hardware IDs, Pi IDs and every schedule field',async()=>{
  const incoming=documents(),local=await applyCloudDocuments(emptyCompanion(),uid,incoming);
  const outgoing=Object.fromEntries(toCloudDocuments(local,uid).map(e=>[e.path,e.data]));
  assert.equal(Object.keys(local.devices).length,2);
  assert.equal(Object.keys(local.memories).length,3);
  assert.equal(new Set(Object.keys(local.memories)).size,3);
  assert.deepEqual(Object.keys(outgoing).sort(),Object.keys(incoming).sort());
  const schedule=outgoing[`users/${uid}/schedules/al_c8e761a3`];
  for(const key of ['scheduleId','at','timeZone','timeOfDay','repeat','deviceIds']) assert.deepEqual(schedule[key],incoming[`users/${uid}/schedules/al_c8e761a3`][key]);
  assert.equal(schedule.lastFired,null);assert.equal(schedule.snoozes,0);
  assert.equal(outgoing[`${d1}/memoryPeople/person_1`].faceEncodingId,'local-face-reference');
  assert.equal(outgoing[`${d1}/memoryFacts/chair`].confidence,0.8);
  assert.equal(outgoing[d1].deviceId,'ADAM-1111');
  assert.ok(!Object.keys(outgoing).some(path=>path.endsWith('/companion')));
});
test('wall-clock alarm intent is unchanged when the importing phone changes timezone',async()=>{
  const previous=process.env.TZ;
  try {
    process.env.TZ='Asia/Kolkata';
    const local=await applyCloudDocuments(emptyCompanion(),uid,documents());
    process.env.TZ='America/New_York';
    const clock=toCloudDocuments(local,uid).find(e=>e.kind==='clocks')!.data;
    assert.equal(clock.timeZone,'Asia/Kolkata');assert.equal(clock.at,'2026-11-01T07:00');assert.equal(clock.timeOfDay,'07:00');
    assert.deepEqual(clock.repeat,['mon','tue']);
  } finally { if(previous===undefined)delete process.env.TZ;else process.env.TZ=previous; }
});
test('equal cloud timestamps do not overwrite; execution fields stay out of desired state',()=>{
  const remote=documents()[`users/${uid}/schedules/al_c8e761a3`];
  assert.equal(chooseCloudWrite({...remote,label:'different'},remote),null);
  const write=chooseCloudWrite({...remote,updatedAt:later,label:'Renamed',lastFired:null},remote)!;
  assert.equal(write.label,'Renamed');assert.equal(write.lastFired,null);
});
test('device rename keeps ownership and existing network/profile metadata',()=>{
  const remote=documents()[d1];
  const update=chooseCloudWrite({...remote,name:'Library',updatedAt:later,wifiSsid:null,status:'setup'},remote,true)!;
  assert.equal(update.name,'Library');assert.ok(!('wifiSsid' in update));assert.ok(!('status' in update));
  assert.equal({...remote,...update}.wifiSsid,'Keep my network');
  assert.throws(()=>chooseCloudWrite({...remote,ownerUid:'other',updatedAt:later},remote,true),/belongs/);
});
test('two room memories with the same cloud fact ID never collapse into one',async()=>{
  const first=await applyCloudDocuments(emptyCompanion(),uid,documents());
  const again=await applyCloudDocuments(first,uid,documents());
  assert.deepEqual(again,first);
  const facts=Object.values(first.memories).filter((m:any)=>m.cloudId==='chair') as any[];
  assert.equal(facts.length,2);assert.notEqual(facts[0].id,facts[1].id);assert.notEqual(facts[0].deviceId,facts[1].deviceId);
});
test('canonical memory and planner deletions retain their original document paths',async()=>{
  const local=await applyCloudDocuments(emptyCompanion(),uid,documents());
  const live=(items:any)=>Object.values(items).filter((i:any)=>!i.deleted).map(({deleted,...i}:any)=>i);
  const before=LocalData.parse({version:1,facts:live(local.memories),todos:live(local.todos),clocks:live(local.clocks),devices:live(local.devices)});
  const after=LocalData.parse({...before,facts:before.facts.filter((i:any)=>i.deviceId!=='ADAM-1111'),todos:[]});
  const changes=capturePhoneChanges(local,before,after,later);
  const outgoing=Object.fromEntries(toCloudDocuments(changes,uid).map(e=>[e.path,e.data]));
  assert.equal(outgoing[`${d1}/memoryFacts/chair`].deleted,true);
  assert.equal(outgoing[`${d1}/memoryPeople/person_1`].deleted,true);
  assert.equal(outgoing[`users/${uid}/todos/td_plan`].deletedAt,later);
  assert.equal(outgoing[`${d2}/memoryFacts/chair`].deleted,false);
  assert.equal(Object.values(mergeCompanions(changes,local).todos)[0]!.deleted,true);
});
test('simulated BLE sends only the selected robot memories and targeted or shared plans',async()=>{
  const local=await applyCloudDocuments(emptyCompanion(),uid,documents());
  const desk=exchangeSimulatedDocuments(local,uid,'ADAM-1111',{});
  const kitchen=exchangeSimulatedDocuments(local,uid,'ADAM-2222',{});
  assert.ok(desk[`${d1}/memoryFacts/chair`]);assert.ok(!desk[`${d2}/memoryFacts/chair`]);
  assert.ok(kitchen[`${d2}/memoryFacts/chair`]);assert.ok(!kitchen[`${d1}/memoryFacts/chair`]);
  assert.ok(desk[`users/${uid}/todos/td_plan`]);assert.ok(!kitchen[`users/${uid}/todos/td_plan`]);
  assert.ok(desk[`users/${uid}/schedules/al_c8e761a3`]);assert.ok(kitchen[`users/${uid}/schedules/al_c8e761a3`]);
});
test('unassigned phone memories and preferences never acquire an invented cloud home',()=>{
  const local=emptyCompanion(),id='00000000-0000-0000-0000-000000000001';
  local.memories[id]={id,title:'Private',text:'On this phone',kind:'fact',createdAt:time,updatedAt:time,deleted:false};
  local.preferences.voice={value:'Puck',updatedAt:time};
  assert.deepEqual(toCloudDocuments(local,uid),[]);
  assert.deepEqual(normalizeCloudMetadata(local).memories,local.memories);
});
test('missing timestamps, invalid wall times and foreign ownership are rejected',async()=>{
  assert.throws(()=>documentUpdatedAt({}),/timestamp/);
  const invalid=documents();invalid[`users/${uid}/schedules/al_c8e761a3`].at='2026-02-30T07:00';
  await assert.rejects(applyCloudDocuments(emptyCompanion(),uid,invalid));
  const foreign=documents();foreign[d1].ownerUid='different-account';
  await assert.rejects(applyCloudDocuments(emptyCompanion(),uid,foreign),/ownership/);
});

test('simulated devices never produce cloud ownership writes',async()=>{
 const local=await applyCloudDocuments(emptyCompanion(),uid,documents());
 for(const record of Object.values(local.devices)) (record as any).simulated=true;
 assert.ok(toCloudDocuments(local,uid).every(e=>e.kind!=='devices'));
});
