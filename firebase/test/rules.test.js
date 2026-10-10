import {readFileSync} from 'node:fs';
import {test,before,after,beforeEach} from 'node:test';
import {initializeTestEnvironment,assertFails,assertSucceeds} from '@firebase/rules-unit-testing';
import {doc,setDoc,updateDoc,getDoc,getDocs,collection,query,where,deleteDoc,Timestamp} from 'firebase/firestore';
let env;const stamp=Timestamp.fromMillis(1700000000000),later=Timestamp.fromMillis(1700000001000);
const metadata={schemaVersion:1,createdAt:stamp,updatedAt:stamp,deleted:false,deletedAt:null,origin:'desktop'};
const todo={...metadata,todoId:'one',text:'Review',done:false,due:null,doneAt:null,deviceIds:['ADAM-TEST']};
const schedule={...metadata,scheduleId:'one',kind:'alarm',label:'Morning',at:'2030-10-10T07:00',timeZone:'Asia/Kolkata',enabled:true,repeat:['mon'],deviceIds:[]};
before(async()=>{env=await initializeTestEnvironment({projectId:'demo-adam',firestore:{host:'127.0.0.1',port:8088,rules:readFileSync(new URL('../../adam-mobile/firestore.rules',import.meta.url),'utf8')}});});
after(async()=>{await env.cleanup();});
beforeEach(async()=>{
 await env.clearFirestore();
 await env.withSecurityRulesDisabled(async c=>{
  await setDoc(doc(c.firestore(),'devices/ADAM-TEST'),{ownerUid:'alice',hardwareSerial:'physical-1',deviceId:'ADAM-TEST',kind:'physical',name:'Desk'});
  await setDoc(doc(c.firestore(),'devices/ADAM-OTHER'),{ownerUid:'bob',hardwareSerial:'physical-2'});
 });
});
const db=(uid='alice')=>env.authenticatedContext(uid).firestore();
test('owner catalog query works; foreign and unauthenticated reads fail',async()=>{
 await assertSucceeds(getDocs(query(collection(db(),'devices'),where('ownerUid','==','alice'))));
 await assertFails(getDoc(doc(db('bob'),'devices/ADAM-TEST')));
 await assertFails(getDocs(collection(db(),'devices')));
 await assertFails(getDoc(doc(env.unauthenticatedContext().firestore(),'devices/ADAM-TEST')));
});
test('no self claim, simulator claim, ownership mutation, identity mutation or hard delete',async()=>{
 for(const row of [{ownerUid:'alice',kind:'physical'},{ownerUid:'alice',simulated:true},{ownerId:'alice'}])await assertFails(setDoc(doc(db(),'devices/ADAM-NEW'),row));
 for(const row of [{ownerUid:'bob'},{hardwareSerial:'fake'},{ownerId:'bob'},{kind:'simulated'}])await assertFails(updateDoc(doc(db(),'devices/ADAM-TEST'),row));
 await assertSucceeds(updateDoc(doc(db(),'devices/ADAM-TEST'),{name:'Office',updatedAt:later}));
 await assertFails(deleteDoc(doc(db(),'devices/ADAM-TEST')));
});
test('validated todo lifecycle preserves creation and tombstones; rejects forged fields and scope',async()=>{
 const ref=doc(db(),'users/alice/todos/one');await assertSucceeds(setDoc(ref,todo));
 for(const change of [{syncToken:'secret'},{deviceIds:['ADAM-OTHER']},{done:'yes'},{todoId:'wrong'},{createdAt:later},{text:'x'.repeat(2001)}])await assertFails(updateDoc(ref,{...change,updatedAt:later}));
 await assertSucceeds(updateDoc(ref,{done:true,doneAt:later,updatedAt:later}));
 const newer=Timestamp.fromMillis(1700000002000);await assertSucceeds(updateDoc(ref,{deleted:true,deletedAt:newer,updatedAt:newer}));
 await assertFails(deleteDoc(ref));await assertFails(getDoc(doc(db('bob'),'users/alice/todos/one')));
});
test('schedules cannot inject execution state and require explicit timezone',async()=>{
 await assertSucceeds(setDoc(doc(db(),'users/alice/schedules/one'),schedule));
 for(const row of [{...schedule,timeZone:null},{...schedule,lastFired:'fake'},{...schedule,deviceIds:['ADAM-OTHER']},{...schedule,repeat:['moon']},{...schedule,repeat:['mon','mon']},{...schedule,kind:'timer',durationSeconds:-1,deadline:'bad'}])await assertFails(setDoc(doc(db(),'users/alice/schedules/two'),{...row,scheduleId:'two'}));
 await assertSucceeds(setDoc(doc(db(),'users/alice/schedules/timer'),{...schedule,scheduleId:'timer',kind:'timer',durationSeconds:30,deadline:'2030-10-10T01:30:00.000Z'}));
});
test('metadata memories work; biometrics, credentials and unsupported fields fail',async()=>{
 const fact={...metadata,factId:'one',category:'preference',content:'Tea',confidence:1,source:'manual',learnedAt:stamp};
 await assertSucceeds(setDoc(doc(db(),'devices/ADAM-TEST/memoryFacts/one'),fact));
 await assertFails(setDoc(doc(db('bob'),'devices/ADAM-TEST/memoryFacts/two'),{...fact,factId:'two'}));
 const person={...metadata,personId:'one',name:'Sam',relationship:'Friend',notes:'',faceEncodingId:null,firstSeen:stamp,lastSeen:stamp};
 await assertSucceeds(setDoc(doc(db(),'devices/ADAM-TEST/memoryPeople/one'),person));
 await assertFails(updateDoc(doc(db(),'devices/ADAM-TEST/memoryPeople/one'),{embedding:[1,2],updatedAt:later}));
 await assertFails(deleteDoc(doc(db(),'devices/ADAM-TEST/memoryFacts/one')));
});
test('safe profile updates preserve legacy backup but cannot write it or ownership mirrors',async()=>{
 const ref=doc(db(),'users/alice');await assertSucceeds(setDoc(ref,{email:'a@example.com',displayName:'Alice',photoUrl:null,createdAt:stamp,linkedDeviceIds:[]}));
 await assertSucceeds(updateDoc(ref,{displayName:'A',updatedAt:later}));
 for(const data of [{companion:{}},{roles:['admin']},{linkedDeviceIds:['ADAM-OTHER']}])await assertFails(updateDoc(ref,data));
});
test('notes/settings/client metadata allowed; backend trust, credits and execution are denied',async()=>{
 await assertSucceeds(setDoc(doc(db(),'users/alice/notes/one'),{...metadata,noteId:'one',content:'Remember'}));
 await assertSucceeds(setDoc(doc(db(),'users/alice/settings/companion'),{schemaVersion:1,updatedAt:stamp,origin:'mobile',brain:'byok',voice:'default',wakeWord:'adam'}));
 await assertFails(updateDoc(doc(db(),'users/alice/settings/companion'),{apiKey:'secret'}));
 await assertSucceeds(setDoc(doc(db(),'users/alice/clients/one'),{platform:'windows',protocolVersion:1,lastSeen:stamp}));
 for(const path of ['devices/ADAM-TEST/executionState/one','creditBalances/ADAM-TEST','hardwareRegistry/one','hardwareProofs/one','unknown/one'])await assertFails(setDoc(doc(db(),path),{ownerUid:'alice',status:'applied'}));
});
