const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {JSDOM}=require('jsdom');
const tick=()=>new Promise(resolve=>setImmediate(resolve));
test('shared planner and device controls preserve IDs and clear editors on account switch',async t=>{
  const root=path.join(__dirname,'../resources/static');
  const dom=new JSDOM(fs.readFileSync(path.join(root,'index.html'),'utf8'),{runScripts:'outside-only',url:'http://127.0.0.1:8642'});t.after(()=>dom.window.close());
  const w=dom.window;await new Promise(resolve=>w.addEventListener('load',resolve,{once:true}));
  const data={todos:[],clocks:[],devices:[]},calls=[];
  w.ADAM={ui:{account:{user:{uid:'one'},signed_in:true}},element:(tag,cls,text)=>{const el=w.document.createElement(tag);el.className=cls||'';if(text!==undefined)el.textContent=text;return el;},text:(id,value)=>w.document.getElementById(id).textContent=value,toast:()=>{},confirmAction:async()=>true,run:async(b,fn)=>{b.disabled=true;try{await fn();}finally{b.disabled=false;}},api:async(url,body)=>{
    calls.push({url,body});const kind=url.split('/')[2];if(body){const item={...body,id:body.id||w.crypto.randomUUID()};data[kind]=[...data[kind].filter(v=>v.id!==item.id),item];}return {items:data[kind]};
  }};
  let init;w.document.addEventListener=(event,fn)=>{if(event==='DOMContentLoaded')init=fn;};w.eval(fs.readFileSync(path.join(root,'js/companion.js'),'utf8'));init();await w.AdamShared.refresh();
  const form=()=>w.document.querySelector('#sharedDevices form');
  w.document.getElementById('shared-deviceName').value='Desk ADAM';form().dispatchEvent(new w.Event('submit',{cancelable:true}));await tick();
  w.document.getElementById('shared-deviceName').value='Studio ADAM';form().dispatchEvent(new w.Event('submit',{cancelable:true}));await tick();
  assert.equal(data.devices.length,2);const first=data.devices[0].id;
  [...w.document.querySelectorAll('#sharedDevices button')].find(b=>b.textContent==='Edit').click();await tick();w.document.getElementById('shared-deviceName').value='Library ADAM';form().dispatchEvent(new w.Event('submit',{cancelable:true}));await tick();assert.equal(data.devices.find(d=>d.id===first).name,'Library ADAM');
  w.document.getElementById('shared-todoText').value='Review';w.document.querySelector('#sharedTodos form').dispatchEvent(new w.Event('submit',{cancelable:true}));await tick();assert.equal(data.todos[0].text,'Review');
  [...w.document.querySelectorAll('#sharedTodos button')].find(b=>b.textContent==='Complete').click();await tick();assert.equal(data.todos[0].done,true);
  w.ADAM.ui.account.user.uid='two';w.AdamShared.accountChanged();assert.equal(w.document.getElementById('sharedDevices').children.length,0);assert.equal(w.document.getElementById('sharedTodos').children.length,0);
});
