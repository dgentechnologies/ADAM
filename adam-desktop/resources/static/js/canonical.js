/* Canonical account data. A cloud commit never claims the robot applied it. */
window.AdamShared=(()=>{
 const A=()=>window.ADAM,$=id=>document.getElementById(id);
 let data={documents:{},outbox:[],conflicts:{}},pending=false;
 const el=(tag,cls,text)=>A().element(tag,cls,text);
 function button(label,fn){const b=el('button','button',label);b.type='button';b.onclick=()=>A().run(b,fn);return b;}
 async function save(path,record){await A().api('/sync/records',{path,record});await refresh();A().toast('Saved locally. Sync to share this change.');}
 function render(root,memories=false){
  root.replaceChildren();const heading=el('div','panel-heading');heading.append(el('h2','',memories?'Your ADAM’s memories':'Your shared plans'),button('Sync now',async()=>{await A().api('/sync/run',{}, {timeout:60000});await refresh();}));heading.append(button('Sync with ADAM',async()=>{await A().api('/sync/robot',{}, {timeout:60000});await refresh();try{const report=await A().api('/sync/execution',{}, {timeout:90000});A().toast(report.more?'ADAM synced. More status updates will be shared on the next sync.':'ADAM synced and its status verified.');}catch(e){A().toast(e.message,true);}}));root.append(heading);
  root.append(el('p','help','Shared through your account. Robot delivery is separate from cloud sync.'));
  const entries=Object.entries(data.documents).filter(([path,d])=>(!d.deleted||data.conflicts[path])&&(memories?path.includes('/memory'):path.includes('/todos/')||path.includes('/schedules/')));
  if(!entries.length)root.append(el('p','empty-state',memories?'No device memories yet. Sync to load your ADAM’s saved people and facts.':'A little room for what comes next. Add a plan below.'));
  for(const [path,d] of entries){
   const row=el('div','list-row'),info=el('div');info.append(el('h3','',d.text||d.label||d.content||d.name||'Untitled'));
   const delivery=Object.values(data.bridge||{}).map(rows=>rows[path]).filter(Boolean);
   const phase=data.conflicts[path]?'Conflict — choose a version below':data.outbox.includes(path)?'Saved locally · waiting for cloud':delivery.some(r=>r.status==='robot_applied')?'Applied by ADAM':delivery.some(r=>r.status==='conflict')?'Cloud / robot conflict — changes preserved':'Synced to cloud · awaiting ADAM';
   info.append(el('p','help',[d.at,d.timeZone,phase].filter(Boolean).join(' · ')));row.append(info);
   if(path.includes('/todos/'))row.append(button(d.done?'Reopen':'Complete',()=>save(path,{...d,done:!d.done,doneAt:d.done?null:new Date().toISOString()})));
   row.append(button('Edit',()=>editRecord(root,path,d)));
   row.append(button('Delete',async()=>{if(await A().confirmAction('Delete this item?','The deletion will sync to your account.','Delete'))await save(path,{...d,deleted:true});}));root.append(row);
   if(data.conflicts[path]){const choices=el('div','button-row');choices.append(button('Keep this computer’s change',()=>resolve(path,'local')),button('Use cloud version',()=>resolve(path,'remote')));root.append(choices);}
  }
  if(!memories)editor(root);else memoryEditor(root);
  const err=data.sync?.error;if(err)root.append(el('p','help error',err));
 }
 async function resolve(path,choice){await A().api('/sync/resolve',{path,choice});await refresh();}
 function editor(root){
  const details=el('details','');details.append(el('summary','','Add a to-do, alarm or reminder'));const form=el('form','form-stack');
  const label=(text,input)=>{const l=el('label','',text);l.append(input);form.append(l);return input;};
  const kind=el('select');for(const [v,t] of [['todos','To-do'],['alarm','Alarm'],['reminder','Reminder'],['timer','Timer']]){const o=el('option','',t);o.value=v;kind.append(o);}label('Type',kind);
  const content=label('What would you like to remember?',el('input'));content.required=true;content.maxLength=2000;
  const when=label('Date and time',el('input'));when.type='datetime-local';when.parentElement.hidden=true;
  const zone=label('Time zone',el('input'));zone.value=Intl.DateTimeFormat().resolvedOptions().timeZone||'Etc/UTC';zone.parentElement.hidden=true;
  const minutes=label('Minutes',el('input'));minutes.type='number';minutes.min=1;minutes.max=10080;minutes.value=5;minutes.parentElement.hidden=true;
  kind.onchange=()=>{when.parentElement.hidden=['todos','timer'].includes(kind.value);zone.parentElement.hidden=kind.value==='todos';when.required=['alarm','reminder'].includes(kind.value);zone.required=kind.value!=='todos';minutes.parentElement.hidden=kind.value!=='timer';};
  const submit=el('button','button primary','Save locally');submit.type='submit';form.append(submit);
  form.onsubmit=event=>{event.preventDefault();if(!form.reportValidity())return;A().run(submit,async()=>{
   const uid=A().ui.account.user?.uid;if(!uid)throw new Error('Sign in first.');const id=crypto.randomUUID();let path,record;
   if(kind.value==='todos'){path=`users/${uid}/todos/${id}`;record={todoId:id,text:content.value.trim(),done:false,due:null,doneAt:null,deviceIds:[]};}
   else{path=`users/${uid}/schedules/${id}`;record={scheduleId:id,kind:kind.value,label:content.value.trim(),at:when.value,timeZone:zone.value.trim(),enabled:true,deviceIds:[],repeat:null,timeOfDay:null,lastFired:null,snoozes:0};}
   if(kind.value==='timer'){const deadline=new Date(Date.now()+Number(minutes.value)*60000);record.at=new Intl.DateTimeFormat('sv-SE',{timeZone:zone.value.trim(),year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hourCycle:'h23'}).format(deadline).replace(' ','T');record.durationSeconds=Number(minutes.value)*60;record.deadline=deadline.toISOString();}
   await save(path,record);
  });};details.append(form);root.append(details);
 }

 function input(form,title,value,type='text'){
  const label=el('label','',title),field=el('input');field.type=type;field.value=value||'';label.append(field);form.append(label);return field;
 }
 function editRecord(root,path,record){
  const form=el('form','form-stack');form.setAttribute('aria-label','Edit saved record');
  const key=path.includes('/todos/')?'text':path.includes('/schedules/')?'label':path.includes('/memoryFacts/')?'content':'name';
  const content=input(form,'Text',record[key]);content.required=true;content.maxLength=2000;
  let at,zone;if(path.includes('/schedules/')){at=input(form,'Date and time',record.at,'datetime-local');zone=input(form,'Time zone',record.timeZone||Intl.DateTimeFormat().resolvedOptions().timeZone);at.required=zone.required=true;}
  const submit=el('button','button primary','Save changes');submit.type='submit';form.append(submit,button('Cancel',()=>form.remove()));
  form.onsubmit=e=>{e.preventDefault();A().run(submit,async()=>{await save(path,{...record,[key]:content.value.trim(),...(at?{at:at.value,timeZone:zone.value.trim()}: {})});});};root.append(form);content.focus();
 }
 function memoryEditor(root){
  const details=el('details');details.append(el('summary','','Add a memory for this ADAM'));const form=el('form','form-stack');
  const title=input(form,'Category or name',''),content=input(form,'What should ADAM remember?','');title.required=content.required=true;title.maxLength=80;content.maxLength=2000;
  const kind=el('select');for(const [v,t] of [['memoryFacts','Fact'],['memoryPeople','Person']]){const o=el('option','',t);o.value=v;kind.append(o);}form.append(kind);
  const submit=el('button','button primary','Save memory');submit.type='submit';form.append(submit);details.append(form);root.append(details);
  form.onsubmit=e=>{e.preventDefault();A().run(submit,async()=>{
   const device=window.AdamOnboarding.deviceId;if(!device)throw new Error('Select your ADAM first.');const id=crypto.randomUUID(),stamp=new Date().toISOString();
   const record=kind.value==='memoryFacts'?{factId:id,category:title.value.trim(),content:content.value.trim(),confidence:1,source:'manual',learnedAt:stamp}:{personId:id,name:title.value.trim(),notes:content.value.trim(),relationship:'',faceEncodingId:null,firstSeen:stamp,lastSeen:stamp};
   await save(`devices/${device}/${kind.value}/${id}`,record);
  });};
 }
 async function refresh(){if(pending)return;pending=true;try{const uid=A().ui.account.user?.uid;const result=await A().api('/sync/records');if(uid!==A().ui.account.user?.uid)return;data=result;render($('canonicalPlans'));render($('canonicalMemories'),true);}catch(e){A().toast(e.message,true);}finally{pending=false;}}
 function accountChanged(){data={documents:{},outbox:[],conflicts:{}};$('canonicalPlans')?.replaceChildren();$('canonicalMemories')?.replaceChildren();}
 document.addEventListener('DOMContentLoaded',()=>{
  $('sharedPlannerPanel').hidden=true;
  const plans=el('article','panel');plans.id='canonicalPlans';$('tab-clock').prepend(plans);
  [...$('tab-memories').children].forEach(n=>n.hidden=true);
  const memory=el('article','panel');memory.id='canonicalMemories';$('tab-memories').prepend(memory);
 });
 return{refresh,accountChanged};
})();
