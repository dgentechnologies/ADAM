/* Account planner and device records share the Android v2 protocol. */
'use strict';
window.AdamShared=(()=>{
  const A=()=>window.ADAM,$=id=>document.getElementById(id);
  let state={todos:[],clocks:[],devices:[]},editing={},pending=false;
  const node=(tag,text='',cls='')=>A().element(tag,cls,text);
  const local=value=>{const d=new Date(value);return new Date(d.getTime()-d.getTimezoneOffset()*60000).toISOString().slice(0,16);};
  function button(text,action){const b=node('button',text,'button subtle');b.type='button';b.onclick=()=>A().run(b,action);return b;}
  function field(form,label,key,type,value='',max=80){const l=node('label',label),i=node('input');i.name=key;i.id='shared-'+key;i.type=type;i.value=value;i.maxLength=max;l.append(i);form.append(l);return i;}
  function select(form,label,key,options,value){const l=node('label',label),s=node('select');s.name=key;options.forEach(([id,name])=>s.append(new Option(name,id)));s.value=value;l.append(s);form.append(l);return s;}
  async function save(kind,value){await A().api('/companion/'+kind,value);editing[kind]=null;await refresh();A().toast('Saved. Ready for account or simulated BLE sync.');}
  async function remove(kind,item){if(!await A().confirmAction('Delete this saved item?','The deletion will sync with your other devices.','Delete'))return;await A().api('/companion/'+kind+'/delete',{id:item.id});await refresh();}
  function renderCollection(kind,root){
    root.replaceChildren();root.append(node('h4',kind==='todos'?'To-dos':kind==='clocks'?'Alarms, timers & reminders':'Saved ADAM devices'));
    state[kind].forEach(item=>{
      const row=node('article','','list-row'),body=node('div');body.append(node('p',item.text||item.name||item.label||item.kind));
      const detail=kind==='devices'?(item.simulated?'Simulated ADAM · ':'ADAM · ')+item.serial:kind==='clocks'?new Date(item.when).toLocaleString()+(item.enabled?' · Enabled':' · Paused'):item.dueAt?'Due '+new Date(item.dueAt).toLocaleString():item.done?'Completed':'Open';body.append(node('p',detail,'help'));row.append(body);
      if(kind==='todos')row.append(button(item.done?'Reopen':'Complete',()=>save(kind,{...item,done:!item.done})));
      if(kind==='clocks')row.append(button(item.enabled?'Pause':'Enable',()=>save(kind,{...item,enabled:!item.enabled})));
      row.append(button('Edit',()=>{editing[kind]=item;renderCollection(kind,root);root.querySelector('form input').focus();}),button('Delete',()=>remove(kind,item)));
      if(kind==='devices'&&item.simulated)row.append(button('Sync simulated BLE',async()=>{await A().api('/companion/ble-sync',{deviceId:item.id});await refresh();A().toast('Simulated BLE transfer complete.');}));
      root.append(row);
    });
    const edit=editing[kind],form=node('form','','form-stack');root.append(form);const controls={};
    if(kind==='devices')controls.name=field(form,'Device name','deviceName','text',edit?.name||'',40);
    else{
      if(kind==='todos'){
        controls.text=field(form,edit?'Edit to-do':'New to-do','todoText','text',edit?.text||'',2000);
        controls.dueAt=field(form,'Due date (optional)','todoDue','datetime-local',edit?.dueAt?local(edit.dueAt):'');
      }else{
        controls.kind=select(form,'Type','clockKind',[['alarm','Alarm'],['timer','Timer'],['reminder','Reminder']],edit?.kind||'alarm');
        controls.label=field(form,'Label','clockLabel','text',edit?.label||'');
        controls.when=field(form,'When','clockWhen','datetime-local',edit?local(edit.when):'');
        controls.minutes=field(form,'Minutes','clockMinutes','number','5');controls.minutes.min=1;controls.minutes.max=10080;
        const toggle=()=>{const timer=controls.kind.value==='timer'&&!edit;controls.minutes.parentElement.hidden=!timer;controls.minutes.required=timer;controls.when.parentElement.hidden=timer;controls.when.required=!timer;controls.label.required=controls.kind.value==='reminder';};controls.kind.onchange=toggle;toggle();
      }
      controls.deviceId=select(form,'ADAM','planDevice',[['','All my ADAMs'],...state.devices.map(d=>[d.id,d.name])],edit?.deviceId||'');
    }
    if(controls.name)controls.name.required=true;if(controls.text)controls.text.required=true;
    const submit=node('button',edit?'Save changes':kind==='devices'?'Add simulated ADAM':kind==='todos'?'Add to-do':'Add plan','button');submit.type='submit';form.append(submit);
    form.onsubmit=event=>{event.preventDefault();A().run(submit,async()=>{
      let record={...edit};
      if(kind==='devices')record={...record,name:controls.name.value,serial:edit?.serial||'SIM-'+crypto.randomUUID().slice(0,8).toUpperCase(),simulated:edit?.simulated??true};
      else if(kind==='todos')record={...record,text:controls.text.value,done:edit?.done??false,dueAt:controls.dueAt.value?new Date(controls.dueAt.value).toISOString():'',deviceId:controls.deviceId.value};
      else record={...record,kind:controls.kind.value,label:controls.label.value,when:controls.kind.value==='timer'&&!edit?new Date(Date.now()+Number(controls.minutes.value)*60000).toISOString():new Date(controls.when.value).toISOString(),enabled:edit?.enabled??true,deviceId:controls.deviceId.value};
      await save(kind,record);
    });};
    if(edit)form.append(button('Cancel',()=>{editing[kind]=null;renderCollection(kind,root);}));
  }
  async function refresh(){
    if(pending)return;pending=true;const uid=A().ui.account.user?.uid;
    try{const results=await Promise.all(['todos','clocks','devices'].map(k=>A().api('/companion/'+k)));if(uid!==A().ui.account.user?.uid)return;
      ['todos','clocks','devices'].forEach((k,i)=>{state[k]=results[i].items;});renderCollection('todos',$('sharedTodos'));renderCollection('clocks',$('sharedClocks'));renderCollection('devices',$('sharedDevices'));
      A().text('sharedPlannerMessage',A().ui.account.signed_in?'Saved to this account. Use Profile & settings to sync with Android.':'Saved on this computer. Sign in and import local items to share them.');
    }catch(err){A().text('sharedPlannerMessage',err.message);A().toast(err.message,true);}finally{pending=false;}
  }
  document.addEventListener('DOMContentLoaded',()=>{
    const grid=node('div','','two-column');['sharedTodos','sharedClocks'].forEach(id=>{const section=node('section','','form-stack');section.id=id;grid.append(section);});$('sharedPlanner').append(grid);
    const devices=node('article','','panel');devices.id='sharedDevices';$('tab-devices').prepend(devices);$('sharedRefresh').onclick=refresh;
  });
  function accountChanged(){state={todos:[],clocks:[],devices:[]};editing={};['sharedTodos','sharedClocks','sharedDevices'].forEach(id=>$(id)?.replaceChildren());}
  return{refresh,accountChanged};
})();
