/* Robot-owned clock data. No optimistic or whole-list writes. */
'use strict';
window.AdamClock=(()=>{
  let pending=false,reachable=false,writable=false,writing=false;
  let connection={};
  const A=()=>window.ADAM;
  const $=id=>document.getElementById(id);
  function updateAccess(state=connection){
    connection=state;
    writable=reachable&&state.connected===true&&state.read_only===false;
    const reason=state.reason||'Add ADAM’s connection key in Connection and reconnect to make changes.';
    document.querySelectorAll('#scheduleForm input, #scheduleForm select, #scheduleForm button, #todoForm input, #todoForm button, #clockScheduleList button, #clockTodoList button, #clockTodoList input').forEach(control=>control.disabled=!writable||writing);
    $('clockConnectionBtn').hidden=writable;
    if(reachable){
      A().text('clockNotice',writable?'Live from your ADAM. Changes are confirmed by your companion.':reason);
      $('clockNotice').classList.toggle('error',!writable);
    }
  }
  function button(label,icon,handler){
    const b=A().element('button','icon-button');b.type='button';b.title=label;b.setAttribute('aria-label',label);b.append(A().icon(icon));
    b.addEventListener('click',()=>A().run(b,handler));return b;
  }
  function when(s){
    const kind=s.kind||'alarm';
    if(kind==='timer'&&s.fire_at)return 'Timer · '+A().formatDate(s.fire_at);
    if(s.time_of_day)return s.time_of_day+(s.repeat?.weekdays?.length?' · '+s.repeat.weekdays.join(', '):'');
    if(s.fire_at)return A().formatDate(s.fire_at);
    if(s.at)return A().formatDate(s.at);
    return kind==='timer'?(s.seconds||0)+' seconds':kind;
  }
  function render(data){
    const schedules=Array.isArray(data.schedules)?data.schedules:[],todos=Array.isArray(data.todos)?data.todos:[];
    const sr=$('clockScheduleList'),tr=$('clockTodoList'),mr=$('clockMemoryList');
    sr.replaceChildren();tr.replaceChildren();mr.replaceChildren();
    A().text('clockScheduleCount',schedules.length);A().text('clockTodoCount',todos.filter(t=>!t.done).length+' open');
    if(!schedules.length)A().empty(sr,'A little quiet.','Add an alarm or timer to your ADAM.');
    schedules.forEach(s=>{
      const row=A().element('div','list-row'),content=A().element('div');
      content.append(A().element('h4','',s.label||s.kind||'Schedule'),A().element('p','',when(s)));
      row.append(A().icon('clock'),content,button('Delete '+(s.label||s.kind||'schedule'),'trash',async()=>{
        if(!await A().confirmAction('Delete this schedule?','This removes it from your ADAM.','Delete'))return;
        await write('schedules',{cancel:s.id});
      }));sr.append(row);
    });
    if(!todos.length)A().empty(tr,'A clear list.','Add something you want ADAM to remember to do.');
    todos.forEach(t=>{
      const row=A().element('div','list-row'),content=A().element('div'),toggle=A().element('input');
      toggle.type='checkbox';toggle.checked=!!t.done;toggle.setAttribute('aria-label',(t.done?'Mark incomplete: ':'Complete: ')+t.text);
      toggle.addEventListener('change',async()=>{toggle.disabled=true;try{await write('todos',{toggle:t.id});}catch(e){toggle.checked=!!t.done;A().toast(e.message,true);}finally{updateAccess();}});
      content.append(A().element('h4',t.done?'done':'',t.text));if(t.due)content.append(A().element('p','','Due '+t.due));
      row.append(toggle,content,button('Delete '+t.text,'trash',async()=>{
        if(!await A().confirmAction('Delete this to-do?',t.text,'Delete'))return;await write('todos',{delete:t.id});
      }));tr.append(row);
    });
    const memories=Object.entries(data.memories||{});A().text('clockMemoryCount',memories.length);
    if(!memories.length)A().empty(mr,'Nothing on ADAM’s mind yet.','Ask your ADAM to remember something and it will appear here.');
    memories.forEach(([key,value])=>{const card=A().element('article','memory-card');card.append(A().element('span','eyebrow',key),A().element('p','',typeof value==='string'?value:JSON.stringify(value)));mr.append(card);});
  }
  async function refresh(){
    if(pending)return;pending=true;
    try{
      const d=await A().api('/pi/snapshot');reachable=true;
      render(d.data||{});
      updateAccess(d.connection||A().ui.status.robot||{});
      A().text('robotMemoryNotice','Live from your ADAM. These memories stay separate from your account notes.');
    }catch(e){
      reachable=false;A().text('clockNotice','Could not refresh from ADAM. '+e.message);$('clockNotice').classList.add('error');
      A().text('robotMemoryNotice','Connect your ADAM to refresh its memories. '+e.message);
      if(!$('clockScheduleList').children.length)render({});
      updateAccess();
    }finally{pending=false;}
  }
  async function write(target,body){
    if(!reachable)throw new Error('Connect ADAM before changing its clock.');
    if(!writable)throw new Error(connection.reason||'Add ADAM’s connection key in Connection and reconnect to make changes.');
    if(writing)throw new Error('Wait for ADAM to confirm the current change.');
    writing=true;updateAccess();
    try{
      await A().api('/pi/write/'+target,body);
      await refresh();A().toast('Updated on ADAM.');
    }catch(error){
      // A rejected key changes server permissions. Refresh them before the
      // user tries another edit, without retrying the write itself.
      await refresh();throw error;
    }finally{writing=false;updateAccess();}
  }
  function init(){
    A().bind('clockRefreshBtn',refresh);
    A().bind('clockConnectionBtn',async()=>{await A().switchView('devices');$('connectionToken').focus();});
    A().bind('robotMemoriesRefreshBtn',refresh);
    $('robotMemoriesPanel').addEventListener('toggle',()=>{if($('robotMemoriesPanel').open)refresh();});
    $('scheduleKind').addEventListener('change',()=>{
      const timer=$('scheduleKind').value==='timer';
      $('scheduleMinutesLabel').hidden=!timer;$('scheduleTimeLabel').hidden=timer;
      $('scheduleMinutes').required=timer;$('scheduleTime').required=!timer;
      $('scheduleLabel').required=$('scheduleKind').value==='reminder';
    });
    A().submit('scheduleForm',async()=>{
      const kind=$('scheduleKind').value,label=$('scheduleLabel').value.trim();
      if(kind==='timer'){
        const minutes=Number($('scheduleMinutes').value);if(!Number.isFinite(minutes)||minutes<1||minutes>10080)throw new Error('Choose a timer between 1 minute and 7 days.');
        await write('schedules',{kind,minutes,label});
      }else{
        const time=$('scheduleTime').value;if(!time)throw new Error('Choose a time.');
        await write('schedules',{kind,when:time,label});
      }
      $('scheduleLabel').value='';
    });
    A().submit('todoForm',async()=>{const text=$('todoText').value.trim();if(!text)throw new Error('Add some text for your to-do.');await write('todos',{text});$('todoText').value='';});
    updateAccess();
  }
  document.addEventListener('DOMContentLoaded',init);
  return{refresh,updateAccess};
})();
