/* Permissions and local control preview. The backend remains authoritative. */
'use strict';
const CONTROL_GROUPS={
  media:{label:'Sound & media',icon:'volume'},
  display:{label:'Display',icon:'sun'},
  clipboard:{label:'Clipboard',icon:'clipboard'},
  security:{label:'Security',icon:'shield'},
  workspace:{label:'Workspace',icon:'code'}
};
const CONTROL_LABELS={
  volume_up:['Volume up','Raise volume by 10%.','volume'],
  volume_down:['Volume down','Lower volume by 10%.','volume'],
  volume_set:['Set volume','Choose an exact listening level.','volume'],
  volume_mute:['Mute audio','Turn the sound off.','mute'],
  volume_unmute:['Unmute audio','Bring the sound back.','volume'],
  media_play_pause:['Play / pause','Control your active media player.','play'],
  media_next:['Next track','Skip to the next track.','next'],
  media_previous:['Previous track','Return to the previous track.','previous'],
  brightness_up:['Brightness up','Brighten a supported display by 10%.','sun'],
  brightness_down:['Brightness down','Dim a supported display by 10%.','sun'],
  brightness_set:['Set brightness','Choose an exact display level.','sun'],
  lock_screen:['Lock computer','Lock Windows and keep your session private.','lock'],
  read_clipboard:['Read clipboard','Read text you have copied on this PC.','clipboard'],
  write_clipboard:['Copy text','Place text on the Windows clipboard.','clipboard'],
  clipboard_paste:['Copy & paste text','Paste text into the currently focused app.','clipboard'],
  dispatch_coding_task:['Start a coding task','Let ADAM use your selected coding workspace.','code'],
  check_coding_task_status:['Check task progress','Read the active or latest task status.','activity'],
  cancel_coding_task:['Stop a coding task','Stop the running assistant process.','pause']
};
function controlGroup(name,spec){
  if(name.startsWith('brightness_'))return 'display';
  if(/clipboard/.test(name))return 'clipboard';
  if(/coding/.test(name))return 'workspace';
  if(spec.category==='Media')return 'media';
  return 'security';
}
function controlPresentation(name,spec){return CONTROL_LABELS[name]||[friendly(name),spec.description||'Control this computer.',CONTROL_GROUPS[controlGroup(name,spec)].icon];}
function filteredControls(){
  const query=$('actionSearch').value.trim().toLowerCase();
  const order=Object.keys(CONTROL_LABELS);
  return Object.entries(ui.actions).sort(([a],[b])=>(order.indexOf(a)<0?999:order.indexOf(a))-(order.indexOf(b)<0?999:order.indexOf(b))).filter(([name,spec])=>{
    const [label,description]=controlPresentation(name,spec);
    return (!query?controlGroup(name,spec)===ui.actionCategory:[name,label,description,spec.description,spec.category].join(' ').toLowerCase().includes(query));
  });
}
function renderControlLibrary(){
  const filters=$('actionCategories');filters.replaceChildren();
  Object.entries(CONTROL_GROUPS).forEach(([key,group])=>{
    const button=actionButton(group.label,()=>{
      ui.actionCategory=key;$('actionSearch').value='';renderControlLibrary();
      $('actionCategories').querySelector('[data-category="'+key+'"]').focus();
    },'category-button',group.icon);
    button.dataset.category=key;button.setAttribute('aria-pressed',String(!$('actionSearch').value.trim()&&ui.actionCategory===key));filters.append(button);
  });
  const root=$('availableActionsList'),actions=filteredControls();root.replaceChildren();
  text('actionCategoryTitle',$('actionSearch').value.trim()?'Search results · '+actions.length:CONTROL_GROUPS[ui.actionCategory].label+' · '+actions.length);
  if(!actions.some(([name])=>name===ui.selectedAction))ui.selectedAction=actions[0]?.[0]||null;
  if(!actions.length)empty(root,'No matching controls','Try a name such as volume, brightness or clipboard.');
  actions.forEach(([name,spec])=>{
    const [label,description,symbol]=controlPresentation(name,spec);
    const row=element('article','control-row');row.dataset.control=name;
    const select=element('button','control-select');select.type='button';select.setAttribute('aria-label','Configure '+label);
    const mark=element('span','control-mark');mark.append(icon(symbol));
    const copy=element('span','control-copy');copy.append(element('strong','',label),element('small','',description));
    select.append(mark,copy);select.addEventListener('click',()=>{ui.selectedAction=name;renderControlInspector();updateControlStates();});
    const permission=element('label','control-permission');permission.append(element('span','sr-only','Allow '+label));
    const toggle=element('input','switch');toggle.type='checkbox';toggle.dataset.actionToggle=name;
    toggle.checked=spec.enabled!==false;toggle.setAttribute('aria-label','Allow '+label);
    toggle.addEventListener('change',async()=>{
      const enabled=toggle.checked;toggle.disabled=true;ui.busy.add('permission:'+name);updateControlStates();
      try{
        await api('/settings',{enabled_actions:{[name]:enabled}});
        ui.actions[name].enabled=enabled;ui.settings.enabled_actions={...ui.settings.enabled_actions,[name]:enabled};
        toast(label+(enabled?' allowed.':' turned off.'));
      }catch(error){toggle.checked=!enabled;toast(error.message,true);}
      finally{ui.busy.delete('permission:'+name);toggle.disabled=false;updateControlStates();updateCodingControls();}
    });
    permission.append(toggle);row.append(select,permission);root.append(row);
  });
  renderControlInspector();updateControlStates();
}
function renderControlInspector(){
  const root=$('actionInspector'),name=ui.selectedAction;root.replaceChildren();
  if(!name){empty(root,'Choose a control','Its options and a local test will appear here.');return;}
  const spec=ui.actions[name],[label,description,symbol]=controlPresentation(name,spec),group=controlGroup(name,spec);
  const mark=element('div','inspector-mark');mark.append(icon(symbol));
  const heading=element('div','inspector-heading');heading.append(element('span','eyebrow',CONTROL_GROUPS[group].label),element('span','permission-badge'));heading.lastChild.id='selectedPermissionState';
  root.append(heading,mark,element('h3','',label),element('p','inspector-description',description));
  const explanation=element('p','inspector-help',group==='display'?'Brightness control depends on your display and its driver.':group==='clipboard'?'Clipboard text stays on this computer. Running this control may read or replace its current contents.':group==='security'?'Locking requires you to sign back in to Windows.':group==='workspace'?'Choose your project and review task output in Workspace. The permissions here also apply to requests from ADAM.':'Runs on this computer. Media controls use the active Windows media session.');
  root.append(explanation);
  if(group==='workspace'){
    root.append(actionButton('Open workspace',()=>switchView('coding'),'button primary inspector-run','arrow'));
  }else{
    const form=element('form','control-test-form');let input;
    if(spec.needs_value){
      const valueLabel=element('label','',/^(volume|brightness)_set$/.test(name)?'Level · 0–100%':'Text to use');
      input=inputForAction(name,spec,ui.controlDrafts[name]);input.id='selectedControlValue';
      input.setAttribute('aria-label',label+' value');input.addEventListener('input',()=>ui.controlDrafts[name]=input.value);valueLabel.append(input);form.append(valueLabel);
    }
    const runButton=element('button','button primary inspector-run');runButton.type='submit';runButton.id='runSelectedControl';runButton.append(icon('play'),document.createTextNode('Try on this computer'));
    const note=element('p','help');note.id='selectedControlHelp';runButton.setAttribute('aria-describedby',note.id);
    const result=element('div','control-result');result.id='selectedControlResult';result.setAttribute('role','status');result.hidden=true;
    form.append(runButton,note,result);form.addEventListener('submit',event=>{
      event.preventDefault();run(runButton,async()=>{
        if(!form.reportValidity())return;
        const value=valueForAction(input,spec);
        if(!await confirmControl(name,spec))return;
        ui.busy.add('control-run');updateControlStates();result.hidden=false;result.classList.remove('error');result.textContent='Running…';
        try{
          const response=await api('/control',{action:name,...(value===undefined?{}:{value})});
          result.textContent=response.result&&typeof response.result==='object'?JSON.stringify(response.result,null,2):typeof response.result==='string'?response.result:response.message||response.details||'Completed on this computer.';
          toast(label+' completed.');await loadStatus();
        }catch(error){result.textContent=error.message;result.classList.add('error');throw error;}
        finally{ui.busy.delete('control-run');updateControlStates();}
      });
    });root.append(form);
  }
  root.append(element('p','inspector-footer','Permissions save automatically. Touch shortcuts are configured on Dashboard.'));
}
function updateControlStates(){
  const paused=!!(ui.status.paused??ui.settings.paused),entries=Object.entries(ui.actions);
  const enabled=entries.filter(([,spec])=>spec.enabled!==false).length;
  text('enabledActionCount',enabled+' of '+entries.length+' allowed');
  text('controlsStateTitle',paused?'Laptop control is paused':'You are in control');
  text('controlsStateDescription',paused?'Resume when you want ADAM to use your allowed controls.':'Only the controls you allow can run on this computer.');
  const pause=$('controlsPauseBtn');if(pause)pause.replaceChildren(icon(paused?'play':'pause'),document.createTextNode(paused?'Resume controls':'Pause controls'));
  document.querySelectorAll('.control-row').forEach(row=>{
    const name=row.dataset.control,active=name===ui.selectedAction;
    row.classList.toggle('selected',active);row.querySelector('.control-select').setAttribute('aria-pressed',String(active));
    const toggle=row.querySelector('[data-action-toggle]');if(!toggle.disabled)toggle.checked=ui.actions[name]?.enabled!==false;
  });
  const spec=ui.actions[ui.selectedAction],allowed=spec&&spec.enabled!==false,saving=ui.busy.has('permission:'+ui.selectedAction);
  text('selectedPermissionState',saving?'Saving…':allowed?'Allowed':'Off');
  $('selectedPermissionState')?.classList.toggle('allowed',!!allowed);
  const runButton=$('runSelectedControl');if(runButton)runButton.disabled=!allowed||paused||saving||ui.busy.has('control-run');
  text('selectedControlHelp',saving?'Saving this permission…':!allowed?'Turn on the permission in the list to try this control.':paused?'Resume laptop control to try this action.':'This runs the selected control immediately.');
}
