/* ADAM desktop controller. Private requests stay on this app's origin. */
'use strict';
const $ = id => document.getElementById(id);
const ui = {
  view: 'dashboard', settings: {}, status: {}, actions: {}, memories: [],
  account: {}, touchDirty: false, memoryEditId: null, createAccount: false,
  taskId: null, codingTools: {}, codingState: 'idle', googlePending: false, syncPending: false, touchLoaded: false, touchRevision: 0, busy: new Set(),
  actionCategory: 'media', selectedAction: null, controlDrafts: {}
};

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}
function icon(name) {
  const wrap = element('i'); wrap.dataset.icon = name; wrap.setAttribute('aria-hidden','true');
  const svg = document.createElementNS('http://www.w3.org/2000/svg','svg');
  svg.setAttribute('viewBox','0 0 24 24'); svg.setAttribute('aria-hidden','true');
  const use = document.createElementNS(svg.namespaceURI,'use');
  use.setAttribute('href','/static/images/icons.svg#'+name);svg.append(use);wrap.append(svg);
  return wrap;
}
function installIcons(root=document) {
  root.querySelectorAll('i[data-icon]').forEach(n => n.replaceWith(icon(n.dataset.icon)));
}
function text(id,value) { const node=$(id); if(node)node.textContent=String(value??''); }
function status(id,message,error=false) { text(id,message); $(id)?.classList.toggle('error',error); }
function empty(root,title,description) {
  const box=element('div','empty-state'); box.append(element('h3','',title),element('p','',description)); root.replaceChildren(box);
}
function actionButton(label, handler, cls='button', symbol) {
  const b=element('button',cls); b.type='button';
  if(symbol)b.append(icon(symbol)); b.append(document.createTextNode(label));
  b.addEventListener('click',()=>run(b,handler)); return b;
}
function toast(message,error=false) {
  const node=element('div','toast'+(error?' error':''),message);
  $('toastRegion').append(node);
  while($('toastRegion').children.length>3)$('toastRegion').firstChild.remove();
  setTimeout(()=>node.remove(),error?8500:4500);
}
async function api(path,body,options={}) {
  if(!path.startsWith('/')||path.startsWith('//'))throw new Error('Invalid companion request.');
  const controller=new AbortController();
  const timer=setTimeout(()=>controller.abort(),options.timeout||20000);
  try {
    const res=await fetch(path,{
      method:body===undefined?'GET':'POST',credentials:'same-origin',cache:'no-store',
      headers:{'Accept':'application/json','X-ADAM-Session':document.querySelector('meta[name="adam-session"]')?.content||'',...(body===undefined?{}:{'Content-Type':'application/json'})},
      ...(body===undefined?{}:{body:JSON.stringify(body)}),signal:controller.signal
    });
    const data=await res.json().catch(()=>({reason:'The companion returned an unreadable response.'}));
    if(!res.ok || data.status==='error' || data.ok===false) {
      throw new Error(data.reason||data.message||data.error||'The request could not be completed.');
    }
    return data;
  } catch(err) {
    if(err.name==='AbortError')throw new Error('This is taking longer than expected. Check the connection and try again.');
    if(err instanceof TypeError)throw new Error('The companion service could not be reached.');
    throw err;
  } finally {clearTimeout(timer);}
}
async function run(button,fn) {
  if(button?.disabled)return;
  if(button)button.disabled=true;
  try {return await fn();}
  catch(e){toast(e.message||'Something went wrong. Please try again.',true);}
  finally{if(button?.isConnected)button.disabled=false;if(button?.id==='codingStartBtn')updateCodingControls();if(button?.id==='runSelectedControl')updateControlStates();if(['authorizeLaptopBtn','revokeLaptopBtn'].includes(button?.id))updatePairingControls();if(button?.closest('#scheduleForm, #todoForm, #clockScheduleList, #clockTodoList'))window.AdamClock?.updateAccess();}
}
function bind(id,fn){$(id)?.addEventListener('click',()=>run($(id),fn));}
function submit(id,fn){$(id)?.addEventListener('submit',e=>{e.preventDefault();if(e.target.reportValidity())run(e.submitter||e.target.querySelector('[type=submit]'),fn);});}
function confirmAction(title,body,accept='Continue') {
  const d=$('confirmDialog'); if(d.open)return Promise.resolve(false);
  text('confirmTitle',title);text('confirmBody',body);text('confirmAcceptBtn',accept);
  return new Promise(resolve=>{
    let result=false;
    const close=()=>{d.removeEventListener('close',close);$('confirmAcceptBtn').removeEventListener('click',yes);$('confirmCancelBtn').removeEventListener('click',no);resolve(result);};
    const yes=()=>{result=true;d.close();};const no=()=>d.close();
    $('confirmAcceptBtn').addEventListener('click',yes);$('confirmCancelBtn').addEventListener('click',no);
    d.addEventListener('close',close);d.showModal();$('confirmCancelBtn').focus();
  });
}
const VIEW_NAMES={dashboard:'Dashboard',actions:'Actions',devices:'Connection',clock:'Clock & to-do',memories:'Memories',coding:'Workspace',activity:'Activity',settings:'Profile & settings'};
const VIEW_GROUPS={dashboard:'dashboard',actions:'controls',activity:'controls',devices:'companion',clock:'companion',memories:'companion',coding:'coding',settings:'settings'};
const GROUP_NAMES={dashboard:'Dashboard',controls:'Controls',companion:'My ADAM',coding:'Workspace',settings:'Profile & settings'};
const SECTION_LINKS={controls:[['actions','Permissions','sliders'],['activity','History','activity']],companion:[['devices','Connection','devices'],['clock','Planner','clock'],['memories','Memories','memory']]};
function renderSectionNav(id){
  const nav=$('sectionNav'),links=SECTION_LINKS[VIEW_GROUPS[id]]||[];nav.replaceChildren();nav.hidden=!links.length;
  links.forEach(([target,label,symbol])=>{
    const button=actionButton(label,()=>switchView(target),'section-link',symbol);
    button.dataset.view=target;button.setAttribute('aria-current',target===id?'page':'false');nav.append(button);
  });
}
async function switchView(id) {
  if(!VIEW_NAMES[id])return;
  if(ui.view==='devices'&&id!=='devices')hideCredentials();
  ui.view=id;
  document.querySelector('.profile-link').setAttribute('aria-current',id==='settings'?'page':'false');
  document.body.dataset.view=id;
  document.querySelectorAll('.sensor-pin[open]').forEach(pin=>pin.open=false);
  document.querySelectorAll('.view').forEach(n=>{n.hidden=n.id!=='tab-'+id;n.classList.toggle('active',!n.hidden);});
  document.querySelectorAll('.nav-link').forEach(n=>{const active=VIEW_GROUPS[n.dataset.view]===VIEW_GROUPS[id];n.classList.toggle('active',active);n.setAttribute('aria-current',active?'page':'false');});
  text('headerTitle',GROUP_NAMES[VIEW_GROUPS[id]]);renderSectionNav(id);history.replaceState(null,'','#'+id);
  window.scrollTo({top:0,behavior:'instant'});
  window.set3DVisible?.(id==='dashboard');
  try {
    if(id==='settings'){await Promise.all([loadSettings(true),loadAccount()]);}
    if(id==='actions')await loadActions();
    if(id==='devices'){await Promise.all([loadSettings(true),loadAccount()]);await loadAccountDevices();}
    if(id==='memories'){await loadMemories();if($('robotMemoriesPanel').open)await window.AdamClock?.refresh();}
    if(id==='activity')await loadLogs();
    if(id==='coding')await loadCoding();
    if(id==='clock'){await window.AdamShared?.refresh();await window.AdamClock?.refresh();}
    if(id==='devices')await window.AdamShared?.refresh();
  }catch(e){toast(e.message,true);}
}
async function loadSettings(fill=false){
  const data=await api('/settings');ui.settings=data.settings||data;
  if(fill){
    $('settingUserName').value=ui.settings.profile?.name||ui.settings.user_name||'';
    $('settingStartup').checked=!!ui.settings.startup_on_login;$('settingPaused').checked=!!ui.settings.paused;
    // The host/port/token inputs were removed with the manual connection form.
    // They used to be filled here; $() returns null for them now, so writing
    // .value would throw a TypeError and abort the rest of loadSettings —
    // taking the workspace field and sensor rendering down with it.
    if(!$('codingCwd').value)$('codingCwd').value=ui.settings.coding_workspace||'';
  }
  if(!ui.touchDirty && Object.keys(ui.actions).length)renderSensors();
  return ui.settings;
}
async function loadStatus(){
  if(ui.busy.has('status'))return;ui.busy.add('status');
  try{
    const data=await api('/status');ui.status=data;ui.settings={...ui.settings,...data.settings};$('backendNotice').hidden=true;
    const robot=data.robot||{},connected=!!robot.connected;
    const paused=!!(data.paused??data.settings?.paused);
    text('headerStatusText',connected?(robot.read_only?'ADAM · read-only':'ADAM connected'):(ui.settings.paired?'ADAM offline':'Connect your ADAM'));
    $('headerStatusDot').className='status-dot'+(connected?' online':'');
    $('modelStatusDot').className='status-dot'+(connected?' online':'');
    $('railServiceDot').className='status-dot'+(paused?'':' online');
    text('railServiceText',paused?'Laptop control paused':'Companion is running');
    text('modelLabel',robot.device_name||'ADAM COMPANION');
    text('robotEmotion',connected?(robot.speaking?'Speaking':robot.listening?'Listening':friendly(robot.emotion||'Ready')):'Ready when you are');
    window.set3DStatus?.({...robot,connected});
    $('pauseAgentBtn').replaceChildren(icon(paused?'play':'pause'),document.createTextNode(paused?'Resume laptop control':'Pause laptop control'));
    text('deviceControlStatus',paused?'Paused':'Available');
    text('deviceHostname',ui.settings.hostname||data.hostname||'This computer');
    text('statDeviceName',ui.settings.hostname||data.hostname||'Desktop companion');
    const version=data.version||ui.settings.version||'—';text('deviceVersion',version);text('aboutVersion',version);
    text('deviceLocalAddress',ui.settings.local_ip||data.local_ip||'Local network');
    text('connectedDeviceName',connected?(robot.device_name||'Your ADAM'):'Connect your companion');
    text('connectionSummary',connected?(robot.read_only?robot.reason||'Connected for viewing. Add ADAM’s connection key to make changes.':'Connected on your local network. Your companion is ready.'):'Set up ADAM with your mobile app, then enter its network address below.');
    window.AdamClock?.updateAccess(robot);
    $('disconnectBtn').hidden=!(ui.settings.paired||connected);
    updatePairingControls();
    const last=data.recent_activity?.[0];if(last)text('statLastActivity',friendly(last.action));
    if(data.enabled_actions)Object.entries(data.enabled_actions).forEach(([name,enabled])=>{
      if(ui.actions[name])ui.actions[name].enabled=enabled;
      const cb=document.querySelector('[data-action-toggle="'+name.replace(/[^a-zA-Z0-9_]/g,'')+'"]');
      if(cb&&!cb.disabled)cb.checked=!!enabled;
    });
    updateControlStates();updateCodingControls();
    updateProfile();
  }finally{ui.busy.delete('status');}
}
function updateProfile(){
  const user=ui.account.user;const name=user?.displayName||user?.display_name||ui.settings.profile?.name||ui.settings.user_name||'Your companion';
  text('sidebarUserName',name);
  if(ui.account.signed_in||ui.settings.user_name)text('sidebarAvatar',name.charAt(0).toUpperCase());else $('sidebarAvatar').replaceChildren(icon('user'));
  text('sidebarUserStatus',ui.account.signed_in?'Account connected':'Local profile');
}
function friendly(name){return String(name||'').replace(/_/g,' ').replace(/\b\w/g,c=>c.toUpperCase());}
function inputForAction(name,spec,value){
  let input;
  if(Array.isArray(spec.choices)&&spec.choices.length){
    input=element('select');spec.choices.forEach(v=>input.append(new Option(String(v),String(v))));
  } else {
    input=element('input');
    input.type=/^(int|integer|float|number)$/.test(spec.value_type)||/^(volume|brightness)_set$/.test(name)?'number':'text';
    if(input.type==='number'){input.min='0';input.max='100';input.step=/float|number/.test(spec.value_type)?'any':'1';}
    input.placeholder=spec.value_hint||'Value';input.maxLength=5000;
  }
  input.value=value??(input.type==='number'?'50':(spec.choices?.[0]??''));
  input.required=true;return input;
}
function valueForAction(input,spec){
  if(!input)return undefined;
  if(!input.reportValidity())throw new Error('Enter a valid value for this action.');
  if(input.type==='number'){
    const n=Number(input.value);if(!Number.isFinite(n))throw new Error('Enter a valid number.');return n;
  }
  return input.value;
}
async function loadActions(){
  const data=await api('/actions');ui.actions=data.actions||{};
  renderActions();if(!ui.touchDirty)renderSensors();
}
function renderActions(){
  renderControlLibrary();
}
async function confirmControl(name,spec={}){
  if(spec.confirm||/lock|clipboard|shutdown|restart|sleep|paste|coding|open_url|open_app/.test(name)){
    const desc=/clipboard|paste/.test(name)?'This action accesses or changes the clipboard on this computer. Clipboard content may appear in the result.':/lock/.test(name)?'This will immediately lock this computer. You will need to sign in again.':'This action will make changes on this computer. Continue only if this is what you intend.';
    return confirmAction('Run '+friendly(name)+'?',desc,'Run action');
  }
  return true;
}
const SENSOR_NAMES={touch1:'Left side',touch2:'Right side',touch3:'Top of head',touch4:'Back of head'};
const TOUCH_EVENTS={double:'Double tap',triple:'Triple tap',hold:'Long press'};
const SENSOR_EVENTS={touch1:['hold'],touch2:['hold'],touch3:['double','triple','hold'],touch4:['hold']};
const ROBOT_ACTIONS={none:'Do nothing'};
function sensorAssignment(sensor,event){
  const saved=ui.settings.touch_assignments?.[sensor];
  if(typeof saved==='string')return {action:'none'};
  return saved?.[event]||{action:'none'};
}
function renderSensors(){
  if(ui.touchDirty)return;
  const signature=JSON.stringify([ui.settings.touch_assignments||{},Object.keys(ui.actions)]);
  if(ui.touchLoaded&&ui.sensorSignature===signature)return;
  ui.sensorSignature=signature;
  const openSensor=document.querySelector('.sensor-pin[open]')?.dataset.point;
  const root=$('sensorControls');root.replaceChildren();
  Object.entries(SENSOR_NAMES).forEach(([sensor,title],index)=>{
    const pin=element('details','sensor-pin');pin.id='pin-'+sensor;pin.dataset.point=sensor;
    const summary=element('summary','sensor-label');summary.setAttribute('aria-label','Configure '+title);
    const chevron=icon('chevron');chevron.classList.add('sensor-chevron');
    summary.append(element('span','sensor-index',String(index+1).padStart(2,'0')),element('span','',title),chevron);
    const card=element('div','sensor-menu');card.id='menu-'+sensor;card.hidden=true;
    summary.setAttribute('aria-controls',card.id);
    summary.setAttribute('aria-expanded','false');
    SENSOR_EVENTS[sensor].forEach(event=>{
      const eventLabel=TOUCH_EVENTS[event];
      const row=element('div','sensor-event'),label=element('label','',eventLabel),select=element('select');
      select.id='touch-'+sensor+'-'+event;select.dataset.sensor=sensor;select.dataset.event=event;select.setAttribute('aria-label',title+' · '+eventLabel);
      Object.entries(ROBOT_ACTIONS).forEach(([v,t])=>select.append(new Option(t,v)));
      Object.keys(ui.actions).filter(n=>!/^dispatch_coding|cancel_coding|check_coding|set_robot_emotion|read_clipboard|write_clipboard|clipboard_paste/.test(n)).forEach(n=>select.append(new Option(CONTROL_LABELS[n]?.[0]||friendly(n),n)));
      const saved=sensorAssignment(sensor,event);
      if(!Array.from(select.options).some(o=>o.value===saved.action)&&saved.action)select.append(new Option(friendly(saved.action),saved.action));
      select.value=saved.action||'none';label.append(select);row.append(label);
      const test=actionButton('',async()=>{
        if(ui.touchDirty){toast('Save your preferences before testing this touch.',true);return;}
        const name=select.value;if(name==='none'){toast('This touch is set to do nothing.');return;}
        if(!(await confirmControl(name,ui.actions[name])))return;
        await api('/touch/test',{sensor,event});toast(title+' · '+eventLabel+' completed.');
      },'icon-button','play');test.title='Test '+eventLabel.toLowerCase();test.setAttribute('aria-label','Test '+title+' '+eventLabel);row.append(test);
      function updateValue(value){
        row.querySelector('.sensor-value')?.remove();
        const spec=ui.actions[select.value];
        if(spec?.needs_value){const inp=inputForAction(select.value,spec,value);inp.classList.add('sensor-value');inp.dataset.touchValue='true';inp.setAttribute('aria-label',title+' '+eventLabel+' value');inp.addEventListener('input',markTouchDirty);row.append(inp);}
        if(pin.open)requestAnimationFrame(()=>positionSensorMenu(pin));
      }
      updateValue(saved.value);select.addEventListener('change',()=>{updateValue();markTouchDirty();});
      card.append(row);
    });
    if(sensor==='touch4'){const rear=actionButton('View rear',()=>window.viewADAMBack?.(),'button text-button','rotate');rear.setAttribute('aria-label','View back of head');card.append(rear);}
    pin.append(summary,card);root.append(pin);
    pin.addEventListener('toggle',()=>{
      card.hidden=!pin.open;
      summary.setAttribute('aria-expanded',String(pin.open));
      if(pin.open){
        root.querySelectorAll('.sensor-pin').forEach(other=>{if(other!==pin)other.open=false;});
        positionSensorMenu(pin);
      }
    });
    if(openSensor===sensor)pin.open=true;
  });
  window.updateSensorAnchors?.();
  ui.touchLoaded=true;text('touchSaveStatus','Saved on this computer · apply to update ADAM');$('touchSaveDot').className='status-dot online';
}
function positionSensorMenu(pin){
  const card=pin.querySelector('.sensor-menu'),label=pin.querySelector('summary').getBoundingClientRect();
  const width=Math.min(288,innerWidth-32);card.style.width=width+'px';
  const height=card.offsetHeight;
  card.style.left=Math.max(16,Math.min(innerWidth-width-16,label.left))+'px';
  const below=label.bottom+9;
  card.style.top=Math.max(16,Math.min(innerHeight-height-16,below))+'px';
}
document.addEventListener('pointerdown',event=>{
  document.querySelectorAll('.sensor-pin[open]').forEach(pin=>{if(!pin.contains(event.target))pin.open=false;});
});
document.addEventListener('keydown',event=>{
  if(event.key==='Escape')document.querySelectorAll('.sensor-pin[open]').forEach(pin=>{pin.open=false;pin.querySelector('summary').focus();});
});
window.addEventListener('resize',()=>document.querySelectorAll('.sensor-pin[open]').forEach(positionSensorMenu));
function markTouchDirty(){ui.touchDirty=true;ui.touchRevision++;text('touchSaveStatus','Unsaved touch preferences');$('touchSaveDot').className='status-dot';}
function collectTouches(){
  const assignments={};
  document.querySelectorAll('[data-sensor]').forEach(select=>{
    const {sensor,event}=select.dataset;assignments[sensor]??={};
    const input=select.closest('.sensor-event').querySelector('[data-touch-value]');
    if(input&&!input.checkValidity()){
      const pin=input.closest('.sensor-pin');pin.open=true;
      requestAnimationFrame(()=>{positionSensorMenu(pin);input.focus();input.reportValidity();});
      throw new Error('Enter a valid value for '+SENSOR_NAMES[sensor]+' · '+TOUCH_EVENTS[event]+'.');
    }
    const value=valueForAction(input,ui.actions[select.value]||{});
    assignments[sensor][event]={action:select.value,...(value===undefined?{}:{value})};
  });return assignments;
}
async function saveTouches(){
  if(!ui.touchLoaded)throw new Error('Touch preferences are still loading.');
  const revision=ui.touchRevision,assignments=collectTouches();await api('/touch/save',{assignments});ui.settings.touch_assignments=assignments;ui.touchDirty=ui.touchRevision!==revision;
  text('touchSaveStatus',ui.touchDirty?'New changes are waiting to be saved.':'Saved on this computer · apply to update ADAM');$('touchSaveDot').className='status-dot'+(ui.touchDirty?'':' online');toast('Touch preferences saved.');
}
async function loadStats(){
  const d=await api('/system_stats');
  text('statSystem',Number.isFinite(d.cpu_pct)&&Number.isFinite(d.ram_pct)?Math.round(d.cpu_pct)+'% CPU · '+Math.round(d.ram_pct)+'% memory':'System status unavailable');
}
// connectionPayload() is gone with the manual form. It read connectionHost /
// connectionToken / the port inputs, none of which exist any more. Connecting
// goes through /pair/adam, which gets the address from mDNS and the key from
// ADAM itself.
let credentialsTimer;
function hideCredentials(){clearTimeout(credentialsTimer);$('credentialToken').value='';$('credentialsPanel').hidden=true;}
function updatePairingControls(){
  const robot=ui.status.robot||{},ready=!!robot.connected,capable=!!robot.capabilities?.laptop_pairing;
  $('authorizeLaptopBtn').disabled=!ready||robot.read_only||!capable||ui.busy.has('pairing');$('revokeLaptopBtn').disabled=!ready||robot.read_only||!capable||ui.busy.has('pairing');
  text('laptopPairingHelp',!ready?'Connect ADAM above, then allow it to use the laptop actions you have enabled.':robot.read_only?robot.reason||'Add ADAM’s connection key above and reconnect to authorize laptop control.':!capable?'This ADAM needs an updated companion service for guided pairing. You can also use manual pairing below.':'Allow this ADAM to run your enabled laptop actions. You can pause control from Dashboard at any time.');
}
async function loadAccountDevices(){
  if(ui.busy.has('devices'))return;ui.busy.add('devices');
  const owner=ui.account.user?.uid,root=$('accountDeviceList');root.replaceChildren();
  status('accountDevicesStatus','Finding your devices…');
  try{
    const data=await api('/account/devices');
    if(owner!==ui.account.user?.uid)return;
    const devices=Array.isArray(data.devices)?data.devices:[];
    if(!ui.account.signed_in){
      status('accountDevicesStatus','Sign in with your mobile account to see the ADAM linked to it. You can also use Find ADAM on this network above.');
      root.append(actionButton('Sign in',()=>switchView('settings'),'button subtle','user'));
      return;
    }
    if(!devices.length){
      status('accountDevicesStatus','No ADAM is linked to this account yet. Set one up in the mobile app, then refresh.');
      return;
    }
    // Cross-reference the account's devices against what is actually on this
    // network. The cloud knows WHICH ADAM is yours; mDNS knows WHERE it is. One
    // without the other is not enough: the cloud has no routable address for a
    // LAN device, and discovery alone cannot tell your ADAM from a flatmate's.
    let nearby=[];
    try{nearby=(await api('/discover/adam')).units||[];}catch(e){/* offline is fine */}
    const byId={};nearby.forEach(u=>{byId[String(u.id).toUpperCase()]=u;});
    let matched=0;
    devices.forEach(device=>{
      const id=String(device.deviceId||device.serial||'').toUpperCase();
      const here=byId[id];
      if(here)matched++;
      const card=element('article','device-card'),info=element('div');
      info.append(element('h4','',device.name||device.deviceId||'Your ADAM'),
                  element('p','',here?(here.connected?'Connected · '+here.host:'On this network · '+here.host):'Not on this network right now'));
      const label=here?(here.connected?'Connected':'Connect'):'Unavailable';
      const choose=actionButton(label,async()=>{
        status('accountDevicesStatus','Connecting to '+(device.name||id)+'…');
        try{
          const r=await api('/pair/adam',{host:here.host,port:here.port});
          if(r.status!=='ok')throw new Error(r.reason||'Pairing failed.');
          status('accountDevicesStatus','Connected to '+(r.id||id)+'.');
          toast('ADAM connected.');
          await loadSettings(true);await loadStatus();await loadAccountDevices();
        }catch(e){status('accountDevicesStatus',e.message,true);throw e;}
      },'button primary','arrow');
      // Disabled when the unit is not reachable here: the cloud record alone
      // gives us no address to dial, so offering Connect would always fail.
      choose.disabled=!here||here.connected;
      card.append(icon('devices'),info,choose);root.append(card);
    });
    status('accountDevicesStatus',matched?'Found '+matched+' of your '+devices.length+' ADAM unit(s) on this network. Select one to connect — nothing to type.':'Your ADAM is linked to this account but is not on this network right now. Put it on the same Wi-Fi, then refresh.');
  }catch(e){status('accountDevicesStatus',e.message+' You can still use Find ADAM on this network above.',true);}
  finally{ui.busy.delete('devices');}
}
async function loadAccount(){
  const previousAccount=ui.account.user?.uid;
  const d=await api('/account/status');ui.account=d.account||d;
  if(previousAccount!==ui.account.user?.uid)window.AdamShared?.accountChanged();
  const signed=!!ui.account.signed_in,user=ui.account.user||{};
  $('accountSignedOut').hidden=signed;$('accountSignedIn').hidden=!signed;
  if(!signed)$('accountDeviceList').replaceChildren();
  text('accountTitle',signed?'Welcome, '+(user.displayName||user.display_name||'you')+'.':'Bring your world together.');
  text('accountDescription',signed?'Your ADAM account connects this desktop with your mobile memories.':'Use the same account as the ADAM mobile app to keep your memories in sync.');
  text('accountUserEmail',user.email||'—');
  const sync=ui.account.sync||{};const syncText=sync.error?'Sync needs attention':sync.syncing?'Syncing':sync.pending?'Changes waiting to sync':sync.lastSynced?'Last synced '+formatDate(sync.lastSynced):signed?'Ready to sync':'Saved on this computer';
  text('accountSyncState',syncText);text('memorySyncStatus',syncText);
  $('importGuestPanel').hidden=!signed||!sync.guestMemories;
  text('importGuestMessage',sync.guestMemories+' local items are available on this computer. Import them only if they belong in this account.');
  $('cancelGoogleBtn').hidden=!ui.account.google?.pending;
  const prefs=ui.account.preferences||{};
  if(!$('sharedPreferences').open){$('preferenceVoice').value=prefs.voice||'Charon';$('preferenceWakeWord').value=prefs.wakeWord||'Hey ADAM';$('preferenceBrain').value=prefs.brain||'lite';}
  if(ui.syncPending&&!sync.syncing){ui.syncPending=false;await loadMemories();if(!sync.error)toast(sync.pending?'Saved locally. Changes are waiting to sync.':'Your memories are up to date.');}
  if(sync.syncing)ui.syncPending=true;
  if(sync.error)status('accountMessage',sync.error,true);
  else if(ui.googlePending&&signed){ui.googlePending=false;status('accountMessage','You are signed in.');toast('Your ADAM account is connected.');await loadMemories();}
  else if(ui.account.error){ui.googlePending=false;status('accountMessage',ui.account.error,true);}
  else if(ui.googlePending&&!ui.account.google?.pending){ui.googlePending=false;status('accountMessage','Browser sign-in ended. You can try again.');}
  else if(!ui.googlePending)status('accountMessage',!signed&&!ui.account.configured?'Account connection needs Firebase setup. You can keep using the desktop locally.':'');
  updateProfile();return ui.account;
}
async function syncAccount(){
  await api('/account/sync',{}, {timeout:45000});ui.syncPending=true;
  await Promise.all([loadAccount(),loadMemories()]);
}
function formatDate(date){
  const d=new Date(typeof date==='number'?(date<1e12?date*1000:date):date);
  return Number.isNaN(d.valueOf())?'recently':d.toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
}
async function loadMemories(){
  const d=await api('/memories');ui.memories=d.memories||d.data?.memories||[];
  renderMemories();if(d.sync?.error)text('memorySyncStatus',d.sync.error);
}
function renderMemories(){
  const root=$('memoryList'),query=$('memorySearch').value.toLowerCase();root.replaceChildren();
  const rows=ui.memories.filter(m=>[m.title,m.text,m.kind].join(' ').toLowerCase().includes(query));
  if(!rows.length){empty(root,query?'No matching memories':'A place for what matters.',query?'Try a different search.':'Add your first memory. It stays on this computer until you choose to sync.');return;}
  rows.forEach(m=>{
    const card=element('article','memory-card');card.append(element('span','eyebrow',m.kind==='person'?'Person':'Note'),element('h3','',m.title||'Memory'),element('p','',m.text||''));
    const footer=element('div','button-row');footer.append(
      actionButton('Edit',()=>editMemory(m),'button subtle','edit'),
      actionButton('Delete',async()=>{if(!(await confirmAction('Delete this memory?','“'+(m.title||'Memory')+'” will be removed from this account when synced.','Delete memory')))return;await api('/memories/delete',{id:m.id});await loadMemories();toast('Memory deleted.');},'icon-button','trash')
    );footer.lastChild.setAttribute('aria-label','Delete '+(m.title||'memory'));card.append(footer);root.append(card);
  });
}
function editMemory(memory){
  ui.memoryEditId=memory?.id||null;text('memoryDialogTitle',memory?'Edit memory':'A new memory');
  $('memoryTitle').value=memory?.title||'';$('memoryText').value=memory?.text||'';
  $('memoryKind').value=memory?.kind||'fact';if(!$('memoryKind').value)$('memoryKind').value='fact';
  $('memoryDialog').showModal();$('memoryTitle').focus();
}
async function loadLogs(){
  const d=await api('/activity_log?limit=100&q='+encodeURIComponent($('logSearchInput').value));const filter=$('logCategoryFilter').value;
  const rows=(d.log||[]).filter(l=>!filter||(filter==='ok'?l.status==='ok':l.status!=='ok'));
  const root=$('activityLogTableBody');root.replaceChildren();text('logEventCount',rows.length+' events');
  if(!rows.length){empty(root,'Nothing here yet.','Actions and connection events appear here as they happen.');return;}
  rows.forEach(l=>{
    const row=element('article','activity-row'+(l.status==='ok'?'':' error')),info=element('div');
    const dt=element('time','',l.time_str||formatDate(l.timestamp));info.append(element('strong','',friendly(l.action)),element('span','badge',l.status==='ok'?'Completed':friendly(l.status||'Needs attention')));
    let details=String(l.details||'');if(l.value!==undefined&&l.value!==null&&l.value!=='')details=String(l.value)+(details?' · '+details:'');
    row.append(dt,info,element('p','',details));root.append(row);
  });
}
async function loadCoding(){
  const data=await api('/coding_task_status');const task=data.active_task||data.task;ui.taskId=task?.id||null;
  const state=task?.state||data.state||'idle';ui.codingState=state;text('codingStateBadge',friendly(state));
  text('codingTaskInfo',task?(task.prompt||task.last_message||'Task in progress'):'Your task output will appear here.');
  const terminal=$('codingTerminal'),stick=terminal.scrollHeight-terminal.scrollTop-terminal.clientHeight<35;
  text('codingTerminal',task?((task.output_tail||[]).map(v=>typeof v==='string'?v:JSON.stringify(v)).join('\n')||task.last_message||'Starting…'):'No task running.');
  if(stick)terminal.scrollTop=terminal.scrollHeight;
  $('cancelTaskBtn').hidden=!['running','needs_input'].includes(state);
  $('codingInputForm').hidden=state!=='needs_input';text('codingInputPromptLabel',task?.input_needed_prompt||'Assistant needs your input');
  if(data.tools&&typeof data.tools==='object'){
    ui.codingTools=data.tools;
    text('codingToolsStatus',Object.entries(data.tools).map(([k,v])=>(v?.name||friendly(k))+': '+(toolAvailable(v)?'available':v?.reason||'not installed')).join(' · '));
    [...$('codingTool').options].forEach(option=>{option.disabled=!toolAvailable(data.tools[option.value]);});
    if($('codingTool').selectedOptions[0]?.disabled)$('codingTool').value=[...$('codingTool').options].find(option=>!option.disabled)?.value||'';
  }
  updateCodingControls();
}
function toolAvailable(value){return value===true||!!(value&&typeof value==='object'&&value.available);}
function updateCodingControls(){
  const paused=!!(ui.status.paused??ui.settings.paused),allowed=ui.actions.dispatch_coding_task?.enabled!==false;
  $('codingStartBtn').disabled=!allowed||paused||!toolAvailable(ui.codingTools[$('codingTool').value])||['running','needs_input'].includes(ui.codingState);
  text('codingPermissionStatus',!allowed?'Allow “Start a coding task” in Controls before starting.':paused?'Laptop control is paused. Resume it from Controls or Dashboard.':'Workspace access is allowed. Every task uses the folder you choose.');
}
// Find ADAM on the local network and pair with one click.
//
// The Pi advertises _adam._tcp over mDNS (pi/adam/discovery.py) and the
// companion browses for it; selecting a unit calls its /api/pair/claim, which
// hands over the sync token while the unit is unclaimed. That is why there is
// nothing to type here — the manual address + key form below exists only for
// an ADAM discovery cannot see (different subnet, VPN).
async function findAdamUnits(){
  const root=$('findAdamList');root.replaceChildren();
  status('findAdamStatus','Looking for ADAM on your network…');
  try{
    const d=await api('/discover/adam');
    const units=Array.isArray(d.units)?d.units:[];
    if(!units.length){
      status('findAdamStatus','No ADAM found yet. Make sure it is powered on and on this same Wi-Fi, then search again. Discovery needs a few seconds after ADAM starts.');
      return;
    }
    status('findAdamStatus',units.length===1?'Found 1 ADAM. Select it to connect.':'Found '+units.length+' ADAM units. Select the one you want.');
    units.forEach(u=>{
      const card=element('article','device-card'),info=element('div');
      const where=u.host+(u.port&&u.port!==8766?':'+u.port:'');
      info.append(element('h4','',u.name&&u.name!=='ADAM'?u.name+' · '+u.id:u.id),
                  element('p','',u.connected?'Connected · '+where:(u.paired?'Already paired with another device · '+where:where+(u.version?' · v'+u.version:''))));
      // An already-paired unit is shown but not offered: claiming it would
      // fail with 409, so a disabled button is honest where a failing one is
      // not. The TXT record is what tells us, so this is accurate before we
      // ever talk to the unit.
      const label=u.connected?'Connected':(u.paired?'Unavailable':'Connect');
      const choose=actionButton(label,async()=>{
        status('findAdamStatus','Connecting to '+u.id+'…');
        try{
          const r=await api('/pair/adam',{host:u.host,port:u.port});
          if(r.status!=='ok')throw new Error(r.reason||'Pairing failed.');
          status('findAdamStatus','Connected to '+(r.id||u.id)+'. Alarms, to-dos and memories are ready.');
          toast('ADAM connected.');
          await loadSettings(true);await loadStatus();
          await findAdamUnits();
        }catch(e){status('findAdamStatus',e.message,true);throw e;}
      },'button primary','arrow');
      choose.disabled=u.connected||u.paired;
      card.append(icon('devices'),info,choose);root.append(card);
    });
  }catch(e){status('findAdamStatus',e.message+' You can still connect by local address below.',true);}
}
function bindUI(){
  document.querySelectorAll('button[data-view],a[data-view]').forEach(n=>n.addEventListener('click',e=>{e.preventDefault();if(n.dataset.controlCategory){ui.actionCategory=n.dataset.controlCategory;$('actionSearch').value='';}switchView(n.dataset.view);}));
  bind('refreshActionsBtn',loadActions);$('actionSearch').addEventListener('input',renderActions);
  bind('resetCameraBtn',()=>window.reset3DCamera?.());bind('saveTouchBtn',saveTouches);
  bind('applyTouchBtn',async()=>{if(ui.touchDirty)await saveTouches();if(ui.touchDirty)throw new Error('Your preferences changed while saving. Save them once more before applying.');const d=await api('/touch/apply',{});text('touchSaveStatus',d.message||'Touch preferences applied to ADAM');toast('Touch preferences applied to ADAM.');});
  bind('pauseAgentBtn',async()=>{const paused=!(ui.status.paused??ui.settings.paused);await api('/settings',{paused});ui.settings.paused=paused;await loadStatus();toast(paused?'Laptop control is paused.':'Laptop control resumed.');});
  bind('controlsPauseBtn',async()=>{const paused=!(ui.status.paused??ui.settings.paused);await api('/settings',{paused});ui.settings.paused=paused;await loadStatus();toast(paused?'Laptop control is paused.':'Laptop control resumed.');});
  // The manual host/key form is gone: connecting is discovery + one click, or
  // the account's own device matched on this network. probeConnectionBtn and
  // connectionForm no longer exist, so their handlers went with them.
  bind('disconnectBtn',async()=>{if(!(await confirmAction('Disconnect ADAM?','This computer will stop connecting to this ADAM. Your memories and preferences stay saved.','Disconnect')))return;await api('/pair/adam/forget',{});await loadSettings(true);await loadStatus();status('connectionResult','ADAM disconnected.');await findAdamUnits();});
  bind('findAdamBtn',findAdamUnits);
  bind('refreshDevicesBtn',async()=>{await loadAccount();await loadAccountDevices();});
  bind('authorizeLaptopBtn',async()=>{ui.busy.add('pairing');updatePairingControls();try{await api('/connection/authorize',{});status('laptopPairingResult','ADAM verified this computer. Your enabled laptop actions are linked.');await loadStatus();}catch(e){status('laptopPairingResult',e.message,true);throw e;}finally{ui.busy.delete('pairing');}});
  bind('revokeLaptopBtn',async()=>{if(!await confirmAction('Revoke laptop access?','ADAM will forget this laptop connection and laptop control will pause on this computer.','Revoke access'))return;ui.busy.add('pairing');updatePairingControls();try{await api('/connection/revoke',{});status('laptopPairingResult','Laptop access revoked. Control is paused on this computer.');await loadStatus();}catch(e){status('laptopPairingResult',e.message,true);throw e;}finally{ui.busy.delete('pairing');}});
  bind('revealCredentialsBtn',async()=>{const d=await api('/connection/credentials',{});$('credentialToken').value=d.token||d.agent_token||'';if(!$('credentialToken').value)throw new Error('No connection key was returned.');$('credentialsPanel').hidden=false;text('deviceLocalAddress',d.host?d.host+':'+(d.port||ui.settings.agent_port||8642):$('deviceLocalAddress').textContent);clearTimeout(credentialsTimer);credentialsTimer=setTimeout(hideCredentials,60000);});
  bind('hideCredentialsBtn',hideCredentials);bind('copyCredentialsBtn',async()=>{if(!$('credentialToken').value)return;await navigator.clipboard.writeText($('credentialToken').value);toast('Connection key copied.');});
  bind('googleSignInBtn',async()=>{const d=await api('/account/google/start',{});ui.googlePending=true;status('accountMessage',d.message||'Continue in your browser, then return here.');});
  bind('cancelGoogleBtn',async()=>{await api('/account/google/cancel',{});ui.googlePending=false;await loadAccount();status('accountMessage','Browser sign-in cancelled.');});
  bind('importGuestBtn',async()=>{if(!await confirmAction('Import local items into this account?','Memories, planner entries and saved devices will be copied to '+(ui.account.user?.email||'your signed-in account')+' and synced with its devices.','Import items'))return;await api('/account/import-guest',{});ui.syncPending=true;await loadAccount();await loadMemories();});
  submit('accountPreferencesForm',async()=>{await api('/account/preferences',{voice:$('preferenceVoice').value,wakeWord:$('preferenceWakeWord').value,brain:$('preferenceBrain').value});toast('Companion preferences saved.');if(ui.account.signed_in)await syncAccount();});
  submit('emailSignInForm',async()=>{try{await api('/account/email',{email:$('accountEmail').value.trim(),password:$('accountPassword').value,create:ui.createAccount,name:$('accountName').value.trim()});$('accountPassword').value='';await loadAccount();await loadMemories();toast('Your ADAM account is connected.');}catch(e){status('accountMessage',e.message,true);throw e;}});
  bind('toggleCreateAccountBtn',()=>{ui.createAccount=!ui.createAccount;$('accountNameLabel').hidden=!ui.createAccount;$('accountPassword').autocomplete=ui.createAccount?'new-password':'current-password';text('emailSignInBtn',ui.createAccount?'Create account':'Sign in');text('toggleCreateAccountBtn',ui.createAccount?'I already have an account':'Create an account');});
  bind('resetPasswordBtn',async()=>{if(!$('accountEmail').reportValidity())return;await api('/account/reset',{email:$('accountEmail').value.trim()});status('accountMessage','If this email has an account, a reset link is on its way.');});
  bind('signOutBtn',async()=>{if(!(await confirmAction('Sign out of ADAM?','Account sync will pause. You can still use this computer locally.','Sign out')))return;await api('/account/signout',{});await loadAccount();await loadMemories();});
  bind('accountSyncBtn',syncAccount);bind('syncMemoriesBtn',syncAccount);
  bind('addMemoryBtn',()=>editMemory());bind('closeMemoryBtn',()=>$('memoryDialog').close());$('memorySearch').addEventListener('input',renderMemories);
  submit('memoryForm',async()=>{const title=$('memoryTitle').value.trim(),content=$('memoryText').value.trim();if(!title||!content)throw new Error('Add a title and some text for your memory.');await api('/memories',{...(ui.memoryEditId?{id:ui.memoryEditId}:{}),title,text:content,kind:$('memoryKind').value});$('memoryDialog').close();await loadMemories();toast('Memory saved.');});
  submit('settingsForm',async()=>{await api('/settings',{user_name:$('settingUserName').value.trim(),startup_on_login:$('settingStartup').checked,paused:$('settingPaused').checked});await loadSettings(true);await loadStatus();toast('Preferences saved.');});
  submit('codingForm',async()=>{if(!(await confirmAction('Start this coding task?','Your coding assistant can read and change files in the selected workspace. Check the folder and task before continuing.','Start task')))return;const cwd=$('codingCwd').value.trim();await api('/settings',{coding_workspace:cwd});await api('/coding/dispatch',{prompt:$('codingPrompt').value.trim(),tool:$('codingTool').value,cwd});await loadCoding();});
  $('codingTool').addEventListener('change',updateCodingControls);
  bind('cancelTaskBtn',async()=>{if(!(await confirmAction('Stop this task?','The running assistant will stop. Files it has already changed will remain changed.','Stop task')))return;await api('/coding/cancel',{task_id:ui.taskId});await loadCoding();});
  submit('codingInputForm',async()=>{await api('/coding/input',{task_id:ui.taskId,input:$('humanResponseInput').value});$('humanResponseInput').value='';await loadCoding();});
  let searchTimer;const refreshLogs=()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>loadLogs().catch(e=>toast(e.message,true)),200);};
  $('logSearchInput').addEventListener('input',refreshLogs);$('logCategoryFilter').addEventListener('change',refreshLogs);
  bind('clearLogsBtn',async()=>{if(!(await confirmAction('Clear activity history?','This removes the activity history shown on this computer.','Clear history')))return;await api('/activity_log/clear',{});await loadLogs();});
  bind('openSetupBtn',()=>$('setupDialog').showModal());
  for(const [id,target] of [['setupAccountBtn','settings'],['setupConnectBtn','devices'],['setupExploreBtn','dashboard']]){
    bind(id,async()=>{await api('/settings',{setup_complete:true});$('setupDialog').close();await switchView(target);});
  }
}
async function poll(){
  if(!document.hidden){
    try{await loadStatus();}catch(e){$('backendNotice').hidden=false;text('headerStatusText','Service offline');$('headerStatusDot').className='status-dot error';}
    try{
      if(ui.view==='activity')await loadLogs();
      if(ui.view==='coding')await loadCoding();
      if(ui.view==='dashboard')await loadStats();
      if(ui.view==='clock')await window.AdamClock?.refresh();
      if(ui.googlePending||ui.syncPending||ui.view==='settings')await loadAccount();
    }catch(e){/* The view retains its last result; interactive refresh reports the error. */}
  }
  setTimeout(poll,4000);
}
async function boot(){
  installIcons();bindUI();
  document.querySelectorAll('.nav-link').forEach(button=>button.title=button.getAttribute('aria-label'));
  const updateTime=()=>{text('headerDate',new Date().toLocaleDateString(undefined,{weekday:'short',month:'short',day:'numeric'}).toUpperCase());text('headerTime',new Date().toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit'}));};
  updateTime();setInterval(updateTime,30000);
  try{window.init3D?.();}catch(e){$('threeCanvas').hidden=true;$('robotFallback').hidden=false;text('modelHelp','ADAM companion preview');}
  const results=await Promise.allSettled([loadSettings(true),loadActions(),loadStatus(),loadAccount()]);
  results.filter(r=>r.status==='rejected').forEach(r=>toast(r.reason.message,true));
  if(!ui.touchLoaded&&Object.keys(ui.actions).length)renderSensors();
  if(results[0].status==='fulfilled'&&!ui.settings.setup_complete)$('setupDialog').showModal();
  const target=location.hash.slice(1);if(VIEW_NAMES[target])await switchView(target);
  poll();
}
window.ADAM={api,element,icon,toast,run,bind,submit,empty,confirmAction,formatDate,ui,text,status,switchView};
document.addEventListener('DOMContentLoaded',boot);

