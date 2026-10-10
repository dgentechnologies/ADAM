/* Mandatory account + verified LAN connection gate. No tokens enter the DOM. */
window.AdamOnboarding=(()=>{
  'use strict';
  const $=id=>document.getElementById(id), A=()=>window.ADAM;
  let ready=false,started=false,create=false,selected='',busy=false,revision=0,uid='',lastRefresh=0;
  const autoTried=new Set();
  function message(value,error=false){$('onboardingMessage').textContent=value||'';$('onboardingMessage').classList.toggle('error',error);}
  function stage(name){
    ready=name==='ready';$('desktopShell').hidden=!ready;$('desktopShell').inert=!ready;$('onboarding').hidden=ready;
    for(const [id,key] of [['onboardingSplash','splash'],['onboardingLogin','login'],['onboardingDevices','devices']])$(id).hidden=name!==key;
    $('stepAccount').classList.toggle('active',name==='login');$('stepDevice').classList.toggle('active',name==='devices');$('stepReady').classList.toggle('active',ready);
    if(!ready)document.querySelectorAll('dialog[open]').forEach(d=>d.close());
  }
  async function enter(state){
    if(!state.ready){stage(state.stage||'devices');return;}
    selected=state.selected||selected;stage('ready');message('');
    if(!started){started=true;await window.bootDashboard();}
  }
  function showDevices(state){
    const root=$('onboardingDeviceList');root.replaceChildren();
    const labels={registration_required:'Robot identity needs registration',offline:'Offline · not found on this network',available:'Available on this network',authorization_required:'Authorization required',ambiguous:'Identity conflict · connection blocked'};
    if(!state.devices?.length)A().empty(root,'No registered ADAM yet.','Use the same account that owns your ADAM. Once it is registered, refresh to find it here.');
    for(const device of state.devices||[]){
      const row=A().element('article','onboarding-device'),info=A().element('div');
      info.append(A().element('h3','',device.name),A().element('p','',labels[device.state]||'Searching…'),A().element('small','',device.deviceId));
      const button=A().element('button','button',device.state==='authorization_required'?'Authorize':'Connect');button.type='button';
      button.disabled=!['available','authorization_required'].includes(device.state);
      button.addEventListener('click',()=>{
        if(busy)return;
        selected=device.deviceId;
        if(device.state==='authorization_required'){
          $('onboardingCodeForm').hidden=false;$('onboardingCode').value='';$('onboardingCode').focus();
          message('Confirm this is your ADAM using its one-time pairing code.');
        }else connect();
      });row.append(info,button);root.append(row);
    }
  }
  async function refresh(automatic=false){
    if(busy)return;busy=true;const rev=revision;
    $('onboardingRefresh').disabled=true;message('Finding your ADAM devices…');
    try{
      const state=await A().api('/onboarding/refresh',{}, {timeout:45000});if(rev!==revision)return;
      lastRefresh=Date.now();showDevices(state);await enter(state);
      if(!ready)message(state.error||'Choose your ADAM to securely connect this computer.');
      const saved=state.devices?.filter(d=>d.savedPairing&&d.state==='available')||[];
      if(automatic&&!ready&&saved.length===1&&!autoTried.has(uid+saved[0].deviceId)){
        selected=saved[0].deviceId;autoTried.add(uid+selected);busy=false;await connect();
      }
    }catch(e){if(rev===revision){stage('devices');message(e.message,true);}}
    finally{busy=false;$('onboardingRefresh').disabled=false;}
  }
  async function connect(code=''){
    if(busy)return;busy=true;const rev=revision;
    $('onboardingCancel').hidden=false;$('onboardingCodeForm').hidden=true;message('Verifying your ADAM and opening a secure connection…');
    $('onboardingDeviceList').querySelectorAll('button').forEach(b=>b.disabled=true);
    try{
      const state=await A().api('/onboarding/connect',{deviceId:selected,code},{timeout:60000});
      if(rev!==revision)return;
      await enter(state);if(!ready)message('Authorized. Waiting for ADAM’s live status channel…');
    }catch(e){if(rev===revision){message(e.message,true);$('onboardingCodeForm').hidden=false;}}
    finally{busy=false;$('onboardingCode').value='';$('onboardingCancel').hidden=true;}
  }
  async function check(){
    const account=await A().api('/account/status');const newUid=account.user?.uid||'';
    if(newUid!==uid){revision++;uid=newUid;selected='';$('onboardingCodeForm').hidden=true;}
    $('onboardingGoogle').disabled=!account.google?.configured||!!account.google?.pending;
    $('onboardingGoogleCancel').hidden=!account.google?.pending;
    const state=await A().api('/onboarding/status');await enter(state);
    if(!uid){
      if(account.error)message(account.error,true);
      else if(account.google?.pending)message('Finish signing in in your browser, then return here.');
      else if(!account.google?.configured)message('Google sign-in needs the desktop OAuth configuration. You can sign in with email.');
    }else if(Date.now()-lastRefresh>30000&&!busy){await refresh(true);}
    return state;
  }
  async function perform(fn){try{await fn();}catch(e){message(e.message,true);}}
  async function init(){
    $('onboardingEmailForm').addEventListener('submit',event=>{event.preventDefault();if(!$('onboardingEmailForm').reportValidity())return;
      perform(async()=>{const button=$('onboardingEmailSubmit');button.disabled=true;
        try{message(create?'Creating your account…':'Signing you in…');await A().api('/account/email',{email:$('onboardingEmail').value.trim(),password:$('onboardingPassword').value,name:$('onboardingName').value.trim(),create});$('onboardingPassword').value='';await check();}
        finally{button.disabled=false;}
      });
    });
    $('onboardingGoogle').onclick=()=>perform(async()=>{$('onboardingGoogle').disabled=true;await A().api('/account/google/start',{});await check();});
    $('onboardingGoogleCancel').onclick=()=>perform(async()=>{await A().api('/account/google/cancel',{});message('Browser sign-in cancelled.');await check();});
    $('onboardingRegister').onclick=()=>{create=!create;$('onboardingNameLabel').hidden=!create;$('onboardingName').required=create;$('onboardingPassword').autocomplete=create?'new-password':'current-password';$('onboardingEmailSubmit').textContent=create?'Create account':'Sign in';$('onboardingRegister').textContent=create?'I already have an account':'Create an account';$('onboardingLoginTitle').textContent=create?'Make yourself at home.':'Welcome to ADAM.';message('');};
    $('onboardingReset').onclick=()=>perform(async()=>{if(!$('onboardingEmail').reportValidity())return;await A().api('/account/reset',{email:$('onboardingEmail').value.trim()});message('If this email has an account, a reset link is on its way.');});
    $('onboardingRefresh').onclick=()=>refresh();
    $('onboardingCodeForm').addEventListener('submit',event=>{event.preventDefault();if($('onboardingCodeForm').reportValidity())connect($('onboardingCode').value);});
    $('onboardingSignout').onclick=()=>perform(async()=>{revision++;stage('login');await A().api('/account/signout',{});location.reload();});
    $('onboardingCancel').onclick=()=>perform(async()=>{revision++;await A().api('/onboarding/cancel',{});busy=false;message('Connection cancelled.');await refresh();});
    $('onboardingRetry').onclick=()=>perform(check);
    try{await check();}catch(e){stage('login');message(e.message,true);$('onboardingRetry').hidden=false;}
    async function tick(){if(!document.hidden&&!busy){try{await check();}catch(e){stage(uid?'devices':'login');message(e.message,true);$('onboardingRetry').hidden=false;}}setTimeout(tick,3000);}setTimeout(tick,3000);
  }
  document.addEventListener('DOMContentLoaded',init);
  async function selectDevice(){
    revision++;await A().api('/onboarding/cancel',{});stage('devices');
    for(const device of (await A().api('/onboarding/status')).devices||[])autoTried.add(uid+device.deviceId);
    await refresh(false);
  }
  return {get ready(){return ready;},get deviceId(){return selected;},refresh,check,selectDevice};
})();
