const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {JSDOM}=require('jsdom');
const root=path.join(__dirname,'../resources/static');
const tick=()=>new Promise(r=>setImmediate(r));
async function fixture(t){
 const dom=new JSDOM(fs.readFileSync(path.join(root,'index.html'),'utf8'),{runScripts:'outside-only',url:'http://127.0.0.1:8642'});t.after(()=>dom.window.close());
 const w=dom.window;await new Promise(r=>w.addEventListener('load',r,{once:true}));
 let init,boots=0;const calls=[];const state={stage:'login',ready:false,devices:[]};const account={signed_in:false,user:null,google:{configured:true,pending:false}};
 w.document.addEventListener=(event,fn)=>{if(event==='DOMContentLoaded')init=fn;};
 const el=(tag,cls,text)=>{const n=w.document.createElement(tag);n.className=cls||'';if(text)n.textContent=text;return n;};
 w.ADAM={element:el,empty:(root,title,description)=>{root.replaceChildren(el('h3','',title),el('p','',description));},api:async(url,body)=>{calls.push({url,body});if(url==='/account/status')return account;if(url==='/account/email'){account.user={uid:'alice'};account.signed_in=true;state.stage='devices';return account;}if(url.startsWith('/onboarding/'))return state;return {};}};
 w.bootDashboard=async()=>{boots++;};w.eval(fs.readFileSync(path.join(root,'js/onboarding.js'),'utf8'));await init();
 return {w,account,state,calls,$:id=>w.document.getElementById(id),boots:()=>boots};
}
test('no dashboard flash or guest bypass, including a dashboard URL fragment',async t=>{
 const f=await fixture(t);f.w.location.hash='#dashboard';assert.equal(f.$('desktopShell').hidden,true);assert.equal(f.$('desktopShell').inert,true);assert.equal(f.boots(),0);assert.equal(f.$('onboardingLogin').hidden,false);
 assert.equal(f.$('onboardingLogin').textContent.includes('Skip'),false);
});
test('email login goes to device selection, not dashboard; offline devices cannot connect',async t=>{
 const f=await fixture(t);f.state.devices=[{deviceId:'ADAM-X',name:'Desk',state:'offline'}];f.$('onboardingEmail').value='alice@example.com';f.$('onboardingPassword').value='password-fixture';
 f.$('onboardingEmailForm').dispatchEvent(new f.w.Event('submit',{cancelable:true}));await tick();await tick();
 assert.equal(f.$('onboardingDevices').hidden,false);assert.equal(f.$('desktopShell').hidden,true);assert.equal(f.$('onboardingPassword').value,'');assert.equal(f.$('onboardingDeviceList').querySelector('button').disabled,true);
});
test('dashboard requires server verification and locks again on telemetry loss',async t=>{
 const f=await fixture(t);f.account.user={uid:'alice'};f.state.stage='ready';f.state.ready=true;await f.w.AdamOnboarding.check();assert.equal(f.boots(),1);assert.equal(f.$('desktopShell').hidden,false);
 f.state.ready=false;f.state.stage='devices';await f.w.AdamOnboarding.check();assert.equal(f.$('desktopShell').hidden,true);
 f.state.ready=true;f.state.stage='ready';await f.w.AdamOnboarding.check();assert.equal(f.boots(),1);
});
test('pairing code is requested explicitly before connection',async t=>{
 const f=await fixture(t);f.account.user={uid:'alice'};f.state.stage='devices';f.state.devices=[{deviceId:'ADAM-X',name:'Desk',state:'authorization_required'}];await f.w.AdamOnboarding.check();
 f.$('onboardingDeviceList').querySelector('button').click();assert.equal(f.$('onboardingCodeForm').hidden,false);assert.equal(f.calls.filter(c=>c.url==='/onboarding/connect').length,0);
 f.$('onboardingCode').value='1234-abcd-5678-123456';f.$('onboardingCodeForm').dispatchEvent(new f.w.Event('submit',{cancelable:true}));await tick();
 assert.equal(f.calls.find(c=>c.url==='/onboarding/connect').body.deviceId,'ADAM-X');assert.equal(f.$('onboardingCode').value,'');assert.equal(f.$('desktopShell').hidden,true);
});
