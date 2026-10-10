/* Run with jsdom installed under artifacts/ui-tests (see tests/README.md). */
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {JSDOM} = require('jsdom');
const root = path.join(__dirname, '../resources/static');
const tick = () => new Promise(resolve => setImmediate(resolve));

async function fixture(t) {
  const dom = new JSDOM(fs.readFileSync(path.join(root, 'index.html'), 'utf8'),
    {runScripts: 'outside-only', url: 'http://127.0.0.1:8642'});
  t.after(() => dom.window.close());
  const w = dom.window;
  await new Promise(resolve => w.addEventListener('load', resolve, {once: true}));
  w.eval(fs.readFileSync(path.join(root, 'js/dashboard.js'), 'utf8') + '\n' +
    fs.readFileSync(path.join(root, 'js/controls.js'), 'utf8') +
    '\nwindow.testHooks = {updatePairingControls, renderControlLibrary};');
  let init;
  const add = w.document.addEventListener.bind(w.document);
  w.document.addEventListener = (event, fn, ...args) => {
    if(event === 'DOMContentLoaded') init = fn;
    else add(event, fn, ...args);
  };
  w.eval(fs.readFileSync(path.join(root, 'js/clock.js'), 'utf8'));
  init();
  const state = {connected: true, read_only: true, write_access: 'missing_key',
    reason: 'Enter the connection key and reconnect.'};
  const calls = [];
  let failWrite = false;
  const data = {schedules: [{id: 'alarm-1', kind: 'alarm', label: 'Wake up'}],
    todos: [{id: 'todo-1', text: 'Buy milk', done: false}]};
  w.ADAM.ui.settings.sync_token_configured = true; // Must not override server permissions.
  w.ADAM.api = async (url, body) => {
    calls.push({url, body});
    if(url === '/pi/snapshot') return {data, connection: {...state}};
    if(url.startsWith('/pi/write/')) {
      if(failWrite) {
        Object.assign(state, {read_only: true, write_access: 'denied', reason: 'The connection key was rejected.'});
        throw new Error(state.reason);
      }
      return {status: 'ok'};
    }
    throw new Error('Unexpected request: ' + url);
  };
  return {w, state, calls, fail: () => failWrite = true,
    $: id => w.document.getElementById(id)};
}

test('read-only planner disables every edit and returns to verified device selection', async t => {
  const f = await fixture(t);
  await f.w.AdamClock.refresh();
  const edits = f.w.document.querySelectorAll('#scheduleForm input, #scheduleForm select, #scheduleForm button, #todoForm input, #todoForm button, #clockScheduleList button, #clockTodoList button, #clockTodoList input');
  assert.ok(edits.length > 8);
  assert.ok([...edits].every(n => n.disabled));
  assert.equal(f.$('clockNotice').textContent, f.state.reason);
  assert.equal(f.$('clockConnectionBtn').hidden, false);
  let selected=false;
  f.w.AdamOnboarding={selectDevice:async()=>{selected=true;}};
  f.$('clockConnectionBtn').click();
  await tick();
  assert.equal(selected,true);
  f.$('scheduleForm').dispatchEvent(new f.w.Event('submit', {bubbles: true, cancelable: true}));
  await tick();
  assert.equal(f.calls.filter(c => c.url.startsWith('/pi/write/')).length, 0);
});

test('reconnecting with write access enables and sends alarm, timer and to-do edits', async t => {
  const f = await fixture(t);
  await f.w.AdamClock.refresh();
  Object.assign(f.state, {read_only: false, write_access: 'available', reason: ''});
  await f.w.AdamClock.refresh();
  assert.equal(f.$('clockConnectionBtn').hidden, true);
  assert.equal(f.$('scheduleKind').disabled, false);
  f.$('scheduleKind').value = 'alarm';
  f.$('scheduleKind').dispatchEvent(new f.w.Event('change'));
  f.$('scheduleTime').value = '07:00';
  f.$('scheduleLabel').value = 'Wake up';
  f.$('scheduleForm').dispatchEvent(new f.w.Event('submit', {bubbles: true, cancelable: true}));
  await tick();
  assert.deepEqual(JSON.parse(JSON.stringify(f.calls.find(c => c.url === '/pi/write/schedules').body)),
    {kind: 'alarm', when: '07:00', label: 'Wake up'});
  f.$('scheduleKind').value = 'timer';
  f.$('scheduleKind').dispatchEvent(new f.w.Event('change'));
  f.$('scheduleMinutes').value = '3';
  f.$('scheduleForm').dispatchEvent(new f.w.Event('submit', {bubbles: true, cancelable: true}));
  await tick();
  assert.equal(f.calls.filter(c => c.url === '/pi/write/schedules')[1].body.minutes, 3);
  f.$('todoText').value = 'Test task';
  f.$('todoForm').dispatchEvent(new f.w.Event('submit', {bubbles: true, cancelable: true}));
  await tick();
  assert.equal(f.calls.find(c => c.url === '/pi/write/todos').body.text, 'Test task');
});

test('rejected key preserves the draft, disables edits and never retries a write', async t => {
  const f = await fixture(t);
  Object.assign(f.state, {read_only: false, write_access: 'available', reason: ''});
  await f.w.AdamClock.refresh();
  f.fail();
  f.$('scheduleLabel').value = 'Keep this draft';
  f.$('scheduleForm').dispatchEvent(new f.w.Event('submit', {bubbles: true, cancelable: true}));
  await tick();
  assert.equal(f.calls.filter(c => c.url.startsWith('/pi/write/')).length, 1);
  assert.equal(f.$('scheduleLabel').value, 'Keep this draft');
  assert.equal(f.$('scheduleForm').querySelector('[type=submit]').disabled, true);
  assert.match(f.$('clockNotice').textContent, /rejected/);
});

test('laptop pairing buttons explain read-only access instead of offering a failing action', async t => {
  const f = await fixture(t);
  f.w.ADAM.ui.status.robot = {...f.state, capabilities: {laptop_pairing: true}};
  f.w.testHooks.updatePairingControls();
  assert.equal(f.$('authorizeLaptopBtn').disabled, true);
  assert.match(f.$('laptopPairingHelp').textContent, /connection key/);
  f.w.ADAM.ui.status.robot.read_only = false;
  f.w.testHooks.updatePairingControls();
  assert.equal(f.$('authorizeLaptopBtn').disabled, false);
});

test('brightness local test sends zero and displays the acknowledged hardware value', async t => {
  const f = await fixture(t);
  const ui = f.w.ADAM.ui;
  ui.actionCategory = 'display';
  ui.actions = {brightness_set: {enabled: true, needs_value: true, value_type: 'int', category: 'System'}};
  ui.controlDrafts.brightness_set = '0';
  f.w.testHooks.renderControlLibrary();
  let command;
  f.w.fetch = async (url, options) => {
    if(url === '/control') command = JSON.parse(options.body);
    assert.ok(['/control', '/status'].includes(url));
    return {ok: true, json: async () => url === '/control'
      ? {status: 'ok', brightness: 5}
      : {status: 'ok', robot: {}, settings: {}}};
  };
  f.$('runSelectedControl').click();
  await tick();
  assert.deepEqual(command, {action: 'brightness_set', value: 0});
  assert.match(f.$('selectedControlResult').textContent, /"brightness": 5/);
});
