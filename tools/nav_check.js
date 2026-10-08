// Verifies the REAL setupPanelTabs() from dashboard.js against a stub DOM.
// Extracts the function from source rather than reimplementing it, so this
// tests the shipped code path, not a copy of it.
const fs = require('fs');

const SRC = fs.readFileSync('pcAPP/static/js/dashboard.js', 'utf8');

// ---- slice out setupPanelTabs by brace matching -------------------------
const start = SRC.indexOf('function setupPanelTabs()');
if (start < 0) { console.error('FAIL: setupPanelTabs not found'); process.exit(1); }
let depth = 0, end = -1;
for (let i = SRC.indexOf('{', start); i < SRC.length; i++) {
  if (SRC[i] === '{') depth++;
  else if (SRC[i] === '}') { depth--; if (depth === 0) { end = i + 1; break; } }
}
const fnSrc = SRC.slice(start, end);

// ---- minimal DOM -------------------------------------------------------
function mkClassList(initial) {
  const s = new Set((initial || '').split(/\s+/).filter(Boolean));
  return {
    add: (...c) => c.forEach(x => s.add(x)),
    remove: (...c) => c.forEach(x => s.delete(x)),
    contains: c => s.has(c),
    toString: () => [...s].join(' '),
  };
}
function el(id, cls, attrs = {}, text = '') {
  const o = {
    id,
    classList: mkClassList(cls),
    textContent: text,
    style: {},
    _h: {},
    getAttribute: k => attrs[k],
  };
  o.addEventListener = (ev, fn) => { (o._h[ev] = o._h[ev] || []).push(fn); };
  o.click = () => (o._h.click || []).forEach(f => f({ preventDefault() {} }));
  return o;
}

const TABS = ['tab-dashboard', 'tab-actions', 'tab-devices', 'tab-clock', 'tab-activity', 'tab-settings'];
const LABELS = { 'tab-dashboard': 'Home', 'tab-actions': 'Actions', 'tab-devices': 'Devices',
                 'tab-clock': 'Clock', 'tab-activity': 'Activity Log', 'tab-settings': 'Settings' };

const navLinks = TABS.map((t, i) =>
  el(`nav-${t}`, i === 0 ? 'nav-link on' : 'nav-link', { 'data-target': t }, LABELS[t]));
const tabEls = TABS.map((t, i) =>
  el(t, i === 0 ? 'tab-content active' : 'tab-content hidden'));

const byId = {
  headerTitle: el('headerTitle', ''),
  headerEyebrow: el('headerEyebrow', ''),
  headerWelcome: el('headerWelcome', ''),
};
tabEls.forEach(e => { byId[e.id] = e; });

global.document = {
  getElementById: id => byId[id] || null,
  querySelectorAll: sel => {
    if (sel === '.nav-link') return navLinks;
    if (sel === '.tab-content') return tabEls;
    return [];
  },
};
global.window = { addEventListener: () => {} };

// stubs for what the handler calls
let called = [];
global.loadAndRenderActions = () => called.push('actions');
global.openSettings = () => called.push('settings');
global.fetchActivityLogs = () => called.push('activity');
global.fetchPiSnapshot = () => called.push('clock');
global.fitStage = () => called.push('dashboard');

eval(fnSrc + '\nsetupPanelTabs();');

// ---- assertions --------------------------------------------------------
let pass = 0, fail = 0;
const ok = (name, cond) => { if (cond) { pass++; console.log('  PASS  ' + name); }
                             else { fail++; console.log('  FAIL  ' + name); } };

ok('handlers bound to all 6 nav links', navLinks.every(l => (l._h.click || []).length === 1));

for (const target of TABS) {
  called = [];
  navLinks.find(l => l.getAttribute('data-target') === target).click();
  const active = tabEls.filter(t => t.classList.contains('active')).map(t => t.id);
  ok(`click ${LABELS[target]} -> only ${target} active`,
     active.length === 1 && active[0] === target);
  ok(`click ${LABELS[target]} -> others hidden`,
     tabEls.filter(t => t.id !== target).every(t => t.classList.contains('hidden')));
  ok(`click ${LABELS[target]} -> nav 'on' moved`,
     navLinks.filter(l => l.classList.contains('on')).length === 1 &&
     navLinks.find(l => l.classList.contains('on')).getAttribute('data-target') === target);
  ok(`click ${LABELS[target]} -> header title = "${LABELS[target]}"`,
     byId.headerTitle.textContent === LABELS[target]);
}

console.log(`\n  ${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
