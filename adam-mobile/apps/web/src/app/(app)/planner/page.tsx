'use client';
import { useState } from 'react';
import { Button } from '@adam/ui';
import { Page, Panel, Notice, Confirm } from '@/components/companion-ui';
import { useLocalData } from '@/lib/use-local-data';
import { saveRecord, deleteRecord, errorMessage } from '@/lib/local-data';
import type { Todo, Clock } from '@/lib/companion-records';

export default function PlannerPage() {
  const { data, error: loadError } = useLocalData();
  const [todo, setTodo] = useState<Todo | null>(null);
  const [clock, setClock] = useState<Clock | null>(null);
  const [text, setText] = useState('');
  const [dueAt, setDueAt] = useState('');
  const [label, setLabel] = useState('');
  const [kind, setKind] = useState<Clock['kind']>('alarm');
  const [when, setWhen] = useState('');
  const [minutes, setMinutes] = useState(5);
  const [deviceId, setDeviceId] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [remove, setRemove] = useState<{ kind: 'todos' | 'clocks'; id: string } | null>(null);
  async function action(work: () => Promise<unknown>) { setBusy(true); setError(''); try { await work(); } catch (e) { setError(errorMessage(e)); } finally { setBusy(false); } }
  const localTime = (value: string) => { const date = new Date(value); return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString().slice(0, 16); };
  return <Page title="Clock & to-do" back="/home">
    <p className="text-fg-muted text-sm leading-6">Save your plans here and sync them through your account or simulated BLE. Timers and alarms are saved plans; robot alerts remain simulated.</p>
    {(error || loadError) && <Notice error>{error || loadError}</Notice>}
    <label className="field-label">ADAM for new plans<select className="field" value={deviceId} onChange={(e) => setDeviceId(e.target.value)}><option value="">All my ADAMs</option>{data.devices.map((device) => <option key={device.id} value={device.id}>{device.name}</option>)}</select></label>
    <Panel><h2 className="font-medium">To-dos</h2>
      <div className="my-4 space-y-3">{data.todos.map((item) => <div key={item.id} className="border-border rounded-xl border p-3">
        <label className="flex items-start gap-3"><input type="checkbox" className="mt-1 h-5 w-5" checked={item.done} disabled={busy} onChange={() => void action(() => saveRecord('todos', { ...item, done: !item.done }))} /><span className={item.done ? 'line-through text-fg-muted' : 'break-words'}>{item.text}</span></label>
        {item.dueAt && <p className="text-fg-muted mt-2 text-xs">Due {new Date(item.dueAt).toLocaleString()}</p>}
        <div className="mt-2 flex gap-2"><Button size="sm" variant="ghost" disabled={busy} onClick={() => { setTodo(item); setText(item.text); setDueAt(item.dueAt ? localTime(item.dueAt) : ''); }}>Edit</Button><Button size="sm" variant="ghost" disabled={busy} onClick={() => setRemove({ kind: 'todos', id: item.id })}>Delete</Button></div>
      </div>)}</div>
      <form className="space-y-3" onSubmit={(e) => { e.preventDefault(); void action(async () => { await saveRecord('todos', { ...todo, text, done: todo?.done ?? false, dueAt: dueAt ? new Date(dueAt).toISOString() : '', deviceId: todo?.deviceId ?? deviceId }); setTodo(null); setText(''); setDueAt(''); }); }}>
        <label className="field-label">{todo ? 'Edit to-do' : 'New to-do'}<input className="field" value={text} onChange={(e) => setText(e.target.value)} required maxLength={2000} /></label>
        <label className="field-label">Due date (optional)<input className="field" type="datetime-local" value={dueAt} onChange={(e) => setDueAt(e.target.value)} /></label>
        <Button block type="submit" disabled={busy || !text.trim()}>{todo ? 'Save to-do' : 'Add to-do'}</Button>
        {todo && <Button variant="ghost" disabled={busy} onClick={() => { setTodo(null); setText(''); setDueAt(''); }}>Cancel edit</Button>}
      </form>
    </Panel>
    <Panel><h2 className="font-medium">Alarms, timers & reminders</h2>
      <div className="my-4 space-y-3">{data.clocks.map((item) => <div key={item.id} className="border-border rounded-xl border p-3">
        <p className="break-words font-medium">{item.label || item.kind}</p><p className="text-fg-muted my-2 text-xs">{item.kind} · {new Date(item.when).toLocaleString()} · {item.enabled ? 'Enabled' : 'Paused'}</p>
        <div className="flex flex-wrap gap-2"><Button size="sm" variant="ghost" disabled={busy} onClick={() => void action(() => saveRecord('clocks', { ...item, enabled: !item.enabled }))}>{item.enabled ? 'Pause' : 'Enable'}</Button><Button size="sm" variant="ghost" disabled={busy} onClick={() => { setClock(item); setLabel(item.label); setKind(item.kind); setWhen(localTime(item.when)); }}>Edit</Button><Button size="sm" variant="ghost" disabled={busy} onClick={() => setRemove({ kind: 'clocks', id: item.id })}>Delete</Button></div>
      </div>)}</div>
      <form className="space-y-3" onSubmit={(e) => { e.preventDefault(); void action(async () => {
        const target = kind === 'timer' && !clock ? new Date(Date.now() + minutes * 60000).toISOString() : new Date(when).toISOString();
        await saveRecord('clocks', { ...clock, kind, label, when: target, enabled: clock?.enabled ?? true, deviceId: clock?.deviceId ?? deviceId }); setClock(null); setLabel(''); setWhen('');
      }); }}>
        <label className="field-label">Type<select className="field" value={kind} onChange={(e) => setKind(e.target.value as Clock['kind'])}><option value="alarm">Alarm</option><option value="timer">Timer</option><option value="reminder">Reminder</option></select></label>
        <label className="field-label">Label<input className="field" value={label} onChange={(e) => setLabel(e.target.value)} maxLength={80} required={kind === 'reminder'} /></label>
        {kind === 'timer' && !clock ? <label className="field-label">Minutes<input className="field" type="number" min={1} max={10080} required value={minutes} onChange={(e) => setMinutes(Number(e.target.value))} /></label> : <label className="field-label">When<input className="field" type="datetime-local" required value={when} onChange={(e) => setWhen(e.target.value)} /></label>}
        <Button block type="submit" disabled={busy}>{clock ? 'Save plan' : 'Add plan'}</Button>
        {clock && <Button variant="ghost" disabled={busy} onClick={() => { setClock(null); setLabel(''); setWhen(''); }}>Cancel edit</Button>}
      </form>
    </Panel>
    {remove && <Confirm title="Delete this plan?" busy={busy} onClose={() => setRemove(null)} onConfirm={() => void action(async () => { await deleteRecord(remove.kind, remove.id); setRemove(null); })}>This deletion will sync with your account and simulated ADAM devices.</Confirm>}
  </Page>;
}
