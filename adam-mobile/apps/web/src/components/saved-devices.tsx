'use client';
import { useState } from 'react';
import { Button } from '@adam/ui';
import { Panel, Notice, Confirm } from './companion-ui';
import { useLocalData } from '@/lib/use-local-data';
import { saveRecord, deleteRecord, errorMessage } from '@/lib/local-data';
import { syncSimulatedDevice } from '@/lib/ble-simulation';
import type { SavedDevice } from '@/lib/companion-records';
import { selectSimulatedDevice } from '@/lib/demo-device';

export function SavedDevices() {
  const { data, error: loadError } = useLocalData();
  const [editing, setEditing] = useState<SavedDevice | null>(null);
  const [name, setName] = useState('');
  const [remove, setRemove] = useState<SavedDevice | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  async function action(work: () => Promise<unknown>, done: string) {
    setBusy(true); setError(''); setMessage('');
    try { await work(); setMessage(done); setRemove(null); } catch (e) { setError(errorMessage(e)); } finally { setBusy(false); }
  }
  return <Panel>
    <h3 className="font-medium">Saved ADAM devices</h3>
    <p className="text-fg-muted my-3 text-sm leading-6">Give each ADAM its own name. These records sync with your account. Bluetooth transfers are simulated until ADAM BLE is available.</p>
    <div className="space-y-3">{data.devices.map((device) => <div key={device.id} className="border-border rounded-xl border p-3">
      <p className="break-words font-medium">{device.name}</p><p className="text-fg-muted mt-1 break-all text-xs">{device.simulated ? 'Simulated ADAM' : 'Saved ADAM'} · {device.serial}</p>
      <div className="mt-3 flex flex-wrap gap-2">
        {device.simulated && <Button size="sm" variant="outline" disabled={busy} onClick={() => void action(() => selectSimulatedDevice(device.id), `${device.name} is selected for the controls below.`)}>Connect simulation</Button>}
        <Button size="sm" variant="outline" disabled={busy} onClick={() => { setEditing(device); setName(device.name); }}>Rename</Button>
        {device.simulated && <Button size="sm" variant="outline" disabled={busy} onClick={() => void action(() => syncSimulatedDevice(device.id), `Synced memories, planner and devices with ${device.name} over simulated BLE.`)}>Sync simulated BLE</Button>}
        <Button size="sm" variant="ghost" disabled={busy} onClick={() => setRemove(device)}>Remove</Button>
      </div>
    </div>)}</div>
    <form className="mt-4 space-y-3" onSubmit={(event) => { event.preventDefault(); void action(async () => {
      await saveRecord('devices', editing ? { ...editing, name } : { name, serial: `SIM-${crypto.randomUUID().slice(0, 8).toUpperCase()}`, simulated: true });
      setName(''); setEditing(null);
    }, editing ? 'Device renamed. Ready to sync.' : 'Simulated ADAM saved.'); }}>
      <label className="field-label">{editing ? 'Device name' : 'Name another ADAM'}<input className="field" value={name} onChange={(e) => setName(e.target.value)} required maxLength={40} placeholder="Desk ADAM" /></label>
      <Button block type="submit" disabled={busy || !name.trim()}>{editing ? 'Save name' : 'Add simulated ADAM'}</Button>
      {editing && <Button block variant="ghost" disabled={busy} onClick={() => { setEditing(null); setName(''); }}>Cancel rename</Button>}
    </form>
    {(error || loadError) && <Notice error>{error || loadError}</Notice>}{message && <Notice>{message}</Notice>}
    {remove && <Confirm title={`Remove ${remove.name}?`} busy={busy} onClose={() => setRemove(null)} onConfirm={() => void action(() => deleteRecord('devices', remove.id), 'Device removed. The removal will sync with your account.')} >Memories and planner entries are kept. This removes the saved device record from your account when synced.</Confirm>}
  </Panel>;
}
