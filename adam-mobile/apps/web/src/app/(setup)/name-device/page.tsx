'use client';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { AdamFaceMark, Button, Screen, ScreenHeader } from '@adam/ui';
import { useSetupStore } from '@/stores/setup-store';
import { readLocalData, saveRecord, errorMessage } from '@/lib/local-data';
import { selectSimulatedDevice } from '@/lib/demo-device';
import { nextStep, setupHref } from '@/lib/setup-flow';

export default function NameDevicePage() {
  const router = useRouter();
  const setup = useSetupStore();
  const [name, setName] = useState(setup.deviceName || 'ADAM');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  async function save(event: React.FormEvent) {
    event.preventDefault(); if (busy) return; setBusy(true); setError('');
    try {
      const serial = setup.selectedSerial || 'SIM-SETUP-ADAM';
      const existing = (await readLocalData()).devices.find((device) => device.serial === serial);
      const updated = await saveRecord('devices', { ...existing, name: name.trim(), serial, simulated: true });
      const device = updated.devices.find((item) => item.serial === serial)!;
      await selectSimulatedDevice(device.id);
      setup.setDeviceName(device.name); setup.complete('name-device');
      const next = nextStep('name-device', setup);
      router.push(next === 'done' ? '/home' : setupHref(next));
    } catch (e) { setError(errorMessage(e)); } finally { setBusy(false); }
  }
  return <Screen className="justify-center gap-8"><AdamFaceMark size="xl" /><ScreenHeader title="What should we call him?" subtitle="Give this ADAM a name. You can change it later." />
    <form onSubmit={save} className="space-y-4"><label className="field-label">ADAM’s name<input className="field" value={name} onChange={(e) => setName(e.target.value)} required maxLength={40} /></label>
      {error && <p role="alert" className="text-sm text-amber-400">{error}</p>}
      <Button block type="submit" disabled={busy || !name.trim()}>{busy ? 'Saving…' : 'Continue'}</Button>
    </form><p className="text-fg-muted text-xs">Demo setup. BLE pairing is simulated.</p>
  </Screen>;
}
