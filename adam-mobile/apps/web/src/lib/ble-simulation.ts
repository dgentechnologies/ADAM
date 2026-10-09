import { getItem, setItem } from './native/preferences';
import { readLocalData, updateLocalData } from './local-data';
import { emptyCompanion, mergeCompanions, validateCompanion, capturePhoneChanges, type Companion } from './firebase/companion-sync';

let pending: Promise<unknown> = Promise.resolve();
/** One isolated simulated robot per account/device. No Bluetooth radio or real robot commands. */
export function syncSimulatedDevice(deviceId: string): Promise<Companion> {
  const operation = pending.catch(() => undefined).then(async () => {
    const { getFirebaseAuth } = await import('./firebase/config');
    const auth = getFirebaseAuth();
    await auth.authStateReady();
    const uid = auth.currentUser?.uid ?? 'guest';
    const check = () => { if ((auth.currentUser?.uid ?? 'guest') !== uid) throw new Error('The account changed. Please sync again.'); };
    const phone = await readLocalData();
    const device = phone.devices.find((item) => item.id === deviceId);
    if (!device?.simulated) throw new Error('BLE is available for simulated ADAM devices only.');
    const key = `adam.ble-simulation.v2.${encodeURIComponent(uid)}.${device.id}`;
    const raw = await getItem(key);
    const saved = raw ? JSON.parse(raw) : null;
    const previous = saved ? validateCompanion(saved.companion) : emptyCompanion();
    const { LocalData } = await import('./local-data');
    const baseline = saved ? LocalData.parse(saved.baseline) : null;
    const outgoing = capturePhoneChanges(previous, baseline, phone, new Date().toISOString());
    const merged = mergeCompanions(previous, outgoing);
    check();
    await setItem(key, JSON.stringify({ companion: merged, baseline: phone }));
    check();
    await updateLocalData((current) => {
      check();
      // Keep edits made while the simulated transfer was in flight.
      const final = capturePhoneChanges(merged, phone, current, new Date().toISOString());
      return applySimulation(current, final);
    });
    return merged;
  });
  pending = operation;
  return operation;
}

function applySimulation(data: Awaited<ReturnType<typeof readLocalData>>, companion: Companion) {
  const live = (items: Record<string, { deleted: boolean }>) => Object.values(items).filter((item) => !item.deleted).map(({ deleted: _, ...item }) => item);
  // The same validator used for cloud imports bounds all received records.
  return { ...data, facts: live(companion.memories), todos: live(companion.todos), clocks: live(companion.clocks), devices: live(companion.devices),
    ...Object.fromEntries(Object.entries(companion.preferences).map(([name, item]) => [name, item.value])) } as typeof data;
}
