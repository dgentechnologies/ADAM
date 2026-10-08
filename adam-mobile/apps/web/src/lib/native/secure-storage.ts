import { Capacitor } from '@capacitor/core';
import { Companion } from './companion';
const memory = new Map<string, string>();
export async function getSecret(key: string): Promise<string | null> {
  return Capacitor.isNativePlatform()
    ? (await Companion.getSecret({ key })).value
    : (memory.get(key) ?? null);
}
export async function setSecret(key: string, value: string): Promise<void> {
  if (Capacitor.isNativePlatform()) {
    await Companion.setSecret({ key, value });
    return;
  }
  memory.set(key, value);
}
export async function removeSecret(key: string): Promise<void> {
  if (Capacitor.isNativePlatform()) {
    await Companion.removeSecret({ key });
    return;
  }
  memory.delete(key);
}
export async function clearSecrets(): Promise<void> {
  if (Capacitor.isNativePlatform()) await Companion.clearSecrets();
  memory.clear();
}
