import { Preferences } from '@capacitor/preferences';

import { isNative } from './platform';

/**
 * Key/value storage for non-sensitive state — the persisted setup step, theme
 * choice, notification preference.
 *
 * Capacitor Preferences on device, `localStorage` in the browser. Both are
 * unencrypted, so auth tokens must not come through here; see `secure-storage`.
 */
export async function getItem(key: string): Promise<string | null> {
  if (isNative()) {
    const { value } = await Preferences.get({ key });
    return value ?? null;
  }
  if (typeof window === 'undefined') return null;
  return window.localStorage.getItem(key);
}

export async function setItem(key: string, value: string): Promise<void> {
  if (isNative()) {
    await Preferences.set({ key, value });
    return;
  }
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(key, value);
}

export async function removeItem(key: string): Promise<void> {
  if (isNative()) {
    await Preferences.remove({ key });
    return;
  }
  if (typeof window === 'undefined') return;
  window.localStorage.removeItem(key);
}
