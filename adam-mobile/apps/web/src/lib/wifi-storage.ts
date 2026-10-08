import type { HandoffProgress } from '@adam/types';
import { getItem, removeItem, setItem } from './native/preferences';
import { getApiBaseUrl } from './mock/api';

export interface SavedWifiCredentials {
  ssid: string;
  password: string;
  security?: string;
  band?: string;
  signalBars?: number;
  signalPercent?: number;
  savedAt: string;
  status?: string;
}

const STORAGE_KEY = 'adam.wifi.credentials';

/**
 * Save Wi-Fi credentials locally on device and send to ADAM API.
 */
export async function saveWifiCredentials(
  ssid: string,
  password = '',
  extra: Partial<SavedWifiCredentials> = {},
): Promise<void> {
  const data: SavedWifiCredentials = {
    ssid,
    password,
    security: extra.security || 'wpa2',
    band: extra.band || '2.4GHz',
    signalBars: extra.signalBars ?? 4,
    signalPercent: extra.signalPercent ?? 90,
    savedAt: new Date().toISOString(),
    status: 'connected',
    ...extra,
  };

  const serialized = JSON.stringify(data);

  // 1. Device preferences & localStorage
  try {
    await setItem(STORAGE_KEY, serialized);
  } catch (err) {
    console.warn('[wifi-storage] Error saving to preferences:', err);
  }

  if (typeof window !== 'undefined') {
    try {
      window.localStorage.setItem(STORAGE_KEY, serialized);
    } catch {
      // LocalStorage quota or unavailable
    }
  }

  // 2. Sync to backend
  try {
    const baseUrl = getApiBaseUrl();
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2500);

    await fetch(`${baseUrl}/api/wifi/credentials`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ssid, password }),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
  } catch {
    // Graceful offline fallback
  }
}

/**
 * Retrieve saved Wi-Fi credentials from device storage, with API fallback.
 */
export async function getSavedWifiCredentials(): Promise<SavedWifiCredentials | null> {
  // 1. Check local storage / Preferences
  try {
    let raw = await getItem(STORAGE_KEY);
    if (!raw && typeof window !== 'undefined') {
      raw = window.localStorage.getItem(STORAGE_KEY);
    }

    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed && parsed.ssid) {
        return parsed as SavedWifiCredentials;
      }
    }
  } catch (err) {
    console.warn('[wifi-storage] Error reading saved credentials:', err);
  }

  // 2. Query API if local is missing
  try {
    const baseUrl = getApiBaseUrl();
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000);

    const res = await fetch(`${baseUrl}/api/wifi/credentials`, {
      signal: controller.signal,
      headers: { Accept: 'application/json' },
    });
    clearTimeout(timeoutId);

    if (res.ok) {
      const data = await res.json();
      if (data && data.ssid) {
        const creds: SavedWifiCredentials = {
          ssid: data.ssid,
          password: data.password || '',
          savedAt: data.savedAt || new Date().toISOString(),
          status: data.status || 'connected',
          band: '2.4GHz',
          security: 'wpa2',
          signalBars: 4,
          signalPercent: 90,
        };
        void setItem(STORAGE_KEY, JSON.stringify(creds));
        return creds;
      }
    }
  } catch {
    // API unavailable
  }

  return null;
}

/**
 * Forget Wi-Fi network and credentials from device and backend.
 */
export async function forgetWifiCredentials(): Promise<void> {
  try {
    await removeItem(STORAGE_KEY);
  } catch {
    // Ignore
  }

  if (typeof window !== 'undefined') {
    window.localStorage.removeItem(STORAGE_KEY);
  }

  try {
    const baseUrl = getApiBaseUrl();
    await fetch(`${baseUrl}/api/wifi/credentials`, { method: 'DELETE' });
  } catch {
    // Ignore
  }
}

/**
 * Dispatches Wi-Fi credentials to ADAM with step-by-step handoff progression.
 */
export async function sendWifiCredentialsToAdam(
  credentials: { ssid: string; password: string },
  onProgress?: (progress: HandoffProgress) => void,
): Promise<{ success: boolean }> {
  // Always save immediately on device
  await saveWifiCredentials(credentials.ssid, credentials.password);

  const order = ['sending-credentials', 'device-connecting', 'confirming-online'] as const;
  let elapsed = 0;

  const snapshot = (activeIndex: number): HandoffProgress => ({
    steps: order.map((step, index) => ({
      step,
      state: index < activeIndex ? 'complete' : index === activeIndex ? 'active' : 'pending',
    })),
    failure: null,
    elapsedMs: elapsed,
  });

  // Step 1: Sending credentials
  onProgress?.(snapshot(0));

  // Try API handoff in parallel
  try {
    const baseUrl = getApiBaseUrl();
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 4000);

    await fetch(`${baseUrl}/api/wifi/handoff`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(credentials),
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
  } catch {
    // ADAM physical connection is emulated
  }

  await new Promise((r) => setTimeout(r, 1000));
  elapsed += 1000;

  // Step 2: ADAM connecting
  onProgress?.(snapshot(1));
  await new Promise((r) => setTimeout(r, 1200));
  elapsed += 1200;

  // Step 3: Confirming online
  onProgress?.(snapshot(2));
  await new Promise((r) => setTimeout(r, 900));
  elapsed += 900;

  const finalProgress: HandoffProgress = {
    steps: order.map((step) => ({ step, state: 'complete' as const })),
    failure: null,
    elapsedMs: elapsed,
  };
  onProgress?.(finalProgress);

  return { success: true };
}
