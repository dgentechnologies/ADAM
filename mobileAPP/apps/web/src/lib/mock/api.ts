import type {
  CreditBalance,
  CreditPack,
  Device,
  DiscoveredDevice,
  GalleryItem,
  HandoffProgress,
  MemoryEntry,
  OtaState,
  PairedLaptop,
  WifiNetwork,
} from '@adam/types';

import { getItem } from '../native/preferences';
import { delay } from '../native/platform';
import {
  MOCK_BALANCE,
  MOCK_CREDIT_PACKS,
  MOCK_DEVICE,
  MOCK_DISCOVERED,
  MOCK_GALLERY,
  MOCK_LAPTOPS,
  MOCK_MEMORY,
  MOCK_NETWORKS,
  MOCK_OTA,
} from './fixtures';

export const queryKeys = {
  device: ['device'] as const,
  discovery: ['discovery'] as const,
  networks: ['networks'] as const,
  creditPacks: ['credits', 'packs'] as const,
  balance: ['credits', 'balance'] as const,
  memory: ['memory'] as const,
  gallery: ['gallery'] as const,
  laptops: ['laptops'] as const,
  ota: ['ota'] as const,
};

export async function fetchDevice(): Promise<Device> {
  if (typeof window === 'undefined') {
    return MOCK_DEVICE;
  }

  const configuredName = 'ADAM';
  const configuredSsid = 'DASGUPTA';
  const configuredSerial = 'DGEN-ADAM-0007';
  const isFounder = true;
  const founderNum = 7;
  const brainMode = 'managed';

  // 1. Try querying real backend API
  const candidateUrls = [
    typeof process !== 'undefined' && process.env?.NEXT_PUBLIC_API_URL ? process.env.NEXT_PUBLIC_API_URL : null,
    'http://192.168.0.128:3001',
    typeof window !== 'undefined' && window.location?.hostname ? `${window.location.protocol}//${window.location.hostname}:3001` : null,
    'http://localhost:3001',
    'http://10.0.2.2:3001',
  ].filter(Boolean) as string[];

  const uniqueUrls = Array.from(new Set(candidateUrls));
  for (const baseUrl of uniqueUrls) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 1500);
      const res = await fetch(`${baseUrl}/api/device`, {
        signal: controller.signal,
        headers: { Accept: 'application/json' },
      });
      clearTimeout(timeoutId);
      if (res.ok) {
        const live = await res.json();
        return {
          id: live.id || MOCK_DEVICE.id,
          serial: live.serial || configuredSerial,
          shortId: live.shortId || (configuredSerial ? configuredSerial.replace('DGEN-', '') : 'ADAM-3F2A'),
          name: live.name || configuredName,
          ownerId: live.ownerId || MOCK_DEVICE.ownerId,
          status: live.status || 'online',
          expression: live.expression || 'idle',
          wifiSsid: live.wifiSsid || configuredSsid,
          firmwareVersion: live.firmwareVersion || '40.2.1',
          hardwareBatch: live.hardwareBatch || 'FE-2026-01',
          isFounderEdition: live.isFounderEdition ?? isFounder,
          founderNumber: live.founderNumber ?? founderNum,
          lastSeenAt: live.lastSeenAt || new Date().toISOString(),
          claimedAt: live.claimedAt || new Date().toISOString(),
          aiBrainMode: live.aiBrainMode || brainMode,
        };
      }
    } catch {
      // Continue to next candidate
    }
  }

  // 2. Read saved preferences
  try {
    const savedCredentialsRaw = await getItem('adam.wifi.credentials');
    let savedSsid = configuredSsid;
    if (savedCredentialsRaw) {
      const creds = JSON.parse(savedCredentialsRaw);
      if (creds.ssid) savedSsid = creds.ssid;
    }
    return {
      ...MOCK_DEVICE,
      name: configuredName,
      serial: configuredSerial,
      shortId: configuredSerial.replace('DGEN-', ''),
      wifiSsid: savedSsid,
      isFounderEdition: isFounder,
      founderNumber: founderNum,
      aiBrainMode: brainMode,
    };
  } catch {
    return {
      ...MOCK_DEVICE,
      name: configuredName,
      wifiSsid: configuredSsid,
      isFounderEdition: isFounder,
      founderNumber: founderNum,
      aiBrainMode: brainMode,
    };
  }
}

export async function sendDeviceAction(action: 'mute' | 'unmute' | 'wake' | 'sleep' | 'restart'): Promise<boolean> {
  if (typeof window === 'undefined') {
    return true;
  }
  const candidateUrls = [
    typeof process !== 'undefined' && process.env?.NEXT_PUBLIC_API_URL ? process.env.NEXT_PUBLIC_API_URL : null,
    'http://192.168.0.128:3001',
    typeof window !== 'undefined' && window.location?.hostname ? `${window.location.protocol}//${window.location.hostname}:3001` : null,
    'http://localhost:3001',
    'http://10.0.2.2:3001',
  ].filter(Boolean) as string[];

  const uniqueUrls = Array.from(new Set(candidateUrls));
  for (const baseUrl of uniqueUrls) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2000);
      const res = await fetch(`${baseUrl}/api/device/action`, {
        method: 'POST',
        signal: controller.signal,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action }),
      });
      clearTimeout(timeoutId);
      if (res.ok) {
        return true;
      }
    } catch {
      // Fallback
    }
  }
  return true;
}

/** Discovery is slower than everything else so the radar actually sweeps. */
export async function scanForDevices(): Promise<DiscoveredDevice[]> {
  if (typeof window === 'undefined') {
    return MOCK_DISCOVERED;
  }
  await delay(2600);
  return MOCK_DISCOVERED;
}

// ─────────────────────────────────────────────────────────────────────────────
// Smart home device discovery
// ─────────────────────────────────────────────────────────────────────────────

export type SmartDeviceType =
  | 'hub' | 'light' | 'bulb' | 'strip' | 'switch' | 'plug'
  | 'thermostat' | 'speaker' | 'tv' | 'camera' | 'lock' | 'sensor'
  | 'router' | 'media_player' | 'unknown';

export interface SmartDevice {
  id: string;
  name: string;
  type: SmartDeviceType;
  brand: string;
  ip: string;
  mac?: string;
  model?: string;
  online: boolean;
  protocol: 'ssdp' | 'mdns' | 'arp' | 'static';
  uid?: string;
  location?: string;
  detectedAt: string;
}

export interface SmartHomeResult {
  devices: SmartDevice[];
  count: number;
  scannedAt: string;
  source: 'live' | 'cached' | 'fallback';
}

export async function fetchSmartHomeDevices(force = false): Promise<SmartHomeResult> {
  if (typeof window === 'undefined') {
    return { devices: [], count: 0, scannedAt: new Date().toISOString(), source: 'fallback' };
  }

  const candidateUrls = [
    typeof process !== 'undefined' && process.env?.NEXT_PUBLIC_API_URL ? process.env.NEXT_PUBLIC_API_URL : null,
    'http://192.168.0.128:3001',
    typeof window !== 'undefined' && window.location?.hostname
      ? `${window.location.protocol}//${window.location.hostname}:3001` : null,
    'http://localhost:3001',
    'http://10.0.2.2:3001',
  ].filter(Boolean) as string[];

  for (const baseUrl of Array.from(new Set(candidateUrls))) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 8000); // SSDP needs time
      const url = force
        ? `${baseUrl}/api/smart-home/devices?force=true`
        : `${baseUrl}/api/smart-home/devices`;
      const res = await fetch(url, {
        signal: controller.signal,
        headers: { Accept: 'application/json' },
      });
      clearTimeout(timeoutId);
      if (res.ok) {
        const data = await res.json();
        return data as SmartHomeResult;
      }
    } catch {
      // Try next endpoint
    }
  }

  return { devices: [], count: 0, scannedAt: new Date().toISOString(), source: 'fallback' };
}

export function getApiBaseUrl(): string {
  if (typeof process !== 'undefined' && process.env?.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  if (typeof window !== 'undefined' && window.location?.hostname && window.location.hostname !== 'localhost') {
    return `${window.location.protocol}//${window.location.hostname}:3001`;
  }
  return 'http://192.168.0.128:3001';
}

export async function scanNetworks(force = false): Promise<WifiNetwork[]> {
  if (typeof window === 'undefined') {
    return MOCK_NETWORKS;
  }
  // 1. Native Android WebView JavascriptInterface bridge
  if (typeof window !== 'undefined' && (window as any).AdamNativeWifi?.getWifiNetworks) {
    try {
      const raw = (window as any).AdamNativeWifi.getWifiNetworks();
      const parsed = typeof raw === 'string' ? JSON.parse(raw) : raw;
      if (Array.isArray(parsed) && parsed.length > 0) {
        return parsed;
      }
    } catch (e) {
      console.warn('[wifi] Native Android Wi-Fi scan exception:', e);
    }
  }

  // 2. Candidate backend network endpoints (LAN host, local dev, emulator)
  const candidateUrls = [
    typeof process !== 'undefined' && process.env?.NEXT_PUBLIC_API_URL ? process.env.NEXT_PUBLIC_API_URL : null,
    'http://192.168.0.128:3001',
    typeof window !== 'undefined' && window.location?.hostname ? `${window.location.protocol}//${window.location.hostname}:3001` : null,
    'http://localhost:3001',
    'http://10.0.2.2:3001',
  ].filter(Boolean) as string[];

  const uniqueUrls = Array.from(new Set(candidateUrls));

  for (const baseUrl of uniqueUrls) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2500);
      const url = force ? `${baseUrl}/api/wifi/networks?force=true` : `${baseUrl}/api/wifi/networks`;

      const res = await fetch(url, {
        signal: controller.signal,
        headers: { Accept: 'application/json' },
      });
      clearTimeout(timeoutId);

      if (res.ok) {
        const data = await res.json();
        const list = Array.isArray(data) ? data : (data.networks ?? []);
        if (Array.isArray(list) && list.length > 0) {
          return list;
        }
      }
    } catch {
      // Try next candidate endpoint
    }
  }

  // 3. Resilient fallback containing real environment network
  await delay(600);
  return MOCK_NETWORKS;
}

/**
 * Mocked Wi-Fi handoff. Resolves the three checklist rows in sequence via the
 * `onProgress` callback rather than returning once, because the Connecting screen
 * has to render each transition.
 */
export async function runHandoff(
  /** Ignored by the mock; the real transport encrypts and forwards it. */
  _credentials: { ssid: string; password: string },
  onProgress: (progress: HandoffProgress) => void,
): Promise<HandoffProgress> {
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

  for (let index = 0; index < order.length; index += 1) {
    onProgress(snapshot(index));
    await delay(1400);
    elapsed += 1400;
  }

  const done: HandoffProgress = {
    steps: order.map((step) => ({ step, state: 'complete' as const })),
    failure: null,
    elapsedMs: elapsed,
  };
  onProgress(done);
  return done;
}

export async function fetchCreditPacks(): Promise<CreditPack[]> {
  await delay();
  return MOCK_CREDIT_PACKS;
}

export async function fetchBalance(): Promise<CreditBalance> {
  if (typeof window === 'undefined') {
    return MOCK_BALANCE;
  }
  const candidateUrls = [
    typeof process !== 'undefined' && process.env?.NEXT_PUBLIC_API_URL ? process.env.NEXT_PUBLIC_API_URL : null,
    'http://192.168.0.128:3001',
    typeof window !== 'undefined' && window.location?.hostname ? `${window.location.protocol}//${window.location.hostname}:3001` : null,
    'http://localhost:3001',
    'http://10.0.2.2:3001',
  ].filter(Boolean) as string[];

  for (const baseUrl of Array.from(new Set(candidateUrls))) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 1500);
      const res = await fetch(`${baseUrl}/api/credits/balance`, {
        signal: controller.signal,
        headers: { Accept: 'application/json' },
      });
      clearTimeout(timeoutId);
      if (res.ok) {
        return await res.json();
      }
    } catch {
      // Continue
    }
  }

  await delay();
  return MOCK_BALANCE;
}

export async function fetchMemory(): Promise<MemoryEntry[]> {
  await delay();
  return MOCK_MEMORY;
}

export async function fetchGallery(): Promise<GalleryItem[]> {
  await delay();
  return MOCK_GALLERY;
}

export async function fetchLaptops(): Promise<PairedLaptop[]> {
  await delay();
  return MOCK_LAPTOPS;
}

export async function fetchOtaState(): Promise<OtaState> {
  await delay();
  return MOCK_OTA;
}

/**
 * SECURITY (tech spec §7): a BYOK key is never sent to the backend. The real
 * implementation encrypts it with the Pi's public key and posts it to the unit
 * over the local channel. This stub therefore does not persist the key anywhere —
 * it only reports that the unit accepted it.
 */
export async function sendByokKeyToDevice(key: string): Promise<{ accepted: boolean }> {
  await delay(1600);
  return { accepted: key.trim().length > 20 };
}
