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

/**
 * Mock API. Same shape the real client will have — async, typed by @adam/types,
 * consumed through TanStack Query — so replacing these bodies with `fetch` calls
 * against `apps/api` is a one-file change per endpoint.
 *
 * `apps/api` is intentionally not involved: this pass is UI-only.
 */
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
  await delay();
  return MOCK_DEVICE;
}

/** Discovery is slower than everything else so the radar actually sweeps. */
export async function scanForDevices(): Promise<DiscoveredDevice[]> {
  await delay(2600);
  return MOCK_DISCOVERED;
}

export async function scanNetworks(): Promise<WifiNetwork[]> {
  await delay(1200);
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
