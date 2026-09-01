import { Capacitor } from '@capacitor/core';

/**
 * Native abstraction layer (tech spec §3.5).
 *
 * Every native capability is reached through this module, never through a
 * Capacitor plugin import in a component. Each function has a browser fallback
 * so the whole app runs in a plain tab — which is how it is developed — and so a
 * missing plugin degrades to a stub instead of a crash.
 *
 * Nothing here is wired to a real plugin yet: BLE, mDNS, Wi-Fi handoff and camera
 * are deliberately mocked for this pass, per the agreed build order.
 */
export function isNative(): boolean {
  return Capacitor.isNativePlatform();
}

export function platform(): 'web' | 'ios' | 'android' {
  const value = Capacitor.getPlatform();
  return value === 'ios' || value === 'android' ? value : 'web';
}

/** Fixed latency for every mocked native call, so loading states are visible. */
export const MOCK_LATENCY_MS = 700;

export function delay(ms: number = MOCK_LATENCY_MS): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}
