import { Capacitor } from '@capacitor/core';

/**
 * Platform detection shared by browser and Capacitor persistence adapters.
 * Legacy delay helpers below remain for dormant prototype integrations;
 * production screens use native plugins or explicit unavailable states.
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
  if (typeof window === 'undefined') {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}
