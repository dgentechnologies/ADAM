'use client';

import {
  INITIAL_SETUP_STATE,
  SETUP_PROGRESS_STEPS,
  SetupState,
  type AiBrainMode,
  type SetupStep,
} from '@adam/types';
import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';

import { getItem, removeItem, setItem } from '../lib/native/preferences';

/**
 * Setup wizard state.
 *
 * Persisted through the native abstraction (Capacitor Preferences on device,
 * localStorage in a browser) because "resume where you left off after a
 * force-close" is an explicit requirement — Next's history stack cannot provide
 * it. The persisted blob is validated against the Zod schema on rehydrate, so a
 * shape change between app versions resets the wizard instead of crashing it.
 */
const STORAGE_KEY = 'adam.setup.v1';

interface SetupActions {
  goTo: (step: SetupStep) => void;
  complete: (step: SetupStep) => void;
  setSignedIn: (signedIn: boolean) => void;
  selectDevice: (serial: string, isFounderEdition: boolean, founderNumber: number | null) => void;
  selectSsid: (ssid: string) => void;
  setDeviceName: (name: string) => void;
  setAiBrainMode: (mode: AiBrainMode) => void;
  setCameraPermission: (granted: boolean) => void;
  setUserNameForFace: (name: string) => void;
  finish: () => void;
  reset: () => void;
}

export type SetupStore = SetupState & SetupActions;

export const useSetupStore = create<SetupStore>()(
  persist(
    (set) => ({
      ...INITIAL_SETUP_STATE,

      goTo: (step) => set({ currentStep: step }),

      complete: (step) =>
        set((state) => ({
          completedSteps: state.completedSteps.includes(step)
            ? state.completedSteps
            : [...state.completedSteps, step],
        })),

      setSignedIn: (signedIn) => set({ signedIn }),

      selectDevice: (serial, isFounderEdition, founderNumber) =>
        set({ selectedSerial: serial, isFounderEdition, founderNumber }),

      selectSsid: (ssid) => set({ selectedSsid: ssid }),
      setDeviceName: (deviceName) => set({ deviceName }),
      setAiBrainMode: (aiBrainMode) => set({ aiBrainMode }),
      setCameraPermission: (cameraPermissionGranted) => set({ cameraPermissionGranted }),
      setUserNameForFace: (userNameForFace) => set({ userNameForFace }),

      finish: () => set({ completedAt: new Date().toISOString() }),

      reset: () => set({ ...INITIAL_SETUP_STATE }),
    }),
    {
      name: STORAGE_KEY,
      storage: createJSONStorage(() => ({
        getItem: (name) => getItem(name),
        setItem: (name, value) => setItem(name, value),
        removeItem: (name) => removeItem(name),
      })),
      /** Actions are recreated on load; only the data half is written. */
      partialize: (state) => SetupState.parse(state satisfies SetupState),
      merge: (persisted, current) => {
        const parsed = SetupState.safeParse(persisted);
        return parsed.success ? { ...current, ...parsed.data } : current;
      },
    },
  ),
);

/** 1-based position within the six progress-bearing steps, or null for the rest. */
export function progressPosition(step: SetupStep): { current: number; total: number } | null {
  const index = SETUP_PROGRESS_STEPS.indexOf(step as (typeof SETUP_PROGRESS_STEPS)[number]);
  if (index === -1) return null;
  return { current: index + 1, total: SETUP_PROGRESS_STEPS.length };
}
