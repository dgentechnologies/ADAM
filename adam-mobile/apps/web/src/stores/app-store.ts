'use client';

import { create } from 'zustand';
import { createJSONStorage, persist } from 'zustand/middleware';

import { getItem, removeItem, setItem } from '../lib/native/preferences';

export type Theme = 'dark' | 'light';

interface AppState {
  /** Dark is the product's identity; light is an explicit opt-in (DESIGN.md). */
  theme: Theme;
  muted: boolean;
  notifyOnUpdate: boolean;
}

interface AppActions {
  toggleTheme: () => void;
  setTheme: (theme: Theme) => void;
  toggleMuted: () => void;
  setNotifyOnUpdate: (value: boolean) => void;
}

export const useAppStore = create<AppState & AppActions>()(
  persist(
    (set) => ({
      theme: 'dark',
      muted: false,
      notifyOnUpdate: true,

      toggleTheme: () => set((state) => ({ theme: state.theme === 'dark' ? 'light' : 'dark' })),
      setTheme: (theme) => set({ theme }),
      toggleMuted: () => set((state) => ({ muted: !state.muted })),
      setNotifyOnUpdate: (notifyOnUpdate) => set({ notifyOnUpdate }),
    }),
    {
      name: 'adam.app.v1',
      storage: createJSONStorage(() => ({
        getItem: (name) => getItem(name),
        setItem: (name, value) => setItem(name, value),
        removeItem: (name) => removeItem(name),
      })),
      partialize: ({ theme, muted, notifyOnUpdate }) => ({ theme, muted, notifyOnUpdate }),
    },
  ),
);
