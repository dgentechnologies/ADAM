import { create } from 'zustand';

// Transient navigation intent; never persisted into a user's backup or account.
export const useMemoryIntent = create<{
  requested: boolean;
  request: () => void;
  clear: () => void;
}>((set) => ({
  requested: false,
  request: () => set({ requested: true }),
  clear: () => set({ requested: false }),
}));
