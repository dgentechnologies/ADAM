import type { StateStorage } from 'zustand/middleware';

/** Only setup state is recoverable here; memories and credentials are untouched. */
export function setupStorage(storage: StateStorage, timeoutMs = 8000): StateStorage {
  return {
    ...storage,
    async getItem(name) {
      let timer: ReturnType<typeof setTimeout> | undefined;
      try {
        const raw = await Promise.race([
          Promise.resolve().then(() => storage.getItem(name)),
          new Promise<never>((_, reject) => {
            timer = setTimeout(() => reject(new Error('Saved setup took too long to load.')), timeoutMs);
          }),
        ]);
        if (!raw) return null;
        try {
          JSON.parse(raw);
          return raw;
        } catch {
          // Preserve the damaged record before the wizard can save fresh state.
          await storage.setItem(`${name}.recovery.${Date.now()}`, raw);
          return null;
        }
      } finally {
        if (timer) clearTimeout(timer);
      }
    },
  };
}
