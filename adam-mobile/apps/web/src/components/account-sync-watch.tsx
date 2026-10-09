'use client';
import { useEffect } from 'react';

/** After the first explicit sync, refresh shared data without blocking navigation. */
export function AccountSyncWatch() {
  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const refresh = async (remote: boolean) => {
      if (stopped || !navigator.onLine || document.visibilityState === 'hidden') return;
      try {
        const { getCompanionSync } = await import('@/lib/firebase/companion-sync');
        const service = await getCompanionSync();
        const status = await service.status();
        if (!stopped && status.enabled && status.lastSynced && !status.syncing && (remote || status.pending)) await service.sync();
      } catch { /* The account panel exposes the retained error and pending data. */ }
    };
    const changes = () => { clearTimeout(timer); timer = setTimeout(() => void refresh(false), 1500); };
    const resume = () => { void refresh(true); };
    window.addEventListener('adam:data', changes);
    window.addEventListener('online', resume);
    document.addEventListener('visibilitychange', resume);
    const interval = setInterval(resume, 60000);
    resume();
    return () => { stopped = true; clearTimeout(timer); clearInterval(interval); window.removeEventListener('adam:data', changes); window.removeEventListener('online', resume); document.removeEventListener('visibilitychange', resume); };
  }, []);
  return null;
}
