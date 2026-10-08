'use client';

import { Wordmark } from '@adam/ui';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';

import { useSetupStore } from '@/stores/setup-store';

/**
 * Entry point. Static export means no server redirect is available, so the
 * decision is made on the client: a finished setup goes to Home, an interrupted
 * one resumes at its persisted step, and a fresh install starts at the splash.
 *
 * The routing decision waits for the persist middleware to finish rehydrating.
 * Capacitor Preferences is an async store, so on device the first render always
 * sees `INITIAL_SETUP_STATE` — deciding then would send a user who is mid-flow
 * (or already finished) back to `/welcome` on every cold start.
 *
 * The markup below is the splash itself rather than a blank screen, so the
 * hand-off is invisible instead of a flash of nothing.
 */
export default function RootPage() {
  const router = useRouter();
  const completedAt = useSetupStore((state) => state.completedAt);
  const currentStep = useSetupStore((state) => state.currentStep);
  const [hydrated, setHydrated] = useState(() => useSetupStore.persist.hasHydrated());

  useEffect(() => {
    if (hydrated) return;
    return useSetupStore.persist.onFinishHydration(() => setHydrated(true));
  }, [hydrated]);

  useEffect(() => {
    if (!hydrated) return;

    // One frame of the wordmark before moving on — the "waking up" beat.
    const timer = setTimeout(() => {
      if (completedAt) {
        router.replace('/home');
        return;
      }
      router.replace(currentStep === 'splash' ? '/welcome' : `/${currentStep}`);
    }, 1200);

    return () => clearTimeout(timer);
  }, [hydrated, completedAt, currentStep, router]);

  return (
    <main className="relative flex min-h-dvh flex-col items-center justify-center">
      <div className="digital-skin pointer-events-none fixed inset-0" aria-hidden />
      <Wordmark size="md" byline />
    </main>
  );
}
