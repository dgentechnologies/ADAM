'use client';

import { Button, Wordmark } from '@adam/ui';
import { useEffect, type ReactNode } from 'react';
import { useSetupLoading, useSetupStore } from '@/stores/setup-store';

let pending: Promise<void> | undefined;
function hydrate() {
  if (!pending) pending = Promise.resolve(useSetupStore.persist.rehydrate()).finally(() => { pending = undefined; });
  return pending;
}

/** Hydrate once before any screen can overwrite the saved wizard position. */
export function SetupLoading({ children }: { children: ReactNode }) {
  const status = useSetupLoading((state) => state.status);
  useEffect(() => { void hydrate(); }, []);
  if (status === 'ready') return children;
  return (
    <main className="flex min-h-dvh flex-col items-center justify-center gap-8 px-6 text-center">
      <Wordmark size="md" byline />
      {status === 'error' && (
        <div className="max-w-sm space-y-4" role="alert">
          <p className="text-fg-muted text-sm">Your saved setup could not be loaded. Your data has been kept. Please try again.</p>
          <Button block onClick={() => void hydrate()}>Try again</Button>
        </div>
      )}
    </main>
  );
}
