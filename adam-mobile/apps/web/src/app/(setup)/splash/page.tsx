'use client';

import { Wordmark } from '@adam/ui';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';

/**
 * `waking_up` — the boot beat. Reachable directly (and from a resumed session
 * whose persisted step is still `splash`), so it forwards to `/welcome` itself
 * rather than relying on the root route.
 *
 * The Stitch screen is the wordmark and byline alone — no face mark. The mark
 * arrives on `/welcome`, which is what makes the hand-off read as ADAM opening
 * his eyes rather than as two frames of the same logo.
 */
export default function SplashPage() {
  const router = useRouter();

  useEffect(() => {
    const timer = setTimeout(() => router.replace('/welcome'), 1400);
    return () => clearTimeout(timer);
  }, [router]);

  return (
    <div className="flex flex-1 flex-col items-center justify-center">
      <Wordmark size="md" byline />
    </div>
  );
}
