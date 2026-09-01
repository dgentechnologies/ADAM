'use client';

import type { ReactNode } from 'react';

import { TabBar } from '@/components/tab-bar';

/**
 * The signed-in shell. The route group adds no path segment, so `/home`,
 * `/gallery`, `/smart-home`, `/memory` and `/settings/*` all live here.
 *
 * Only the tab bar is shared: each screen owns its own app bar because the title,
 * the back affordance and the trailing action differ per screen, and hoisting them
 * here would mean a lookup table keyed by pathname — the thing that made the
 * Stitch chrome inconsistent in the first place.
 */
export default function AppLayout({ children }: { children: ReactNode }) {
  return (
    <>
      {children}
      <TabBar />
    </>
  );
}
