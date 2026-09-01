import type { ReactNode } from 'react';

import { SetupShell } from '@/components/setup-shell';

/**
 * `(setup)` is a route group, so it adds no path segment: `/welcome`, `/discover`
 * and friends stay top-level, which keeps the persisted step-to-route mapping a
 * plain string.
 *
 * No tab bar here — the Stitch export put one on several setup screens, but both
 * specs require the wizard to be modal.
 */
export default function SetupLayout({ children }: { children: ReactNode }) {
  return <SetupShell>{children}</SetupShell>;
}
