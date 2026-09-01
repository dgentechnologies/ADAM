'use client';

import { StepProgress, cn } from '@adam/ui';
import { ChevronLeft } from 'lucide-react';
import { usePathname, useRouter } from 'next/navigation';
import { useEffect, type ReactNode } from 'react';

import { stepFromPathname } from '../lib/setup-flow';
import { progressPosition, useSetupStore } from '../stores/setup-store';
import { SetupTransition } from './setup-transition';

/**
 * Wizard shell for the `(setup)` route group.
 *
 * Two jobs: render the step eyebrow + progress track for the six steps that carry
 * one, and keep `currentStep` in the persisted store in sync with the route so a
 * force-close resumes here. Screens that are full-bleed (splash, discover,
 * face-capture) opt out of the chrome via the route list below.
 */
const NO_CHROME = new Set(['splash', 'discover', 'connecting', 'founder-reveal', 'face-capture']);
const NO_BACK = new Set(['splash', 'welcome', 'connecting', 'founder-reveal']);

export function SetupShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const goTo = useSetupStore((state) => state.goTo);

  const step = stepFromPathname(pathname);

  useEffect(() => {
    if (step) goTo(step);
  }, [step, goTo]);

  const bare = !step || NO_CHROME.has(step);
  const progress = step ? progressPosition(step) : null;

  return (
    <div className="relative flex h-dvh max-h-dvh flex-col overflow-hidden">
      <div className="digital-skin pointer-events-none fixed inset-0" aria-hidden />

      {bare ? null : (
        <div className="relative z-10 flex shrink-0 flex-col gap-stack-sm px-container pt-safe">
          <div className={cn('flex items-center', NO_BACK.has(step) && 'invisible')}>
            <button
              type="button"
              aria-label="Back"
              onClick={() => router.back()}
              className="-ml-2 flex h-10 w-10 items-center justify-center text-fg"
            >
              <ChevronLeft className="h-6 w-6" />
            </button>
          </div>
          {progress ? <StepProgress current={progress.current} total={progress.total} /> : null}
        </div>
      )}

      <div className="relative z-10 flex min-h-0 flex-1 flex-col">
        <SetupTransition>{children}</SetupTransition>
      </div>
    </div>
  );
}
