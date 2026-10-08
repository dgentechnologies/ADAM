'use client';

import { cn } from '../lib/cn';

/**
 * The ADAM wordmark. Michroma is scoped to this component, the eyebrow labels,
 * and the Founder Edition reveal — everything else is Inter, per the confirmed
 * type split.
 */
export function Wordmark({
  size = 'md',
  byline = false,
  className,
}: {
  size?: 'sm' | 'md' | 'lg';
  byline?: boolean;
  className?: string;
}) {
  return (
    <div className={cn('flex flex-col items-center gap-stack-sm', className)}>
      <p
        className={cn(
          'font-display uppercase text-fg',
          size === 'sm' && 'text-title-md tracking-[0.24em]',
          size === 'md' && 'text-headline-sm tracking-[0.28em]',
          size === 'lg' && 'text-headline-md tracking-[0.3em]',
        )}
      >
        ADAM
      </p>
      {byline ? (
        <p className="text-label-sm uppercase text-fg-subtle">by DGEN Technologies</p>
      ) : null}
    </div>
  );
}
