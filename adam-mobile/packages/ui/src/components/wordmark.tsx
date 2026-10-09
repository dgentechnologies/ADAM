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
      <img
        src="/assets/adam-wordmark.png"
        alt="ADAM"
        width={420}
        height={105}
        className={cn(
          'object-contain [html[data-theme=light]_&]:invert',
          size === 'sm' && 'h-auto w-28',
          size === 'md' && 'h-auto w-40',
          size === 'lg' && 'h-auto w-52',
        )}
      />
      {byline ? (
        <p className="text-label-sm uppercase text-fg-subtle">by DGEN Technologies</p>
      ) : null}
    </div>
  );
}
