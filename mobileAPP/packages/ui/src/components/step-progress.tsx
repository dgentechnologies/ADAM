'use client';

import { Check } from 'lucide-react';

import { cn } from '../lib/cn';

export interface StepProgressProps {
  /** 1-based. */
  current: number;
  total: number;
  className?: string;
}

/**
 * Wizard progress: a Michroma eyebrow ("STEP 2 OF 6") over a segmented hairline
 * track. Segments rather than a continuous bar so the remaining count is
 * readable at a glance without a percentage.
 */
export function StepProgress({ current, total, className }: StepProgressProps) {
  return (
    <div className={cn('flex flex-col gap-1.5', className)}>
      <p className="font-display text-label-sm font-bold tracking-widest uppercase text-fg">
        Step {current} of {total}
      </p>
      <div
        className="flex gap-1.5"
        role="progressbar"
        aria-valuemin={1}
        aria-valuemax={total}
        aria-valuenow={current}
        aria-label={`Setup step ${current} of ${total}`}
      >
        {Array.from({ length: total }, (_, index) => (
          <span
            key={index}
            className={cn(
              'h-1 flex-1 rounded-full transition-colors duration-base ease-standard',
              index < current ? 'bg-fg shadow-[0_0_8px_rgba(255,255,255,0.4)]' : 'bg-white/20',
            )}
          />
        ))}
      </div>
    </div>
  );
}

export type ChecklistState = 'pending' | 'active' | 'done' | 'failed';

export interface ChecklistItem {
  id: string;
  label: string;
  state: ChecklistState;
}

/**
 * The three-step handoff checklist on the "connecting" screen.
 *
 * Marker shapes follow the Stitch screen exactly: done is a filled white disc
 * with a black check, active is a white ring over a grey fill, pending is a
 * hollow grey ring. A failed step is an outlined ring carrying `!` — there is no
 * red available, so failure has to be legible from shape plus the copy beneath.
 */
export function StepChecklist({
  items,
  className,
}: {
  items: readonly ChecklistItem[];
  className?: string;
}) {
  return (
    <ol className={cn('flex flex-col gap-stack-md', className)}>
      {items.map((item) => (
        <li
          key={item.id}
          className={cn(
            'flex items-center gap-gutter transition-opacity duration-base ease-standard',
            item.state === 'pending' ? 'opacity-45' : 'opacity-100',
          )}
        >
          <span
            aria-hidden
            className={cn(
              'flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-label-xs',
              item.state === 'done' && 'bg-fg text-fg-inverse',
              item.state === 'active' && 'animate-breathe border-2 border-fg bg-grey-mid',
              item.state === 'pending' && 'border border-border-strong bg-transparent',
              item.state === 'failed' && 'border border-fg bg-transparent text-fg',
            )}
          >
            {item.state === 'done' ? (
              <Check className="h-3.5 w-3.5" strokeWidth={3} />
            ) : item.state === 'failed' ? (
              '!'
            ) : null}
          </span>
          <span
            className={cn(
              'text-body-md',
              item.state === 'pending' ? 'text-fg-muted' : 'text-fg',
            )}
          >
            {item.label}
          </span>
        </li>
      ))}
    </ol>
  );
}
