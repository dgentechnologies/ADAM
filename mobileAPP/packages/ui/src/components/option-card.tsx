'use client';

import { Check, ChevronRight } from 'lucide-react';
import type { ReactNode } from 'react';

import { cn } from '../lib/cn';

export interface OptionCardProps {
  title: ReactNode;
  description?: ReactNode;
  /** "Recommended" / "Most Popular" — a hairline pill, never a coloured flag. */
  badge?: string;
  /** Left-hand price or number, set in display type. */
  lead?: ReactNode;
  selected?: boolean;
  /** `check` marks a chosen option; `chevron` marks navigation to a sub-flow. */
  affordance?: 'check' | 'chevron' | 'none';
  /**
   * `inline` is the brain-choice row (title + description + affordance).
   * `stacked` is the credit-pack card: a large `lead` price with the tier name
   * beneath it, the badge top-right, and the check in the bottom-right corner.
   */
  layout?: 'inline' | 'stacked';
  onSelect?: () => void;
  disabled?: boolean;
  className?: string;
}

/**
 * The selectable card behind "Choose ADAM's brain" and "Choose a credit pack".
 * Selection is shown by a white hairline plus a filled check — with no accent
 * colour available, the border weight has to do the work.
 */
export function OptionCard({
  title,
  description,
  badge,
  lead,
  selected = false,
  affordance = 'check',
  layout = 'inline',
  onSelect,
  disabled = false,
  className,
}: OptionCardProps) {
  const check =
    affordance === 'chevron' ? (
      <ChevronRight className="h-5 w-5 shrink-0 text-fg-subtle" />
    ) : affordance === 'check' ? (
      <span
        className={cn(
          'flex h-6 w-6 shrink-0 items-center justify-center rounded-full border',
          selected ? 'border-fg bg-fg text-fg-inverse' : 'border-border-strong text-transparent',
        )}
        aria-hidden
      >
        <Check className="h-4 w-4" strokeWidth={2.5} />
      </span>
    ) : null;

  const shell = cn(
    'group relative flex w-full rounded-card border bg-surface-raised p-stack-md text-left',
    'transition-all duration-fast ease-standard active:scale-[0.98]',
    selected ? 'border-fg' : 'border-border',
    disabled && 'pointer-events-none opacity-40',
    className,
  );

  if (layout === 'stacked') {
    return (
      <button
        type="button"
        onClick={onSelect}
        disabled={disabled}
        aria-pressed={affordance === 'check' ? selected : undefined}
        className={cn(shell, 'flex-col gap-unit')}
      >
        <span className="flex w-full items-start justify-between gap-gutter">
          {lead ? <span className="text-headline-sm text-fg">{lead}</span> : null}
          {badge ? (
            <span className="mt-1 shrink-0 rounded-full border border-border-strong px-3 py-1 text-label-xs uppercase text-fg-muted">
              {badge}
            </span>
          ) : null}
        </span>

        <span className="text-label-sm uppercase text-fg">{title}</span>

        <span className="flex w-full items-end justify-between gap-gutter">
          <span className="text-label-md text-fg-muted">{description}</span>
          {check}
        </span>
      </button>
    );
  }

  return (
    <button
      type="button"
      onClick={onSelect}
      disabled={disabled}
      aria-pressed={affordance === 'check' ? selected : undefined}
      className={cn(shell, 'items-center gap-gutter')}
    >
      {lead ? <span className="shrink-0 text-title-md text-fg">{lead}</span> : null}

      <span className="flex min-w-0 flex-1 flex-col gap-1">
        <span className="flex flex-wrap items-center gap-stack-sm">
          <span className="text-body-lg text-fg">{title}</span>
          {badge ? (
            <span className="rounded-full border border-border-strong px-3 py-0.5 text-label-xs uppercase text-fg-muted">
              {badge}
            </span>
          ) : null}
        </span>
        {description ? (
          <span className="text-label-md text-fg-muted">{description}</span>
        ) : null}
      </span>

      {check}
    </button>
  );
}
