'use client';

import { cn } from '../lib/cn';

export interface SegmentedControlProps<T extends string> {
  options: ReadonlyArray<{ value: T; label: string }>;
  value: T;
  onChange: (value: T) => void;
  className?: string;
  'aria-label': string;
}

/**
 * Filter row for Moments (All / Starred / This Week) and Memory (People / Facts).
 * The Stitch export drew these as loose pills; unified here so both screens
 * share one selected treatment — filled white on the active segment.
 */
export function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
  className,
  'aria-label': ariaLabel,
}: SegmentedControlProps<T>) {
  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      className={cn('flex gap-stack-sm overflow-x-auto', className)}
    >
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => onChange(option.value)}
            className={cn(
              'shrink-0 rounded-full border px-5 py-2 text-label-md',
              'transition-colors duration-fast ease-standard active:scale-95',
              active
                ? 'border-fg bg-fg text-fg-inverse'
                : 'border-border text-fg-muted hover:text-fg',
            )}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
