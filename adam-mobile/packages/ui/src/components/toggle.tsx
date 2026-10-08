'use client';

import { useId } from 'react';

import { cn } from '../lib/cn';

export interface ToggleProps {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  label: string;
  /** Hides the visible label but keeps it for assistive tech. */
  hideLabel?: boolean;
  description?: string;
  disabled?: boolean;
  className?: string;
}

/**
 * Switch built on a real checkbox so it is keyboard- and AT-native. The track is
 * mid-grey when off and white when on — the only two states the palette allows.
 */
export function Toggle({
  checked,
  onCheckedChange,
  label,
  hideLabel = false,
  description,
  disabled = false,
  className,
}: ToggleProps) {
  const id = useId();

  return (
    <div className={cn('flex items-center gap-gutter', className)}>
      <label
        htmlFor={id}
        className={cn('flex min-w-0 flex-1 flex-col', hideLabel && 'sr-only')}
      >
        <span className="text-body-md text-fg">{label}</span>
        {description ? <span className="text-label-md text-fg-muted">{description}</span> : null}
      </label>

      <span className="relative inline-flex shrink-0">
        <input
          id={id}
          type="checkbox"
          role="switch"
          checked={checked}
          disabled={disabled}
          onChange={(event) => onCheckedChange(event.target.checked)}
          className="peer h-8 w-14 cursor-pointer appearance-none rounded-full bg-grey-mid transition-colors duration-base ease-standard checked:bg-fg disabled:opacity-40"
        />
        <span
          aria-hidden
          className={cn(
            'pointer-events-none absolute left-1 top-1 h-6 w-6 rounded-full',
            'bg-fg transition-all duration-base ease-standard',
            'peer-checked:left-7 peer-checked:bg-fg-inverse',
          )}
        />
      </span>
    </div>
  );
}
