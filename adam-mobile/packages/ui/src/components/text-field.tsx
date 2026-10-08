'use client';

import { forwardRef, useId, type InputHTMLAttributes, type ReactNode } from 'react';

import { cn } from '../lib/cn';

export interface TextFieldProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'size'> {
  label?: string;
  /** Shown under the field. Errors replace it rather than stacking. */
  hint?: string;
  error?: string;
  /** Trailing control — the password reveal eye, a Paste button, a unit. */
  trailing?: ReactNode;
  /**
   * `outline` is the default bordered control. `underline` is the borderless rule
   * used by the Wi-Fi password entry, where the placeholder carries the label.
   * `filled` is the grey pill on the face-capture name field.
   */
  variant?: 'outline' | 'underline' | 'filled';
}

export const TextField = forwardRef<HTMLInputElement, TextFieldProps>(function TextField(
  { label, hint, error, trailing, variant = 'outline', className, id, ...props },
  ref,
) {
  const autoId = useId();
  const inputId = id ?? autoId;
  const describedBy = error || hint ? `${inputId}-desc` : undefined;

  return (
    <div className="flex w-full flex-col gap-unit">
      {label ? (
        <label
          htmlFor={inputId}
          className="font-display text-label-xs uppercase text-fg-subtle"
        >
          {label}
        </label>
      ) : null}

      <div
        className={cn(
          'flex items-center gap-stack-sm',
          'transition-colors duration-fast ease-standard',
          variant === 'outline' && [
            'rounded-control border bg-surface px-5 focus-within:border-fg',
            error ? 'border-fg' : 'border-border',
          ],
          variant === 'filled' && 'rounded-full bg-surface-pressed px-6',
          variant === 'underline' && [
            'border-b bg-transparent focus-within:border-fg',
            error ? 'border-fg' : 'border-border-strong',
          ],
          className,
        )}
      >
        <input
          ref={ref}
          id={inputId}
          aria-invalid={error ? true : undefined}
          aria-describedby={describedBy}
          className={cn(
            'min-w-0 flex-1 bg-transparent text-fg outline-none placeholder:text-fg-subtle',
            variant === 'underline' ? 'h-14 text-title-md' : 'h-14 text-body-lg',
          )}
          {...props}
        />
        {trailing}
      </div>

      {error || hint ? (
        <p
          id={describedBy}
          className={cn('text-label-md', error ? 'text-fg' : 'text-fg-muted')}
        >
          {error ?? hint}
        </p>
      ) : null}
    </div>
  );
});

/**
 * The oversized, centred, borderless field used by "What should we call him?"
 * and "What should ADAM call you?" — the input *is* the headline on those
 * screens, so it gets its own component rather than a variant.
 */
export const DisplayField = forwardRef<HTMLInputElement, TextFieldProps>(function DisplayField(
  { label, hint, error, className, id, ...props },
  ref,
) {
  const autoId = useId();
  const inputId = id ?? autoId;

  return (
    <div className="flex w-full flex-col items-center gap-stack-sm">
      {label ? (
        <label htmlFor={inputId} className="sr-only">
          {label}
        </label>
      ) : null}
      <input
        ref={ref}
        id={inputId}
        className={cn(
          'w-full border-b border-border bg-transparent pb-stack-sm text-center',
          'text-headline-sm text-fg caret-white outline-none',
          'transition-colors duration-fast ease-standard focus:border-fg',
          className,
        )}
        {...props}
      />
      {error || hint ? (
        <p className="text-label-md text-fg-muted">{error ?? hint}</p>
      ) : null}
    </div>
  );
});
