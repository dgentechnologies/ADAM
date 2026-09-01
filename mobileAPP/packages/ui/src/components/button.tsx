'use client';

import { cva, type VariantProps } from 'class-variance-authority';
import { forwardRef, type ButtonHTMLAttributes } from 'react';

import { cn } from '../lib/cn';

/**
 * Pill button. Three intents only — filled white, hairline outline, bare text —
 * because the system has no accent colour to spend on a fourth.
 *
 * Destructive actions reuse `outline` and carry their meaning in the label and a
 * warning glyph; the theme deliberately has no `error` token.
 */
const button = cva(  [
    'relative inline-flex select-none items-center justify-center gap-unit',
    'rounded-full font-sans transition-all duration-fast ease-standard',
    'active:scale-[0.97] disabled:pointer-events-none disabled:opacity-40',
  ],
  {
    variants: {
      variant: {
        primary: 'bg-fg text-fg-inverse text-black bg-white hover:opacity-90',
        outline: 'border border-border-strong text-fg hover:bg-surface-pressed',
        ghost: 'text-fg-muted hover:text-fg',
      },
      size: {
        lg: 'h-14 px-8 text-body-lg',
        md: 'h-12 px-6 text-body-md',
        sm: 'h-10 px-5 text-label-md',
      },
      block: {
        true: 'w-full',
        false: '',
      },
    },
    defaultVariants: { variant: 'primary', size: 'lg', block: false },
  },
);

/**
 * Exported so a router-aware `<Link>` can wear the exact same pill without the
 * library depending on Next. `apps/web`'s `LinkButton` is the only consumer.
 */
export const buttonVariants = button;

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof button> {}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { className, variant, size, block, type = 'button', ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cn(button({ variant, size, block }), className)}
      {...props}
    />
  );
});

/** Circular icon-only control — the mic / theme / shutter row on Home. */
const iconButton = cva(
  'inline-flex items-center justify-center rounded-full transition-all duration-fast ease-standard active:scale-95 disabled:pointer-events-none disabled:opacity-40',
  {
    variants: {
      variant: {
        primary: 'bg-fg text-fg-inverse hover:opacity-90',
        outline: 'border border-grey-light text-fg hover:bg-surface-pressed',
        ghost: 'text-fg-muted hover:text-fg',
      },
      size: {
        lg: 'h-14 w-14',
        md: 'h-11 w-11',
        sm: 'h-8 w-8',
      },
    },
    defaultVariants: { variant: 'outline', size: 'lg' },
  },
);

export interface IconButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof iconButton> {
  /** Icon-only controls have no text, so a label is mandatory. */
  'aria-label': string;
}

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(function IconButton(
  { className, variant, size, type = 'button', ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cn(iconButton({ variant, size }), className)}
      {...props}
    />
  );
});
