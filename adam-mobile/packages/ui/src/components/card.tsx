'use client';

import { cva, type VariantProps } from 'class-variance-authority';
import type { HTMLAttributes } from 'react';

import { cn } from '../lib/cn';

/**
 * 24px-radius charcoal card on a 1px hairline — the single container shape in
 * the system. `texture` layers the Digital Skin behind the content, which is why
 * the card is always `relative` and the overlay always `pointer-events-none`.
 */
const card = cva('relative overflow-hidden rounded-card border border-border', {
  variants: {
    surface: {
      raised: 'bg-surface-raised',
      recessed: 'bg-surface',
      flat: 'bg-transparent',
    },
    padding: {
      none: '',
      md: 'p-stack-md',
      lg: 'p-stack-lg',
    },
    interactive: {
      true: 'transition-transform duration-fast ease-standard active:scale-[0.98]',
      false: '',
    },
  },
  defaultVariants: { surface: 'raised', padding: 'md', interactive: false },
});

export interface CardProps extends HTMLAttributes<HTMLDivElement>, VariantProps<typeof card> {
  texture?: boolean;
}

export function Card({
  className,
  surface,
  padding,
  interactive,
  texture = false,
  children,
  ...props
}: CardProps) {
  return (
    <div className={cn(card({ surface, padding, interactive }), className)} {...props}>
      {texture ? (
        <div className="digital-skin pointer-events-none absolute inset-0" aria-hidden />
      ) : null}
      <div className="relative">{children}</div>
    </div>
  );
}

/**
 * Grouped list container. Rows are separated by hairlines rather than gaps, the
 * iOS-settings pattern both specs describe.
 */
export function CardGroup({ className, children, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        'overflow-hidden rounded-card border border-border bg-surface-raised',
        '[&>*+*]:hairline-t',
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
}
