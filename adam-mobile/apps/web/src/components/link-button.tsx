'use client';

import { buttonVariants, cn } from '@adam/ui';
import Link from 'next/link';
import type { ComponentProps } from 'react';

type Variants = Parameters<typeof buttonVariants>[0];

/**
 * A `next/link` that renders as an ADAM pill button.
 *
 * The alternative — a `<Button>` with an `asChild` escape hatch — would pull a
 * Slot implementation into @adam/ui and let any element inherit button styling.
 * Sharing the cva recipe keeps one source of truth for the pill while leaving
 * routing in the app.
 */
export function LinkButton({
  href,
  variant,
  size,
  block,
  className,
  children,
  ...props
}: ComponentProps<typeof Link> & Variants) {
  return (
    <Link href={href} className={cn(buttonVariants({ variant, size, block }), className)} {...props}>
      {children}
    </Link>
  );
}
