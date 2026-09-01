'use client';

import { cn } from '@adam/ui';
import { ChevronLeft } from 'lucide-react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import type { ReactNode } from 'react';

/**
 * Fixed frosted app bar.
 *
 * The Stitch export shipped four inconsistent variants of this — including two
 * screens titled "Credit Selection" that were not the credit screen, and a stray
 * person avatar on the memory screen. It is unified here: title on the left, one
 * optional action on the right, back arrow only where there is somewhere to go.
 */
export function AppBar({
  title,
  back,
  action,
  className,
}: {
  title: string;
  /** `true` uses history.back(); a string pushes that route. */
  back?: boolean | string;
  action?: ReactNode;
  className?: string;
}) {
  const router = useRouter();

  return (
    <header
      className={cn(
        'chrome-blur fixed inset-x-0 top-0 z-50 pt-safe',
        'shadow-[0_1px_8px_rgba(0,0,0,0.4)]',
        className,
      )}
    >
      <div className="flex h-appbar-h items-center gap-stack-sm px-container">
        {back ? (
          typeof back === 'string' ? (
            <Link
              href={back}
              aria-label="Back"
              className="-ml-2 flex h-10 w-10 items-center justify-center text-fg"
            >
              <ChevronLeft className="h-6 w-6" />
            </Link>
          ) : (
            <button
              type="button"
              aria-label="Back"
              onClick={() => router.back()}
              className="-ml-2 flex h-10 w-10 items-center justify-center text-fg"
            >
              <ChevronLeft className="h-6 w-6" />
            </button>
          )
        ) : null}

        <h1 className="min-w-0 flex-1 truncate text-title-md text-fg">{title}</h1>
        {action}
      </div>
    </header>
  );
}
