'use client';

import type { ReactNode } from 'react';

import { cn } from '../lib/cn';

/**
 * Page shell. Owns the 24px gutter, the safe-area padding, and the ambient
 * texture so no screen re-derives them. `chrome` reserves room for the fixed app
 * bar and tab bar that `apps/web` renders.
 */
export function Screen({
  children,
  className,
  chrome = 'none',
  center = false,
  texture = false,
}: {
  children: ReactNode;
  className?: string;
  chrome?: 'none' | 'top' | 'both';
  center?: boolean;
  texture?: boolean;
}) {
  return (
    <main
      className={cn(
        'relative flex min-h-dvh w-full flex-col px-container',
        chrome === 'none' && 'pb-stack-lg pt-safe',
        chrome === 'top' && 'pb-stack-lg pt-[calc(theme(spacing.appbar-h)+env(safe-area-inset-top,0px))]',
        chrome === 'both' &&
          'pt-[calc(theme(spacing.appbar-h)+env(safe-area-inset-top,0px))] pb-[calc(theme(spacing.tabbar-h)+env(safe-area-inset-bottom,0px)+theme(spacing.stack-md))]',
        center && 'justify-center',
        className,
      )}
    >
      {texture ? (
        <div className="digital-skin pointer-events-none fixed inset-0" aria-hidden />
      ) : null}
      <div className="relative flex w-full flex-1 min-h-0 flex-col justify-between">{children}</div>
    </main>
  );
}

/**
 * Headline block. `eyebrow` is the only place Michroma appears outside the
 * wordmark, per the confirmed type split.
 */
export function ScreenHeader({
  eyebrow,
  title,
  subtitle,
  size = 'md',
  align = 'start',
  className,
}: {
  eyebrow?: ReactNode;
  title: ReactNode;
  subtitle?: ReactNode;
  /**
   * `xs` exists for the two Stitch screens whose "headline" is really a
   * body-weight caption over a list (`choose_adam_s_brain`, `connect_your_key`).
   */
  size?: 'lg' | 'md' | 'sm' | 'xs';
  align?: 'start' | 'center';
  className?: string;
}) {
  return (
    <header
      className={cn(
        'flex flex-col gap-stack-sm',
        align === 'center' && 'items-center text-center',
        className,
      )}
    >
      {eyebrow ? (
        <p className="font-display text-label-xs uppercase text-fg-subtle">{eyebrow}</p>
      ) : null}
      <h1
        className={cn(
          'text-fg',
          size === 'lg' && 'text-display-lg',
          size === 'md' && 'text-headline-md',
          size === 'sm' && 'text-headline-sm',
          size === 'xs' && 'text-title-md',
        )}
      >
        {title}
      </h1>
      {subtitle ? (
        <p
          className={cn(
            'max-w-md text-fg-muted',
            size === 'xs' ? 'text-body-md' : 'text-body-lg',
          )}
        >
          {subtitle}
        </p>
      ) : null}
    </header>
  );
}

/**
 * Bottom action rail. Setup screens pin their primary action here so the CTA sits
 * in the same place on every step regardless of content height.
 */
export function ScreenActions({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn('mt-auto flex flex-col gap-stack-sm pt-stack-lg', className)}>
      {children}
    </div>
  );
}
