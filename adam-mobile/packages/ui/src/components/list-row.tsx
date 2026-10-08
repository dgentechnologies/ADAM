'use client';

import { ChevronRight } from 'lucide-react';
import type { ReactNode } from 'react';

import { cn } from '../lib/cn';

export interface ListRowProps {
  icon?: ReactNode;
  title: ReactNode;
  subtitle?: ReactNode;
  /** Right-hand slot: a value, a toggle, a badge. Suppresses the chevron. */
  trailing?: ReactNode;
  chevron?: boolean;
  onClick?: () => void;
  /** Renders as a link when provided — used with Next's `Link asChild` pattern. */
  as?: 'div' | 'button';
  className?: string;
  disabled?: boolean;
}

/**
 * One row of a grouped list. Deliberately unaware of routing: `apps/web` wraps it
 * in a `<Link>` where navigation is needed, so the library stays framework-free.
 */
export function ListRow({
  icon,
  title,
  subtitle,
  trailing,
  chevron = false,
  onClick,
  as,
  className,
  disabled = false,
}: ListRowProps) {
  const Tag = as ?? (onClick ? 'button' : 'div');

  return (
    <Tag
      {...(Tag === 'button' ? { type: 'button' as const, disabled } : {})}
      onClick={onClick}
      className={cn(
        'flex w-full items-center gap-gutter px-stack-md py-gutter text-left',
        'transition-colors duration-fast ease-standard',
        (onClick || chevron) && 'active:bg-surface-pressed',
        disabled && 'opacity-40',
        className,
      )}
    >
      {icon ? (
        <span className="flex h-10 w-10 shrink-0 items-center justify-center text-fg">{icon}</span>
      ) : null}

      <span className="flex min-w-0 flex-1 flex-col">
        <span className="truncate text-body-md text-fg">{title}</span>
        {subtitle ? (
          <span className="truncate text-label-md text-fg-muted">{subtitle}</span>
        ) : null}
      </span>

      {trailing ?? (chevron ? <ChevronRight className="h-5 w-5 shrink-0 text-fg-subtle" /> : null)}
    </Tag>
  );
}
