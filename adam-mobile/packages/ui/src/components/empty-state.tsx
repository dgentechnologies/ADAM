'use client';

import type { LucideIcon } from 'lucide-react';
import type { ReactNode } from 'react';

import { cn } from '../lib/cn';
import { AdamFaceMark } from './adam-face-mark';

export interface EmptyStateProps {
  icon?: LucideIcon;
  title: string;
  description?: string;
  action?: ReactNode;
  className?: string;
}

export function EmptyState({ icon: Icon, title, description, action, className }: EmptyStateProps) {
  return (
    <div
      className={cn(
        'flex flex-1 flex-col items-center justify-center gap-stack-md py-stack-lg text-center',
        className,
      )}
    >
      {Icon ? (
        <Icon className="h-10 w-10 text-fg-faint" strokeWidth={1.5} aria-hidden />
      ) : (
        <AdamFaceMark expression="asleep" size="md" bloom={false} />
      )}
      <div className="flex flex-col gap-unit">
        <p className="text-title-md text-fg">{title}</p>
        {description ? (
          <p className="max-w-xs text-body-md text-fg-muted">{description}</p>
        ) : null}
      </div>
      {action}
    </div>
  );
}

/**
 * Honest stand-in for the seven routes the Stitch export never designed
 * (`smart-home` and the six Settings sub-pages). It states what will live here
 * rather than inventing UI, which was the agreed approach.
 */
export function NotYetDesigned({
  title,
  purpose,
  bullets,
  className,
}: {
  title: string;
  purpose: string;
  bullets?: readonly string[];
  className?: string;
}) {
  return (
    <div className={cn('flex flex-col gap-stack-md', className)}>
      <div className="relative overflow-hidden rounded-card border border-dashed border-border-strong bg-surface p-stack-md">
        <div className="digital-skin-coarse pointer-events-none absolute inset-0" aria-hidden />
        <div className="relative flex flex-col gap-stack-sm">
          <p className="font-display text-label-xs uppercase text-fg-subtle">Not yet designed</p>
          <p className="text-title-md text-fg">{title}</p>
          <p className="text-body-md text-fg-muted">{purpose}</p>
          {bullets?.length ? (
            <ul className="flex flex-col gap-unit pt-unit">
              {bullets.map((bullet) => (
                <li key={bullet} className="flex gap-stack-sm text-label-md text-fg-muted">
                  <span aria-hidden className="text-fg-faint">
                    —
                  </span>
                  <span>{bullet}</span>
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      </div>
    </div>
  );
}
