'use client';

import { cn } from '../lib/cn';

export type DeviceStatusKind = 'online' | 'offline' | 'thinking' | 'updating';

/**
 * Status is carried by fill and motion, never hue (DESIGN.md): filled + bloom is
 * online, hollow is offline, breathing is thinking, sweeping is updating.
 */
const DOT: Record<DeviceStatusKind, string> = {
  online: 'bg-fg bloom',
  offline: 'border border-fg-muted bg-transparent',
  thinking: 'bg-fg animate-breathe',
  updating: 'bg-fg animate-pulse',
};

const LABEL: Record<DeviceStatusKind, string> = {
  online: 'Online',
  offline: 'Offline',
  thinking: 'Thinking',
  updating: 'Updating',
};

export interface StatusDotProps {
  status: DeviceStatusKind;
  /** Renders the uppercase label beside the dot. */
  withLabel?: boolean;
  label?: string;
  className?: string;
}

export function StatusDot({ status, withLabel = false, label, className }: StatusDotProps) {
  const text = label ?? LABEL[status];

  if (!withLabel) {
    return (
      <span
        className={cn('block h-2.5 w-2.5 rounded-full', DOT[status], className)}
        role="img"
        aria-label={text}
      />
    );
  }

  return (
    <span className={cn('inline-flex items-center gap-unit', className)}>
      <span className={cn('block h-2.5 w-2.5 rounded-full', DOT[status])} aria-hidden />
      <span className="text-label-sm uppercase text-fg">{text}</span>
    </span>
  );
}
