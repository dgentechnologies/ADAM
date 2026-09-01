'use client';

import type { DeviceStatus, FaceExpression as DeviceExpression } from '@adam/types';
import {
  AdamFaceMark,
  Card,
  Screen,
  StatusDot,
  type DeviceStatusKind,
  type FaceExpression,
} from '@adam/ui';
import { useQuery } from '@tanstack/react-query';
import { Brain, Images, LampCeiling, Mic, MicOff, Moon, Settings, Zap } from 'lucide-react';
import Link from 'next/link';
import { useState } from 'react';

import { AppBar } from '@/components/app-bar';
import { fetchBalance, fetchDevice, queryKeys } from '@/lib/mock/api';
import { useAppStore } from '@/stores/app-store';
import { useSetupStore } from '@/stores/setup-store';

/**
 * `dashboard` — the screen the app opens to after setup.
 *
 * The Stitch screenshot for this route is one of the three 28-byte
 * `<FIFE Image failed to fetch>` stubs, so the layout comes from `code.html`:
 * hero card, status line, a row of instant controls, then a 2×2 grid of the
 * secondary destinations.
 */

/**
 * The unit reports seven expressions; the face mark draws five. `speaking` maps to
 * `happy` (mouth-analogue open eyes) and `annoyed` falls back to `idle` rather
 * than inventing a sixth drawing.
 */
const EXPRESSION: Record<DeviceExpression, FaceExpression> = {
  idle: 'idle',
  happy: 'happy',
  listening: 'listening',
  thinking: 'thinking',
  speaking: 'happy',
  sleeping: 'asleep',
  annoyed: 'idle',
};

const STATUS: Record<DeviceStatus, DeviceStatusKind> = {
  online: 'online',
  offline: 'offline',
  updating: 'updating',
  sleeping: 'offline',
};

const DESTINATIONS = [
  { href: '/gallery', label: 'Moments', hint: 'What he saw', Icon: Images },
  { href: '/smart-home', label: 'Smart Home', hint: 'Lights & scenes', Icon: LampCeiling },
  { href: '/memory', label: 'Memory', hint: 'People & facts', Icon: Brain },
  { href: '/settings', label: 'Settings', hint: 'Device & account', Icon: Settings },
] as const;

export default function HomePage() {
  const { data: device } = useQuery({ queryKey: queryKeys.device, queryFn: fetchDevice });
  const { data: balance } = useQuery({ queryKey: queryKeys.balance, queryFn: fetchBalance });

  const muted = useAppStore((state) => state.muted);
  const toggleMuted = useAppStore((state) => state.toggleMuted);
  const [asleep, setAsleep] = useState(false);

  /**
   * The brain mode chosen during setup wins over the one the mock device reports:
   * the unit is the source of truth in production, but in a mock/demo build the
   * fixture is fixed, and showing "Own key" to someone who just bought credits
   * makes the walkthrough look broken.
   */
  const chosenMode = useSetupStore((state) => state.aiBrainMode);
  const brainMode = chosenMode ?? device?.aiBrainMode ?? null;

  const status: DeviceStatusKind = asleep ? 'offline' : STATUS[device?.status ?? 'offline'];
  const expression: FaceExpression = asleep
    ? 'asleep'
    : EXPRESSION[device?.expression ?? 'idle'];

  return (
    <>
      <AppBar
        title={device?.name ?? 'ADAM'}
        action={<StatusDot status={status} withLabel />}
      />

      <Screen chrome="both" texture>
        <div className="flex flex-col gap-stack-lg">
          <Card padding="lg" texture className="flex flex-col items-center gap-stack-md">
            <AdamFaceMark expression={expression} size="xl" className="animate-float" />
            <div className="flex flex-col items-center gap-unit text-center">
              <p className="text-title-md text-fg">
                {asleep
                  ? 'Sleeping.'
                  : muted
                    ? 'Mic muted.'
                    : status === 'online'
                      ? 'Listening for “Hey ADAM”.'
                      : 'Can’t reach him right now.'}
              </p>
              <p className="text-label-md text-fg-muted">
                {device?.wifiSsid ? `On ${device.wifiSsid}` : 'Not connected to Wi-Fi'}
                {device ? ` · v${device.firmwareVersion}` : ''}
              </p>
            </div>
          </Card>

          <div className="grid grid-cols-3 gap-stack-sm">
            <ControlTile
              label={muted ? 'Unmute' : 'Mute mic'}
              Icon={muted ? MicOff : Mic}
              active={muted}
              onClick={() => toggleMuted()}
            />
            <ControlTile
              label={asleep ? 'Wake' : 'Sleep'}
              Icon={Moon}
              active={asleep}
              onClick={() => setAsleep((value) => !value)}
            />
            <ControlTile
              label="Credits"
              Icon={Zap}
              value={
                brainMode === 'managed'
                  ? `${Math.round(balance?.remainingMinutes ?? 0)}m`
                  : brainMode === 'byok'
                    ? 'Own key'
                    : 'Lite'
              }
            />
          </div>

          <div className="grid grid-cols-2 gap-stack-sm">
            {DESTINATIONS.map(({ href, label, hint, Icon }) => (
              <Link
                key={href}
                href={href}
                className="flex flex-col gap-stack-sm rounded-card border border-border bg-surface-raised p-stack-md transition-transform duration-fast ease-standard active:scale-[0.98]"
              >
                <Icon className="h-6 w-6 text-fg" strokeWidth={1.5} aria-hidden />
                <span className="flex flex-col">
                  <span className="text-body-md text-fg">{label}</span>
                  <span className="text-label-md text-fg-muted">{hint}</span>
                </span>
              </Link>
            ))}
          </div>
        </div>
      </Screen>
    </>
  );
}

/**
 * A single instant control. Rendered as a button only when it does something —
 * the credits tile is a readout, so it must not look pressable.
 */
function ControlTile({
  label,
  Icon,
  active = false,
  value,
  onClick,
}: {
  label: string;
  Icon: typeof Mic;
  active?: boolean;
  value?: string;
  onClick?: () => void;
}) {
  const content = (
    <>
      <Icon className="h-5 w-5" strokeWidth={1.5} aria-hidden />
      <span className="text-label-md">{value ?? label}</span>
    </>
  );

  const shared =
    'flex h-24 flex-col items-center justify-center gap-stack-sm rounded-card border p-stack-sm text-center';

  if (!onClick) {
    return (
      <div className={`${shared} border-border bg-surface text-fg-muted`}>
        {content}
        <span className="sr-only">{label}</span>
      </div>
    );
  }

  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`${shared} transition-colors duration-fast ease-standard active:scale-[0.98] ${
        active ? 'border-fg bg-fg text-fg-inverse' : 'border-border bg-surface-raised text-fg'
      }`}
    >
      {content}
    </button>
  );
}
