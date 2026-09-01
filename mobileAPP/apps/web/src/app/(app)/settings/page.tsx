'use client';

import { Button, CardGroup, ListRow, Screen, StatusDot, Toggle } from '@adam/ui';
import { useQuery } from '@tanstack/react-query';
import {
  Brain,
  Cpu,
  Info,
  Laptop,
  Mic,
  Moon,
  Sun,
  User,
  Wifi,
} from 'lucide-react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useState } from 'react';

import { AppBar } from '@/components/app-bar';
import { MOCK_USER } from '@/lib/mock/fixtures';
import { fetchDevice, queryKeys } from '@/lib/mock/api';
import { useAppStore } from '@/stores/app-store';
import { useSetupStore } from '@/stores/setup-store';

/**
 * `settings` — the hub. Eight rows plus the two in-place switches and Factory
 * Reset, per spec §3.4.
 *
 * Six of the eight destinations were never designed by Stitch and render an
 * honest `NotYetDesigned` panel; Software Update is the one real sub-screen.
 */
const ROWS = [
  { href: '/settings/account', label: 'Account', Icon: User },
  { href: '/settings/ai-brain', label: 'AI Brain', Icon: Brain },
  { href: '/settings/wifi', label: 'Wi-Fi', Icon: Wifi },
  { href: '/settings/voice', label: 'Voice & Wake Word', Icon: Mic },
  { href: '/settings/laptops', label: 'Connected Laptops', Icon: Laptop },
  { href: '/settings/software-update', label: 'Software Update', Icon: Cpu },
  { href: '/settings/about', label: 'About', Icon: Info },
] as const;

export default function SettingsPage() {
  const router = useRouter();
  const { data: device } = useQuery({ queryKey: queryKeys.device, queryFn: fetchDevice });

  const theme = useAppStore((state) => state.theme);
  const toggleTheme = useAppStore((state) => state.toggleTheme);
  const notifyOnUpdate = useAppStore((state) => state.notifyOnUpdate);
  const setNotifyOnUpdate = useAppStore((state) => state.setNotifyOnUpdate);
  const reset = useSetupStore((state) => state.reset);

  const [confirmReset, setConfirmReset] = useState(false);

  function factoryReset() {
    // Local only: this clears the app's own wizard state so setup can be re-run.
    // Wiping the unit needs the local channel, which is not wired up in this pass.
    reset();
    router.replace('/');
  }

  return (
    <>
      <AppBar title="Settings" />

      <Screen chrome="both">
        <div className="flex flex-col gap-stack-md">
          <CardGroup>
            <ListRow
              icon={<User className="h-5 w-5" strokeWidth={1.5} aria-hidden />}
              title={MOCK_USER.name}
              subtitle={MOCK_USER.email}
            />
            <ListRow
              icon={<Cpu className="h-5 w-5" strokeWidth={1.5} aria-hidden />}
              title={device?.name ?? 'ADAM'}
              subtitle={`${device?.shortId ?? '—'} · v${device?.firmwareVersion ?? '—'}`}
              trailing={<StatusDot status={device?.status === 'online' ? 'online' : 'offline'} />}
            />
          </CardGroup>

          <CardGroup>
            {ROWS.map(({ href, label, Icon }) => (
              <Link key={href} href={href} className="block">
                <ListRow
                  icon={<Icon className="h-5 w-5" strokeWidth={1.5} aria-hidden />}
                  title={label}
                  chevron
                />
              </Link>
            ))}
          </CardGroup>

          <CardGroup>
            <div className="px-stack-md py-gutter">
              <Toggle
                label="Light theme"
                description="Dark is the default. Light is an explicit opt-in."
                checked={theme === 'light'}
                onCheckedChange={() => toggleTheme()}
              />
            </div>
            <div className="px-stack-md py-gutter">
              <Toggle
                label="Notify me about updates"
                checked={notifyOnUpdate}
                onCheckedChange={setNotifyOnUpdate}
              />
            </div>
          </CardGroup>

          <div className="flex flex-col gap-stack-sm pt-stack-sm">
            {confirmReset ? (
              <>
                <p className="text-body-md text-fg">
                  This clears this phone’s setup state and returns you to the start.
                </p>
                <Button block variant="primary" onClick={factoryReset}>
                  Yes, reset
                </Button>
                <Button block variant="ghost" size="md" onClick={() => setConfirmReset(false)}>
                  Cancel
                </Button>
              </>
            ) : (
              <Button block variant="outline" onClick={() => setConfirmReset(true)}>
                Factory Reset
              </Button>
            )}
          </div>

          <p className="flex items-center gap-stack-sm pb-stack-md text-label-md text-fg-faint">
            {theme === 'light' ? (
              <Sun className="h-4 w-4" strokeWidth={1.5} aria-hidden />
            ) : (
              <Moon className="h-4 w-4" strokeWidth={1.5} aria-hidden />
            )}
            ADAM Companion · placeholder build, no backend connected.
          </p>
        </div>
      </Screen>
    </>
  );
}
