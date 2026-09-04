'use client';

import {
  Button,
  Card,
  ListRow,
  Screen,
  ScreenActions,
  ScreenHeader,
  cn,
} from '@adam/ui';
import { useQuery } from '@tanstack/react-query';
import { Check, Info, Lock, Wifi } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useState } from 'react';

import { queryKeys, scanNetworks } from '@/lib/mock/api';
import { useSetupStore } from '@/stores/setup-store';

/**
 * `connecting_to_wi_fi` — the network list ("Get him online.").
 *
 * Designed for mobile:
 * - Fixed header and bottom actions.
 * - Scrollable list of networks so buttons are never pushed below fold.
 */
export default function WifiSelectPage() {
  const router = useRouter();
  const selectSsid = useSetupStore((state) => state.selectSsid);
  const complete = useSetupStore((state) => state.complete);
  const [selected, setSelected] = useState<string | null>(null);

  const { data: networks = [], isPending } = useQuery({
    queryKey: queryKeys.networks,
    queryFn: scanNetworks,
  });

  function submit() {
    if (!selected) return;
    const network = networks.find((item) => item.ssid === selected);
    selectSsid(selected);
    complete('wifi-select');
    // An open network has no password step.
    router.push(network?.security === 'open' ? '/connecting' : '/wifi-password');
  }

  return (
    <Screen className="pt-0 flex-1 min-h-0 flex flex-col justify-between" texture={false}>
      <div className="shrink-0 pt-1 pb-2">
        <ScreenHeader
          size="md"
          title="Get him online."
          subtitle="Select a network to connect ADAM to your local environment."
        />
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto py-2 flex flex-col gap-stack-md no-scrollbar">
        <div className="flex flex-col gap-2.5">
          {isPending
            ? Array.from({ length: 4 }, (_, index) => (
                <Card key={index} padding="none">
                  <div className="flex items-center gap-gutter px-stack-md py-gutter">
                    <span className="h-10 w-10 animate-breathe rounded-full bg-surface-pressed" />
                    <span className="h-4 w-32 animate-breathe rounded-full bg-surface-pressed" />
                  </div>
                </Card>
              ))
            : networks.map((network) => {
                const isSelected = selected === network.ssid;
                return (
                  <Card
                    key={network.ssid}
                    padding="none"
                    className={cn(
                      'transition-colors duration-fast',
                      isSelected ? 'border-fg bg-surface-raised' : 'border-border/60 bg-surface-raised/90',
                    )}
                  >
                    <ListRow
                      disabled={network.unsupported}
                      onClick={network.unsupported ? undefined : () => setSelected(network.ssid)}
                      icon={
                        <Wifi
                          className={cn('h-5 w-5', network.signalBars <= 1 && 'opacity-40')}
                          strokeWidth={1.5}
                        />
                      }
                      title={network.ssid}
                      trailing={
                        <span className="flex items-center gap-stack-sm text-fg-subtle">
                          {network.security === 'open' ? null : (
                            <Lock className="h-4 w-4" strokeWidth={1.5} aria-label="Secured" />
                          )}
                          {isSelected ? (
                            <Check
                              className="h-5 w-5 text-fg"
                              strokeWidth={2}
                              aria-label="Selected"
                            />
                          ) : null}
                        </span>
                      }
                    />
                  </Card>
                );
              })}
        </div>

        <p className="flex items-center gap-stack-sm text-label-sm text-fg-muted pb-2">
          <Info className="h-4 w-4 shrink-0" strokeWidth={1.5} aria-hidden />
          ADAM only supports 2.4GHz networks.
        </p>
      </div>

      <ScreenActions className="mt-auto pb-safe shrink-0 pt-2">
        <Button block variant="primary" size="lg" disabled={!selected} onClick={submit}>
          Continue
        </Button>
      </ScreenActions>
    </Screen>
  );
}
