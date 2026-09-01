'use client';

import { CardGroup, ListRow, NotYetDesigned, Screen, StatusDot } from '@adam/ui';
import { useQuery } from '@tanstack/react-query';
import { Laptop } from 'lucide-react';

import { AppBar } from '@/components/app-bar';
import { fetchLaptops, queryKeys } from '@/lib/mock/api';

/**
 * `settings/laptops`. Stitch never designed this screen, so the panel below states
 * what belongs here — but the paired-laptop fixture already exists and satisfies
 * the schema, so the list itself is rendered read-only rather than withheld.
 * Pairing needs the local channel and is out of scope for this pass.
 */
const OS_LABEL = { windows: 'Windows', macos: 'macOS', linux: 'Linux' } as const;

export default function LaptopsSettingsPage() {
  const { data: laptops = [] } = useQuery({
    queryKey: queryKeys.laptops,
    queryFn: fetchLaptops,
  });

  return (
    <>
      <AppBar title="Connected Laptops" back="/settings" />
      <Screen chrome="both">
        <div className="flex flex-col gap-stack-md">
          {laptops.length ? (
            <CardGroup>
              {laptops.map((laptop) => (
                <ListRow
                  key={laptop.pairingId}
                  icon={<Laptop className="h-5 w-5" strokeWidth={1.5} aria-hidden />}
                  title={laptop.hostname}
                  subtitle={OS_LABEL[laptop.os]}
                  trailing={<StatusDot status={laptop.online ? 'online' : 'offline'} withLabel />}
                />
              ))}
            </CardGroup>
          ) : null}

          <NotYetDesigned
            title="Pairing flow"
            purpose="Adding and removing a laptop agent. Undesigned in the Stitch export; the list above is read-only."
            bullets={[
              'Six-character pairing code entry',
              'Per-laptop permissions (screen, files, shell)',
              'Unpair and revoke',
              'What the agent is allowed to do while you are away',
            ]}
          />
        </div>
      </Screen>
    </>
  );
}
