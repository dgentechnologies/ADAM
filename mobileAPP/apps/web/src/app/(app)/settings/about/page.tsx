'use client';

import { Card, NotYetDesigned, Screen, Wordmark } from '@adam/ui';
import { useQuery } from '@tanstack/react-query';

import { AppBar } from '@/components/app-bar';
import { fetchDevice, queryKeys } from '@/lib/mock/api';

/**
 * `settings/about`. Undesigned by Stitch; the identity block and the device facts
 * are real because both are already available, and the panel states what is still
 * missing (licences, legal, support).
 */
export default function AboutSettingsPage() {
  const { data: device } = useQuery({ queryKey: queryKeys.device, queryFn: fetchDevice });

  const facts: ReadonlyArray<[string, string]> = [
    ['Device', device?.name ?? 'ADAM'],
    ['Short ID', device?.shortId ?? '—'],
    ['Serial', device?.serial ?? '—'],
    ['Firmware', device ? `v${device.firmwareVersion}` : '—'],
    ['Batch', device?.hardwareBatch ?? '—'],
    [
      'Edition',
      device?.isFounderEdition
        ? `Founder № ${String(device.founderNumber ?? 1).padStart(3, '0')}`
        : 'Standard',
    ],
  ];

  return (
    <>
      <AppBar title="About" back="/settings" />
      <Screen chrome="both">
        <div className="flex flex-col gap-stack-lg">
          <div className="flex justify-center pt-stack-md">
            <Wordmark byline />
          </div>

          <Card surface="recessed" padding="md">
            <dl className="flex flex-col gap-stack-sm">
              {facts.map(([label, value]) => (
                <div key={label} className="flex items-baseline justify-between gap-gutter">
                  <dt className="text-label-md text-fg-muted">{label}</dt>
                  <dd className="text-body-md text-fg">{value}</dd>
                </div>
              ))}
            </dl>
          </Card>

          <NotYetDesigned
            title="Legal & support"
            purpose="Undesigned in the Stitch export."
            bullets={[
              'Open-source licences',
              'Privacy policy and terms',
              'Support contact and diagnostics bundle',
              'Regulatory / compliance marks',
            ]}
          />
        </div>
      </Screen>
    </>
  );
}
