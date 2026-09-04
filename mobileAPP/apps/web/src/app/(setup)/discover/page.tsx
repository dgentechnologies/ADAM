'use client';

import { Button, Screen, ScreenActions, ScreenHeader } from '@adam/ui';
import { useQuery } from '@tanstack/react-query';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';

import { AdamSearchingDotPulse } from '@/components/AdamSearchingDotPulse';
import { queryKeys, scanForDevices } from '@/lib/mock/api';
import { useSetupStore } from '@/stores/setup-store';

/**
 * `finding_adam` — "Looking for ADAM…".
 *
 * Uses AdamSearchingDotPulse:
 * Centered blinking eyes with subtle expanding dot-matrix pulse rings.
 * Discovery resolves to one Founder unit after ~2.6s, then hands off to `/device-found`.
 */
export default function DiscoverPage() {
  const router = useRouter();
  const selectDevice = useSetupStore((state) => state.selectDevice);

  const { data, isSuccess } = useQuery({
    queryKey: queryKeys.discovery,
    queryFn: scanForDevices,
    staleTime: 0,
  });

  const first = data?.[0];

  useEffect(() => {
    if (!isSuccess || !first) return;
    selectDevice(first.serial, first.isFounderEdition, first.isFounderEdition ? 7 : null);
    const timer = setTimeout(() => router.push('/device-found'), 15000);
    return () => clearTimeout(timer);
  }, [isSuccess, first, router, selectDevice]);

  return (
    <Screen className="relative pt-safe min-h-0 flex-1" texture={false}>
      {/* Background Dot Pulse Animation */}
      <AdamSearchingDotPulse />

      {/* Foreground Content */}
      <div className="relative z-10 flex flex-1 flex-col justify-between">
        <div className="pt-stack-md">
          <ScreenHeader
            size="md"
            title={first ? 'Found him.' : 'Looking for ADAM...'}
            subtitle={
              first
                ? `${first.shortId} is responding over ${first.transport.toUpperCase()}.`
                : 'Make sure he’s powered on and the eyes are open.'
            }
          />
        </div>

        <ScreenActions className="relative z-10">
          {/* Manual entry is the documented fallback when BLE/mDNS find nothing. */}
          <Button block variant="ghost" size="md" onClick={() => router.push('/device-found')}>
            Having trouble? Connect manually.
          </Button>
        </ScreenActions>
      </div>
    </Screen>
  );
}

