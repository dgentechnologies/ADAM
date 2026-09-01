'use client';

import { Button, RadarSweep, Screen, ScreenActions, ScreenHeader } from '@adam/ui';
import { useQuery } from '@tanstack/react-query';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';

import { queryKeys, scanForDevices } from '@/lib/mock/api';
import { useSetupStore } from '@/stores/setup-store';

/**
 * `finding_adam` — "Looking for ADAM…".
 *
 * The Stitch original drew its radar with a WebGL shader; this is the agreed SVG
 * + Framer Motion rebuild (see `RadarSweep`). Discovery itself is mocked and
 * resolves to one Founder unit after ~2.6s, then hands off to `/device-found`.
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
    const timer = setTimeout(() => router.push('/device-found'), 900);
    return () => clearTimeout(timer);
  }, [isSuccess, first, router, selectDevice]);

  return (
    <Screen className="pt-safe" texture>
      <div className="flex flex-1 flex-col justify-center gap-stack-lg">
        <ScreenHeader
          size="md"
          title={first ? 'Found him.' : 'Looking for ADAM...'}
          subtitle={
            first
              ? `${first.shortId} is responding over ${first.transport.toUpperCase()}.`
              : 'Make sure he’s powered on and the eyes are open.'
          }
        />
        <RadarSweep shape="square" found={Boolean(first)} className="self-center" />
      </div>

      <ScreenActions>
        {/* Manual entry is the documented fallback when BLE/mDNS find nothing. */}
        <Button block variant="ghost" size="md" onClick={() => router.push('/device-found')}>
          Having trouble? Connect manually.
        </Button>
      </ScreenActions>
    </Screen>
  );
}
