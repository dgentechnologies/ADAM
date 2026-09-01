'use client';

import { NotYetDesigned, Screen } from '@adam/ui';

import { AppBar } from '@/components/app-bar';

export default function SmartHomePage() {
  return (
    <>
      <AppBar title="Smart Home" />
      <Screen chrome="both">
        <NotYetDesigned
          title="Smart Home"
          purpose="Rooms, devices and scenes ADAM can control by voice — the tab exists in both specs but the Stitch export never designed it."
          bullets={[
            'Room list with per-device on/off and brightness',
            'Scene shortcuts (“Movie night”, “Wind down”)',
            'Which integration each device came from',
            'Local-only control when the cloud is unreachable',
          ]}
        />
      </Screen>
    </>
  );
}
