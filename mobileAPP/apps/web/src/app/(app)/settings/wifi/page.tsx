'use client';

import { NotYetDesigned, Screen } from '@adam/ui';

import { AppBar } from '@/components/app-bar';

export default function WifiSettingsPage() {
  return (
    <>
      <AppBar title="Wi-Fi" back="/settings" />
      <Screen chrome="both">
        <NotYetDesigned
          title="Wi-Fi"
          purpose="Change the network ADAM is on, after setup. Undesigned in the Stitch export."
          bullets={[
            'Current SSID, band and signal',
            'Re-run the handoff on a different network',
            'Forget the saved network',
            'The 2.4GHz-only constraint restated at the point of choice',
          ]}
        />
      </Screen>
    </>
  );
}
