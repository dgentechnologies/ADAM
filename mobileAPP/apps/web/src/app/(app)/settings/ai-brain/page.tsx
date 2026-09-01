'use client';

import { NotYetDesigned, Screen } from '@adam/ui';

import { AppBar } from '@/components/app-bar';

export default function AiBrainSettingsPage() {
  return (
    <>
      <AppBar title="AI Brain" back="/settings" />
      <Screen chrome="both">
        <NotYetDesigned
          title="AI Brain"
          purpose="Switch between your own key, DGEN credits and Lite Mode after setup. Undesigned in the Stitch export."
          bullets={[
            'Current mode and who is paying for inference',
            'Replace or remove the BYOK key (device-local, never uploaded)',
            'Credit balance and top-up',
            'Drop to Lite Mode without losing the device',
          ]}
        />
      </Screen>
    </>
  );
}
