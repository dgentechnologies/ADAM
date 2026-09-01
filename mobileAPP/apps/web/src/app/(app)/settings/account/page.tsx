'use client';

import { NotYetDesigned, Screen } from '@adam/ui';

import { AppBar } from '@/components/app-bar';

export default function AccountSettingsPage() {
  return (
    <>
      <AppBar title="Account" back="/settings" />
      <Screen chrome="both">
        <NotYetDesigned
          title="Account"
          purpose="Identity, sign-out and data export. Undesigned in the Stitch export; listed in spec §3.4."
          bullets={[
            'Signed-in email and provider',
            'Sign out (clears the secure-storage session)',
            'Export or delete your account data',
            'Transfer device ownership',
          ]}
        />
      </Screen>
    </>
  );
}
