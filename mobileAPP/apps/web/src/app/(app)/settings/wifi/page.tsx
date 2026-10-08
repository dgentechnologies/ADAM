'use client';
import { Button } from '@adam/ui';
import { Wifi } from 'lucide-react';
import { useState } from 'react';
import { Page, Empty, Notice, ConnectionNote } from '@/components/companion-ui';
import { openWifiSettings } from '@/lib/native/companion';
import { errorMessage } from '@/lib/local-data';
export default function WifiPage() {
  const [error, setError] = useState('');
  return (
    <Page title="Wi-Fi" back="/settings">
      <Empty
        icon={Wifi}
        title="Stay connected"
        action={
          <Button
            onClick={async () => {
              try {
                await openWifiSettings();
              } catch (e) {
                setError(errorMessage(e));
              }
            }}
          >
            Open phone Wi-Fi settings
          </Button>
        }
      >
        Choose or change your phone’s network securely in Android settings. ADAM never needs to read
        or store your phone’s Wi-Fi password.
      </Empty>
      {error && <Notice>{error}</Notice>}
      <ConnectionNote />
      <p className="text-fg-muted text-xs leading-6">
        Sending Wi-Fi credentials to your robot will be available with ADAM pairing.
      </p>
    </Page>
  );
}
