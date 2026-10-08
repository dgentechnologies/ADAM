'use client';
import { Button } from '@adam/ui';
import { Capacitor } from '@capacitor/core';
import { useState } from 'react';
import { Notice } from './companion-ui';
import { Companion } from '@/lib/native/companion';
import { errorMessage } from '@/lib/local-data';

export function CameraNotice({ error }: { error: string }) {
  const [settingsError, setSettingsError] = useState('');
  const denied =
    Capacitor.getPlatform() === 'android' && /permission|denied|access to camera/i.test(error);
  return (
    <Notice error>
      {denied
        ? 'Camera access is off. Try taking a photo again, or allow Camera in Android app permissions.'
        : error}
      {denied && (
        <Button
          variant="outline"
          size="md"
          className="mt-3"
          onClick={() =>
            void Companion.openAppSettings().catch((e) => setSettingsError(errorMessage(e)))
          }
        >
          Open camera permissions
        </Button>
      )}
      {settingsError && <p className="mt-2">{settingsError}</p>}
    </Notice>
  );
}
