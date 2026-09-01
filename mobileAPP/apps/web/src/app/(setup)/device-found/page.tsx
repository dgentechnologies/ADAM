'use client';

import { Button, Card, Screen, ScreenActions } from '@adam/ui';
import { Bot } from 'lucide-react';
import { useRouter } from 'next/navigation';

import { MOCK_DISCOVERED } from '@/lib/mock/fixtures';
import { useSetupStore } from '@/stores/setup-store';

/**
 * `adam_found` — confirmation before claiming. "Not my device" returns to the
 * scan rather than dead-ending, which the Stitch export left unhandled.
 */
export default function DeviceFoundPage() {
  const router = useRouter();
  const complete = useSetupStore((state) => state.complete);
  const found = MOCK_DISCOVERED[0];

  function confirm() {
    complete('device-found');
    router.push('/wifi-select');
  }

  return (
    <Screen className="pt-stack-lg">
      <div className="flex flex-1 flex-col justify-center">
        <Card texture padding="lg" className="flex flex-col items-center gap-stack-md text-center">
          {/* Stitch draws a black disc with a robot glyph here rather than the face
              mark — the unit as an object, not as a personality yet. */}
          <span className="flex h-20 w-20 items-center justify-center rounded-full border border-border-strong bg-black">
            <Bot className="h-9 w-9 text-fg" strokeWidth={1.5} aria-hidden />
          </span>

          <div className="flex flex-col gap-unit">
            <p className="text-headline-sm text-fg">{found?.shortId ?? 'ADAM'} found nearby</p>
            <p className="text-label-md text-fg-muted">Serial: {found?.serial ?? '—'}</p>
          </div>

          {found?.isFounderEdition ? (
            <p className="font-display text-label-xs uppercase text-fg-muted">Founder Edition</p>
          ) : null}
        </Card>
      </div>

      <ScreenActions>
        <Button block variant="primary" onClick={confirm}>
          Yes, this is my ADAM
        </Button>
        <Button block variant="outline" size="lg" onClick={() => router.replace('/discover')}>
          Not my device
        </Button>
      </ScreenActions>
    </Screen>
  );
}
