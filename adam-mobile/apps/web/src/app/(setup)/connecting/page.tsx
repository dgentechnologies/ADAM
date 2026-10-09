'use client';

import {
  AdamFaceMark,
  Button,
  Screen,
  ScreenActions,
  StepChecklist,
  type ChecklistItem,
} from '@adam/ui';
import type { HandoffProgress, HandoffStep } from '@adam/types';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';

import { runHandoff } from '@/lib/mock/api';
import { useSetupStore } from '@/stores/setup-store';

const LABELS: Record<HandoffStep, string> = {
  'sending-credentials': 'Sending credentials',
  'device-connecting': 'ADAM connecting',
  'confirming-online': 'Confirming online',
};

/**
 * `connecting` — the Wi-Fi handoff.
 *
 * Stitch implemented the step transitions as a chain of `setTimeout`s mutating
 * classes; here the state comes from the (mocked) handoff itself, so the same
 * screen will render a real transport's progress and its failures without change.
 */
export default function ConnectingPage() {
  const router = useRouter();
  const ssid = useSetupStore((state) => state.selectedSsid);
  const complete = useSetupStore((state) => state.complete);
  const [progress, setProgress] = useState<HandoffProgress | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    void runHandoff({ ssid: ssid ?? '', password: '' }, (next) => {
      if (!cancelled) setProgress(next);
    }).then(() => {
      if (cancelled) return;
      complete('connecting');
      timer = setTimeout(() => router.push('/name-device'), 700);
    }).catch(() => { if (!cancelled) setError('Setup could not finish. Please try the Wi-Fi step again.'); });

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [ssid, complete, router]);

  const items: ChecklistItem[] = (
    progress?.steps ?? [
      { step: 'sending-credentials' as const, state: 'active' as const },
      { step: 'device-connecting' as const, state: 'pending' as const },
      { step: 'confirming-online' as const, state: 'pending' as const },
    ]
  ).map(({ step, state }) => ({
    id: step,
    label: LABELS[step],
    state: state === 'complete' ? 'done' : state,
  }));

  const failure = progress?.failure ?? (error || null);

  return (
    <Screen className="pt-safe">
      <div className="flex flex-1 flex-col justify-center gap-stack-lg">
        <AdamFaceMark expression={failure ? 'idle' : 'thinking'} size="xl" />
        <StepChecklist items={items} className="w-full" />
        {failure ? (
          <p className="max-w-xs text-body-md text-fg-muted">
            Handoff failed ({failure.replace(/-/g, ' ')}). Check the password and try again.
          </p>
        ) : null}
      </div>

      {failure ? (
        <ScreenActions>
          <Button block variant="primary" onClick={() => router.replace('/wifi-password')}>
            Try again
          </Button>
          <Button block variant="ghost" size="md" onClick={() => router.replace('/wifi-select')}>
            Pick a different network
          </Button>
        </ScreenActions>
      ) : null}
    </Screen>
  );
}
