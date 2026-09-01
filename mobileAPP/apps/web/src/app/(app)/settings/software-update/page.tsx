'use client';

import type { OtaStage } from '@adam/types';
import {
  AdamFaceMark,
  Button,
  Card,
  ProgressTrack,
  Screen,
  ScreenHeader,
  Toggle,
} from '@adam/ui';
import { useQuery } from '@tanstack/react-query';
import { CheckCircle2 } from 'lucide-react';
import { useEffect, useState } from 'react';

import { AppBar } from '@/components/app-bar';
import { fetchOtaState, queryKeys } from '@/lib/mock/api';
import { useAppStore } from '@/stores/app-store';

/**
 * `settings/software-update` — both Stitch states behind `MOCK_UPDATE_AVAILABLE`
 * in the fixtures: "up to date" and "update available".
 *
 * The install run is simulated locally (no OTA transport in this pass) but it
 * walks the real `OtaStage` sequence, so wiring the device channel later means
 * replacing the timer with a subscription and nothing else.
 *
 * VERSION DISCREPANCY (flagged): the Stitch export showed v40.2 in one state and
 * v40.2.1 in the other. v40.2.1 is treated as the installed version.
 */
const STAGE_COPY: Record<OtaStage, string> = {
  idle: '',
  checking: 'Checking with DGEN…',
  downloading: 'Downloading…',
  verifying: 'Verifying signature…',
  installing: 'Installing…',
  rebooting: 'ADAM is restarting. His eyes will go dark for a moment.',
  confirming: 'Confirming the new version…',
  complete: 'Updated.',
  failed: 'The update failed. ADAM is still on the old version.',
  'rolled-back': 'The update was rolled back. Nothing changed.',
};

const RUN: ReadonlyArray<{ stage: OtaStage; ms: number }> = [
  { stage: 'downloading', ms: 2400 },
  { stage: 'verifying', ms: 1200 },
  { stage: 'installing', ms: 2000 },
  { stage: 'rebooting', ms: 1800 },
  { stage: 'confirming', ms: 1200 },
  { stage: 'complete', ms: 0 },
];

function formatSize(bytes: number): string {
  return `${(bytes / 1_000_000_000).toFixed(1)} GB`;
}

export default function SoftwareUpdatePage() {
  const { data: ota, refetch, isFetching } = useQuery({
    queryKey: queryKeys.ota,
    queryFn: fetchOtaState,
  });
  const notifyOnUpdate = useAppStore((state) => state.notifyOnUpdate);
  const setNotifyOnUpdate = useAppStore((state) => state.setNotifyOnUpdate);

  const [stage, setStage] = useState<OtaStage>('idle');
  const [step, setStep] = useState(-1);

  useEffect(() => {
    if (step < 0 || step >= RUN.length) return;
    const current = RUN[step];
    if (!current) return;
    setStage(current.stage);
    if (current.ms === 0) return;
    const timer = setTimeout(() => setStep(step + 1), current.ms);
    return () => clearTimeout(timer);
  }, [step]);

  const running = stage !== 'idle' && stage !== 'complete';
  const progress = step < 0 ? 0 : Math.round(((step + 1) / RUN.length) * 100);
  const manifest = ota?.manifest ?? null;

  return (
    <>
      <AppBar title="Software Update" back="/settings" />

      <Screen chrome="both" texture>
        <div className="flex flex-col gap-stack-lg pt-stack-md">
          <div className="flex flex-col items-center gap-stack-md text-center">
            <AdamFaceMark
              expression={running ? 'thinking' : stage === 'complete' ? 'happy' : 'idle'}
              size="xl"
            />
            {stage === 'complete' ? (
              <ScreenHeader
                align="center"
                size="sm"
                title="Updated."
                subtitle={`ADAM is now on v${manifest?.latestVersion ?? ota?.currentVersion ?? '—'}.`}
              />
            ) : running ? (
              <ScreenHeader align="center" size="sm" title={STAGE_COPY[stage]} />
            ) : ota?.updateAvailable && manifest ? (
              <ScreenHeader
                align="center"
                size="sm"
                eyebrow={`v${ota.currentVersion} → v${manifest.latestVersion}`}
                title="An update is ready."
                subtitle={`${formatSize(manifest.packageSizeBytes)} · ADAM will restart once.`}
              />
            ) : (
              <ScreenHeader
                align="center"
                size="sm"
                title="ADAM is up to date."
                subtitle={`Running v${ota?.currentVersion ?? '—'}.`}
              />
            )}
          </div>

          {running ? <ProgressTrack value={progress} /> : null}

          {manifest && !running && stage !== 'complete' ? (
            <Card surface="recessed" padding="md">
              <p className="pb-stack-sm font-display text-label-xs uppercase text-fg-subtle">
                What’s new
              </p>
              <ul className="flex flex-col gap-stack-sm">
                {manifest.changelog.map((entry) => (
                  <li key={entry.text} className="flex gap-stack-sm text-body-md text-fg-muted">
                    <span aria-hidden className="text-fg-faint">
                      —
                    </span>
                    <span>{entry.text}</span>
                  </li>
                ))}
              </ul>
            </Card>
          ) : null}

          {stage === 'complete' ? (
            <p className="flex items-center justify-center gap-stack-sm text-label-md text-fg-muted">
              <CheckCircle2 className="h-4 w-4" strokeWidth={1.5} aria-hidden />
              No action needed.
            </p>
          ) : null}

          {!running && stage !== 'complete' ? (
            <div className="flex flex-col gap-stack-md">
              {ota?.updateAvailable ? (
                <Button block variant="primary" onClick={() => setStep(0)}>
                  Install now
                </Button>
              ) : (
                <Button block variant="outline" onClick={() => void refetch()} disabled={isFetching}>
                  {isFetching ? 'Checking…' : 'Check again'}
                </Button>
              )}

              <Card surface="recessed" padding="md">
                <Toggle
                  label="Notify me about updates"
                  description="ADAM never installs on his own."
                  checked={notifyOnUpdate}
                  onCheckedChange={setNotifyOnUpdate}
                />
              </Card>
            </div>
          ) : null}

          {running ? (
            <p className="text-center text-label-md text-fg-faint">
              Keep ADAM plugged in until this finishes.
            </p>
          ) : null}
        </div>
      </Screen>
    </>
  );
}
