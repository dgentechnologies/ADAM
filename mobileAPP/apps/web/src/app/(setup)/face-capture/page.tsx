'use client';

import {
  AdamFaceMark,
  Button,
  ProgressTrack,
  Screen,
  ScreenActions,
  TextField,
} from '@adam/ui';
import { ArrowRight, CheckCircle2, Pencil } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';

import { useSetupStore } from '@/stores/setup-store';

/**
 * `face_capture` → `got_it`, one route with two phases.
 *
 * The live camera preview needs the native camera plugin, which is out of scope
 * for this pass; the capture phase is therefore a timed placeholder and the frame
 * is drawn empty rather than faked with a stock face.
 *
 * COPY (ruled): Stitch's success screen reads "Biometric sync complete. Your
 * profile has been securely mapped." That contradicts the on-device, no-upload
 * framing used on the consent screen — "sync" and "profile" both imply a server —
 * so the plainer confirmation below is the accepted wording.
 */
export default function FaceCapturePage() {
  const router = useRouter();
  const [phase, setPhase] = useState<'capturing' | 'done'>('capturing');
  const [name, setName] = useState('');
  const setUserNameForFace = useSetupStore((state) => state.setUserNameForFace);
  const complete = useSetupStore((state) => state.complete);
  const finish = useSetupStore((state) => state.finish);

  useEffect(() => {
    const timer = setTimeout(() => setPhase('done'), 3200);
    return () => clearTimeout(timer);
  }, []);

  function save() {
    if (name.trim()) setUserNameForFace(name.trim());
    complete('face-capture');
    finish();
    router.push('/home');
  }

  if (phase === 'capturing') {
    return (
      <Screen className="pt-safe" texture>
        <div className="flex flex-1 flex-col items-center justify-center gap-stack-lg">
          {/* Stands in for the camera preview: an empty frame with corner marks. */}
          <div className="relative flex h-64 w-64 items-center justify-center overflow-hidden rounded-full border border-border-strong bg-surface">
            <div className="digital-skin-coarse pointer-events-none absolute inset-0" aria-hidden />
            <AdamFaceMark expression="listening" size="lg" />
          </div>
          <div className="flex w-full max-w-xs flex-col gap-stack-sm text-center">
            <p className="text-body-lg text-fg">Hold still.</p>
            <ProgressTrack />
            <p className="text-label-md text-fg-muted">Learning your face on the device…</p>
          </div>
        </div>
      </Screen>
    );
  }

  return (
    <Screen className="pt-safe" texture>
      <div className="flex flex-1 flex-col justify-center gap-stack-lg">
        <div className="flex justify-center">
          <AdamFaceMark expression="happy" size="xl" />
        </div>

        {/* Stitch sets the confirmation as an icon + word on one line, so the
            check reads as part of the sentence rather than a separate badge. */}
        <div className="flex flex-col items-center gap-stack-sm text-center">
          <span className="flex items-center gap-stack-sm">
            <CheckCircle2 className="h-7 w-7 text-fg" strokeWidth={1.5} aria-hidden />
            <span className="text-headline-sm text-fg">Got it.</span>
          </span>
          <p className="max-w-md text-body-lg text-fg-muted">
            Your face is stored on ADAM only — nothing left the device.
          </p>
        </div>

        <div className="flex flex-col gap-stack-sm">
          <span className="flex items-center gap-2 text-label-xs uppercase tracking-wide text-fg-muted">
            <span className="h-1.5 w-1.5 rounded-full bg-fg" aria-hidden />
            What should ADAM call you?
          </span>
          <TextField
            variant="filled"
            placeholder="Your name"
            aria-label="What should ADAM call you?"
            maxLength={32}
            value={name}
            onChange={(event) => setName(event.target.value)}
            trailing={
              <Pencil className="h-4 w-4 shrink-0 text-fg-muted" strokeWidth={1.5} aria-hidden />
            }
          />
        </div>
      </div>

      <ScreenActions>
        <Button block variant="primary" onClick={save}>
          <span className="uppercase tracking-wide">Save</span>
          <ArrowRight className="h-5 w-5" strokeWidth={1.5} aria-hidden />
        </Button>
      </ScreenActions>
    </Screen>
  );
}
