'use client';

import { Button, DisplayField, Screen, ScreenActions, ScreenHeader } from '@adam/ui';
import { useRouter } from 'next/navigation';
import { useState } from 'react';

import { nextStep, setupHref } from '@/lib/setup-flow';
import { useSetupStore } from '@/stores/setup-store';

/**
 * `what_should_we_call_him`.
 *
 * The name defaults to "ADAM" if left blank (spec §2.5), so the field is optional
 * and the CTA is never disabled. Stitch drew a custom blinking caret over a hidden
 * input; the real caret is used instead, which keeps selection and IME working.
 */
export default function NameDevicePage() {
  const router = useRouter();
  const [name, setName] = useState('');
  const setDeviceName = useSetupStore((state) => state.setDeviceName);
  const complete = useSetupStore((state) => state.complete);
  const isFounderEdition = useSetupStore((state) => state.isFounderEdition);
  const aiBrainMode = useSetupStore((state) => state.aiBrainMode);

  function submit() {
    setDeviceName(name.trim() || 'ADAM');
    complete('name-device');
    const next = nextStep('name-device', { isFounderEdition, aiBrainMode });
    router.push(next === 'done' ? '/home' : setupHref(next));
  }

  return (
    <Screen className="pt-stack-md">
      <div className="flex flex-1 flex-col justify-center gap-stack-lg">
        <ScreenHeader
          align="center"
          size="md"
          title={
            <>
              What should
              <br />
              we call him?
            </>
          }
        />
        <DisplayField
          label="Device name"
          placeholder="ADAM"
          autoFocus
          maxLength={32}
          value={name}
          onChange={(event) => setName(event.target.value)}
          hint="You can change this anytime."
        />
      </div>

      <ScreenActions>
        <Button block variant="primary" onClick={submit}>
          Continue
        </Button>
      </ScreenActions>
    </Screen>
  );
}
