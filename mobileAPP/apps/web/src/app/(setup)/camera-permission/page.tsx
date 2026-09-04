'use client';

import { AdamFaceMark, Button, Screen, ScreenActions, ScreenHeader } from '@adam/ui';
import { useRouter } from 'next/navigation';

import { useSetupStore } from '@/stores/setup-store';

/**
 * `let_adam_see_you` — camera consent.
 *
 * On tap of "Let ADAM meet you" we immediately invoke `getUserMedia` so the
 * native permission dialog fires here (at the moment of informed consent)
 * rather than a screen later.  The obtained stream is immediately stopped —
 * the face-capture page opens its own stream fresh.
 */
export default function CameraPermissionPage() {
  const router = useRouter();
  const setCameraPermission = useSetupStore((state) => state.setCameraPermission);
  const complete = useSetupStore((state) => state.complete);
  const finish = useSetupStore((state) => state.finish);

  async function grant() {
    setCameraPermission(true);
    complete('camera-permission');

    // Fire the browser permission prompt NOW so the dialog appears at the
    // informed-consent moment rather than in the middle of the next screen.
    if (typeof navigator !== 'undefined' && navigator.mediaDevices?.getUserMedia) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        // Permission obtained — immediately release the stream.
        // face-capture/page.tsx will open a fresh stream with optimal constraints.
        stream.getTracks().forEach((t) => t.stop());
      } catch {
        // User denied or no camera — face-capture page handles this gracefully.
      }
    }

    router.push('/face-capture');
  }

  function decline() {
    setCameraPermission(false);
    complete('camera-permission');
    finish();
    router.push('/home');
  }

  return (
    <Screen className="pt-stack-md" texture>
      <div className="flex flex-1 flex-col justify-center gap-stack-lg">
        <div className="flex justify-center">
          <AdamFaceMark expression="idle" size="xl" />
        </div>

        <ScreenHeader
          align="center"
          size="md"
          title="Let ADAM see you"
          subtitle="He learns your face so he knows who he's talking to. Face data stays on the device and is never uploaded — no photos, only a local mathematical signature you can delete any time from Settings → Memory."
        />
      </div>

      <ScreenActions>
        <Button block variant="primary" onClick={grant}>
          Let ADAM meet you
        </Button>
        <Button block variant="ghost" size="md" className="mt-unit" onClick={decline}>
          Not now
        </Button>
      </ScreenActions>
    </Screen>
  );
}
