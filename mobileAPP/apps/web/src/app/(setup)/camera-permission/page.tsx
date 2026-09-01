'use client';

import { AdamFaceMark, Button, Screen, ScreenActions, ScreenHeader } from '@adam/ui';
import { useRouter } from 'next/navigation';

import { useSetupStore } from '@/stores/setup-store';

/**
 * `let_adam_see_you` — camera consent.
 *
 * The on-device framing is stated on this screen because consent has to be
 * informed at the moment it is given, not on a later legal page. "Not now" is a
 * real, equally-weighted exit: face recognition is optional (spec §1).
 *
 * The native camera permission plugin is out of scope for this pass, so both
 * buttons only record the choice in the store.
 */
export default function CameraPermissionPage() {
  const router = useRouter();
  const setCameraPermission = useSetupStore((state) => state.setCameraPermission);
  const complete = useSetupStore((state) => state.complete);
  const finish = useSetupStore((state) => state.finish);

  function grant() {
    setCameraPermission(true);
    complete('camera-permission');
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
          subtitle="He learns your face so he knows who he’s talking to. Face data stays on the device and is never uploaded — no photos, only a local mathematical signature you can delete any time from Settings → Memory."
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
