'use client';

import { AdamFaceMark, Button, Screen, ScreenActions } from '@adam/ui';
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
    <Screen className="pt-0 flex-1 min-h-0 flex flex-col justify-between" texture={false}>
      {/* Centered Device Discovery Card */}
      <div className="flex flex-1 flex-col items-center justify-center py-4">
        <div
          className="relative w-full max-w-[342px] flex flex-col items-center justify-between text-center overflow-hidden"
          style={{
            minHeight: 430,
            borderRadius: 28,
            background: 'linear-gradient(180deg, #18181b 0%, #0d0d0f 100%)',
            border: '1px solid rgba(255, 255, 255, 0.12)',
            boxShadow: '0 24px 50px rgba(0, 0, 0, 0.8), inset 0 1px 0 rgba(255, 255, 255, 0.15)',
            padding: '36px 20px 28px 20px',
          }}
        >
          {/* Subtle top interior ambient light */}
          <div
            className="pointer-events-none absolute -top-12 left-1/2 -translate-x-1/2 w-48 h-24 rounded-full blur-xl"
            style={{ background: 'rgba(255, 255, 255, 0.07)' }}
          />

          {/* Connection Status Badge */}
          <div
            className="inline-flex items-center gap-1.5 px-3 py-1 text-[10px] font-mono font-medium tracking-wider uppercase"
            style={{
              borderRadius: 9999,
              background: 'rgba(16, 185, 129, 0.1)',
              border: '1px solid rgba(16, 185, 129, 0.25)',
              color: '#34d399',
            }}
          >
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
            <span>Nearby • BLE Connected</span>
          </div>

          {/* ADAM Signature Glowing Eyes (Clean, borderless, no container outline) */}
          <div className="flex items-center justify-center my-6 py-2">
            <AdamFaceMark size="lg" expression="idle" animated bloom />
          </div>

          {/* Device Identity */}
          <div className="flex flex-col items-center gap-1.5 mb-6">
            <h2 className="text-2xl font-bold tracking-tight text-white font-sans">
              {found?.shortId ?? 'ADAM-3F2A'}
            </h2>
            <p className="text-xs font-medium" style={{ color: '#a1a1aa' }}>
              Hardware Companion Found
            </p>
          </div>

          {/* Hardware Specs & Metadata Footer */}
          <div
            className="w-full pt-5 mt-auto flex items-center justify-between"
            style={{ borderTop: '1px solid rgba(255, 255, 255, 0.08)' }}
          >
            <div className="flex flex-col text-left">
              <span className="text-[10px] font-mono tracking-widest uppercase" style={{ color: '#71717a' }}>
                Serial
              </span>
              <span className="font-mono text-xs font-semibold text-neutral-200 mt-0.5 tracking-wider">
                {found?.serial ?? 'DGEN-ADAM-0007'}
              </span>
            </div>

            {found?.isFounderEdition ? (
              <div className="flex flex-col items-end text-right">
                <span className="text-[10px] font-mono tracking-widest uppercase" style={{ color: '#71717a' }}>
                  Edition
                </span>
                <span
                  className="inline-flex items-center text-[10px] font-mono uppercase tracking-widest font-semibold mt-0.5"
                  style={{
                    padding: '2px 8px',
                    borderRadius: 6,
                    background: 'rgba(255, 255, 255, 0.06)',
                    border: '1px solid rgba(255, 255, 255, 0.18)',
                    color: '#ffffff',
                  }}
                >
                  FOUNDER #007
                </span>
              </div>
            ) : null}
          </div>
        </div>
      </div>

      {/* Action Buttons spanning full width of the screen with proper padding */}
      <ScreenActions className="mt-auto pb-safe flex flex-col gap-3 w-full max-w-none px-0">
        <Button block variant="primary" size="lg" onClick={confirm} className="w-full h-14">
          Yes, this is my ADAM
        </Button>
        <Button block variant="outline" size="lg" onClick={() => router.replace('/discover')} className="w-full h-14">
          Not my device
        </Button>
      </ScreenActions>
    </Screen>
  );
}
