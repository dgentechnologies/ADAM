'use client';
import { AdamFaceMark, Button, Screen } from '@adam/ui';
import { Bluetooth, Check, ChevronRight, Loader2, Radio, Wifi } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Notice, Panel } from './companion-ui';
import { errorMessage, updateLocalData } from '@/lib/local-data';
import { updateDemoDevice, useDemoDevice, type DemoDevice } from '@/lib/demo-device';

type Step = 'start' | 'scanning' | 'found' | 'network' | 'connecting' | 'ready';
export function PairingDemo() {
  const router = useRouter();
  const { device, loading, error: loadError } = useDemoDevice();
  const [step, setStep] = useState<Step>('start');
  const [name, setName] = useState('Demo ADAM');
  const [network, setNetwork] = useState<DemoDevice['network']>('Demo Home');
  const [progress, setProgress] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const saving = useRef(false);
  useEffect(() => {
    const back = (event: Event) => {
      if (step === 'start' && !saving.current) return;
      event.preventDefault();
      if (saving.current) return;
      setError('');
      setStep(step === 'network' ? 'found' : step === 'connecting' ? 'network' : 'start');
    };
    window.addEventListener('adam:back', back);
    return () => window.removeEventListener('adam:back', back);
  }, [step]);
  useEffect(() => {
    if (step !== 'scanning') return;
    const timer = setTimeout(() => setStep('found'), 1200);
    return () => clearTimeout(timer);
  }, [step]);
  useEffect(() => {
    if (step !== 'connecting') return;
    let active = true;
    const timers = [
      setTimeout(() => setProgress(1), 500),
      setTimeout(() => setProgress(2), 1100),
      setTimeout(() => {
        saving.current = true;
        void updateDemoDevice({
          connected: true,
          name: name.trim() || 'Demo ADAM',
          network,
          expression: 'happy',
        })
          .then(() => {
            if (active) setStep('ready');
          })
          .catch((e) => {
            if (active) {
              setError(errorMessage(e));
              setStep('network');
            }
          })
          .finally(() => {
            saving.current = false;
          });
      }, 1700),
    ];
    return () => {
      active = false;
      timers.forEach(clearTimeout);
    };
  }, [step, name, network]);
  async function finish(destination: string) {
    if (saving.current) return;
    saving.current = true;
    setBusy(true);
    setError('');
    try {
      await updateLocalData((d) => ({ ...d, onboardingComplete: true }));
      router.replace(destination);
    } catch (e) {
      setError(errorMessage(e));
      setBusy(false);
      saving.current = false;
    }
  }
  return (
    <Screen>
      <div className="page-stack max-w-md pb-8">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="eyebrow">MEET YOUR COMPANION</p>
          <span className="border-border text-fg-muted rounded-full border px-3 py-1.5 text-[10px] tracking-widest">
            DEMO EXPERIENCE
          </span>
        </div>
        <div className="hero-halo flex h-40 items-center justify-center">
          <AdamFaceMark
            size="xl"
            expression={step === 'ready' ? 'happy' : step === 'connecting' ? 'thinking' : 'idle'}
          />
        </div>
        {step === 'start' && (
          <>
            <div>
              <h1 className="page-title">
                A little hello.
                <br />A new connection.
              </h1>
              <p className="text-fg-muted mt-4 text-sm leading-7">
                Explore discovery, pairing, and life with a simulated ADAM. Everything in this demo
                stays on your phone.
              </p>
            </div>
            <Button block disabled={loading || busy} onClick={() => setStep('scanning')}>
              <Bluetooth size={18} />
              Find my ADAM
            </Button>
            {device.connected && (
              <Button
                block
                variant="outline"
                disabled={busy}
                onClick={() => void finish('/device')}
              >
                <span className="truncate">Open {device.name}</span>
              </Button>
            )}
          </>
        )}
        {step === 'scanning' && (
          <Panel>
            <div className="flex items-center gap-3">
              <Radio className="animate-pulse" size={22} />
              <h1 className="text-lg">Looking for your companion…</h1>
            </div>
            <p className="text-fg-muted mt-4 text-sm leading-7">
              Simulating nearby discovery. Keep ADAM close during setup.
            </p>
            <Button block variant="ghost" className="mt-4" onClick={() => setStep('start')}>
              Cancel
            </Button>
          </Panel>
        )}
        {step === 'found' && (
          <>
            <div>
              <h1 className="page-title">There you are.</h1>
              <p className="text-fg-muted mt-3 text-sm">Choose your companion to begin.</p>
            </div>
            <button
              className="panel flex items-center gap-4 text-left"
              onClick={() => setStep('network')}
            >
              <Bluetooth size={26} />
              <span className="flex-1">
                <span className="block font-medium">Demo ADAM</span>
                <span className="text-fg-muted mt-2 block text-xs">
                  DEMO-001 · Strong signal · 82% battery
                </span>
              </span>
              <ChevronRight size={20} />
            </button>
            <Button block variant="ghost" onClick={() => setStep('scanning')}>
              Search again
            </Button>
          </>
        )}
        {step === 'network' && (
          <>
            <div>
              <h1 className="page-title">Make it your ADAM.</h1>
              <p className="text-fg-muted mt-3 text-sm leading-7">
                Give your demo companion a name and choose a sample network.
              </p>
            </div>
            <label className="field-label">
              Companion name
              <input
                className="field"
                value={name}
                maxLength={40}
                onChange={(e) => setName(e.target.value)}
              />
            </label>
            <div className="space-y-3" role="group" aria-label="Demo Wi-Fi network">
              {(['Demo Home', 'Demo Studio'] as const).map((ssid) => (
                <button
                  key={ssid}
                  className="panel flex w-full items-center gap-3 text-left"
                  aria-pressed={network === ssid}
                  onClick={() => setNetwork(ssid)}
                >
                  <Wifi size={20} />
                  <span className="flex-1 text-sm">{ssid}</span>
                  {network === ssid && <Check size={18} />}
                </button>
              ))}
            </div>
            <p className="text-fg-muted text-xs leading-6">
              Sample networks use no password. Your phone’s Wi-Fi connection stays as it is.
            </p>
            <Button
              block
              disabled={!name.trim()}
              onClick={() => {
                setError('');
                setProgress(0);
                setStep('connecting');
              }}
            >
              Pair Demo ADAM
            </Button>
          </>
        )}
        {step === 'connecting' && (
          <>
            <h1 className="page-title">Bringing it all together.</h1>
            <Panel>
              <p className="eyebrow mb-5">SIMULATED SETUP</p>
              {[
                'Pairing your companion',
                'Joining your chosen network',
                'Getting your preferences ready',
              ].map((label, index) => (
                <div key={label} className="flex items-center gap-3 py-3 text-sm">
                  {progress > index ? (
                    <Check size={18} />
                  ) : progress === index ? (
                    <Loader2 size={18} className="animate-spin" />
                  ) : (
                    <span className="border-border h-4 w-4 rounded-full border" />
                  )}
                  {label}
                </div>
              ))}
            </Panel>
          </>
        )}
        {step === 'ready' && (
          <>
            <div>
              <h1 className="page-title break-words">Hello, {device.name}.</h1>
              <p className="text-fg-muted mt-4 text-sm leading-7">
                Your demo companion is paired. Try expressions, adjust its display, or explore your
                connected desk.
              </p>
            </div>
            <Panel>
              <p className="flex items-center gap-2 text-sm">
                <Check size={18} />
                Demo connection ready
              </p>
              <p className="text-fg-muted mt-3 text-xs">{device.network} · DEMO-001</p>
            </Panel>
            <Button block disabled={busy} onClick={() => void finish('/device')}>
              Meet your ADAM
            </Button>
            <Button block variant="ghost" disabled={busy} onClick={() => void finish('/home')}>
              Go home
            </Button>
          </>
        )}
        {(error || loadError) && <Notice error>{error || loadError}</Notice>}
        {step !== 'connecting' && step !== 'ready' && (
          <Button block variant="ghost" disabled={busy} onClick={() => void finish('/home')}>
            Set up later
          </Button>
        )}
      </div>
    </Screen>
  );
}
