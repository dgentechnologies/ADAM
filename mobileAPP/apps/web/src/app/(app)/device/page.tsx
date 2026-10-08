'use client';
import { AdamFaceMark, Button } from '@adam/ui';
import { Battery, Bluetooth, Moon, Smile, Sun, Volume2, Wifi } from 'lucide-react';
import Link from 'next/link';
import { useRef, useState } from 'react';
import { Confirm, Loading, Notice, Page, Panel, Row } from '@/components/companion-ui';
import { useDemoDevice, updateDemoDevice, type DemoDevice } from '@/lib/demo-device';
import { errorMessage } from '@/lib/local-data';
import { DemoRange } from '@/components/demo-range';
export default function DevicePage() {
  const { device, loading, error: loadError } = useDemoDevice();
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [disconnect, setDisconnect] = useState(false);
  const [message, setMessage] = useState('');
  const pendingChanges = useRef(0);
  async function change(patch: Partial<DemoDevice>, text = '') {
    pendingChanges.current += 1;
    setBusy(true);
    setError('');
    try {
      await updateDemoDevice(patch);
      setMessage(text);
      setDisconnect(false);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      pendingChanges.current -= 1;
      setBusy(pendingChanges.current > 0);
    }
  }
  return (
    <Page title="Your ADAM" back="/home">
      <div>
        <p className="eyebrow mb-3">DEMO COMPANION</p>
        <h2 className="page-title break-words">{device.name}</h2>
        <p className="text-fg-muted mt-3 text-sm leading-6">
          Explore how your companion responds. Status and controls on this screen are simulated.
        </p>
      </div>
      {loading ? (
        <Loading />
      ) : !device.connected ? (
        <Panel>
          <Bluetooth size={26} />
          <h3 className="mt-4 font-medium">Ready to meet?</h3>
          <p className="text-fg-muted my-4 text-sm leading-6">
            Take a moment to pair your demo companion.
          </p>
          <Link href="/discover" className="block min-h-11 py-3 text-sm underline">
            Start demo setup
          </Link>
        </Panel>
      ) : (
        <>
          <Panel>
            <div
              className="flex h-36 items-center justify-center"
              style={{ opacity: device.brightness / 150 + 0.33 }}
            >
              <AdamFaceMark expression={device.expression} size="xl" />
            </div>
            <div className="text-fg-muted flex flex-wrap items-center justify-center gap-4 text-xs">
              <span className="flex items-center gap-1.5">
                <Bluetooth size={14} />
                Demo connected
              </span>
              <span className="flex items-center gap-1.5">
                <Battery size={16} />
                82%
              </span>
              <span className="flex items-center gap-1.5">
                <Wifi size={14} />
                {device.network}
              </span>
            </div>
          </Panel>
          <div className="grid grid-cols-2 gap-3">
            <Button
              size="md"
              variant="outline"
              disabled={busy}
              onClick={() => void change({ expression: 'happy' }, 'Demo ADAM is happy to see you.')}
            >
              <Smile size={17} />
              Say hello
            </Button>
            <Button
              size="md"
              variant="outline"
              disabled={busy}
              onClick={() =>
                void change({ expression: device.expression === 'asleep' ? 'idle' : 'asleep' })
              }
            >
              {device.expression === 'asleep' ? <Sun size={17} /> : <Moon size={17} />}
              {device.expression === 'asleep' ? 'Wake up' : 'Rest'}
            </Button>
          </div>
          <Panel>
            <DemoRange
              label="Display brightness"
              icon={Sun}
              min={10}
              value={device.brightness}
              onChange={(brightness) => void change({ brightness })}
            />
            <div className="mt-6">
              <DemoRange
                label="Voice volume"
                icon={Volume2}
                value={device.volume}
                onChange={(volume) => void change({ volume })}
              />
            </div>
            <p className="text-fg-muted mt-3 text-xs leading-6">
              Volume changes the demo setting; it does not play audio or change your phone’s volume.
            </p>
          </Panel>
          <Panel className="flush">
            <Row
              href="/settings/voice"
              icon={Volume2}
              title="Voice & wake word"
              detail="Your companion’s personality"
            />
            <Row
              href="/settings/laptops"
              icon={Bluetooth}
              title="Connected desk"
              detail="Try laptop controls in the demo"
            />
          </Panel>
          <Button block variant="outline" disabled={busy} onClick={() => setDisconnect(true)}>
            Disconnect demo
          </Button>
        </>
      )}
      {(error || loadError) && <Notice error>{error || loadError}</Notice>}
      {message && <Notice>{message}</Notice>}
      {disconnect && (
        <Confirm
          title="Disconnect Demo ADAM?"
          busy={busy}
          onClose={() => setDisconnect(false)}
          onConfirm={() =>
            void change(
              { connected: false, laptop: false, playing: false },
              'Demo disconnected. You can pair it again any time.',
            )
          }
        >
          Your phone’s memories, photos and notifications stay saved.
        </Confirm>
      )}
    </Page>
  );
}
