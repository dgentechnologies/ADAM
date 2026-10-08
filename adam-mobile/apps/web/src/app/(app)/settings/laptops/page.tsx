'use client';
import { Button } from '@adam/ui';
import { Laptop, Pause, Play, Volume2 } from 'lucide-react';
import Link from 'next/link';
import { useRef, useState } from 'react';
import { Page, Panel, Notice, Loading } from '@/components/companion-ui';
import { DemoRange } from '@/components/demo-range';
import { useDemoDevice, updateDemoDevice, type DemoDevice } from '@/lib/demo-device';
import { errorMessage } from '@/lib/local-data';
export default function LaptopPage() {
  const { device, loading, error: loadError } = useDemoDevice();
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const pendingChanges = useRef(0);
  async function change(patch: Partial<DemoDevice>) {
    pendingChanges.current += 1;
    setBusy(true);
    setError('');
    try {
      await updateDemoDevice(patch);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      pendingChanges.current -= 1;
      setBusy(pendingChanges.current > 0);
    }
  }
  return (
    <Page title="Laptop companion" back="/settings">
      <div>
        <p className="eyebrow mb-3">DEMO CONNECTED DESK</p>
        <h2 className="page-title">Your desk, in reach.</h2>
        <p className="text-fg-muted mt-3 text-sm leading-7">
          Try how ADAM would control a paired laptop. These sample controls only update the demo.
        </p>
      </div>
      {loading ? (
        <Loading />
      ) : (
        <Panel>
          <Laptop size={32} strokeWidth={1.3} />
          <h3 className="mt-5 font-medium">Studio laptop</h3>
          <p className="text-fg-muted mt-2 text-xs">
            {device.connected && device.laptop
              ? 'Demo laptop connected'
              : 'Sample laptop · Ready to pair'}
          </p>
          {!device.connected ? (
            <Link href="/discover" className="mt-4 block min-h-11 py-3 text-sm underline">
              Pair Demo ADAM first
            </Link>
          ) : (
            <Button
              block
              variant="outline"
              className="mt-5"
              disabled={busy}
              onClick={() => void change({ laptop: !device.laptop, playing: false })}
            >
              {device.laptop ? 'Disconnect demo laptop' : 'Pair demo laptop'}
            </Button>
          )}
        </Panel>
      )}
      {!loading && device.connected && device.laptop && (
        <Panel>
          <DemoRange
            label="Laptop volume"
            icon={Volume2}
            value={device.laptopVolume}
            onChange={(laptopVolume) => void change({ laptopVolume })}
          />
          <div className="border-border mt-6 flex items-center gap-4 border-t pt-6">
            <span className="flex-1">
              <span className="block text-sm font-medium">A quiet afternoon</span>
              <span className="text-fg-muted mt-2 block text-xs">
                Sample track · {device.playing ? 'Playing' : 'Paused'}
              </span>
            </span>
            <Button
              size="md"
              variant="outline"
              aria-label={device.playing ? 'Pause demo media' : 'Play demo media'}
              disabled={busy}
              onClick={() => void change({ playing: !device.playing })}
            >
              {device.playing ? <Pause size={19} /> : <Play size={19} />}
            </Button>
          </div>
        </Panel>
      )}
      {(error || loadError) && <Notice error>{error || loadError}</Notice>}
    </Page>
  );
}
