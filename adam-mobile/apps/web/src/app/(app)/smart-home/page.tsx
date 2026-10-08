'use client';
import { Button, IconButton } from '@adam/ui';
import { Capacitor } from '@capacitor/core';
import { Home, LampCeiling, Power, RefreshCw, Settings2 } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import { Dialog, Empty, Loading, Notice, Page, Panel } from '@/components/companion-ui';
import {
  readHome,
  savedHomeToken,
  setHomePower,
  validateHomeUrl,
  type Entity,
} from '@/lib/home-assistant';
import { errorMessage, updateLocalData } from '@/lib/local-data';
import { setSecret, removeSecret } from '@/lib/native/secure-storage';
import { useLocalData } from '@/lib/use-local-data';
export default function SmartHomePage() {
  const { data, loading: localLoading } = useLocalData();
  const [entities, setEntities] = useState<Entity[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [settings, setSettings] = useState(false);
  const [busy, setBusy] = useState('');
  const refresh = useCallback(async (url: string) => {
    if (!url) return;
    setLoading(true);
    setError('');
    try {
      setEntities(await readHome(url));
    } catch (e) {
      setError(errorMessage(e));
      setEntities([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (data.homeAssistantUrl) void refresh(data.homeAssistantUrl);
    else setEntities([]);
  }, [data.homeAssistantUrl, refresh]);
  async function toggle(entity: Entity) {
    setBusy(entity.entity_id);
    setError('');
    try {
      await setHomePower(data.homeAssistantUrl, entity);
      await refresh(data.homeAssistantUrl);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy('');
    }
  }
  return (
    <Page
      title="Smart Home"
      action={
        <IconButton
          aria-label="Smart home connection"
          size="md"
          variant="ghost"
          onClick={() => setSettings(true)}
        >
          <Settings2 size={21} />
        </IconButton>
      }
    >
      <div>
        <p className="eyebrow mb-3">YOUR CONNECTED SPACE</p>
        <h2 className="page-title">
          A home that
          <br />
          feels like yours.
        </h2>
        <p className="text-fg-muted mt-3 text-sm leading-6">
          Control lights, switches, fans, and scenes through your own Home Assistant server.
        </p>
      </div>
      {error && (
        <Notice error>
          {error}
          <button
            className="mt-2 block underline"
            onClick={() => void refresh(data.homeAssistantUrl)}
          >
            Try again
          </button>
        </Notice>
      )}
      {localLoading || loading ? (
        <Loading />
      ) : !data.homeAssistantUrl ? (
        <Empty
          icon={Home}
          title="Bring your home together"
          action={<Button onClick={() => setSettings(true)}>Connect Home Assistant</Button>}
        >
          Connect securely to an existing Home Assistant installation. No ADAM hardware or developer
          laptop needed.
        </Empty>
      ) : (
        <>
          <div className="flex items-center justify-between">
            <p className="text-fg-muted text-xs">
              {entities.length} entities ·{' '}
              {error ? 'Connection unavailable' : 'Last refreshed just now'}
            </p>
            <IconButton
              size="md"
              variant="ghost"
              aria-label="Refresh smart devices"
              onClick={() => void refresh(data.homeAssistantUrl)}
            >
              <RefreshCw size={18} />
            </IconButton>
          </div>
          {!entities.length && !error && (
            <Empty icon={LampCeiling} title="No supported devices yet">
              Add lights, switches, fans, scenes, or sensors in Home Assistant, then refresh.
            </Empty>
          )}
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {entities.map((entity) => {
              const controllable = /^(light|switch|fan|scene)\./.test(entity.entity_id);
              return (
                <Panel key={entity.entity_id}>
                  <div className="flex items-center gap-3">
                    <LampCeiling size={22} strokeWidth={1.3} />
                    <div className="min-w-0 flex-1">
                      <h3 className="break-words text-sm font-medium">
                        {entity.attributes.friendly_name ?? entity.entity_id}
                      </h3>
                      <p className="text-fg-muted mt-1 text-xs">{entity.state}</p>
                    </div>
                    {controllable && (
                      <IconButton
                        aria-label={`${entity.state === 'on' ? 'Turn off' : 'Turn on'} ${entity.attributes.friendly_name ?? entity.entity_id}`}
                        size="md"
                        disabled={!!busy || ['unavailable', 'unknown'].includes(entity.state)}
                        variant={entity.state === 'on' ? 'primary' : 'outline'}
                        onClick={() => void toggle(entity)}
                      >
                        <Power size={18} />
                      </IconButton>
                    )}
                  </div>
                </Panel>
              );
            })}
          </div>
        </>
      )}
      {settings && (
        <ConnectionForm initialUrl={data.homeAssistantUrl} onClose={() => setSettings(false)} />
      )}
    </Page>
  );
}
function ConnectionForm({ initialUrl, onClose }: { initialUrl: string; onClose: () => void }) {
  const [url, setUrl] = useState(initialUrl);
  const [token, setToken] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError('');
    try {
      const validUrl = validateHomeUrl(url);
      const existing = validUrl === initialUrl ? await savedHomeToken(validUrl) : null;
      const value = token.trim() || existing;
      if (!value) throw new Error('Enter an access token from your Home Assistant profile.');
      await readHome(validUrl, value);
      await setSecret('home-assistant', JSON.stringify({ url: validUrl, token: value }));
      await updateLocalData((d) => ({ ...d, homeAssistantUrl: validUrl }));
      onClose();
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }
  async function disconnect() {
    setBusy(true);
    try {
      await updateLocalData((d) => ({ ...d, homeAssistantUrl: '' }));
      await removeSecret('home-assistant');
      onClose();
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <Dialog
      title="Connect Home Assistant"
      onClose={() => {
        if (!busy) onClose();
      }}
    >
      <p className="text-fg-muted text-sm leading-6">
        In Home Assistant, open your profile → Security and create a long-lived access token. Your
        token is kept on this phone.
      </p>
      <form onSubmit={submit} className="space-y-4">
        <label className="field-label">
          Server address
          <input
            className="field"
            type="url"
            required
            placeholder="http://homeassistant.local:8123"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            autoCapitalize="none"
            spellCheck={false}
          />
        </label>
        <label className="field-label">
          Access token
          <input
            className="field"
            type="password"
            autoComplete="off"
            placeholder={initialUrl ? 'Leave blank to keep saved token' : 'Paste your access token'}
            value={token}
            onChange={(e) => setToken(e.target.value)}
          />
        </label>
        {!Capacitor.isNativePlatform() && (
          <Notice>
            In a browser, allow this site in Home Assistant’s CORS settings. Tokens last for this
            tab only. The Android app stores them securely.
          </Notice>
        )}
        {error && <Notice error>{error}</Notice>}
        <Button block type="submit" disabled={busy}>
          {busy ? 'Connecting…' : 'Test & save connection'}
        </Button>
        {initialUrl && (
          <Button block variant="outline" disabled={busy} onClick={disconnect}>
            Disconnect
          </Button>
        )}
        <Button block variant="ghost" disabled={busy} onClick={onClose}>
          Cancel
        </Button>
      </form>
    </Dialog>
  );
}
