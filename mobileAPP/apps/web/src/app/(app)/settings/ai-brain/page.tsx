'use client';
import { Button } from '@adam/ui';
import { Capacitor, CapacitorHttp } from '@capacitor/core';
import { KeyRound, ShieldCheck } from 'lucide-react';
import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Page, Panel, Notice, ConnectionNote } from '@/components/companion-ui';
import { useLocalData } from '@/lib/use-local-data';
import { updateLocalData, errorMessage } from '@/lib/local-data';
import { getSecret, setSecret, removeSecret } from '@/lib/native/secure-storage';
export default function BrainPage() {
  const { data } = useLocalData();
  const [key, setKey] = useState('');
  const [saved, setSaved] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  useEffect(() => {
    void getSecret('gemini-api-key')
      .then((v) => setSaved(!!v))
      .catch(() => setError('Secure storage is unavailable.'));
  }, []);
  async function save() {
    setBusy(true);
    setError('');
    setMessage('');
    try {
      const value = key.trim();
      if (value.length < 20) throw new Error('Paste a valid Gemini API key.');
      const url = 'https://generativelanguage.googleapis.com/v1beta/models';
      let status = 0;
      if (Capacitor.isNativePlatform()) {
        const r = await CapacitorHttp.get({
          url,
          headers: { 'x-goog-api-key': value },
          connectTimeout: 10000,
          readTimeout: 10000,
        });
        status = r.status;
      } else {
        const c = new AbortController();
        const t = setTimeout(() => c.abort(), 10000);
        try {
          status = (await fetch(url, { headers: { 'x-goog-api-key': value }, signal: c.signal }))
            .status;
        } finally {
          clearTimeout(t);
        }
      }
      if (status !== 200)
        throw new Error('Google could not verify this key. Check its API access and try again.');
      await setSecret('gemini-api-key', value);
      await updateLocalData((d) => ({ ...d, brain: 'byok' }));
      setKey('');
      setSaved(true);
      setMessage('Key verified and saved. It has not been sent to an ADAM robot.');
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  async function remove() {
    setBusy(true);
    try {
      await removeSecret('gemini-api-key');
      await updateLocalData((d) => ({ ...d, brain: 'lite' }));
      setSaved(false);
      setMessage('API key removed from this phone.');
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <Page title="AI preferences" back="/settings">
      <div>
        <p className="eyebrow mb-3">YOUR CHOICE OF INTELLIGENCE</p>
        <h2 className="page-title">
          A mind of its own.
          <br />
          On your terms.
        </h2>
      </div>
      <Panel>
        <div className="mb-3 flex items-center gap-3">
          <KeyRound size={22} />
          <h3 className="text-sm font-medium">Use your Gemini key</h3>
        </div>
        <p className="text-fg-muted mb-4 text-sm leading-6">
          Verify your key with Google and save it securely for future pairing. Google bills usage
          directly to your account.
        </p>
        <label className="field-label">
          API key
          <input
            className="field"
            type="password"
            autoComplete="off"
            value={key}
            onChange={(e) => setKey(e.target.value)}
            placeholder={
              saved ? 'A key is saved · enter a replacement' : 'Paste your Gemini API key'
            }
          />
        </label>
        <Button block size="md" className="mt-4" disabled={busy || !key.trim()} onClick={save}>
          {busy ? 'Checking…' : 'Verify & save key'}
        </Button>
        {saved && (
          <Button block variant="ghost" className="mt-2" disabled={busy} onClick={remove}>
            Remove saved key
          </Button>
        )}
        <a
          className="mt-4 block py-2 text-center text-xs underline"
          href="https://aistudio.google.com/apikey"
          target="_blank"
          rel="noopener noreferrer"
        >
          Get a key from Google AI Studio
        </a>
      </Panel>
      {error && <Notice error>{error}</Notice>}
      {message && <Notice>{message}</Notice>}
      <Panel>
        <p className="flex items-center gap-2 text-sm">
          <ShieldCheck size={18} />
          {saved ? 'Your own key is ready' : 'No paid AI service configured'}
        </p>
        <p className="text-fg-muted mt-3 text-xs leading-6">
          {Capacitor.isNativePlatform()
            ? 'Credentials are encrypted with Android Keystore.'
            : 'In a browser, keys are held for this tab only.'}{' '}
          Current preference: {data.brain === 'byok' ? 'your own key' : 'Lite'}.
        </p>
      </Panel>
      <Link href="/credits" className="text-fg-muted py-3 text-center text-sm">
        About managed credits
      </Link>
      <ConnectionNote />
    </Page>
  );
}
