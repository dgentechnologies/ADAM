'use client';
import { Button } from '@adam/ui';
import { useEffect, useState } from 'react';
import { Page, Notice, ConnectionNote } from '@/components/companion-ui';
import { useLocalData } from '@/lib/use-local-data';
import { updateLocalData, errorMessage, type LocalData } from '@/lib/local-data';
export default function VoicePage() {
  const { data } = useLocalData();
  const [voice, setVoice] = useState<LocalData['voice']>('Charon');
  const [wake, setWake] = useState<LocalData['wakeWord']>('Hey ADAM');
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    setVoice(data.voice);
    setWake(data.wakeWord);
  }, [data.voice, data.wakeWord]);
  async function save() {
    setBusy(true);
    try {
      await updateLocalData((d) => ({ ...d, voice, wakeWord: wake }));
      setMessage('Your voice and wake-word preferences are saved on this phone.');
      setError('');
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <Page title="Voice & wake word" back="/settings">
      <div>
        <p className="eyebrow mb-3">A FAMILIAR VOICE</p>
        <h2 className="page-title">The way ADAM sounds.</h2>
      </div>
      <label className="field-label">
        Preferred voice
        <select
          className="field"
          value={voice}
          onChange={(e) => setVoice(e.target.value as LocalData['voice'])}
        >
          {['Charon', 'Aoede', 'Kore', 'Puck', 'Fenrir'].map((v) => (
            <option key={v}>{v}</option>
          ))}
        </select>
      </label>
      <label className="field-label">
        Wake phrase
        <select
          className="field"
          value={wake}
          onChange={(e) => setWake(e.target.value as LocalData['wakeWord'])}
        >
          <option>Hey ADAM</option>
          <option>ADAM</option>
        </select>
      </label>
      <Button block disabled={busy} onClick={save}>
        {busy ? 'Saving…' : 'Save preferences'}
      </Button>
      {error && <Notice error>{error}</Notice>}
      {message && <Notice>{message}</Notice>}
      <ConnectionNote />
    </Page>
  );
}
