'use client';
import { Button } from '@adam/ui';
import { Download, FileUp, Shield, Trash2 } from 'lucide-react';
import { Capacitor } from '@capacitor/core';
import { Preferences } from '@capacitor/preferences';
import { useRef, useState } from 'react';
import { Confirm, Notice, Page, Panel } from '@/components/companion-ui';
import { LocalData, readLocalData, restoreLocalData, errorMessage } from '@/lib/local-data';
import { clearShareCache, shareFile } from '@/lib/native/share';
import { clearSecrets } from '@/lib/native/secure-storage';
import { clearMoments } from '@/lib/gallery-store';
import { signOutUser } from '@/lib/firebase/auth';
import { Companion } from '@/lib/native/companion';
export default function DataPage() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [erase, setErase] = useState(false);
  const [restore, setRestore] = useState<LocalData | null>(null);
  const input = useRef<HTMLInputElement>(null);
  async function backup() {
    setBusy(true);
    setError('');
    try {
      const data = await readLocalData();
      const blob = new Blob(
        [
          JSON.stringify(
            {
              format: 'adam-backup',
              version: 1,
              exportedAt: new Date().toISOString(),
              data: { ...data, homeAssistantUrl: '' },
            },
            null,
            2,
          ),
        ],
        { type: 'application/json' },
      );
      await shareFile(blob, 'adam-backup.json', 'ADAM memory backup');
    } catch (e) {
      if (!/cancel|abort/i.test(errorMessage(e))) setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  async function choose(file: File | undefined) {
    if (!file) return;
    setError('');
    try {
      if (file.size > 5 * 1024 * 1024) throw new Error('Choose an ADAM backup smaller than 5 MB.');
      const parsed = JSON.parse(await file.text());
      if (parsed.format !== 'adam-backup' || parsed.version !== 1)
        throw new Error('This is not a supported ADAM backup.');
      setRestore(LocalData.parse(parsed.data));
    } catch (e) {
      setError(errorMessage(e));
    }
    if (input.current) input.current.value = '';
  }
  async function apply() {
    if (!restore) return;
    setBusy(true);
    try {
      await restoreLocalData(restore);
      setRestore(null);
      setMessage('Your memories and preferences have been restored.');
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  async function clear() {
    setBusy(true);
    setError('');
    try {
      await signOutUser();
      await clearMoments();
      await clearShareCache();
      if (Capacitor.getPlatform() === 'android')
        await Companion.clearNotifications({ reset: true });
      await clearSecrets();
      if (Capacitor.isNativePlatform()) {
        const { keys } = await Preferences.keys();
        for (const key of keys.filter((k) => k.startsWith('adam.')))
          await Preferences.remove({ key });
      }
      for (const key of Object.keys(localStorage).filter((k) => k.startsWith('adam.')))
        localStorage.removeItem(key);
      window.location.replace('/welcome/');
    } catch (e) {
      setError(errorMessage(e));
      setBusy(false);
    }
  }
  return (
    <Page title="Your data" back="/settings">
      <div>
        <p className="eyebrow mb-3">PRIVATE BY DEFAULT</p>
        <h2 className="page-title">
          Yours to keep.
          <br />
          Yours to control.
        </h2>
      </div>
      <Panel>
        <Shield size={25} strokeWidth={1.3} />
        <p className="text-fg-muted mt-4 text-sm leading-7">
          Your data stays on this phone by default. In Your profile, you can choose to sync memories,
            people, planner entries, saved ADAM devices, and selected preferences with your desktop account. Photos, face profiles,
          notification history, and private keys stay on this phone.
        </p>
      </Panel>
      {error && <Notice error>{error}</Notice>}
      {message && <Notice>{message}</Notice>}
      <Panel>
        <h3 className="text-sm font-semibold">Memory backup</h3>
        <p className="text-fg-muted my-4 text-sm leading-6">
            Export your memories, planner, saved devices, name, and preferences. API keys, access tokens, photos, and face
          images are excluded. Share photos individually from Moments.
        </p>
        <div className="space-y-3">
          <Button block variant="outline" disabled={busy} onClick={backup}>
            <Download size={17} />
            Export backup
          </Button>
          <Button block variant="outline" disabled={busy} onClick={() => input.current?.click()}>
            <FileUp size={17} />
            Restore backup
          </Button>
        </div>
        <input
          ref={input}
          type="file"
          accept="application/json,.json"
          className="hidden"
          aria-label="Choose ADAM backup"
          onChange={(e) => void choose(e.target.files?.[0])}
        />
      </Panel>
      <Button block variant="outline" disabled={busy} onClick={() => setErase(true)}>
        <Trash2 size={17} />
        Erase app data
      </Button>
      <p className="text-fg-muted text-xs leading-6">
        Uninstalling also removes this app’s local data. Export anything you want to keep first.
      </p>
      {restore && (
        <Confirm
          title="Restore this backup?"
          onClose={() => setRestore(null)}
          onConfirm={apply}
          busy={busy}
        >
            Replace the memories, planner, saved devices and preferences on this phone with {restore.facts.length} memories,
            {restore.todos.length} to-dos, {restore.clocks.length} clocks and {restore.devices.length} devices from this backup? Photos and credentials will stay as they are.
          {error && <span className="mt-2 block">{error}</span>}
        </Confirm>
      )}
      {erase && (
        <Confirm
          title="Erase data from this phone?"
          onClose={() => setErase(false)}
          onConfirm={clear}
          busy={busy}
        >
          This signs you out and permanently removes this app’s memories, photos, face profile,
          settings, saved notifications, and keys. Notification reading is paused. Your cloud
          account is not deleted, and Android access can be revoked in system settings.
          {error && <span className="mt-2 block">{error}</span>}
        </Confirm>
      )}
    </Page>
  );
}
