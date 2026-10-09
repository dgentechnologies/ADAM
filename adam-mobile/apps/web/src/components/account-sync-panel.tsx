'use client';

import { Button } from '@adam/ui';
import { Cloud, RefreshCw } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';
import { Confirm, Notice, Panel } from './companion-ui';
import { getCompanionSync, type CompanionSyncStatus } from '../lib/firebase/companion-sync';

const INITIAL: CompanionSyncStatus = {
  enabled: false, syncing: false, pending: false, lastSynced: null, error: '',
};

export function AccountSyncPanel({ uid, email }: { uid: string; email: string }) {
  const [status, setStatus] = useState(INITIAL);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [confirm, setConfirm] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const refresh = useCallback(async () => {
    const service = await getCompanionSync();
    setStatus(await service.status());
  }, []);
  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const service = await getCompanionSync();
        const next = await service.status();
        if (active) setStatus(next);
      } catch {
        if (active) setError('Sync settings could not be loaded. Please reopen this screen.');
      } finally {
        if (active) setLoading(false);
      }
    };
    void load();
    window.addEventListener('adam:account-sync', load);
    window.addEventListener('adam:data', load);
    return () => {
      active = false;
      window.removeEventListener('adam:account-sync', load);
      window.removeEventListener('adam:data', load);
    };
  }, [uid]);
  async function action(kind: 'enable' | 'sync' | 'disable') {
    setBusy(true);
    setError('');
    setMessage('');
    try {
      const service = await getCompanionSync();
      await service[kind](uid);
      await refresh();
      setConfirm(false);
      setMessage(kind === 'enable'
        ? 'Ready to sync. Tap Sync now to save to your account.'
        : kind === 'disable'
          ? 'Sync is off. Existing phone and account data have been kept.'
          : 'Account sync complete. Memories without an ADAM selected remain on this phone.');
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Sync could not finish. Please try again.');
      await refresh().catch(() => undefined);
    } finally {
      setBusy(false);
    }
  }
  return (
    <Panel>
      <div className="flex items-center gap-3">
        <Cloud size={20} aria-hidden="true" />
        <div>
          <h3 className="text-sm font-medium">Account cloud sync</h3>
          <p className="text-fg-muted mt-1 text-xs">
            {loading ? 'Checking sync settings…' : status.enabled ? 'Connected to your account' : 'Your choice, always'}
          </p>
        </div>
      </div>
      <p className="text-fg-muted my-4 text-sm leading-6">
        Sync your named ADAM devices, to-dos and schedules with your account.
        In Memory, choose the ADAM that each memory belongs to; unassigned memories stay on this phone.
        Photos, face profiles, notifications, voice preferences and private keys stay on this phone.
        After your first sync, changes sync automatically while the app is open and online.
      </p>
      {status.enabled ? (
        <div className="space-y-3">
          <p className="text-fg-muted text-xs leading-5" role="status">
            {status.syncing ? 'Syncing your account…'
              : status.pending ? 'Phone changes are ready to sync.'
                : status.lastSynced ? `Last synced ${new Date(status.lastSynced).toLocaleString()}.`
                  : 'Tap Sync now when you are ready.'}
          </p>
          <Button block disabled={busy || status.syncing} onClick={() => void action('sync')}>
            <RefreshCw size={16} className={status.syncing ? 'animate-spin' : ''} aria-hidden="true" />
            {status.syncing ? 'Syncing…' : 'Sync now'}
          </Button>
          <Button block variant="ghost" disabled={busy || status.syncing} onClick={() => void action('disable')}>
            Turn off sync
          </Button>
        </div>
      ) : (
        <Button block variant="outline" disabled={loading || busy} onClick={() => setConfirm(true)}>
          Enable account sync
        </Button>
      )}
      {(error || status.error) && <div className="mt-3"><Notice error>{error || status.error}</Notice></div>}
      {message && <div className="mt-3"><Notice>{message}</Notice></div>}
      {confirm && (
        <Confirm title="Sync this phone with your account?" busy={busy}
          onClose={() => setConfirm(false)} onConfirm={() => void action('enable')}>
          Enable sync for <strong className="break-all">{email || 'this signed-in account'}</strong>.
          Your device-assigned memories, planner and saved ADAM devices will be included when you tap Sync now.
          Account changes and deletions will also appear here after syncing. Photos and other
          private phone data are excluded. You can turn sync off at any time.
          {error && <span className="mt-2 block">{error}</span>}
        </Confirm>
      )}
    </Panel>
  );
}
