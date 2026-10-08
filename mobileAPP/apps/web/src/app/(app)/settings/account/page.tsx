'use client';
import { Button } from '@adam/ui';
import { LogOut, UserRound } from 'lucide-react';
import { deleteUser, updateProfile } from 'firebase/auth';
import { deleteDoc, doc } from 'firebase/firestore';
import type { User } from 'firebase/auth';
import { useEffect, useState } from 'react';
import Link from 'next/link';
import { Confirm, Loading, Notice, Page, Panel } from '@/components/companion-ui';
import { AccountSyncPanel } from '@/components/account-sync-panel';
import { subscribeToAuthState, signOutUser } from '@/lib/firebase/auth';
import { getFirebaseFirestore } from '@/lib/firebase/config';
import { getCompanionSync } from '@/lib/firebase/companion-sync';
import { errorMessage, updateLocalData } from '@/lib/local-data';
import { useLocalData } from '@/lib/use-local-data';
export default function AccountPage() {
  const { data, loading: localLoading } = useLocalData();
  const [user, setUser] = useState<User | null>(null);
  const [checking, setChecking] = useState(true);
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [remove, setRemove] = useState(false);
  useEffect(() => {
    setName(data.name);
  }, [data.name]);
  useEffect(() => {
    let unsubscribe = () => {};
    try {
      unsubscribe = subscribeToAuthState((u) => {
        setUser(u);
        setChecking(false);
      });
    } catch {
      setChecking(false);
    }
    return unsubscribe;
  }, []);
  async function save(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError('');
    setMessage('');
    try {
      await updateLocalData((d) => ({ ...d, name: name.trim() }));
      if (user) await updateProfile(user, { displayName: name.trim() });
      setMessage('Your profile is saved.');
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  async function logout() {
    setBusy(true);
    setError('');
    try {
      await signOutUser();
      setMessage('Signed out. Your private data remains on this phone.');
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  async function deleteAccount() {
    if (!user) return;
    setBusy(true);
    setError('');
    try {
      // Fence any in-flight sync before deleting the shared document.
      await (await getCompanionSync()).disable(user.uid);
      await deleteDoc(doc(getFirebaseFirestore(), 'users', user.uid));
      await deleteUser(user);
      setRemove(false);
      setMessage('Your account has been deleted. Local memories remain on this phone.');
    } catch {
      setError(
        'Account deletion could not finish. Sign in again, then retry while connected to the internet.',
      );
    } finally {
      setBusy(false);
    }
  }
  return (
    <Page title="Your profile" back="/settings">
      <Panel>
        <div className="flex items-center gap-4">
          <span className="icon-orbit !m-0 !h-14 !w-14">
            <UserRound size={25} />
          </span>
          <div>
            <h2 className="text-lg font-medium">
              {data.name || user?.displayName || 'Make it personal'}
            </h2>
            <p className="text-fg-muted mt-1 break-all text-xs">
              {user?.email || 'Private profile on this phone'}
            </p>
          </div>
        </div>
      </Panel>
      {localLoading ? (
        <Loading />
      ) : (
        <form onSubmit={save} className="space-y-5">
          <label className="field-label">
            What should we call you?
            <input
              className="field"
              autoComplete="name"
              maxLength={80}
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Your name"
              required
            />
          </label>
          <Button block type="submit" disabled={busy || !name.trim()}>
            {busy ? 'Saving…' : 'Save profile'}
          </Button>
        </form>
      )}
      {error && <Notice error>{error}</Notice>}
      {message && <Notice>{message}</Notice>}
      <Panel>
        <h3 className="text-sm font-medium">Your account</h3>
        <p className="text-fg-muted my-3 text-sm leading-6">
          {checking
            ? 'Checking your session…'
            : user
              ? 'You are signed in. Choose below whether to sync memories with your desktop.'
              : 'Sign-in is optional. Your memories, photos, and settings work without an account.'}
        </p>
        {user ? (
          <Button block variant="outline" disabled={busy} onClick={logout}>
            <LogOut size={16} />
            Sign out
          </Button>
        ) : (
          <Link
            href="/sign-in"
            className="border-border-strong block rounded-full border px-5 py-3 text-center text-sm"
          >
            Sign in or create account
          </Link>
        )}
      </Panel>
      {user && <AccountSyncPanel key={user.uid} uid={user.uid} email={user.email ?? ''} />}
      {user && (
        <Button variant="ghost" disabled={busy} onClick={() => setRemove(true)}>
          Delete account
        </Button>
      )}
      {remove && (
        <Confirm
          title="Delete your account?"
          busy={busy}
          onClose={() => setRemove(false)}
          onConfirm={deleteAccount}
        >
          This permanently deletes your sign-in account, cloud profile, and synced memories. Local data remains until
          you erase it in Your data.{error && <span className="mt-2 block">{error}</span>}
        </Confirm>
      )}
    </Page>
  );
}
