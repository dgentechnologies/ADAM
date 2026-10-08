'use client';

import { App, type RestoredListenerEvent } from '@capacitor/app';
import { Capacitor } from '@capacitor/core';
import { AlertCircle, Camera, Loader2, X } from 'lucide-react';
import Link from 'next/link';
import { useCallback, useEffect, useState } from 'react';
import { addMoment } from '@/lib/gallery-store';
import { errorMessage } from '@/lib/local-data';

const recoveries = new Map<string, Promise<void>>();

function saveRecoveredPhoto(webPath: string): Promise<void> {
  const existing = recoveries.get(webPath);
  if (existing) return existing;
  const work = (async () => {
    const url = new URL(webPath, window.location.origin);
    if (url.origin !== window.location.origin)
      throw new Error(
        'The camera photo is no longer available on this phone. Please take it again.',
      );
    const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(url.href));
    const photoId = `recovered-${Array.from(new Uint8Array(digest), (byte) =>
      byte.toString(16).padStart(2, '0'),
    ).join('')}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch(url.href, { signal: controller.signal });
      if (!response.ok)
        throw new Error('The camera photo could not be opened. Please take it again.');
      // A restored native result may be delivered again after another restart. A stable
      // ID makes saving it idempotent, including when the first save already committed.
      await addMoment(await response.blob(), 'Recovered camera photo', {
        id: photoId,
      });
      window.dispatchEvent(new Event('adam:moments-changed'));
    } finally {
      clearTimeout(timer);
    }
  })();
  recoveries.set(webPath, work);
  void work.catch(() => recoveries.delete(webPath));
  if (recoveries.size > 16) recoveries.delete(recoveries.keys().next().value!);
  return work;
}

type RecoveryNotice = {
  kind: 'saving' | 'saved' | 'error';
  message: string;
  retryPath?: string;
};

/** Recovers an external camera result if Android recreated the app in the meantime. */
export function CameraRecovery() {
  const [notice, setNotice] = useState<RecoveryNotice | null>(null);
  const recover = useCallback(async (webPath: string) => {
    setNotice({ kind: 'saving', message: 'Recovering your camera photo…' });
    try {
      await saveRecoveredPhoto(webPath);
      setNotice({
        kind: 'saved',
        message:
          'Your photo is safe in Moments after ADAM restarted. For face setup, review or retake your photos in Face profile.',
      });
    } catch (error) {
      setNotice({
        kind: 'error',
        message: `Your camera photo has not been saved. ${errorMessage(error)}`,
        retryPath: webPath,
      });
    }
  }, []);

  useEffect(() => {
    if (Capacitor.getPlatform() !== 'android') return;
    let disposed = false;
    const listener = App.addListener('appRestoredResult', (result: RestoredListenerEvent) => {
      if (disposed || result.pluginId !== 'Camera' || result.methodName !== 'getPhoto') return;
      if (!result.success) {
        if (/cancel/i.test(result.error?.message ?? '')) return;
        setNotice({
          kind: 'error',
          message: 'The camera closed before your photo could be saved. Please take it again.',
        });
        return;
      }
      const webPath: unknown = result.data?.webPath;
      if (typeof webPath !== 'string' || !webPath || webPath.length > 4096) {
        setNotice({
          kind: 'error',
          message: 'The camera did not return a recoverable photo. Please take it again.',
        });
        return;
      }
      void recover(webPath);
    });
    // Registering a result listener never requests camera or other permissions.
    void listener.catch(() => undefined);
    return () => {
      disposed = true;
      void listener.then((handle) => handle.remove()).catch(() => undefined);
    };
  }, [recover]);

  if (!notice) return null;
  const Icon = notice.kind === 'saving' ? Loader2 : notice.kind === 'saved' ? Camera : AlertCircle;
  return (
    <aside
      className="panel fixed inset-x-4 top-[calc(76px+env(safe-area-inset-top,0px))] z-[60] mx-auto max-w-lg shadow-xl"
      role={notice.kind === 'error' ? 'alert' : 'status'}
    >
      <div className="flex items-start gap-3">
        <Icon
          size={20}
          className={`mt-0.5 shrink-0 ${notice.kind === 'saving' ? 'animate-spin' : ''}`}
          aria-hidden
        />
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium">
            {notice.kind === 'saved'
              ? 'Photo recovered'
              : notice.kind === 'saving'
                ? 'One moment'
                : 'Camera recovery'}
          </p>
          <p className="text-fg-muted mt-2 text-xs leading-6">{notice.message}</p>
          {notice.kind === 'saved' && (
            <Link
              href="/gallery"
              className="mt-2 inline-flex min-h-11 items-center text-xs font-medium underline underline-offset-4"
              onClick={() => setNotice(null)}
            >
              Review in Moments
            </Link>
          )}
          {notice.kind === 'error' && notice.retryPath && (
            <button
              className="mt-2 inline-flex min-h-11 items-center text-xs font-medium underline underline-offset-4"
              onClick={() => void recover(notice.retryPath!)}
            >
              Try recovery again
            </button>
          )}
        </div>
        <button
          type="button"
          aria-label="Dismiss camera recovery notice"
          className="-mr-2 -mt-2 flex h-11 w-11 shrink-0 items-center justify-center"
          onClick={() => setNotice(null)}
        >
          <X size={18} />
        </button>
      </div>
    </aside>
  );
}
