'use client';
import { Button, IconButton } from '@adam/ui';
import { Capacitor } from '@capacitor/core';
import {
  BatteryCharging,
  Bell,
  BellOff,
  Check,
  Pause,
  Play,
  RefreshCw,
  Search,
  ShieldCheck,
  Trash2,
} from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Confirm, Dialog, Empty, Notice, Page, Panel } from '@/components/companion-ui';
import { Companion, type NotificationAccess, type PhoneNotification } from '@/lib/native/companion';
import { errorMessage } from '@/lib/local-data';

const OFF: NotificationAccess = {
  accessGranted: false,
  captureEnabled: false,
  connected: false,
  error: '',
};
export default function NotificationsPage() {
  const native = Capacitor.getPlatform() === 'android';
  const [access, setAccess] = useState(OFF);
  const [items, setItems] = useState<PhoneNotification[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [search, setSearch] = useState('');
  const [selected, setSelected] = useState<PhoneNotification | null>(null);
  const [clearing, setClearing] = useState(false);
  const enableAfterReturn = useRef(false);
  const refresh = useCallback(
    async (start = false) => {
      if (!native) return;
      try {
        let status = await Companion.notificationStatus();
        if (start && status.accessGranted && !status.captureEnabled) {
          await Companion.setNotificationCapture({ enabled: true });
          status = await Companion.notificationStatus();
        }
        setAccess(status);
        setItems((await Companion.getNotifications()).items);
        setError(status.error);
      } catch (e) {
        setError(errorMessage(e));
      }
    },
    [native],
  );
  useEffect(() => {
    void refresh();
    const timer = setInterval(() => {
      if (document.visibilityState === 'visible') void refresh();
    }, 5000);
    let disposed = false;
    let remove: (() => Promise<void>) | undefined;
    if (native)
      void import('@capacitor/app').then(async ({ App }) => {
        const listener = await App.addListener('appStateChange', ({ isActive }) => {
          if (isActive) {
            const start = enableAfterReturn.current;
            enableAfterReturn.current = false;
            void refresh(start);
          }
        });
        if (disposed) await listener.remove();
        else remove = () => listener.remove();
      });
    return () => {
      disposed = true;
      clearInterval(timer);
      void remove?.();
    };
  }, [native, refresh]);
  async function openAccess(start: boolean) {
    setError('');
    enableAfterReturn.current = start;
    try {
      await Companion.openNotificationAccessSettings();
    } catch (e) {
      enableAfterReturn.current = false;
      setError(errorMessage(e));
    }
  }
  async function toggle() {
    setBusy(true);
    setError('');
    try {
      await Companion.setNotificationCapture({ enabled: !access.captureEnabled });
      await refresh();
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  async function clearHistory() {
    setBusy(true);
    try {
      await Companion.clearNotifications({});
      setClearing(false);
      await refresh();
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }
  const running = access.accessGranted && access.captureEnabled && access.connected;
  const status = !access.accessGranted
    ? 'Access off'
    : !access.captureEnabled
      ? 'Reading paused'
      : running
        ? 'Running in background'
        : 'Waiting for Android';
  const filtered = items.filter((item) =>
    `${item.appName} ${item.title} ${item.body}`.toLowerCase().includes(search.toLowerCase()),
  );
  return (
    <Page
      title="Notifications"
      back="/settings"
      action={
        <IconButton
          size="md"
          variant="ghost"
          aria-label="Refresh notifications"
          disabled={!native || busy}
          onClick={() => void refresh()}
        >
          <RefreshCw size={19} />
        </IconButton>
      }
    >
      <div>
        <p className="eyebrow mb-3">STAY IN THE LOOP</p>
        <h2 className="page-title">A little more connected.</h2>
        <p className="text-fg-muted mt-3 text-sm leading-6">
          Your recent notifications, together in one private place.
        </p>
      </div>
      <Panel>
        <div className="flex items-center gap-3">
          <span className="border-border flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl border">
            <Bell size={22} strokeWidth={1.4} />
          </span>
          <div>
            <h3 className="text-sm font-medium">Notification access</h3>
            <p className="text-fg-muted mt-1 flex items-center gap-1.5 text-xs">
              {running && <Check size={12} />}
              {status}
            </p>
          </div>
        </div>
        <p className="text-fg-muted my-4 text-sm leading-6">
          ADAM can read notifications from your apps, including private messages. The latest 100
          stay on this phone. Nothing is sent to your robot or uploaded.
        </p>
        {!native ? (
          <Notice>Notification access is available in the Android app.</Notice>
        ) : !access.accessGranted ? (
          <>
            <Button block onClick={() => void openAccess(true)}>
              <Bell size={17} />
              Enable notification access
            </Button>
            <p className="text-fg-muted mt-3 text-xs leading-6">
              Android will open its notification access screen. Choose ADAM and confirm Allow. No
              access is granted until you approve.
            </p>
            <details className="text-fg-muted mt-3 text-xs leading-6">
              <summary className="cursor-pointer py-2">If Android blocks access</summary>
              <p>
                For an APK installation, open ADAM’s App info and use the menu to allow restricted
                settings, then return here to enable notification access. Availability depends on
                your Android version.
              </p>
              <Button
                block
                variant="outline"
                size="md"
                className="mt-3"
                onClick={() =>
                  void Companion.openAppSettings().catch((e) => setError(errorMessage(e)))
                }
              >
                Open access settings
              </Button>
            </details>
          </>
        ) : (
          <div className="space-y-2">
            <Button block variant="outline" disabled={busy} onClick={toggle}>
              {access.captureEnabled ? <Pause size={17} /> : <Play size={17} />}
              {access.captureEnabled ? 'Pause reading' : 'Start reading'}
            </Button>
            <Button block variant="ghost" size="md" onClick={() => void openAccess(false)}>
              Manage Android access
            </Button>
          </div>
        )}
      </Panel>
      <Panel>
        <div className="mb-3 flex items-center gap-3">
          <BatteryCharging size={22} strokeWidth={1.4} />
          <h3 className="text-sm font-medium">Ready in the background</h3>
        </div>
        <p className="text-fg-muted text-xs leading-6">
          When reading is enabled, Android keeps the notification service available after you leave
          ADAM. Force stopping the app or battery restrictions can pause updates. Allow background
          battery use in Android’s app settings if needed.
        </p>
        {native && (
          <Button
            block
            size="md"
            variant="outline"
            className="mt-4"
            onClick={() => void Companion.openAppSettings().catch((e) => setError(errorMessage(e)))}
          >
            Open Android app settings
          </Button>
        )}
      </Panel>
      {error && <Notice error>{error}</Notice>}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="eyebrow">RECENT NOTIFICATIONS</p>
            <p className="text-fg-muted mt-2 text-xs">{items.length} saved on this phone</p>
          </div>
          <IconButton
            size="md"
            variant="ghost"
            aria-label="Clear notification history"
            disabled={(!items.length && !error) || busy || !native}
            onClick={() => setClearing(true)}
          >
            <Trash2 size={18} />
          </IconButton>
        </div>
        {items.length > 0 && (
          <label className="relative block">
            <Search size={17} className="text-fg-muted absolute left-4 top-4" />
            <input
              aria-label="Search notifications"
              className="field pl-11"
              placeholder="Search apps or messages"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </label>
        )}
        {!filtered.length ? (
          <Empty
            icon={search ? Search : BellOff}
            title={search ? 'Nothing found' : 'A quiet inbox'}
          >
            {search
              ? 'Try another app name or word.'
              : 'New notifications appear here after you enable access and start reading. Android may hide sensitive content.'}
          </Empty>
        ) : (
          <div className="space-y-3">
            {filtered.map((item) => (
              <button
                key={item.id}
                className="panel block w-full text-left"
                onClick={() => setSelected(item)}
                aria-label={`Read ${item.title || item.appName}`}
              >
                <div className="mb-3 flex items-center gap-2">
                  <Bell size={14} className="text-fg-muted shrink-0" />
                  <span className="min-w-0 flex-1 truncate text-xs font-medium">
                    {item.appName}
                  </span>
                  <time className="text-fg-muted text-[10px]">
                    {new Date(item.postedAt).toLocaleTimeString([], {
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </time>
                </div>
                <h3 className="break-words text-sm font-medium">{item.title || 'Notification'}</h3>
                <p className="text-fg-muted mt-2 line-clamp-3 break-words text-xs leading-6">
                  {item.body}
                </p>
              </button>
            ))}
          </div>
        )}
      </section>
      <p className="text-fg-muted flex items-start gap-2 text-xs leading-6">
        <ShieldCheck size={16} className="mt-1 shrink-0" />
        You control access in Android. Clearing history here does not dismiss notifications in other
        apps.
      </p>
      {selected && (
        <Dialog title={selected.title || selected.appName} onClose={() => setSelected(null)}>
          <p className="text-fg-muted text-xs">
            {selected.appName} · {new Date(selected.postedAt).toLocaleString()}
          </p>
          <p className="whitespace-pre-wrap break-words text-sm leading-7">
            {selected.body || 'No preview text was shared by this app.'}
          </p>
          <Button block variant="outline" onClick={() => setSelected(null)}>
            Close
          </Button>
        </Dialog>
      )}
      {clearing && (
        <Confirm
          title="Clear notification history?"
          busy={busy}
          onClose={() => setClearing(false)}
          onConfirm={clearHistory}
        >
          Remove all saved notification content from this phone? New notifications will still be
          saved while reading is enabled.{error && <span className="mt-2 block">{error}</span>}
        </Confirm>
      )}
    </Page>
  );
}
