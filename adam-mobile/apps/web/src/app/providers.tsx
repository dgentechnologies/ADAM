'use client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Capacitor } from '@capacitor/core';
import { usePathname, useRouter } from 'next/navigation';
import { useEffect, useState, type ReactNode } from 'react';
import { useAppStore } from '@/stores/app-store';
import { Companion } from '@/lib/native/companion';
import { CameraRecovery } from '@/components/camera-recovery';
import { SetupLoading } from '@/components/setup-loading';
import { AccountSyncWatch } from '@/components/account-sync-watch';
import { previousSetupHref } from '@/lib/setup-flow';
import { useSetupStore } from '@/stores/setup-store';
function NativeShell() {
  const theme = useAppStore((s) => s.theme);
  const path = usePathname().replace(/\/+$/, '') || '/';
  const router = useRouter();
  useEffect(() => {
    document.documentElement.dataset['theme'] = theme;
    if (Capacitor.isNativePlatform())
      void Companion.setAppearance({ theme }).catch(() => undefined);
    if (Capacitor.isNativePlatform())
      void import('@capacitor/status-bar')
        .then(async ({ StatusBar, Style }) => {
          await StatusBar.setOverlaysWebView({ overlay: false });
          await StatusBar.setStyle({ style: theme === 'dark' ? Style.Dark : Style.Light });
          await StatusBar.setBackgroundColor({ color: theme === 'dark' ? '#000000' : '#ffffff' });
        })
        .catch(() => undefined);
  }, [theme]);
  useEffect(() => {
    const handle = () => {
      const dialog = document.querySelector('dialog[open]');
      if (dialog) {
        dialog.dispatchEvent(new Event('cancel', { cancelable: true, bubbles: true }));
        return 'handled';
      }
      if (!window.dispatchEvent(new Event('adam:back', { cancelable: true }))) return 'handled';
      if (['/privacy', '/terms'].includes(path) && !useSetupStore.getState().completedAt) { router.replace('/sign-in'); return 'handled'; }
      const previous = previousSetupHref(path, useSetupStore.getState());
      if (previous) { router.replace(previous); return 'handled'; }
      if (path === '/' || path === '/home' || path === '/welcome' || path === '/sign-in')
        return 'minimize';
      if (path.startsWith('/settings/')) router.replace('/settings');
      else if (
        ['/settings', '/memory', '/gallery', '/smart-home', '/privacy', '/terms'].includes(path)
      )
        router.replace('/home');
      else router.replace('/home');
      return 'handled';
    };
    window.__adamHandleBack = handle;
    return () => {
      delete window.__adamHandleBack;
    };
  }, [path, router]);
  useEffect(() => {
    if (!Capacitor.isNativePlatform()) return;
    let active = true;
    const cleanup: Array<() => Promise<void>> = [];
    void import('@capacitor/keyboard').then(async ({ Keyboard }) => {
      const show = await Keyboard.addListener('keyboardDidShow', () =>
        document.documentElement.classList.add('keyboard-open'),
      );
      const hide = await Keyboard.addListener('keyboardDidHide', () =>
        document.documentElement.classList.remove('keyboard-open'),
      );
      if (!active) {
        await show.remove();
        await hide.remove();
      } else
        cleanup.push(
          () => show.remove(),
          () => hide.remove(),
        );
    }).catch(() => undefined);
    return () => {
      active = false;
      cleanup.forEach((fn) => void fn());
    };
  }, []);
  return null;
}
declare global {
  interface Window {
    __adamHandleBack?: () => string;
  }
}
export function Providers({ children }: { children: ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: { queries: { refetchOnWindowFocus: false, staleTime: 30000, retry: 1 } },
      }),
  );
  return (
    <QueryClientProvider client={client}>
      <SetupLoading>
      <NativeShell />
      <CameraRecovery />
      <AccountSyncWatch />
      {children}
      </SetupLoading>
    </QueryClientProvider>
  );
}
