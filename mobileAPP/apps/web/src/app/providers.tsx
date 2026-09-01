'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useEffect, useState, type ReactNode } from 'react';

import { useAppStore } from '../stores/app-store';

/**
 * Applies the persisted theme to `<html data-theme>`. Kept out of layout.tsx so
 * that file can stay a server component and keep exporting `metadata`.
 */
function ThemeSync() {
  const theme = useAppStore((state) => state.theme);

  useEffect(() => {
    document.documentElement.dataset['theme'] = theme;
  }, [theme]);

  return null;
}

export function Providers({ children }: { children: ReactNode }) {
  /**
   * One client per mount, created in state rather than at module scope: a module
   * -level client would be shared across a static export's hydration boundary and
   * leak cache between users in a shared WebView.
   */
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            // The device is on a LAN; refetching on every focus is noise.
            refetchOnWindowFocus: false,
            staleTime: 30_000,
            retry: 1,
          },
        },
      }),
  );

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeSync />
      {children}
    </QueryClientProvider>
  );
}
