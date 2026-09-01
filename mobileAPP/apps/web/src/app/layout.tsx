import type { Metadata, Viewport } from 'next';
import { fontVariables } from '@/lib/fonts';
import { Providers } from './providers';
import './globals.css';

export const metadata: Metadata = {
  title: 'ADAM',
  applicationName: 'ADAM',
  description: 'Companion app for ADAM — an AI desk companion by DGEN Technologies.',
  manifest: '/manifest.webmanifest',
  appleWebApp: {
    capable: true,
    title: 'ADAM',
    statusBarStyle: 'black-translucent',
  },
  formatDetection: {
    telephone: false,
    date: false,
    address: false,
    email: false,
  },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  /** Lets content sit under the notch; safe-area utilities pad it back. */
  viewportFit: 'cover',
  themeColor: '#000000',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    /**
     * data-theme is set here rather than resolved from the OS: dark is the
     * default and primary experience, and light mode is an explicit user choice
     * in Settings (spec / DESIGN.md). suppressHydrationWarning covers the
     * client-side theme swap once that toggle lands.
     */
    <html lang="en" data-theme="dark" suppressHydrationWarning>
      <body className={`${fontVariables} font-sans antialiased`}>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
