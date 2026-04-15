import type { Metadata } from 'next';
import './globals.css';
import { Navbar } from '@/components/Navbar';
import { Footer } from '@/components/Footer';
import { FirebaseAuthProvider } from '@/components/FirebaseAuthProvider';

export const metadata: Metadata = {
  title: {
    default:  'DGEN Technologies Pvt. Ltd.',
    template: '%s | DGEN Technologies',
  },
  description:
    'DGEN Technologies — Innovate. Integrate. Inspire. Smart city lighting, IoT, AI, and ADAM: the Autonomous Desktop AI Module. Made in India.',
  metadataBase: new URL('https://dgentechnologies.com'),
  openGraph: { siteName: 'DGEN Technologies', locale: 'en_IN' },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen flex flex-col bg-[#0a0a0a] text-white">
        <FirebaseAuthProvider>
          <Navbar />
          <main className="flex-1">{children}</main>
          <Footer />
        </FirebaseAuthProvider>
      </body>
    </html>
  );
}
