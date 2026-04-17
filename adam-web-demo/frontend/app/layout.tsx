import type { Metadata } from 'next';
import './globals.css';
import { FirebaseAuthProvider } from '@/components/FirebaseAuthProvider';

export const metadata: Metadata = {
  title: 'ADAM — Live AI Demo | DGEN Technologies',
  description:
    'Talk to ADAM — the Autonomous Desktop AI Module by DGEN Technologies. Powered by Gemini Live. Free 5-minute session.',
  metadataBase: new URL('https://dgentechnologies.com'),
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#0a0a0a] text-white">
        <FirebaseAuthProvider>
          {children}
        </FirebaseAuthProvider>
      </body>
    </html>
  );
}
