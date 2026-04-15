import type { Metadata } from 'next';
import { WaitlistForm } from '@/components/waitlist/WaitlistForm';

export const metadata: Metadata = {
  title: 'ADAM Waitlist — Be First',
  description: 'Join the ADAM waitlist. Get early access when the hardware launches from DGEN Technologies.',
};

export default function WaitlistPage() {
  return (
    <div className="min-h-[80vh] flex flex-col items-center justify-center px-6 py-16">
      <div className="max-w-lg w-full space-y-8">
        <div className="text-center space-y-3">
          <p className="text-sky-400 text-sm font-semibold uppercase tracking-widest">
            Early Access
          </p>
          <h1 className="text-4xl font-black">Join the ADAM Waitlist</h1>
          <p className="text-gray-400">
            ADAM ships soon. Sign up to be notified first — no spam, no dates, just DGEN.
          </p>
        </div>

        <WaitlistForm />

        <p className="text-center text-xs text-gray-600">
          By signing up you agree to our{' '}
          <a href="/privacy-policy" className="underline hover:text-gray-400">
            Privacy Policy
          </a>
          .
        </p>
      </div>
    </div>
  );
}
