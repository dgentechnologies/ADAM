'use client';

import { useEffect, useState } from 'react';
import type { User } from 'firebase/auth';
import { useAuth } from '@/components/FirebaseAuthProvider';
import { DemoSession } from '@/components/adam/DemoSession';
import { AdamFace } from '@/components/adam/AdamFace';
import Image from 'next/image';

export default function AdamDemoPage() {
  const { user, loading, signInWithGoogle } = useAuth();
  const [signingIn, setSigningIn] = useState(false);
  const [authError, setAuthError] = useState('');

  // Update document title client-side (page is client component)
  useEffect(() => {
    document.title = 'ADAM Demo — Live Voice Session | DGEN Technologies';
  }, []);

  const handleSignIn = async () => {
    setSigningIn(true);
    setAuthError('');
    try {
      await signInWithGoogle();
    } catch (err) {
      setAuthError('Sign-in failed. Please try again.');
      console.error('[demo] Sign-in error:', err);
    } finally {
      setSigningIn(false);
    }
  };

  // ── Loading ───────────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="min-h-[80vh] flex flex-col items-center justify-center gap-4 text-gray-400">
        <div className="w-8 h-8 border-2 border-sky-400 border-t-transparent rounded-full animate-spin" />
        <p className="text-sm">Loading…</p>
      </div>
    );
  }

  // ── Not signed in ─────────────────────────────────────────────────────────
  if (!user) {
    return (
      <div className="min-h-[80vh] flex flex-col items-center justify-center px-6">
        <div className="max-w-md w-full text-center space-y-8">
          <div className="flex justify-center">
            <AdamFace emotion="idle" faceState="idle" size={160} />
          </div>

          <div className="space-y-3">
            <h1 className="text-3xl font-black">
              Talk to <span className="text-sky-400">ADAM</span>
            </h1>
            <p className="text-gray-400">
              Sign in with Google to start a live voice session. 5 minutes · 20 turns · Free.
            </p>
          </div>

          <div className="space-y-3">
            <button
              onClick={handleSignIn}
              disabled={signingIn}
              className="w-full flex items-center justify-center gap-3 px-6 py-3 bg-white hover:bg-gray-100 disabled:opacity-60 text-gray-900 rounded-xl font-semibold text-sm transition"
            >
              {signingIn ? (
                <div className="w-5 h-5 border-2 border-gray-400 border-t-transparent rounded-full animate-spin" />
              ) : (
                <Image src="/images/google-logo.svg" alt="Google" width={20} height={20} />
              )}
              {signingIn ? 'Signing in…' : 'Continue with Google'}
            </button>

            {authError && (
              <p className="text-red-400 text-sm">{authError}</p>
            )}
          </div>

          <p className="text-xs text-gray-600">
            By signing in you agree to our{' '}
            <a href="/privacy-policy" className="underline hover:text-gray-400">
              Privacy Policy
            </a>
            . We only use your Google profile to identify your session.
          </p>
        </div>
      </div>
    );
  }

  // ── Signed in — show demo ─────────────────────────────────────────────────
  return (
    <div className="min-h-[90vh] flex flex-col items-center justify-center px-4 py-12">
      <div className="w-full max-w-2xl space-y-4">
        <div className="text-center space-y-1 mb-6">
          <h1 className="text-3xl font-black tracking-tight">
            ADAM <span className="text-sky-400">Live</span>
          </h1>
          <p className="text-gray-400 text-sm">
            5-minute session · 20 turns · Powered by Gemini Live
          </p>
        </div>

        <DemoSession user={user} />

        <p className="text-center text-xs text-gray-600 pt-4">
          Signed in as {user.email}
        </p>
      </div>
    </div>
  );
}
