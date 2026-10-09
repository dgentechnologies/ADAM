'use client';

import { AdamLogo, Button, Screen, ScreenActions, ScreenHeader, cn } from '@adam/ui';
import { Check, Mail, X, Loader2 } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useState } from 'react';
import Link from 'next/link';
import type { User } from 'firebase/auth';

import { useSetupStore } from '@/stores/setup-store';
import { CanvasRevealEffect } from '@/components/canvas-reveal-effect';

/**
 * `authentication` — "Who am I working for?".
 *
 * Provides Google Sign-In with official colorful G logo and Email option,
 * updating the user state and smoothly advancing to `/discover`.
 */
export default function SignInPage() {
  const router = useRouter();
  const setSignedIn = useSetupStore((state) => state.setSignedIn);
  const complete = useSetupStore((state) => state.complete);
  const setUserNameForFace = useSetupStore((state) => state.setUserNameForFace);
  const completedAt = useSetupStore((state) => state.completedAt);

  const [accepted, setAccepted] = useState(false);
  const [busy, setBusy] = useState(false);
  const [showEmailModal, setShowEmailModal] = useState(false);
  const [email, setEmail] = useState('');
  const [emailSent, setEmailSent] = useState(false);
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [createAccount, setCreateAccount] = useState(false);
  const [authNotice, setAuthNotice] = useState<string | null>(null);

  function finishSignIn(user: User) {
    if (user.displayName) setUserNameForFace(user.displayName);
    setSignedIn(true);
    complete('sign-in');
    router.push(completedAt ? '/settings/account' : '/discover');
  }
  async function showAuthError(error: unknown) {
    const { authError } = await import('@/lib/firebase/auth');
    setAuthNotice(authError(error));
  }

  async function handleGoogleSignIn() {
    if (!accepted || busy) return;
    setBusy(true);
    setAuthNotice(null);

    try {
      const { signInWithGoogle } = await import('@/lib/firebase/auth');
      const result = await signInWithGoogle();
      finishSignIn(result.user);
    } catch (err: unknown) {
      await showAuthError(err);
    } finally {
      setBusy(false);
    }
  }

  async function handleEmailSignIn(e: React.FormEvent) {
    e.preventDefault();
    if (!accepted || busy || !email.trim() || !password) return;
    setBusy(true);
    setAuthNotice(null);

    try {
      const { signInWithEmail } = await import('@/lib/firebase/auth');
      const result = await signInWithEmail(email.trim(), password, createAccount, name);
      finishSignIn(result.user);
    } catch (error) {
      await showAuthError(error);
    } finally {
      setBusy(false);
    }
  }

  async function handleResetPassword() {
    if (!email.trim() || busy) return;
    setBusy(true);
    setAuthNotice(null);
    try {
      const { resetPassword } = await import('@/lib/firebase/auth');
      await resetPassword(email.trim());
      setEmailSent(true);
    } catch (error) {
      await showAuthError(error);
    } finally { setBusy(false); }
  }

  function handleSkip() {
    if (busy) return;
    setSignedIn(false);
    complete('sign-in');
    router.push(completedAt ? '/settings/account' : '/discover');
  }

  return (
    <Screen id="sign-in-screen" data-page="sign-in" className="relative min-h-0 flex-1 justify-between pt-0 pb-safe pb-stack-md">
      {/* Dynamic CanvasRevealEffect Dot Matrix Background */}
      <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden [mask-image:radial-gradient(ellipse_44%_24%_at_38%_43%,black_15%,transparent_75%)]" aria-hidden="true">
        <CanvasRevealEffect
          animationSpeed={3}
          colors={[
            [255, 255, 255],
            [255, 255, 255],
          ]}
          dotSize={4}
          showGradient={true}
        />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_rgba(0,0,0,0.85)_0%,_rgba(0,0,0,0)_100%)] pointer-events-none" />
        <div className="absolute top-0 left-0 right-0 h-1/3 bg-gradient-to-b from-black to-transparent pointer-events-none" />
      </div>

      <div className="relative z-10 flex flex-1 flex-col justify-center gap-6">
        {/* Generous, beautifully padded hardware icon box for ADAM's eyes */}
        <div
          className="flex h-20 w-20 items-center justify-center overflow-hidden"
          style={{
            borderRadius: 20,
            background: 'linear-gradient(180deg, #18181b 0%, #0d0d0f 100%)',
            border: '1px solid rgba(255, 255, 255, 0.15)',
            boxShadow: '0 12px 30px rgba(0, 0, 0, 0.7), inset 0 1px 0 rgba(255, 255, 255, 0.15)',
            padding: '16px',
          }}
        >
          <AdamLogo size={64} />
        </div>
        <ScreenHeader size="md" title="Who am I working for?" />
        {authNotice && (
          <p role="alert" className="text-xs text-amber-400 bg-amber-400/10 border border-amber-400/20 rounded-lg p-2.5">
            {authNotice}
          </p>
        )}
      </div>

      <ScreenActions className="relative z-10 pt-2 gap-2.5">
        <Button
          block
          variant="primary"
          disabled={!accepted || busy}
          onClick={handleGoogleSignIn}
          className="gap-3 h-11"
        >
          {busy ? (
            <Loader2 className="h-5 w-5 animate-spin" />
          ) : (
            <svg
              viewBox="0 0 24 24"
              width="20"
              height="20"
              className="h-5 w-5 shrink-0"
              aria-hidden="true"
            >
              <path
                d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                fill="#4285F4"
              />
              <path
                d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                fill="#34A853"
              />
              <path
                d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"
                fill="#FBBC05"
              />
              <path
                d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"
                fill="#EA4335"
              />
            </svg>
          )}
          Continue with Google
        </Button>
        <Button
          block
          variant="ghost"
          size="md"
          disabled={!accepted || busy}
          onClick={() => setShowEmailModal(true)}
          className="h-11"
        >
          Use email instead
        </Button>
        <Button block variant="ghost" disabled={busy} onClick={handleSkip} className="h-10 text-xs">
          Continue on this phone
        </Button>

        <label className="mt-2 flex cursor-pointer items-start gap-stack-sm">
          <span className="relative mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center">
            <input
              type="checkbox"
              checked={accepted}
              onChange={(event) => setAccepted(event.target.checked)}
              className="peer h-6 w-6 cursor-pointer appearance-none rounded-sm border border-border-strong checked:border-fg checked:bg-fg"
            />
            <Check
              className={cn(
                'pointer-events-none absolute h-4 w-4 text-fg-inverse transition-opacity',
                accepted ? 'opacity-100' : 'opacity-0',
              )}
              strokeWidth={2.5}
              aria-hidden
            />
          </span>
          <span className="text-label-md text-fg-muted">
            By continuing you agree to DGEN’s{' '}
            <Link href="/terms" className="text-fg underline">Terms</Link> and{' '}
            <Link href="/privacy" className="text-fg underline">Privacy Policy</Link>.
          </span>
        </label>
      </ScreenActions>

      {/* Email Sign-In Modal */}
      {showEmailModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
          <div role="dialog" aria-modal="true" aria-labelledby="email-title" className="w-full max-w-sm max-h-[90dvh] overflow-y-auto rounded-2xl bg-[#141416] border border-white/10 p-6 flex flex-col gap-4 shadow-2xl relative">
            <button
              onClick={() => setShowEmailModal(false)}
              className="absolute top-4 right-4 text-white/50 hover:text-white p-1"
              aria-label="Close"
              disabled={busy}
            >
              <X size={20} />
            </button>

            <div className="flex items-center gap-3">
              <div className="h-10 w-10 rounded-full bg-white/5 border border-white/10 flex items-center justify-center">
                <Mail size={18} className="text-white" />
              </div>
              <div>
                <h3 id="email-title" className="text-base font-semibold text-white">{createAccount ? 'Create an account' : 'Sign in with Email'}</h3>
                <p className="text-xs text-white/50">Enter your email to connect with ADAM</p>
              </div>
            </div>

            {authNotice && <p role="alert" className="text-sm text-amber-400">{authNotice}</p>}
            {emailSent && (
              <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-center">
                <p className="text-sm text-emerald-400 font-medium">Check your inbox</p>
                <p className="text-xs text-emerald-400/70 mt-1">
                  If an account exists for {email}, a password-reset email is on its way. Reset your password, then sign in below.
                </p>
              </div>
            )}
              <form onSubmit={handleEmailSignIn} className="flex flex-col gap-3">
                {createAccount && <input aria-label="Your name" autoComplete="name" maxLength={80} value={name} onChange={(e) => setName(e.target.value)} placeholder="Your name" className="field" />}
                <input
                  type="email"
                  aria-label="Email"
                  autoComplete="email"
                  required
                  placeholder="name@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full rounded-xl bg-black/50 border border-white/10 px-4 py-3 text-sm text-white placeholder-white/30 outline-none focus:border-white/30"
                />
                <input type="password" aria-label="Password" autoComplete={createAccount ? 'new-password' : 'current-password'} minLength={createAccount ? 8 : undefined} required value={password} onChange={(e) => setPassword(e.target.value)} placeholder={createAccount ? 'Password (at least 8 characters)' : 'Password'} className="w-full rounded-xl bg-black/50 border border-white/10 px-4 py-3 text-sm text-white placeholder-white/30 outline-none focus:border-white/30" />
                <Button block variant="primary" type="submit" disabled={busy || !email.trim() || !password}>
                  {busy ? <Loader2 className="h-4 w-4 animate-spin mx-auto" /> : createAccount ? 'Create account' : 'Sign in'}
                </Button>
                <Button block variant="ghost" type="button" disabled={busy} onClick={() => { setCreateAccount(!createAccount); setAuthNotice(null); setEmailSent(false); }} className="text-xs text-white/70">
                  {createAccount ? 'Already have an account? Sign in' : 'Create an account'}
                </Button>
                {!createAccount && <Button block variant="ghost" type="button" disabled={busy || !email.trim()} onClick={() => void handleResetPassword()} className="text-xs text-white/70">Forgot password?</Button>}
              </form>
          </div>
        </div>
      )}
    </Screen>
  );
}
