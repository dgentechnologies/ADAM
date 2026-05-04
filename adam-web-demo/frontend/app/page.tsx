'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { doc, getDoc } from 'firebase/firestore';
import { useAuth } from '@/components/FirebaseAuthProvider';
import { ADAMLoginPage } from '@/components/ui/adam-login-page';
import { getClientDb } from '@/lib/firebase';

export default function HomePage() {
  const { user, loading, signInWithGoogle, signInWithEmail, signUpWithEmail } = useAuth();
  const router = useRouter();
  const [authError, setAuthError] = useState('');
  const [authLoading, setAuthLoading] = useState(false);

  // Already logged in → check onboarding → redirect
  useEffect(() => {
    if (loading || !user) return;
    getDoc(doc(getClientDb(), 'onboarding', user.uid))
      .then((snap) => {
        router.replace(snap.exists() && snap.data()?.completed ? '/demo' : '/form');
      })
      .catch(() => router.replace('/demo'));
  }, [user, loading, router]);

  const handleGoogle = async () => {
    setAuthError('');
    setAuthLoading(true);
    try { await signInWithGoogle(); }
    catch { setAuthError('Google sign-in failed. Please try again.'); }
    finally { setAuthLoading(false); }
  };

  const handleEmailSubmit = async (
    email: string,
    password: string,
    mode: 'signin' | 'signup',
    name?: string,
  ) => {
    setAuthError('');
    setAuthLoading(true);
    try {
      if (mode === 'signin') await signInWithEmail(email, password);
      else                   await signUpWithEmail(email, password, name);
    } catch (err: unknown) {
      const code = (err as { code?: string }).code ?? '';
      if (code === 'auth/user-not-found' || code === 'auth/wrong-password' || code === 'auth/invalid-credential')
        setAuthError('Invalid email or password.');
      else if (code === 'auth/email-already-in-use')
        setAuthError('Account already exists — sign in instead.');
      else if (code === 'auth/weak-password')
        setAuthError('Password must be at least 6 characters.');
      else if (code === 'auth/invalid-email')
        setAuthError('Enter a valid email address.');
      else
        setAuthError('Authentication failed. Please try again.');
    } finally { setAuthLoading(false); }
  };

  // While auth is loading or user is being redirected, render nothing
  if (loading || user) return null;

  return (
    <ADAMLoginPage
      onGoogleSignIn={handleGoogle}
      onEmailSubmit={handleEmailSubmit}
      externalError={authError}
      externalLoading={authLoading}
    />
  );
}
