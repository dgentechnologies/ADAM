'use client';

import { useEffect, useState } from 'react';
import Image from 'next/image';
import { doc, getDoc } from 'firebase/firestore';
import { useAuth } from '@/components/FirebaseAuthProvider';
import { DemoSession } from '@/components/adam/DemoSession';
import { AdamFace } from '@/components/adam/AdamFace';
import { OnboardingForm } from '@/components/adam/OnboardingForm';
import { DemoAuthGate } from '@/components/adam/DemoAuthGate';
import { db } from '@/lib/firebase';

const WAITLIST_URL = 'https://dgentechnologies.com/adam#waitlist';

type Step        = 'auth' | 'checking' | 'onboarding' | 'demo';
type EmailMode   = 'signin' | 'signup';
type DemoOverlay = 'welcome' | 'active' | 'ended';

const FONTS = `@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;600&family=Share+Tech+Mono&family=DM+Sans:wght@300;400;500&display=swap');`;

// ── Shared input styles ───────────────────────────────────────────────────────
const INPUT: React.CSSProperties = {
  width: '100%', padding: '12px 16px',
  background: 'rgba(255,255,255,0.04)',
  border: '1px solid rgba(255,255,255,0.09)',
  borderRadius: 10, color: '#f0f0f0',
  fontFamily: '"DM Sans", sans-serif', fontSize: 14,
  outline: 'none', boxSizing: 'border-box',
  transition: 'border-color 0.2s',
};

// ── Spinner ───────────────────────────────────────────────────────────────────
function Spinner({ label }: { label?: string }) {
  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: '#0a0a0a', gap: 20 }}>
      <div style={{ filter: 'drop-shadow(0 0 18px rgba(74,240,255,0.3))' }}>
        <AdamFace emotion="idle" faceState="idle" size={160} />
      </div>
      <div style={{ width: 24, height: 24, border: '1.5px solid rgba(74,240,255,0.3)', borderTopColor: '#4AF0FF', borderRadius: '50%', animation: 'pgSpin 0.8s linear infinite' }} />
      {label && <p style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 11, color: '#444', letterSpacing: '0.12em' }}>{label}</p>}
      <style>{`@keyframes pgSpin { to { transform: rotate(360deg); } } ${FONTS}`}</style>
    </div>
  );
}

// ── Welcome overlay ───────────────────────────────────────────────────────────
function WelcomeOverlay({ onDismiss }: { onDismiss: () => void }) {
  const [visible, setVisible] = useState(false);
  useEffect(() => { const t = setTimeout(() => setVisible(true), 60); return () => clearTimeout(t); }, []);

  return (
    <div
      style={{
        position: 'fixed', inset: 0, zIndex: 200,
        background: '#0a0a0a',
        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
        padding: '32px 24px',
        opacity: visible ? 1 : 0,
        transition: 'opacity 0.5s ease',
        overflow: 'hidden',
      }}
    >
      {/* Ambient glow behind face */}
      <div style={{ position: 'absolute', top: '35%', left: '50%', transform: 'translate(-50%, -50%)', width: 520, height: 520, borderRadius: '50%', background: 'radial-gradient(circle, rgba(74,240,255,0.06) 0%, transparent 65%)', pointerEvents: 'none' }} />

      <div style={{ width: '100%', maxWidth: 420, textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 24, position: 'relative', zIndex: 1 }}>

        {/* DGEN badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '5px 12px', background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 20 }}>
          <span style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 9, color: '#555', letterSpacing: '0.14em' }}>DGEN TECHNOLOGIES</span>
        </div>

        {/* Animated face */}
        <div style={{ filter: 'drop-shadow(0 0 28px rgba(74,240,255,0.4))' }}>
          <AdamFace emotion="happy" faceState="idle" size={190} />
        </div>

        {/* Greeting */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <h1 style={{ fontFamily: '"Rajdhani", sans-serif', fontWeight: 600, fontSize: 40, letterSpacing: '0.06em', color: '#f0f0f0', margin: 0 }}>
            Hello. I&apos;m <span style={{ color: '#4AF0FF' }}>ADAM</span>.
          </h1>
          <p style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 10, color: '#444', letterSpacing: '0.1em', margin: 0 }}>
            AUTONOMOUS DESKTOP AI MODULE
          </p>
        </div>

        {/* Message */}
        <div style={{ background: 'rgba(255,255,255,0.025)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 16, padding: '20px 22px', textAlign: 'left' }}>
          <p style={{ fontFamily: '"DM Sans", sans-serif', fontSize: 14, lineHeight: 1.8, color: '#aaa', margin: 0 }}>
            I can hear your voice, process what you say in real time, and respond like a person — because I&apos;m powered by{' '}
            <span style={{ color: '#f0f0f0', fontWeight: 500 }}>Gemini Live</span>. You have{' '}
            <span style={{ color: '#4AF0FF', fontWeight: 500 }}>5 minutes</span> with me right now.
          </p>
          <p style={{ fontFamily: '"DM Sans", sans-serif', fontSize: 14, lineHeight: 1.8, color: '#aaa', margin: '10px 0 0' }}>
            Hold the <span style={{ color: '#4AF0FF', fontWeight: 500 }}>mic button</span> and speak clearly. Release when you&apos;re done.
          </p>
        </div>

        {/* CTA */}
        <button
          onClick={onDismiss}
          style={{
            width: '100%', padding: '15px 0',
            background: '#4AF0FF', color: '#0a0a0a',
            border: 'none', borderRadius: 14,
            fontFamily: '"Rajdhani", sans-serif', fontWeight: 600,
            fontSize: 16, letterSpacing: '0.09em',
            cursor: 'pointer',
            boxShadow: '0 0 32px rgba(74,240,255,0.35)',
            transition: 'transform 0.1s ease, box-shadow 0.2s ease',
          }}
          onMouseEnter={(e) => { e.currentTarget.style.boxShadow = '0 0 48px rgba(74,240,255,0.55)'; e.currentTarget.style.transform = 'translateY(-1px)'; }}
          onMouseLeave={(e) => { e.currentTarget.style.boxShadow = '0 0 32px rgba(74,240,255,0.35)'; e.currentTarget.style.transform = 'translateY(0)'; }}
        >
          UNDERSTOOD →
        </button>

        <p style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 9, color: 'rgba(255,255,255,0.15)', letterSpacing: '0.08em' }}>
          MICROPHONE ACCESS REQUIRED
        </p>
      </div>
      <style>{FONTS}</style>
    </div>
  );
}

// ── End overlay ───────────────────────────────────────────────────────────────
function EndOverlay({ reason }: { reason: string }) {
  const isTimeout  = reason === 'timeout' || reason === 'cap_reached';
  const isUserExit = reason === 'user_disconnect';

  return (
    <div
      style={{
        position: 'fixed', inset: 0, zIndex: 200,
        background: '#0a0a0a',
        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
        padding: '32px 24px',
      }}
    >
      {/* Subtle ambient */}
      <div style={{ position: 'absolute', top: '40%', left: '50%', transform: 'translate(-50%,-50%)', width: 400, height: 400, borderRadius: '50%', background: 'radial-gradient(circle, rgba(74,240,255,0.04) 0%, transparent 65%)', pointerEvents: 'none' }} />

      <div style={{ width: '100%', maxWidth: 380, textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 22, position: 'relative', zIndex: 1 }}>

        <div style={{ filter: 'drop-shadow(0 0 20px rgba(74,240,255,0.2))' }}>
          <AdamFace emotion="sad" faceState="idle" size={150} />
        </div>

        <div>
          <p style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 10, color: '#444', letterSpacing: '0.12em', margin: '0 0 8px' }}>
            {isTimeout ? 'PREVIEW ENDED' : isUserExit ? 'SESSION CLOSED' : 'SESSION ENDED'}
          </p>
          <h2 style={{ fontFamily: '"Rajdhani", sans-serif', fontWeight: 600, fontSize: 30, letterSpacing: '0.05em', color: '#f0f0f0', margin: 0 }}>
            {isTimeout
              ? "That's your 5-minute preview."
              : isUserExit
              ? 'You ended the session.'
              : 'Session complete.'}
          </h2>
          <p style={{ fontFamily: '"DM Sans", sans-serif', fontSize: 14, color: '#666', marginTop: 10, lineHeight: 1.7 }}>
            {isTimeout
              ? 'Want ADAM on your desk, available all day? Be first when it ships.'
              : 'Ready to own one? Join the waitlist now.'}
          </p>
        </div>

        {/* Waitlist CTA */}
        <a
          href={WAITLIST_URL}
          target="_top"
          style={{
            display: 'block', width: '100%', padding: '15px 0',
            background: '#4AF0FF', color: '#0a0a0a',
            borderRadius: 14, textDecoration: 'none',
            fontFamily: '"Rajdhani", sans-serif', fontWeight: 600,
            fontSize: 15, letterSpacing: '0.09em', textAlign: 'center',
            boxShadow: '0 0 32px rgba(74,240,255,0.3)',
            transition: 'box-shadow 0.2s',
          }}
          onMouseEnter={(e) => { (e.currentTarget as HTMLAnchorElement).style.boxShadow = '0 0 48px rgba(74,240,255,0.5)'; }}
          onMouseLeave={(e) => { (e.currentTarget as HTMLAnchorElement).style.boxShadow = '0 0 32px rgba(74,240,255,0.3)'; }}
        >
          → JOIN THE ADAM WAITLIST
        </a>

        {/* Try again */}
        <button
          onClick={() => window.location.reload()}
          style={{
            background: 'transparent',
            border: '1px solid rgba(255,255,255,0.08)',
            color: '#444', borderRadius: 12,
            padding: '11px 32px', cursor: 'pointer',
            fontFamily: '"Share Tech Mono", monospace', fontSize: 11,
            letterSpacing: '0.07em',
            transition: 'border-color 0.2s, color 0.2s',
          }}
          onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.22)'; e.currentTarget.style.color = '#f0f0f0'; }}
          onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)'; e.currentTarget.style.color = '#444'; }}
        >
          TRY AGAIN ↺
        </button>
      </div>
      <style>{FONTS}</style>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function AdamDemoPage() {
  const { user, loading, signInWithGoogle, signInWithEmail, signUpWithEmail } = useAuth();

  const [step,        setStep]        = useState<Step>('auth');
  const [demoOverlay, setDemoOverlay] = useState<DemoOverlay>('welcome');
  const [endReason,   setEndReason]   = useState('');

  // Auth form
  const [emailMode, setEmailMode] = useState<EmailMode>('signup');
  const [name,      setName]      = useState('');
  const [email,     setEmail]     = useState('');
  const [password,  setPassword]  = useState('');
  const [busy,      setBusy]      = useState(false);
  const [error,     setError]     = useState('');

  useEffect(() => { document.title = 'ADAM Live Demo — DGEN Technologies'; }, []);

  // After auth, check onboarding
  useEffect(() => {
    if (!user || loading) return;
    setStep('checking');
    getDoc(doc(db, 'onboarding', user.uid))
      .then((snap) => setStep(snap.exists() && snap.data()?.completed ? 'demo' : 'onboarding'))
      .catch(() => setStep('demo'));
  }, [user, loading]);

  const clearError = () => setError('');

  const handleGoogle = async () => {
    setBusy(true); clearError();
    try { await signInWithGoogle(); }
    catch { setError('Google sign-in failed. Please try again.'); }
    finally { setBusy(false); }
  };

  const handleEmailSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (emailMode === 'signup' && !name.trim()) { setError('Please enter your name.'); return; }
    if (!email.trim() || !password) { setError('Enter email and password.'); return; }
    setBusy(true); clearError();
    try {
      if (emailMode === 'signin') await signInWithEmail(email, password);
      else                        await signUpWithEmail(email, password, name.trim());
    } catch (err: unknown) {
      const code = (err as { code?: string }).code ?? '';
      if (code === 'auth/user-not-found' || code === 'auth/wrong-password' || code === 'auth/invalid-credential')
        setError('Invalid email or password.');
      else if (code === 'auth/email-already-in-use')
        setError('Account already exists — sign in instead.');
      else if (code === 'auth/weak-password')
        setError('Password must be at least 6 characters.');
      else if (code === 'auth/invalid-email')
        setError('Enter a valid email address.');
      else
        setError('Authentication failed. Please try again.');
    } finally { setBusy(false); }
  };

  // ── Loading ───────────────────────────────────────────────────────────────
  if (loading)              return <Spinner />;
  if (step === 'checking')  return <Spinner label="PREPARING ADAM…" />;

  // ── Auth gate ─────────────────────────────────────────────────────────────
  if (!user) {
    return (
      <DemoAuthGate
        busy={busy}
        error={error}
        emailMode={emailMode}
        name={name}
        email={email}
        password={password}
        onGoogleClick={handleGoogle}
        onEmailSubmit={handleEmailSubmit}
        onNameChange={setName}
        onEmailChange={setEmail}
        onPasswordChange={setPassword}
        onModeToggle={() => { setEmailMode(emailMode === 'signup' ? 'signin' : 'signup'); clearError(); }}
      />
    );
  }

  // ── Onboarding ────────────────────────────────────────────────────────────
  if (step === 'onboarding') {
    return <OnboardingForm user={user} onComplete={() => setStep('demo')} />;
  }

  // ── Demo — welcome overlay ────────────────────────────────────────────────
  if (demoOverlay === 'welcome') {
    return <WelcomeOverlay onDismiss={() => setDemoOverlay('active')} />;
  }

  // ── Demo — active session ─────────────────────────────────────────────────
  if (demoOverlay === 'active') {
    return (
      <DemoSession
        user={user}
        fullscreen
        onSessionEnded={(reason) => { setEndReason(reason); setDemoOverlay('ended'); }}
      />
    );
  }

  // ── Demo — ended overlay ──────────────────────────────────────────────────
  return <EndOverlay reason={endReason} />;
}
