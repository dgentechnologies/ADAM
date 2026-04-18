'use client';

import { createPortal } from 'react-dom';
import { useEffect, useState } from 'react';
import { doc, getDoc } from 'firebase/firestore';
import { useAuth } from '@/components/FirebaseAuthProvider';
import { DemoSession } from '@/components/adam/DemoSession';
import { AdamFace } from '@/components/adam/AdamFace';
import { OnboardingForm } from '@/components/adam/OnboardingForm';
import { DemoAuthGate } from '@/components/adam/DemoAuthGate';
import { getClientDb } from '@/lib/firebase';

const WAITLIST_URL = 'https://dgentechnologies.com/adam#waitlist';

type Step        = 'auth' | 'checking' | 'onboarding' | 'demo';
type EmailMode   = 'signin' | 'signup';
type DemoOverlay = 'welcome' | 'active' | 'ended';

const FONTS = `@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;600&family=Share+Tech+Mono&family=DM+Sans:wght@300;400;500&display=swap');`;

// ── Portal: renders children into document.body to avoid transform stacking ──
function Portal({ children }: { children: React.ReactNode }) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) return null;
  return createPortal(children, document.body);
}

// ── Ambient background shared by overlays ─────────────────────────────────────
const OVERLAY_BG_STYLES: React.CSSProperties = {
  position: 'fixed', inset: 0, zIndex: 9999,
  background: '#080a0c',
  display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
  padding: '24px 20px',
  overflow: 'hidden',
};

// ── Spinner ───────────────────────────────────────────────────────────────────
function Spinner({ label }: { label?: string }) {
  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: '#080a0c', gap: 20 }}>
      <style>{`@keyframes pgSpin { to { transform: rotate(360deg); } } ${FONTS}`}</style>
      {/* Ambient grid */}
      <div style={{ position: 'fixed', inset: 0, backgroundImage: 'linear-gradient(rgba(74,240,255,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(74,240,255,0.025) 1px, transparent 1px)', backgroundSize: '48px 48px', pointerEvents: 'none' }} />
      <div style={{ filter: 'drop-shadow(0 0 24px rgba(74,240,255,0.4))', position: 'relative', zIndex: 1 }}>
        <AdamFace emotion="idle" faceState="idle" size={160} />
      </div>
      <div style={{ width: 22, height: 22, border: '1.5px solid rgba(74,240,255,0.25)', borderTopColor: '#4AF0FF', borderRadius: '50%', animation: 'pgSpin 0.8s linear infinite', position: 'relative', zIndex: 1 }} />
      {label && <p style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 10, color: '#333', letterSpacing: '0.14em', position: 'relative', zIndex: 1 }}>{label}</p>}
    </div>
  );
}

// ── Welcome overlay (rendered via Portal to avoid parent transform issues) ────
function WelcomeOverlay({ onDismiss }: { onDismiss: () => void }) {
  const [visible, setVisible] = useState(false);
  useEffect(() => { const t = setTimeout(() => setVisible(true), 80); return () => clearTimeout(t); }, []);

  return (
    <Portal>
      <style>{`
        ${FONTS}
        @keyframes wFadeUp {
          from { opacity: 0; transform: translateY(28px) scale(0.97); }
          to   { opacity: 1; transform: translateY(0)   scale(1);    }
        }
        @keyframes wPulse {
          0%, 100% { opacity: 0.5; transform: scale(1);    }
          50%       { opacity: 1;   transform: scale(1.07); }
        }
      `}</style>
      <div style={{ ...OVERLAY_BG_STYLES, opacity: visible ? 1 : 0, transition: 'opacity 0.45s ease' }}>
        {/* Grid texture */}
        <div style={{ position: 'absolute', inset: 0, backgroundImage: 'linear-gradient(rgba(74,240,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(74,240,255,0.03) 1px, transparent 1px)', backgroundSize: '48px 48px', pointerEvents: 'none' }} />
        {/* Radial glow */}
        <div style={{ position: 'absolute', top: '30%', left: '50%', transform: 'translate(-50%,-50%)', width: 700, height: 700, borderRadius: '50%', background: 'radial-gradient(circle, rgba(74,240,255,0.07) 0%, transparent 60%)', pointerEvents: 'none' }} />

        {/* Glass card */}
        <div style={{
          width: '100%', maxWidth: 460, position: 'relative', zIndex: 1,
          background: 'rgba(10, 14, 18, 0.72)',
          backdropFilter: 'blur(28px) saturate(180%)',
          WebkitBackdropFilter: 'blur(28px) saturate(180%)',
          border: '1px solid rgba(255,255,255,0.07)',
          borderTop: '1px solid rgba(255,255,255,0.13)',
          borderRadius: 28,
          boxShadow: '0 0 0 1px rgba(255,255,255,0.03), 0 32px 80px rgba(0,0,0,0.85), inset 0 1px 0 rgba(255,255,255,0.07)',
          padding: '36px 32px',
          display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 22,
          animation: visible ? 'wFadeUp 0.65s cubic-bezier(0.22,1,0.36,1) both' : 'none',
        }}>
          {/* DGEN badge */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 7, padding: '5px 14px', background: 'rgba(74,240,255,0.07)', border: '1px solid rgba(74,240,255,0.18)', borderRadius: 20 }}>
            <span style={{ width: 5, height: 5, borderRadius: '50%', background: '#4AF0FF', boxShadow: '0 0 7px #4AF0FF', flexShrink: 0 }} />
            <span style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 9, color: 'rgba(74,240,255,0.75)', letterSpacing: '0.18em' }}>DGEN TECHNOLOGIES</span>
          </div>

          {/* Face with animated glow ring */}
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ position: 'absolute', width: 230, height: 230, borderRadius: '50%', background: 'radial-gradient(circle, rgba(74,240,255,0.13) 0%, transparent 65%)', animation: 'wPulse 2.8s ease-in-out infinite', pointerEvents: 'none' }} />
            <div style={{ position: 'absolute', width: 205, height: 205, borderRadius: '50%', border: '1px solid rgba(74,240,255,0.14)', animation: 'wPulse 2.8s ease-in-out infinite 0.3s', pointerEvents: 'none' }} />
            <div style={{ filter: 'drop-shadow(0 0 26px rgba(74,240,255,0.5))', position: 'relative', zIndex: 1 }}>
              <AdamFace emotion="happy" faceState="idle" size={178} />
            </div>
          </div>

          {/* Heading */}
          <div style={{ textAlign: 'center' }}>
            <h1 style={{ fontFamily: '"Rajdhani", sans-serif', fontWeight: 600, fontSize: 44, letterSpacing: '0.06em', color: '#f0f0f0', margin: 0, lineHeight: 1.05 }}>
              Hello. I&apos;m <span style={{ color: '#4AF0FF', textShadow: '0 0 22px rgba(74,240,255,0.55)' }}>ADAM</span>.
            </h1>
            <p style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 9, color: 'rgba(255,255,255,0.22)', letterSpacing: '0.2em', margin: '7px 0 0' }}>
              AUTONOMOUS DESKTOP AI MODULE
            </p>
          </div>

          {/* Info glass box */}
          <div style={{ width: '100%', background: 'rgba(255,255,255,0.025)', border: '1px solid rgba(255,255,255,0.055)', borderRadius: 16, padding: '18px 20px', textAlign: 'left' }}>
            <p style={{ fontFamily: '"DM Sans", sans-serif', fontSize: 14, lineHeight: 1.85, color: '#888', margin: 0 }}>
              I can hear your voice, process what you say in real time, and respond like a person — powered by{' '}
              <span style={{ color: '#d4d4d4', fontWeight: 500 }}>Gemini Live</span>. You have a{' '}
              <span style={{ color: '#4AF0FF', fontWeight: 500 }}>5-minute session</span> right now.
            </p>
            <p style={{ fontFamily: '"DM Sans", sans-serif', fontSize: 14, lineHeight: 1.85, color: '#888', margin: '10px 0 0' }}>
              Hold the <span style={{ color: '#4AF0FF', fontWeight: 500 }}>mic button</span> and speak. Release when done.
            </p>
          </div>

            {/* CTA */}
          <button
            onClick={onDismiss}
            style={{
              width: '100%', padding: '16px 0',
              background: 'linear-gradient(135deg, #4AF0FF 0%, #00c8e0 100%)',
              color: '#080a0c',
              border: 'none', borderRadius: 16,
              fontFamily: '"Rajdhani", sans-serif', fontWeight: 600,
              fontSize: 17, letterSpacing: '0.13em',
              cursor: 'pointer',
              boxShadow: '0 0 0 1px rgba(74,240,255,0.25), 0 8px 36px rgba(74,240,255,0.42)',
              transition: 'transform 0.15s ease, box-shadow 0.2s ease',
            }}
            onMouseEnter={(e) => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = '0 0 0 1px rgba(74,240,255,0.35), 0 14px 52px rgba(74,240,255,0.58)'; }}
            onMouseLeave={(e) => { e.currentTarget.style.transform = 'translateY(0)'; e.currentTarget.style.boxShadow = '0 0 0 1px rgba(74,240,255,0.25), 0 8px 36px rgba(74,240,255,0.42)'; }}
          >
            START SESSION →
          </button>

          <p style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 9, color: 'rgba(255,255,255,0.14)', letterSpacing: '0.12em', margin: 0 }}>
            MICROPHONE ACCESS REQUIRED
          </p>
        </div>
      </div>
    </Portal>
  );
}

// ── Star rating ───────────────────────────────────────────────────────────────
function StarRating() {
  const [rating, setRating] = useState(0);
  const [hover,  setHover]  = useState(0);
  return (
    <div style={{ display: 'flex', gap: 6 }}>
      {[1, 2, 3, 4, 5].map((n) => (
        <button
          key={n}
          onClick={() => setRating(n)}
          onMouseEnter={() => setHover(n)}
          onMouseLeave={() => setHover(0)}
          style={{
            background: 'none', border: 'none', cursor: 'pointer', padding: '2px',
            fontSize: 26,
            color: n <= (hover || rating) ? '#4AF0FF' : 'rgba(255,255,255,0.13)',
            textShadow: n <= (hover || rating) ? '0 0 14px rgba(74,240,255,0.65)' : 'none',
            transition: 'color 0.12s, text-shadow 0.12s',
          }}
        >★</button>
      ))}
    </div>
  );
}

// ── End overlay ───────────────────────────────────────────────────────────────
function EndOverlay({ reason }: { reason: string }) {
  const [visible, setVisible] = useState(false);
  useEffect(() => { const t = setTimeout(() => setVisible(true), 80); return () => clearTimeout(t); }, []);

  const isTimeout  = reason === 'timeout' || reason === 'cap_reached';
  const isUserExit = reason === 'user_disconnect';

  return (
    <Portal>
      <style>{`
        ${FONTS}
        @keyframes eFadeUp {
          from { opacity: 0; transform: translateY(20px) scale(0.98); }
          to   { opacity: 1; transform: translateY(0)   scale(1);    }
        }
      `}</style>
      <div style={{ ...OVERLAY_BG_STYLES, opacity: visible ? 1 : 0, transition: 'opacity 0.45s ease' }}>
        {/* Grid */}
        <div style={{ position: 'absolute', inset: 0, backgroundImage: 'linear-gradient(rgba(74,240,255,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(74,240,255,0.025) 1px, transparent 1px)', backgroundSize: '48px 48px', pointerEvents: 'none' }} />
        {/* Ambient glow */}
        <div style={{ position: 'absolute', top: '40%', left: '50%', transform: 'translate(-50%,-50%)', width: 500, height: 500, borderRadius: '50%', background: 'radial-gradient(circle, rgba(74,240,255,0.06) 0%, transparent 65%)', pointerEvents: 'none' }} />

        {/* Glass card */}
        <div style={{
          width: '100%', maxWidth: 420, position: 'relative', zIndex: 1,
          background: 'rgba(10, 14, 18, 0.75)',
          backdropFilter: 'blur(28px) saturate(180%)',
          WebkitBackdropFilter: 'blur(28px) saturate(180%)',
          border: '1px solid rgba(255,255,255,0.07)',
          borderTop: '1px solid rgba(255,255,255,0.12)',
          borderRadius: 28,
          boxShadow: '0 0 0 1px rgba(255,255,255,0.03), 0 32px 80px rgba(0,0,0,0.85), inset 0 1px 0 rgba(255,255,255,0.06)',
          padding: '36px 32px',
          display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 20,
          animation: visible ? 'eFadeUp 0.6s cubic-bezier(0.22,1,0.36,1) both' : 'none',
          textAlign: 'center',
        }}>
          {/* Face */}
          <div style={{ filter: 'drop-shadow(0 0 20px rgba(74,240,255,0.22))' }}>
            <AdamFace emotion="sad" faceState="idle" size={145} />
          </div>

          {/* Text */}
          <div>
            <p style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 9, color: 'rgba(255,255,255,0.22)', letterSpacing: '0.18em', margin: '0 0 8px' }}>
              {isTimeout ? '■ PREVIEW COMPLETE' : isUserExit ? '■ SESSION CLOSED' : '■ SESSION ENDED'}
            </p>
            <h2 style={{ fontFamily: '"Rajdhani", sans-serif', fontWeight: 600, fontSize: 30, letterSpacing: '0.05em', color: '#f0f0f0', margin: 0 }}>
              {isTimeout ? "That's your 5-minute preview." : isUserExit ? 'You ended the session.' : 'Session complete.'}
            </h2>
            <p style={{ fontFamily: '"DM Sans", sans-serif', fontSize: 13, color: '#555', margin: '8px 0 0', lineHeight: 1.7 }}>
              {isTimeout ? 'Want ADAM on your desk, available all day? Be first when it ships.' : 'Ready to own one? Join the waitlist.'}
            </p>
          </div>

          {/* Stars */}
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
            <p style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 9, color: '#333', letterSpacing: '0.12em', margin: 0 }}>RATE THIS EXPERIENCE</p>
            <StarRating />
          </div>

          {/* Divider */}
          <div style={{ width: '100%', height: 1, background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent)' }} />

          {/* Waitlist CTA */}
          <a
            href={WAITLIST_URL}
            target="_top"
            style={{
              display: 'block', width: '100%', padding: '15px 0',
              background: 'linear-gradient(135deg, #4AF0FF 0%, #00c8e0 100%)',
              color: '#080a0c',
              borderRadius: 16, textDecoration: 'none',
              fontFamily: '"Rajdhani", sans-serif', fontWeight: 600,
              fontSize: 16, letterSpacing: '0.11em', textAlign: 'center',
              boxShadow: '0 0 0 1px rgba(74,240,255,0.22), 0 8px 28px rgba(74,240,255,0.38)',
              transition: 'transform 0.15s ease, box-shadow 0.2s ease',
            }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLAnchorElement).style.transform = 'translateY(-2px)'; (e.currentTarget as HTMLAnchorElement).style.boxShadow = '0 0 0 1px rgba(74,240,255,0.35), 0 14px 44px rgba(74,240,255,0.52)'; }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLAnchorElement).style.transform = 'translateY(0)'; (e.currentTarget as HTMLAnchorElement).style.boxShadow = '0 0 0 1px rgba(74,240,255,0.22), 0 8px 28px rgba(74,240,255,0.38)'; }}
          >
            → JOIN THE ADAM WAITLIST
          </a>

          {/* Try again */}
          <button
            onClick={() => window.location.reload()}
            style={{
              background: 'transparent',
              border: '1px solid rgba(255,255,255,0.07)',
              color: '#444', borderRadius: 12,
              padding: '11px 0', cursor: 'pointer', width: '100%',
              fontFamily: '"Share Tech Mono", monospace', fontSize: 11,
              letterSpacing: '0.07em',
              transition: 'border-color 0.2s, color 0.2s',
            }}
            onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.18)'; e.currentTarget.style.color = '#f0f0f0'; }}
            onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.07)'; e.currentTarget.style.color = '#444'; }}
          >
            TRY AGAIN ↺
          </button>
        </div>
      </div>
    </Portal>
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
    getDoc(doc(getClientDb(), 'onboarding', user.uid))
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

  // ── Demo flow ─────────────────────────────────────────────────────────────
  // Always render DemoSession once active (it returns null when ended + fullscreen).
  // Overlays (welcome & ended) render on top via portals.
  return (
    <>
      {/* Active session — only mount when the welcome has been dismissed */}
      {demoOverlay !== 'welcome' && (
        <DemoSession
          user={user}
          fullscreen
          onSessionEnded={(reason) => {
            setEndReason(reason);
            setDemoOverlay('ended');
          }}
        />
      )}

      {/* Welcome overlay — portal so position:fixed is never trapped by parent transforms */}
      {demoOverlay === 'welcome' && (
        <WelcomeOverlay onDismiss={() => setDemoOverlay('active')} />
      )}

      {/* End overlay — renders on top of the (null-returning) DemoSession */}
      {demoOverlay === 'ended' && (
        <EndOverlay reason={endReason} />
      )}
    </>
  );
}
