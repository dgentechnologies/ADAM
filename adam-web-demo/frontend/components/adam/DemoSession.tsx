'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { User } from 'firebase/auth';
import { AdamFace } from './AdamFace';
import { AdamModelViewer } from './AdamModelViewer';
import { AudioCapture } from './AudioCapture';
import { SessionTimer } from './SessionTimer';
import type {
  ClientMessage,
  ServerMessage,
  TranscriptEntry,
  SessionState,
  FaceState,
  Emotion,
  MouthIntensity,
} from '@/types';

interface DemoSessionProps {
  user:              User;
  onSessionEnded?:   (reason: string) => void;
  fullscreen?:       boolean;
}

const RELAY_URL = process.env.NEXT_PUBLIC_RELAY_URL!;

export function DemoSession({ user, onSessionEnded, fullscreen }: DemoSessionProps) {
  const [state,          setState]          = useState<SessionState>('connecting');
  const [faceState,      setFaceState]      = useState<FaceState>('idle');
  const [emotion,        setEmotion]        = useState<Emotion>('idle');
  const [mouthIntensity, setMouthIntensity] = useState<MouthIntensity>('closed');
  const [transcripts,    setTranscripts]    = useState<TranscriptEntry[]>([]);
  const [isRecording,    setIsRecording]    = useState(false);
  const [turnCount,      setTurnCount]      = useState(0);
  const [turnsAllowed,   setTurnsAllowed]   = useState(1);
  const [durationMs,     setDurationMs]     = useState(300_000);
  const [endReason,      setEndReason]      = useState<string | null>(null);
  const [errorMsg,       setErrorMsg]       = useState<string | null>(null);
  const [micPermission,  setMicPermission]  = useState<'requesting' | 'granted' | 'denied'>('requesting');
  const [adamSpeaking,   setAdamSpeaking]   = useState(false);

  const wsRef            = useRef<WebSocket | null>(null);
  const audioCtxRef      = useRef<AudioContext | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const speechEndTimerRef= useRef<ReturnType<typeof setTimeout> | null>(null);
  const transcriptRef    = useRef<HTMLDivElement>(null);

  // ── Audio playback (gapless scheduled) ──────────────────────────────────

  const enqueueAudio = useCallback(async (base64: string) => {
    const ctx = (audioCtxRef.current ??= new AudioContext({ sampleRate: 24000 }));
    // Resume if browser suspended the context (autoplay policy)
    if (ctx.state === 'suspended') await ctx.resume();

    adamSpeakingRef.current = true;
    setAdamSpeaking(true);
    setIsRecording(false);

    const binary = atob(base64);
    const bytes  = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const pcm16  = new Int16Array(bytes.buffer);
    const f32    = new Float32Array(pcm16.length);
    for (let i = 0; i < pcm16.length; i++) f32[i] = pcm16[i] / 32768;

    const buffer = ctx.createBuffer(1, f32.length, 24000);
    buffer.copyToChannel(f32, 0);

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);

    // Schedule gapless: start exactly when the previous chunk ends.
    // A 10ms lookahead guards against scheduling jitter.
    const now       = ctx.currentTime;
    const startTime = Math.max(now + 0.01, nextStartTimeRef.current);
    source.start(startTime);
    nextStartTimeRef.current = startTime + buffer.duration;

    if (speechEndTimerRef.current) clearTimeout(speechEndTimerRef.current);
    const msUntilSpeechEnds = Math.max(0, (nextStartTimeRef.current - ctx.currentTime) * 1000) + 420;
    speechEndTimerRef.current = setTimeout(() => {
      adamSpeakingRef.current = false;
      setAdamSpeaking(false);
      if (stateRef.current === 'active' && micPermissionRef.current === 'granted') {
        setIsRecording(true);
      }
    }, msUntilSpeechEnds);
  }, []);

  // ── WS message dispatch ───────────────────────────────────────────────────

  const handleMessage = useCallback((msg: ServerMessage) => {
    switch (msg.type) {
      case 'session_ready':
        setState('active');
        setTurnsAllowed(msg.turnsAllowed);
        setDurationMs(msg.durationMs);
        setIsRecording(micPermissionRef.current === 'granted');
        setFaceState('listening');
        break;
      case 'audio':
        enqueueAudio(msg.data);
        break;
      case 'transcript':
        if (msg.role === 'adam') {
          // Accumulate all fragments into a single in-progress bubble
          setTranscripts((prev) => {
            const last = prev[prev.length - 1];
            if (last && last.role === 'adam' && last.inProgress) {
              return [...prev.slice(0, -1), { ...last, text: last.text + msg.text }];
            }
            return [...prev, { role: 'adam', text: msg.text, ts: Date.now(), inProgress: true }];
          });
        } else {
          setTranscripts((prev) => [...prev, { role: 'user', text: msg.text, ts: Date.now() }]);
          setTurnCount((n) => n + 1);
        }
        break;
      case 'face_state':
        setFaceState(msg.state);
        if (msg.state === 'speaking') {
          adamSpeakingRef.current = true;
          setAdamSpeaking(true);
          setIsRecording(false);
        }
        if (
          (msg.state === 'idle' || msg.state === 'listening')
          && !adamSpeakingRef.current
          && micPermissionRef.current === 'granted'
          && stateRef.current === 'active'
        ) {
          setIsRecording(true);
        }
        break;
      case 'emotion':
        setEmotion(msg.emotion);
        break;
      case 'mouth_sync':
        setMouthIntensity(msg.intensity);
        break;
      case 'turn_complete':
        setFaceState('idle');
        setMouthIntensity('closed');
        // Finalize the in-progress ADAM bubble
        setTranscripts((prev) => {
          if (!prev.length) return prev;
          const last = prev[prev.length - 1];
          if (last.role === 'adam' && last.inProgress) {
            return [...prev.slice(0, -1), { ...last, inProgress: false }];
          }
          return prev;
        });
        // Reset audio schedule so next response starts immediately
        nextStartTimeRef.current = 0;
        setTimeout(() => {
          if (
            stateRef.current === 'active'
            && !adamSpeakingRef.current
            && micPermissionRef.current === 'granted'
          ) {
            setIsRecording(true);
          }
        }, 400);
        break;
      case 'session_end':
        setState('ended');
        setEndReason(msg.reason);
        setIsRecording(false);
        adamSpeakingRef.current = false;
        setAdamSpeaking(false);
        if (speechEndTimerRef.current) clearTimeout(speechEndTimerRef.current);
        break;
      case 'error':
        setErrorMsg(`${msg.code}: ${msg.message}`);
        if (msg.code === 'auth_failed' || msg.code === 'cap_exceeded') setState('error');
        break;
    }
  }, [enqueueAudio]);

  const stateRef = useRef<SessionState>('connecting');
  const adamSpeakingRef = useRef(false);
  const micPermissionRef = useRef<'requesting' | 'granted' | 'denied'>('requesting');

  // Keep stateRef in sync so ws.onclose can read the current value without a
  // stale closure (the useEffect has empty deps, so `state` would be frozen).
  useEffect(() => { stateRef.current = state; }, [state]);
  useEffect(() => { adamSpeakingRef.current = adamSpeaking; }, [adamSpeaking]);
  useEffect(() => {
    micPermissionRef.current = micPermission;
    if (stateRef.current !== 'active') return;
    if (micPermission === 'denied') {
      setIsRecording(false);
      return;
    }
    if (micPermission === 'granted' && !adamSpeakingRef.current) {
      setIsRecording(true);
    }
  }, [micPermission]);

  // ── Connect on mount ──────────────────────────────────────────────────────

  useEffect(() => {
    let ws: WebSocket;

    (async () => {
      try {
        if (!RELAY_URL) {
          setErrorMsg('Demo relay not configured. Please try again later.');
          setState('error');
          return;
        }

        // 1. Get fresh Firebase ID token
        const idToken = await user.getIdToken(/* forceRefresh */ true);

        // 2. Exchange for short-lived relay JWT (server verifies Firebase token)
        const tokenRes = await fetch('/api/relay-token', {
          method:  'POST',
          headers: { 'Content-Type': 'application/json' },
          body:    JSON.stringify({ idToken }),
        });

        if (tokenRes.status === 429) {
          const { error } = await tokenRes.json() as { error: string };
          setErrorMsg(error);
          setState('error');
          return;
        }

        if (!tokenRes.ok) throw new Error('Failed to get relay token');
        const { token } = await tokenRes.json() as { token: string };

        // 3. Open WebSocket to relay
        ws = new WebSocket(RELAY_URL);
        wsRef.current = ws;

        ws.onopen = () => {
          const authMsg: ClientMessage = { type: 'auth', token };
          ws.send(JSON.stringify(authMsg));
        };

        ws.onmessage = (e) => {
          try { handleMessage(JSON.parse(e.data) as ServerMessage); } catch { /* ignore */ }
        };

        ws.onclose = () => {
          // If we never finished connecting, treat as an error (not a completed session).
          // This prevents the page from jumping straight to the waitlist EndOverlay.
          if (stateRef.current === 'connecting') {
            setState('error');
            setErrorMsg('Could not connect to ADAM. Check your connection and try again.');
          } else if (stateRef.current !== 'ended' && stateRef.current !== 'error') {
            setState('ended');
            setEndReason('connection_closed');
          }
        };

        ws.onerror = () => setErrorMsg('WebSocket connection failed');
      } catch (err) {
        setErrorMsg((err as Error).message);
        setState('error');
      }
    })();

    return () => {
      ws?.close();
      if (speechEndTimerRef.current) clearTimeout(speechEndTimerRef.current);
      audioCtxRef.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-scroll transcript
  useEffect(() => {
    transcriptRef.current?.scrollTo({ top: transcriptRef.current.scrollHeight, behavior: 'smooth' });
  }, [transcripts]);

  // Notify parent when session ends (fullscreen mode)
  useEffect(() => {
    if (state === 'ended' && onSessionEnded) {
      onSessionEnded(endReason ?? 'unknown');
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state]);

  // ── Controls ──────────────────────────────────────────────────────────────

  const send = (msg: ClientMessage) => wsRef.current?.send(JSON.stringify(msg));

  const endSession     = () => { send({ type: 'disconnect' }); setState('ended'); setEndReason('user_disconnect'); };
  const sendAudioChunk = (base64: string) => {
    if (
      stateRef.current !== 'active'
      || micPermissionRef.current !== 'granted'
      || adamSpeakingRef.current
    ) {
      return;
    }
    send({ type: 'audio', data: base64 });
  };

  // ── Render states ─────────────────────────────────────────────────────────

  if (state === 'connecting') {
    if (fullscreen) {
      return (
        <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: '#080a0c', gap: 24, position: 'relative', overflow: 'hidden' }}>
          <div style={{ position: 'absolute', inset: 0, backgroundImage: 'linear-gradient(rgba(74,240,255,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(74,240,255,0.025) 1px, transparent 1px)', backgroundSize: '48px 48px', pointerEvents: 'none' }} />
          <div style={{ position: 'absolute', top: '40%', left: '50%', transform: 'translate(-50%,-50%)', width: 500, height: 500, borderRadius: '50%', background: 'radial-gradient(circle, rgba(74,240,255,0.06) 0%, transparent 65%)', pointerEvents: 'none' }} />
          <div style={{ filter: 'drop-shadow(0 0 28px rgba(74,240,255,0.45))', position: 'relative', zIndex: 1 }}>
            <AdamFace emotion="idle" faceState="idle" size={200} />
          </div>
          <div style={{ width: 26, height: 26, border: '2px solid rgba(74,240,255,0.3)', borderTopColor: '#4AF0FF', borderRadius: '50%', animation: 'adamSpin 0.8s linear infinite', position: 'relative', zIndex: 1 }} />
          <p style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 11, color: '#444', letterSpacing: '0.12em', position: 'relative', zIndex: 1 }}>CONNECTING TO ADAM…</p>
          <style>{`@keyframes adamSpin { to { transform: rotate(360deg); } }`}</style>
        </div>
      );
    }
    return (
      <div className="flex flex-col items-center gap-4 py-16 text-gray-400">
        <div className="w-8 h-8 border-2 border-sky-400 border-t-transparent rounded-full animate-spin" />
        <p>Connecting to ADAM…</p>
      </div>
    );
  }

  if (state === 'error') {
    if (fullscreen) {
      return (
        <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: '#080a0c', gap: 22, position: 'relative', overflow: 'hidden', padding: '24px 20px' }}>
          <div style={{ position: 'absolute', inset: 0, backgroundImage: 'linear-gradient(rgba(74,240,255,0.02) 1px, transparent 1px), linear-gradient(90deg, rgba(74,240,255,0.02) 1px, transparent 1px)', backgroundSize: '48px 48px', pointerEvents: 'none' }} />
          {/* Glass error card */}
          <div style={{
            width: '100%', maxWidth: 400, position: 'relative', zIndex: 1,
            background: 'rgba(10, 14, 18, 0.72)',
            backdropFilter: 'blur(24px) saturate(160%)',
            WebkitBackdropFilter: 'blur(24px) saturate(160%)',
            border: '1px solid rgba(220,80,80,0.18)',
            borderTop: '1px solid rgba(220,80,80,0.28)',
            borderRadius: 24,
            boxShadow: '0 0 0 1px rgba(220,80,80,0.08), 0 24px 60px rgba(0,0,0,0.8)',
            padding: '36px 28px',
            display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 18, textAlign: 'center',
          }}>
            <div style={{ filter: 'drop-shadow(0 0 16px rgba(220,80,80,0.35))' }}>
              <AdamFace emotion="sad" faceState="idle" size={160} />
            </div>
            <div>
              <p style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 10, color: 'rgba(220,80,80,0.7)', letterSpacing: '0.14em', margin: '0 0 6px' }}>CONNECTION ERROR</p>
              {errorMsg && <p style={{ fontFamily: '"DM Sans", sans-serif', fontSize: 13, color: '#666', lineHeight: 1.6, margin: 0 }}>{errorMsg}</p>}
            </div>
            <button
              onClick={() => window.location.reload()}
              style={{ padding: '13px 40px', background: 'linear-gradient(135deg, #4AF0FF, #00c8e0)', color: '#080a0c', border: 'none', borderRadius: 14, fontFamily: '"Rajdhani", sans-serif', fontWeight: 600, fontSize: 15, letterSpacing: '0.08em', cursor: 'pointer', boxShadow: '0 6px 24px rgba(74,240,255,0.35)' }}
            >
              RETRY
            </button>
          </div>
          <style>{`@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600&family=Share+Tech+Mono&family=DM+Sans:wght@400&display=swap');`}</style>
        </div>
      );
    }
    return (
      <div className="text-center space-y-4 py-16">
        <p className="text-red-400 font-semibold">Connection failed</p>
        <p className="text-gray-500 text-sm">{errorMsg}</p>
        <button onClick={() => window.location.reload()} className="px-6 py-2 bg-sky-500 hover:bg-sky-400 text-white rounded-lg text-sm font-semibold transition">Try Again</button>
      </div>
    );
  }

  if (state === 'ended') {
    // fullscreen: parent handles overlay via onSessionEnded callback
    if (onSessionEnded) return null;
    return (
      <div className="text-center space-y-6 py-16">
        <div className="flex justify-center">
          <AdamFace emotion="happy" faceState="idle" size={120} />
        </div>
        <div>
          <p className="text-xl font-bold mb-1">Session ended</p>
          <p className="text-gray-400 text-sm">
            {endReason === 'cap_reached'
              ? 'Session complete.'
              : endReason === 'timeout'
              ? '5 minutes are up.'
              : 'Goodbye.'}
          </p>
        </div>
        <div className="flex gap-3 justify-center">
          <button
            onClick={() => window.location.reload()}
            className="px-6 py-2 bg-sky-500 hover:bg-sky-400 text-white rounded-lg text-sm font-semibold transition"
          >
            New Session
          </button>
        </div>
      </div>
    );
  }

  // ── Fullscreen active layout — full-viewport dark with floating glass panels ──
  if (fullscreen) {
    const recentTranscripts = transcripts.slice(-8);

    const statusLabel =
      faceState === 'listening' ? 'LISTENING'
      : faceState === 'speaking' ? 'SPEAKING'
      : 'IDLE';
    const statusColor =
      faceState === 'listening' ? '#4AF0FF'
      : faceState === 'speaking' ? '#ff9f0a'
      : '#555';

    const micPermissionLabel =
      micPermission === 'requesting' ? 'MIC PERMISSION: REQUESTING'
      : micPermission === 'granted'   ? 'MIC PERMISSION: GRANTED'
      : 'MIC PERMISSION: DENIED';
    const micStateLabel =
      micPermission !== 'granted'                        ? 'MIC STATE: BLOCKED'
      : adamSpeaking || faceState === 'speaking'         ? 'MIC STATE: SPEAKING (MIC OFF)'
      : isRecording                                      ? 'MIC STATE: LISTENING'
      : 'MIC STATE: LISTENING (INITIALIZING)';

    // Shared glass morphism style object
    const glass: React.CSSProperties = {
      background: 'rgba(10, 14, 18, 0.58)',
      backdropFilter: 'blur(24px) saturate(160%)',
      WebkitBackdropFilter: 'blur(24px) saturate(160%)',
      border: '1px solid rgba(255,255,255,0.08)',
      borderTop: '1px solid rgba(255,255,255,0.13)',
      borderRadius: 18,
      boxShadow: '0 0 0 1px rgba(255,255,255,0.03), 0 20px 60px rgba(0,0,0,0.7)',
    };

    return (
      <div style={{ position: 'fixed', inset: 0, background: '#080a0c', overflow: 'hidden', fontFamily: '"DM Sans", sans-serif' }}>
        <style>{`
          @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600&family=Share+Tech+Mono&family=DM+Sans:wght@300;400;500&display=swap');
          @keyframes adamCursorBlink { 0%,100%{opacity:1} 50%{opacity:0} }
          @keyframes statusGlow { 0%,100%{opacity:0.7} 50%{opacity:1} }
          ::-webkit-scrollbar { width: 3px; } ::-webkit-scrollbar-track { background: transparent; } ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
        `}</style>

        {/* ── Full-viewport 3D model as background, left-aligned ── */}
        <div style={{ position: 'absolute', inset: 0, zIndex: 0 }}>
          <AdamModelViewer modelPath="/models/adam-body.fbx" faceState={faceState} />
        </div>

        {/* ── Right-side gradient vignette — darkens the right to contrast panels ── */}
        <div style={{
          position: 'absolute', inset: 0, zIndex: 1, pointerEvents: 'none',
          background: 'linear-gradient(to right, transparent 18%, rgba(8,10,12,0.45) 44%, rgba(8,10,12,0.82) 66%, rgba(8,10,12,0.97) 85%)',
        }} />
        {/* Top vignette */}
        <div style={{ position: 'absolute', inset: 0, zIndex: 1, pointerEvents: 'none', background: 'linear-gradient(to bottom, rgba(8,10,12,0.55) 0%, transparent 12%)' }} />

        {/* ── Top bar (floats over everything) ── */}
        <div style={{ position: 'absolute', top: 16, left: 20, right: 20, zIndex: 30, display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
          {/* Brand badge */}
          <div style={{ ...glass, display: 'inline-flex', alignItems: 'center', gap: 8, padding: '8px 16px', borderRadius: 999 }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#4AF0FF', boxShadow: '0 0 8px #4AF0FF', flexShrink: 0 }} />
            <span style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 10, color: 'rgba(255,255,255,0.55)', letterSpacing: '0.18em' }}>ADAM LIVE DEMO</span>
          </div>
          {/* Timer pill */}
          <div style={{ ...glass, padding: '6px 14px', borderRadius: 999 }}>
            <SessionTimer durationMs={durationMs} turnsAllowed={turnsAllowed} turnCount={turnCount} onExpire={endSession} compact />
          </div>
        </div>

        {/* ── Status badge — bottom-left, over model ── */}
        <div style={{ position: 'absolute', bottom: 28, left: 28, zIndex: 20 }}>
          <div style={{ ...glass, display: 'inline-flex', alignItems: 'center', gap: 8, padding: '8px 18px', borderRadius: 999 }}>
            <span style={{
              width: 7, height: 7, borderRadius: '50%',
              background: statusColor, boxShadow: `0 0 10px ${statusColor}`,
              flexShrink: 0,
              animation: faceState === 'listening' ? 'statusGlow 1.6s ease-in-out infinite' : 'none',
            }} />
            <span style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 10, letterSpacing: '0.16em', color: statusColor }}>{statusLabel}</span>
          </div>
        </div>

        {/* ── Right floating panel column ── */}
        <div style={{
          position: 'absolute', top: 76, right: 18, bottom: 18,
          width: 'clamp(300px, 28vw, 400px)', maxWidth: 'calc(100vw - 40px)',
          zIndex: 20, display: 'flex', flexDirection: 'column', gap: 10,
        }}>

          {/* Conversation card */}
          <div style={{ ...glass, flex: 1, display: 'flex', flexDirection: 'column', padding: '18px 16px', gap: 10, minHeight: 0 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', paddingBottom: 10, borderBottom: '1px solid rgba(255,255,255,0.07)', flexShrink: 0 }}>
              <div>
                <p style={{ margin: 0, fontFamily: '"Rajdhani", sans-serif', fontSize: 24, fontWeight: 600, letterSpacing: '0.02em', color: '#f0f0f0', lineHeight: 1.1 }}>Live Conversation</p>
                <p style={{ margin: '3px 0 0', fontFamily: '"Share Tech Mono", monospace', fontSize: 9, color: 'rgba(255,255,255,0.28)', letterSpacing: '0.1em' }}>REAL-TIME VOICE TRANSCRIPT</p>
              </div>
              <button
                onClick={endSession}
                title="End session"
                aria-label="End session"
                style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', color: 'rgba(255,255,255,0.45)', borderRadius: 10, padding: '6px 14px', fontFamily: '"Share Tech Mono", monospace', fontSize: 9, letterSpacing: '0.1em', cursor: 'pointer', flexShrink: 0, transition: 'background 0.15s, color 0.15s' }}
                onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(255,80,80,0.12)'; e.currentTarget.style.color = '#ff6b6b'; e.currentTarget.style.borderColor = 'rgba(255,80,80,0.3)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.05)'; e.currentTarget.style.color = 'rgba(255,255,255,0.45)'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)'; }}
                onFocus={(e) => { e.currentTarget.style.outline = '2px solid rgba(255,80,80,0.5)'; e.currentTarget.style.outlineOffset = '2px'; }}
                onBlur={(e) => { e.currentTarget.style.outline = 'none'; }}
              >
                END
              </button>
            </div>

            {/* Transcript scroll area */}
            <div
              ref={transcriptRef}
              style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 9, minHeight: 0 }}
              aria-live="polite"
            >
              {recentTranscripts.length === 0 && (
                <p style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 9, color: 'rgba(255,255,255,0.18)', letterSpacing: '0.08em', textAlign: 'center', marginTop: 24, padding: '0 12px' }}>
                  Mic activates automatically after permission is granted.
                </p>
              )}
              {recentTranscripts.map((t, i) => (
                <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: t.role === 'user' ? 'flex-end' : 'flex-start' }}>
                  {t.role === 'adam' && (
                    <span style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 8, letterSpacing: '0.14em', color: '#4AF0FF', marginBottom: 4, paddingLeft: 2 }}>ADAM</span>
                  )}
                  <div style={{
                    maxWidth: '88%',
                    padding: '9px 13px',
                    borderRadius: t.role === 'adam' ? '4px 16px 16px 16px' : '16px 4px 16px 16px',
                    background: t.role === 'adam' ? 'rgba(74,240,255,0.07)' : 'rgba(255,255,255,0.06)',
                    border: t.role === 'adam' ? '1px solid rgba(74,240,255,0.14)' : '1px solid rgba(255,255,255,0.07)',
                    fontSize: 13, color: t.role === 'adam' ? '#dff6ff' : '#c8c8c8', lineHeight: 1.6,
                  }}>
                    {t.text}
                    {t.role === 'adam' && t.inProgress && (
                      <span aria-hidden="true" style={{ display: 'inline-block', width: 2, height: '0.85em', background: '#4AF0FF', marginLeft: 3, verticalAlign: 'middle', borderRadius: 1, animation: 'adamCursorBlink 0.8s step-end infinite' }} />
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Mic status card */}
          <div style={{ ...glass, padding: '10px 14px', display: 'flex', flexDirection: 'column', gap: 4 }} role="status" aria-live="polite">
            <span style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 9, letterSpacing: '0.1em', color: micPermission === 'denied' ? '#ff453a' : micPermission === 'granted' ? 'rgba(74,240,255,0.65)' : 'rgba(255,255,255,0.3)' }}>{micPermissionLabel}</span>
            <span style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 9, letterSpacing: '0.1em', color: isRecording && !adamSpeaking ? '#4AF0FF' : 'rgba(255,255,255,0.28)' }}>{micStateLabel}</span>
          </div>

          {/* Meta cards row */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <div style={{ ...glass, padding: '10px 14px' }}>
              <p style={{ margin: 0, fontFamily: '"Share Tech Mono", monospace', fontSize: 8, letterSpacing: '0.12em', color: 'rgba(255,255,255,0.3)' }}>TURNS</p>
              <p style={{ margin: '5px 0 0', fontFamily: '"DM Sans", sans-serif', fontSize: 15, fontWeight: 500, color: '#f0f0f0' }}>{turnCount} <span style={{ color: 'rgba(255,255,255,0.3)', fontSize: 11 }}>/ {turnsAllowed}</span></p>
            </div>
            <div style={{ ...glass, padding: '10px 14px' }}>
              <p style={{ margin: 0, fontFamily: '"Share Tech Mono", monospace', fontSize: 8, letterSpacing: '0.12em', color: 'rgba(255,255,255,0.3)' }}>CONNECTION</p>
              <p style={{ margin: '5px 0 0', fontFamily: '"DM Sans", sans-serif', fontSize: 15, fontWeight: 500, color: state === 'active' ? '#32d74b' : '#ff9f0a' }}>{state === 'active' ? 'Stable' : 'Pending'}</p>
            </div>
          </div>

          {errorMsg && (
            <p style={{ margin: 0, fontFamily: '"Share Tech Mono", monospace', fontSize: 10, color: '#ff453a', letterSpacing: '0.06em', paddingLeft: 4 }}>{errorMsg}</p>
          )}
        </div>

        <AudioCapture
          isRecording={isRecording}
          onAudioChunk={sendAudioChunk}
          onPermissionChange={setMicPermission}
        />
      </div>
    );
  }

  // ── Default (non-fullscreen) active layout ────────────────────────────────
  return (
    <div className="space-y-6">
      {/* Face */}
      <div className="flex flex-col items-center gap-3">
        <AdamFace emotion={emotion} faceState={faceState} mouthIntensity={mouthIntensity} size={200} />
        <p className="text-xs text-gray-500 uppercase tracking-widest font-semibold">
          {faceState === 'listening' ? '● Listening' : faceState === 'speaking' ? '▶ Speaking' : '— Idle'}
        </p>
      </div>

      {/* Timer */}
      <SessionTimer
        durationMs={durationMs}
        turnsAllowed={turnsAllowed}
        turnCount={turnCount}
        onExpire={endSession}
      />

      {/* Transcript */}
      <div
        ref={transcriptRef}
        className="h-48 overflow-y-auto rounded-xl border border-white/10 bg-white/5 p-4 space-y-3 text-sm"
      >
        {transcripts.length === 0 && (
          <p className="text-gray-600 text-center pt-4">Say something to ADAM…</p>
        )}
        {transcripts.map((t, i) => (
          <div key={i} className={`flex gap-2 ${t.role === 'adam' ? '' : 'justify-end'}`}>
            {t.role === 'adam' && <span className="text-sky-400 font-bold shrink-0">ADAM</span>}
            <p
              className={`rounded-lg px-3 py-1.5 max-w-xs text-sm ${
                t.role === 'adam' ? 'bg-sky-950/50 text-gray-100' : 'bg-white/10 text-gray-200'
              }`}
            >
              {t.text}
              {t.role === 'adam' && t.inProgress && (
                <span aria-hidden="true" className="adam-typing-cursor" />
              )}
            </p>
          </div>
        ))}
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-4">
        <div
          className="w-full max-w-xl rounded-xl border border-white/15 bg-white/5 px-4 py-3 text-sm text-gray-200"
          role="status"
          aria-live="polite"
        >
          <p className="font-mono text-xs tracking-wider text-gray-400">
            {micPermission === 'requesting'
              ? 'MIC PERMISSION: REQUESTING'
              : micPermission === 'granted'
              ? 'MIC PERMISSION: GRANTED'
              : 'MIC PERMISSION: DENIED'}
          </p>
          <p className="mt-1 font-mono text-xs tracking-wider text-sky-300">
            {micPermission !== 'granted'
              ? 'MIC STATE: BLOCKED'
              : adamSpeaking || faceState === 'speaking'
              ? 'MIC STATE: SPEAKING (MIC OFF)'
              : isRecording
              ? 'MIC STATE: LISTENING'
              : 'MIC STATE: LISTENING (INITIALIZING)'}
          </p>
        </div>
        <button
          onClick={endSession}
          className="px-4 py-3 border border-white/20 hover:border-red-500/50 hover:text-red-400 text-gray-400 rounded-xl text-sm font-semibold transition"
        >
          ■ End
        </button>
      </div>

      {errorMsg && <p className="text-center text-xs text-red-400">{errorMsg}</p>}

      <AudioCapture
        isRecording={isRecording}
        onAudioChunk={sendAudioChunk}
        onPermissionChange={setMicPermission}
      />
    </div>
  );
}
