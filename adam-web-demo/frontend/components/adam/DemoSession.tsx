'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { User } from 'firebase/auth';
import { AdamFace } from './AdamFace';
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

  const wsRef         = useRef<WebSocket | null>(null);
  const audioQueueRef = useRef<AudioBuffer[]>([]);
  const isPlayingRef  = useRef(false);
  const audioCtxRef   = useRef<AudioContext | null>(null);
  const transcriptRef = useRef<HTMLDivElement>(null);

  // ── Audio playback ────────────────────────────────────────────────────────

  const playNextAudio = useCallback(async () => {
    if (isPlayingRef.current || audioQueueRef.current.length === 0) return;
    isPlayingRef.current = true;
    const ctx    = (audioCtxRef.current ??= new AudioContext({ sampleRate: 24000 }));
    const buffer = audioQueueRef.current.shift()!;
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    source.onended = () => { isPlayingRef.current = false; playNextAudio(); };
    source.start();
  }, []);

  const enqueueAudio = useCallback(async (base64: string) => {
    const ctx    = (audioCtxRef.current ??= new AudioContext({ sampleRate: 24000 }));
    const binary = atob(base64);
    const bytes  = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const pcm16  = new Int16Array(bytes.buffer);
    const f32    = new Float32Array(pcm16.length);
    for (let i = 0; i < pcm16.length; i++) f32[i] = pcm16[i] / 32768;
    const buffer = ctx.createBuffer(1, f32.length, 24000);
    buffer.copyToChannel(f32, 0);
    audioQueueRef.current.push(buffer);
    playNextAudio();
  }, [playNextAudio]);

  // ── WS message dispatch ───────────────────────────────────────────────────

  const handleMessage = useCallback((msg: ServerMessage) => {
    switch (msg.type) {
      case 'session_ready':
        setState('active');
        setTurnsAllowed(msg.turnsAllowed);
        setDurationMs(msg.durationMs);
        break;
      case 'audio':
        enqueueAudio(msg.data);
        break;
      case 'transcript':
        setTranscripts((prev) => [...prev, { role: msg.role, text: msg.text, ts: Date.now() }]);
        if (msg.role === 'user') setTurnCount((n) => n + 1);
        break;
      case 'face_state':
        setFaceState(msg.state);
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
        break;
      case 'session_end':
        setState('ended');
        setEndReason(msg.reason);
        setIsRecording(false);
        break;
      case 'error':
        setErrorMsg(`${msg.code}: ${msg.message}`);
        if (msg.code === 'auth_failed' || msg.code === 'cap_exceeded') setState('error');
        break;
    }
  }, [enqueueAudio]);

  const stateRef = useRef<SessionState>('connecting');

  // Keep stateRef in sync so ws.onclose can read the current value without a
  // stale closure (the useEffect has empty deps, so `state` would be frozen).
  useEffect(() => { stateRef.current = state; }, [state]);

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

  const startRecording = () => { setIsRecording(true); setFaceState('listening'); };
  const stopRecording  = () => { setIsRecording(false); setFaceState('idle'); send({ type: 'end_turn' }); };
  const endSession     = () => { send({ type: 'disconnect' }); setState('ended'); setEndReason('user_disconnect'); };
  const sendAudioChunk = (base64: string) => send({ type: 'audio', data: base64 });

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

  // ── Fullscreen active layout ──────────────────────────────────────────────
  if (fullscreen) {
    const statusLabel =
      faceState === 'listening' ? '● LISTENING'
      : faceState === 'speaking' ? '▶ SPEAKING'
      : '— IDLE';
    const statusColor =
      faceState === 'listening' ? '#4AF0FF'
      : faceState === 'speaking' ? '#9a9a9a'
      : 'rgba(255,255,255,0.18)';
    const glowColor =
      faceState === 'listening' ? 'rgba(74,240,255,0.28)'
      : faceState === 'speaking' ? 'rgba(74,240,255,0.18)'
      : 'rgba(74,240,255,0.09)';

    return (
      <div style={{ minHeight: '100vh', background: '#080a0c', position: 'relative', display: 'flex', flexDirection: 'column', alignItems: 'center', overflow: 'hidden' }}>
        <style>{`
          @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600&family=Share+Tech+Mono&family=DM+Sans:wght@300;400&display=swap');
          @keyframes fsGlow  { 0%,100% { opacity:.7; transform:scale(1); }    50% { opacity:1; transform:scale(1.07); } }
          @keyframes fsPulse { 0%,100% { box-shadow:0 0 0 0 rgba(74,240,255,0); } 50% { box-shadow:0 0 0 14px rgba(74,240,255,0); } }
          @keyframes fsSpin  { to { transform:rotate(360deg); } }
          @keyframes fsRecPulse { 0%,100% { box-shadow:0 0 0 0 rgba(220,80,80,0.6), 0 8px 28px rgba(220,80,80,0.4); } 50% { box-shadow:0 0 0 10px rgba(220,80,80,0), 0 8px 28px rgba(220,80,80,0.55); } }
        `}</style>

        {/* Background grid */}
        <div style={{ position: 'absolute', inset: 0, backgroundImage: 'linear-gradient(rgba(74,240,255,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(74,240,255,0.025) 1px, transparent 1px)', backgroundSize: '48px 48px', pointerEvents: 'none' }} />
        {/* Ambient top glow */}
        <div style={{ position: 'absolute', top: '10%', left: '55%', transform: 'translateX(-50%)', width: 600, height: 600, borderRadius: '50%', background: `radial-gradient(circle, ${glowColor} 0%, transparent 65%)`, transition: 'background 0.8s ease', pointerEvents: 'none' }} />
        {/* Ambient bottom glow */}
        <div style={{ position: 'absolute', bottom: '0%', left: '35%', transform: 'translateX(-50%)', width: 400, height: 400, borderRadius: '50%', background: 'radial-gradient(circle, rgba(74,240,255,0.03) 0%, transparent 65%)', pointerEvents: 'none' }} />

        {/* ── Top bar: timer + end button ──────────────────────────────── */}
        <div style={{ position: 'absolute', top: 0, left: 0, right: 0, zIndex: 20, display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '18px 24px' }}>
          {/* ADAM label */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 14px', background: 'rgba(10,14,18,0.6)', backdropFilter: 'blur(16px)', WebkitBackdropFilter: 'blur(16px)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 20 }}>
            <span style={{ width: 5, height: 5, borderRadius: '50%', background: state === 'active' ? '#4AF0FF' : '#333', boxShadow: state === 'active' ? '0 0 7px #4AF0FF' : 'none', transition: 'all 0.4s', flexShrink: 0 }} />
            <span style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 9, color: 'rgba(255,255,255,0.3)', letterSpacing: '0.16em' }}>ADAM DEMO</span>
          </div>

          {/* Timer glass badge */}
          <div style={{ padding: '6px 14px', background: 'rgba(10,14,18,0.6)', backdropFilter: 'blur(16px)', WebkitBackdropFilter: 'blur(16px)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 20 }}>
            <SessionTimer durationMs={durationMs} turnsAllowed={turnsAllowed} turnCount={turnCount} onExpire={endSession} compact />
          </div>
        </div>

        {/* ── Center: face ──────────────────────────────────────────────── */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 18, padding: '88px 24px 16px', width: '100%' }}>
          {/* Face glow ring */}
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ position: 'absolute', width: 310, height: 310, borderRadius: '50%', background: `radial-gradient(circle, ${glowColor} 0%, transparent 65%)`, transition: 'background 0.6s ease', animation: faceState !== 'idle' ? 'fsGlow 2.2s ease-in-out infinite' : 'none', pointerEvents: 'none' }} />
            <div style={{ position: 'absolute', width: 278, height: 278, borderRadius: '50%', border: `1px solid ${faceState === 'idle' ? 'rgba(255,255,255,0.04)' : 'rgba(74,240,255,0.16)'}`, transition: 'border-color 0.5s ease', pointerEvents: 'none' }} />
            <div style={{ filter: `drop-shadow(0 0 ${faceState === 'idle' ? '18px' : '32px'} rgba(74,240,255,${faceState === 'idle' ? '0.25' : '0.5'}))`, transition: 'filter 0.5s ease', position: 'relative', zIndex: 1 }}>
              <AdamFace emotion={emotion} faceState={faceState} mouthIntensity={mouthIntensity} size={260} />
            </div>
          </div>

          {/* Status pill */}
          <div style={{ padding: '7px 18px', background: 'rgba(10,14,18,0.65)', backdropFilter: 'blur(16px)', WebkitBackdropFilter: 'blur(16px)', border: `1px solid ${faceState === 'idle' ? 'rgba(255,255,255,0.06)' : 'rgba(74,240,255,0.18)'}`, borderRadius: 20, transition: 'border-color 0.4s ease' }}>
            <span style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 10, color: statusColor, letterSpacing: '0.14em', transition: 'color 0.3s ease' }}>
              {statusLabel}
            </span>
          </div>
        </div>

        {/* ── Bottom: transcript + controls ────────────────────────────── */}
        <div style={{ width: '100%', maxWidth: 580, padding: '0 20px 36px', display: 'flex', flexDirection: 'column', gap: 12, position: 'relative', zIndex: 10 }}>

          {/* Transcript glass panel */}
          {transcripts.length > 0 && (
            <div
              ref={transcriptRef}
              style={{
                maxHeight: 148,
                overflowY: 'auto',
                background: 'rgba(10,14,18,0.65)',
                backdropFilter: 'blur(20px) saturate(160%)',
                WebkitBackdropFilter: 'blur(20px) saturate(160%)',
                border: '1px solid rgba(255,255,255,0.06)',
                borderTop: '1px solid rgba(255,255,255,0.1)',
                borderRadius: 20,
                padding: '14px 16px',
                display: 'flex', flexDirection: 'column', gap: 7,
              }}
            >
              {transcripts.slice(-5).map((t, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: t.role === 'adam' ? 'flex-start' : 'flex-end' }}>
                  <span
                    style={{
                      fontFamily: '"DM Sans", sans-serif', fontSize: 12, lineHeight: 1.6,
                      color: t.role === 'adam' ? '#9a9a9a' : '#e8e8e8',
                      background: t.role === 'adam' ? 'rgba(74,240,255,0.06)' : 'rgba(255,255,255,0.07)',
                      border: `1px solid ${t.role === 'adam' ? 'rgba(74,240,255,0.12)' : 'rgba(255,255,255,0.08)'}`,
                      padding: '5px 12px', borderRadius: 10, maxWidth: '82%',
                    }}
                  >
                    {t.role === 'adam' && (
                      <span style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 8, color: '#4AF0FF', marginRight: 6, letterSpacing: '0.08em' }}>ADAM </span>
                    )}
                    {t.text}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Controls glass strip */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 10,
            background: 'rgba(10,14,18,0.72)',
            backdropFilter: 'blur(24px) saturate(180%)',
            WebkitBackdropFilter: 'blur(24px) saturate(180%)',
            border: '1px solid rgba(255,255,255,0.07)',
            borderTop: '1px solid rgba(255,255,255,0.12)',
            borderRadius: 22,
            padding: '12px 16px',
            boxShadow: '0 8px 32px rgba(0,0,0,0.55), inset 0 1px 0 rgba(255,255,255,0.06)',
          }}>
            {/* Mic button */}
            <button
              onPointerDown={startRecording}
              onPointerUp={stopRecording}
              onPointerLeave={stopRecording}
              style={{
                flex: 1,
                padding: '14px 0',
                background: isRecording
                  ? 'linear-gradient(135deg, rgba(220,80,80,0.9), rgba(180,40,40,0.9))'
                  : 'linear-gradient(135deg, #4AF0FF, #00c8e0)',
                color: isRecording ? '#fff' : '#080a0c',
                border: 'none',
                borderRadius: 14,
                fontFamily: '"Rajdhani", sans-serif',
                fontWeight: 600,
                fontSize: 15,
                letterSpacing: '0.09em',
                cursor: 'pointer',
                boxShadow: isRecording
                  ? '0 0 0 1px rgba(220,80,80,0.3), 0 8px 24px rgba(220,80,80,0.45)'
                  : '0 0 0 1px rgba(74,240,255,0.2), 0 8px 24px rgba(74,240,255,0.35)',
                transition: 'all 0.18s ease',
                userSelect: 'none',
                WebkitUserSelect: 'none',
                animation: isRecording ? 'fsRecPulse 1.4s ease-in-out infinite' : 'none',
              }}
            >
              {isRecording ? '● LISTENING…' : '🎤 HOLD TO SPEAK'}
            </button>

            {/* End button */}
            <button
              onClick={endSession}
              title="End session"
              style={{
                padding: '14px 18px',
                background: 'rgba(255,255,255,0.04)',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: 14,
                color: 'rgba(255,255,255,0.35)',
                cursor: 'pointer',
                fontSize: 15,
                transition: 'all 0.18s ease',
                flexShrink: 0,
              }}
              onMouseEnter={(e) => { e.currentTarget.style.background = 'rgba(220,80,80,0.12)'; e.currentTarget.style.borderColor = 'rgba(220,80,80,0.35)'; e.currentTarget.style.color = '#ff6b6b'; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = 'rgba(255,255,255,0.04)'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)'; e.currentTarget.style.color = 'rgba(255,255,255,0.35)'; }}
            >
              ■
            </button>
          </div>

          {errorMsg && (
            <p style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 10, color: 'rgba(220,80,80,0.75)', textAlign: 'center', letterSpacing: '0.06em' }}>
              {errorMsg}
            </p>
          )}
        </div>

        <AudioCapture isRecording={isRecording} onAudioChunk={sendAudioChunk} />
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
            </p>
          </div>
        ))}
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-4">
        <button
          onPointerDown={startRecording}
          onPointerUp={stopRecording}
          onPointerLeave={stopRecording}
          className={`flex items-center gap-2 px-8 py-3 rounded-xl text-sm font-bold transition select-none ${
            isRecording
              ? 'bg-red-500 text-white shadow-lg shadow-red-500/30 scale-95'
              : 'bg-sky-500 hover:bg-sky-400 text-white shadow-lg shadow-sky-500/20'
          }`}
        >
          <span>{isRecording ? '🔴' : '🎤'}</span>
          {isRecording ? 'Listening…' : 'Hold to Speak'}
        </button>
        <button
          onClick={endSession}
          className="px-4 py-3 border border-white/20 hover:border-red-500/50 hover:text-red-400 text-gray-400 rounded-xl text-sm font-semibold transition"
        >
          ■ End
        </button>
      </div>

      {errorMsg && <p className="text-center text-xs text-red-400">{errorMsg}</p>}

      <AudioCapture isRecording={isRecording} onAudioChunk={sendAudioChunk} />
    </div>
  );
}
