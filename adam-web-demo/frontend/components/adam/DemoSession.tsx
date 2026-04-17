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
          if (stateRef.current !== 'ended' && stateRef.current !== 'error') {
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
        <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: '#0a0a0a', gap: 24 }}>
          <AdamFace emotion="idle" faceState="idle" size={200} />
          <div style={{ width: 28, height: 28, border: '2px solid #4AF0FF', borderTopColor: 'transparent', borderRadius: '50%', animation: 'adamSpin 0.8s linear infinite' }} />
          <p style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 12, color: '#9a9a9a', letterSpacing: '0.1em' }}>CONNECTING TO ADAM…</p>
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
        <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: '#0a0a0a', gap: 20 }}>
          <AdamFace emotion="sad" faceState="idle" size={180} />
          <p style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 12, color: '#ff6b6b', letterSpacing: '0.1em' }}>CONNECTION ERROR</p>
          {errorMsg && <p style={{ fontFamily: '"DM Sans", sans-serif', fontSize: 13, color: '#9a9a9a', maxWidth: 320, textAlign: 'center' }}>{errorMsg}</p>}
          <button
            onClick={() => window.location.reload()}
            style={{ padding: '10px 28px', background: '#4AF0FF', color: '#0a0a0a', border: 'none', borderRadius: 10, fontFamily: '"Rajdhani", sans-serif', fontWeight: 600, fontSize: 14, letterSpacing: '0.06em', cursor: 'pointer' }}
          >
            RETRY
          </button>
        </div>
      );
    }
    return (
      <div className="text-center space-y-4 py-16">
        <p className="text-red-400 font-semibold">Connection failed</p>
        <p className="text-gray-500 text-sm">{errorMsg}</p>
        <button
          onClick={() => window.location.reload()}
          className="px-6 py-2 bg-sky-500 hover:bg-sky-400 text-white rounded-lg text-sm font-semibold transition"
        >
          Try Again
        </button>
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
    return (
      <div
        style={{
          minHeight: '100vh',
          background: '#0a0a0a',
          position: 'relative',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
        }}
      >
        {/* Timer — top-left corner */}
        <div style={{ position: 'absolute', top: 16, left: 16, zIndex: 10 }}>
          <SessionTimer
            durationMs={durationMs}
            turnsAllowed={turnsAllowed}
            turnCount={turnCount}
            onExpire={endSession}
            compact
          />
        </div>

        {/* ADAM face — centered */}
        <div
          style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 14,
            padding: '72px 24px 16px',
            width: '100%',
          }}
        >
          <AdamFace emotion={emotion} faceState={faceState} mouthIntensity={mouthIntensity} size={240} />
          <p
            style={{
              fontFamily: '"Share Tech Mono", monospace',
              fontSize: 11,
              letterSpacing: '0.12em',
              color: faceState === 'listening' ? '#4AF0FF'
                   : faceState === 'speaking'  ? '#9a9a9a'
                   : '#2a2a2a',
            }}
          >
            {faceState === 'listening' ? '● LISTENING'
             : faceState === 'speaking'  ? '▶ SPEAKING'
             : '— IDLE'}
          </p>
        </div>

        {/* Bottom: transcript + controls */}
        <div
          style={{
            width: '100%',
            maxWidth: 560,
            padding: '0 24px 36px',
            display: 'flex',
            flexDirection: 'column',
            gap: 14,
          }}
        >
          {/* Compact transcript — last 4 messages */}
          {transcripts.length > 0 && (
            <div
              ref={transcriptRef}
              style={{
                maxHeight: 100,
                overflowY: 'auto',
                display: 'flex',
                flexDirection: 'column',
                gap: 5,
              }}
            >
              {transcripts.slice(-4).map((t, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: t.role === 'adam' ? 'flex-start' : 'flex-end' }}>
                  <span
                    style={{
                      fontFamily: '"DM Sans", sans-serif',
                      fontSize: 12,
                      color: t.role === 'adam' ? '#9a9a9a' : '#f0f0f0',
                      background: t.role === 'adam' ? 'rgba(74,240,255,0.06)' : 'rgba(255,255,255,0.07)',
                      border: `1px solid ${t.role === 'adam' ? 'rgba(74,240,255,0.12)' : 'rgba(255,255,255,0.08)'}`,
                      padding: '4px 10px',
                      borderRadius: 8,
                      maxWidth: '82%',
                      lineHeight: 1.5,
                    }}
                  >
                    {t.role === 'adam' && (
                      <span style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 9, color: '#4AF0FF', marginRight: 5, letterSpacing: '0.06em' }}>ADAM </span>
                    )}
                    {t.text}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Controls */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 12 }}>
            <button
              onPointerDown={startRecording}
              onPointerUp={stopRecording}
              onPointerLeave={stopRecording}
              style={{
                padding: '14px 40px',
                background: isRecording ? 'rgba(220,80,80,0.9)' : '#4AF0FF',
                color: isRecording ? '#fff' : '#0a0a0a',
                border: 'none',
                borderRadius: 14,
                fontFamily: '"Rajdhani", sans-serif',
                fontWeight: 600,
                fontSize: 15,
                letterSpacing: '0.07em',
                cursor: 'pointer',
                boxShadow: isRecording ? '0 0 22px rgba(220,80,80,0.45)' : '0 0 22px rgba(74,240,255,0.3)',
                transition: 'all 0.15s ease',
                userSelect: 'none',
                WebkitUserSelect: 'none',
              }}
            >
              {isRecording ? '🔴 LISTENING…' : '🎤 HOLD TO SPEAK'}
            </button>
            <button
              onClick={endSession}
              title="End session"
              style={{
                padding: '14px 18px',
                background: 'transparent',
                border: '1px solid rgba(255,255,255,0.12)',
                borderRadius: 14,
                color: '#555',
                cursor: 'pointer',
                fontSize: 16,
                transition: 'all 0.15s ease',
              }}
              onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'rgba(220,80,80,0.5)'; e.currentTarget.style.color = '#ff6b6b'; }}
              onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'rgba(255,255,255,0.12)'; e.currentTarget.style.color = '#555'; }}
            >
              ■
            </button>
          </div>

          {errorMsg && (
            <p style={{ fontFamily: '"Share Tech Mono", monospace', fontSize: 11, color: '#ff6b6b', textAlign: 'center' }}>
              {errorMsg}
            </p>
          )}
        </div>

        <AudioCapture isRecording={isRecording} onAudioChunk={sendAudioChunk} />

        <style>{`
          @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600&family=Share+Tech+Mono&family=DM+Sans:wght@300;400&display=swap');
        `}</style>
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
