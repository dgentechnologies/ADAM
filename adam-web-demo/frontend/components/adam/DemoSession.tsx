'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { User } from 'firebase/auth';
import { AdamFace } from './AdamFace';
import { AudioCapture } from './AudioCapture';
import { SessionTimer } from './SessionTimer';
import styles from './DemoSession.module.css';
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
        setIsRecording(true);
        setFaceState('listening');
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

  const toggleRecording = () => {
    setIsRecording((prev) => {
      const next = !prev;
      if (!next) setFaceState('idle');
      return next;
    });
  };
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
    const recentTranscripts = transcripts.slice(-8);
    const statusLabel =
      faceState === 'listening' ? '● LISTENING'
      : faceState === 'speaking' ? '▶ SPEAKING'
      : '— IDLE';
    const statusTone =
      faceState === 'listening' ? '#4AF0FF'
      : faceState === 'speaking' ? '#ffb347'
      : 'rgba(255,255,255,0.26)';

    return (
      <div className={styles.shell}>
        <div className={styles.ambientTop} />
        <div className={styles.ambientBottom} />
        <div className={styles.gridTexture} />

        <div className={styles.topBar}>
          <div className={styles.brandBadge}>
            <span className={styles.brandDot} />
            <span className={styles.brandText}>ADAM LIVE DEMO</span>
          </div>
          <div className={styles.timerWrap}>
            <SessionTimer durationMs={durationMs} turnsAllowed={turnsAllowed} turnCount={turnCount} onExpire={endSession} compact />
          </div>
        </div>

        <main className={styles.mainLayout}>
          <section className={styles.stagePane}>
            <div className={styles.stageSubhead}>AUTONOMOUS DESKTOP AI MODULE</div>

            <div className={styles.chassisWrap}>
              <div className={styles.chassisHalo} />
              <div className={styles.chassisBody}>
                <div className={styles.chassisRim} />
                <div className={styles.faceDock}>
                  <AdamFace emotion={emotion} faceState={faceState} mouthIntensity={mouthIntensity} size={254} />
                </div>
                <div className={styles.chassisBase} />
              </div>
            </div>

            <div className={styles.statusRow}>
              <span className={styles.statusText} style={{ color: statusTone }}>{statusLabel}</span>
            </div>
          </section>

          <aside className={styles.chatPane}>
            <header className={styles.chatHeader}>
              <div>
                <p className={styles.chatTitle}>Live Conversation</p>
                <p className={styles.chatHint}>Your messages and ADAM responses appear in real time.</p>
              </div>
              <button
                onClick={endSession}
                title="End session"
                aria-label="End session"
                className={styles.endSessionBtn}
              >
                End
              </button>
            </header>

            <div ref={transcriptRef} className={styles.transcriptPanel} aria-live="polite">
              {recentTranscripts.length === 0 && (
                <p className={styles.emptyState}>Tap the mic and start speaking to ADAM.</p>
              )}
              {recentTranscripts.map((t, i) => (
                <article key={i} className={`${styles.msgRow} ${t.role === 'user' ? styles.msgRight : styles.msgLeft}`}>
                  <div className={`${styles.msgBubble} ${t.role === 'user' ? styles.userBubble : styles.adamBubble}`}>
                    {t.role === 'adam' && <span className={styles.msgTag}>ADAM</span>}
                    <span>{t.text}</span>
                  </div>
                </article>
              ))}
            </div>

            <div className={styles.controlsPanel}>
              <button
                onClick={toggleRecording}
                aria-label={isRecording ? 'Turn microphone off' : 'Turn microphone on'}
                className={`${styles.micBtn} ${isRecording ? styles.micOn : styles.micOff}`}
              >
                {isRecording ? 'Mic On • Auto Listen' : 'Mic Off'}
              </button>

              <div className={styles.metaGrid}>
                <div className={styles.metaCard}>
                  <span className={styles.metaLabel}>Turns</span>
                  <span className={styles.metaValue}>{turnCount}/{turnsAllowed}</span>
                </div>
                <div className={styles.metaCard}>
                  <span className={styles.metaLabel}>Connection</span>
                  <span className={styles.metaValue}>{state === 'active' ? 'Stable' : 'Pending'}</span>
                </div>
              </div>

              {errorMsg && <p className={styles.errorText}>{errorMsg}</p>}
            </div>
          </aside>
        </main>

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
          onClick={toggleRecording}
          className={`flex items-center gap-2 px-8 py-3 rounded-xl text-sm font-bold transition select-none ${
            isRecording
              ? 'bg-sky-500 hover:bg-sky-400 text-white shadow-lg shadow-sky-500/20'
              : 'bg-red-500 text-white shadow-lg shadow-red-500/30'
          }`}
        >
          <span>{isRecording ? '🎤' : '🔇'}</span>
          {isRecording ? 'Mic On (Auto Listen)' : 'Mic Off'}
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
