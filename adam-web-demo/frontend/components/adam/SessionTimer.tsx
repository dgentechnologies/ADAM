'use client';

import { useEffect, useRef, useState } from 'react';

interface SessionTimerProps {
  durationMs:   number;
  turnsAllowed: number;
  turnCount:    number;
  onExpire:     () => void;
  compact?:     boolean;
}

function fmt(ms: number) {
  const s   = Math.max(0, Math.floor(ms / 1000));
  const m   = Math.floor(s / 60);
  const sec = s % 60;
  return `${m}:${String(sec).padStart(2, '0')}`;
}

export function SessionTimer({ durationMs, turnsAllowed, turnCount, onExpire, compact }: SessionTimerProps) {
  const [remaining, setRemaining] = useState(durationMs);
  const startRef   = useRef(Date.now());
  const onExpireRef= useRef(onExpire);
  onExpireRef.current = onExpire;

  useEffect(() => {
    startRef.current = Date.now();
    const id = setInterval(() => {
      const elapsed = Date.now() - startRef.current;
      const left    = durationMs - elapsed;
      if (left <= 0) { clearInterval(id); setRemaining(0); onExpireRef.current(); return; }
      setRemaining(left);
    }, 500);
    return () => clearInterval(id);
  }, [durationMs]);

  const pct    = remaining / durationMs;
  const isWarn = pct < 0.25;

  if (compact) {
    return (
      <div
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: 8,
          padding: '5px 10px',
          background: 'rgba(0,0,0,0.7)',
          backdropFilter: 'blur(8px)',
          border: `1px solid ${isWarn ? 'rgba(220,80,80,0.5)' : 'rgba(74,240,255,0.25)'}`,
          borderRadius: 8,
        }}
      >
        <span
          style={{
            fontFamily: '"Share Tech Mono", monospace',
            fontSize: 13,
            fontWeight: 700,
            color: isWarn ? '#ff6b6b' : '#4AF0FF',
            letterSpacing: '0.05em',
          }}
        >
          ⏱ {fmt(remaining)}
        </span>

      </div>
    );
  }

  return (
    <div className="flex items-center justify-between px-4 py-2 rounded-xl bg-white/5 border border-white/10 text-sm">
      <span className={`font-mono font-bold ${isWarn ? 'text-red-400' : 'text-sky-400'}`}>
        ⏱ {fmt(remaining)}
      </span>
      <div className="flex-1 mx-4 h-1.5 rounded-full bg-white/10 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ease-linear ${isWarn ? 'bg-red-500' : 'bg-sky-500'}`}
          style={{ width: `${pct * 100}%` }}
        />
      </div>

    </div>
  );
}
