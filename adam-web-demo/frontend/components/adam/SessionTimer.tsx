'use client';

import { useEffect, useRef, useState } from 'react';

interface SessionTimerProps {
  durationMs:   number;
  turnsAllowed: number;
  turnCount:    number;
  onExpire:     () => void;
}

function fmt(ms: number) {
  const s   = Math.max(0, Math.floor(ms / 1000));
  const m   = Math.floor(s / 60);
  const sec = s % 60;
  return `${m}:${String(sec).padStart(2, '0')}`;
}

export function SessionTimer({ durationMs, turnsAllowed, turnCount, onExpire }: SessionTimerProps) {
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

  const pct      = remaining / durationMs;
  const isWarn   = pct < 0.25;
  const turnsLeft = turnsAllowed - turnCount;

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
      <span className={`font-semibold ${turnsLeft <= 3 ? 'text-red-400' : 'text-gray-300'}`}>
        💬 {turnCount} / {turnsAllowed}
      </span>
    </div>
  );
}
