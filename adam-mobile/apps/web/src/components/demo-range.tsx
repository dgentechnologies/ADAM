'use client';
import type { LucideIcon } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

/** Keep a dragged value responsive while native preference writes finish. */
export function DemoRange({
  label,
  icon: Icon,
  value,
  min = 0,
  onChange,
}: {
  label: string;
  icon: LucideIcon;
  value: number;
  min?: number;
  onChange: (value: number) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [draft, setDraft] = useState(value);
  useEffect(() => {
    if (document.activeElement !== input.current) setDraft(value);
  }, [value]);
  return (
    <label className="field-label">
      <span className="flex items-center justify-between gap-3">
        <span className="flex items-center gap-2">
          <Icon size={17} className="shrink-0" />
          {label}
        </span>
        <span>{draft}%</span>
      </span>
      <input
        ref={input}
        aria-label={`Demo ${label.toLowerCase()}`}
        className="mt-3 h-8 w-full accent-current"
        type="range"
        min={min}
        max={100}
        value={draft}
        onBlur={() => setDraft(value)}
        onChange={(event) => {
          const next = Number(event.target.value);
          setDraft(next);
          onChange(next);
        }}
      />
    </label>
  );
}
