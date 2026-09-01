'use client';

import { motion } from 'framer-motion';

import { cn } from '../lib/cn';
import { AdamFaceMark } from './adam-face-mark';

export interface RadarSweepProps {
  /** Stops the sweep and settles the rings once a device is found. */
  found?: boolean;
  /**
   * `square` frames the sweep in a large rounded square with a hairline, which is
   * how `finding_adam` draws it. `circle` leaves it unframed.
   */
  shape?: 'circle' | 'square';
  className?: string;
}

const RINGS = [0.36, 0.58, 0.8, 1] as const;

/**
 * The "Looking for ADAM…" radar.
 *
 * Stitch shipped this as a WebGL fragment shader. It is rebuilt as SVG plus
 * Framer Motion instead: a shader cannot inherit the theme's CSS variables, adds
 * a canvas context to a WebView that also has to render camera preview, and
 * fails closed on devices with no GL. Reduced-motion is honoured for free
 * because the plugin neutralises animation duration globally.
 */
export function RadarSweep({ found = false, shape = 'circle', className }: RadarSweepProps) {
  return (
    <div
      className={cn(
        'relative aspect-square w-full max-w-sm',
        shape === 'square' && 'overflow-hidden rounded-[2rem] border border-border bg-surface',
        className,
      )}
      aria-hidden
    >
      <svg viewBox="0 0 200 200" className="absolute inset-0 h-full w-full">
        {RINGS.map((scale, index) => (
          <motion.circle
            key={scale}
            cx="100"
            cy="100"
            r={94 * scale}
            fill="none"
            stroke="var(--adam-fg)"
            strokeWidth="0.5"
            initial={{ opacity: 0.08 }}
            animate={
              found
                ? { opacity: 0.24, scale: 1 }
                : { opacity: [0.06, 0.22, 0.06], scale: [1, 1.015, 1] }
            }
            transition={{
              duration: 3.4,
              delay: index * 0.28,
              repeat: found ? 0 : Infinity,
              ease: 'easeInOut',
            }}
            style={{ transformOrigin: '100px 100px' }}
          />
        ))}

        {/* Crosshairs — static, so the sweep has something to read against. */}
        <line x1="6" y1="100" x2="194" y2="100" stroke="var(--adam-fg)" strokeWidth="0.5" opacity="0.08" />
        <line x1="100" y1="6" x2="100" y2="194" stroke="var(--adam-fg)" strokeWidth="0.5" opacity="0.08" />
      </svg>

      {/* Rotating wedge. A conic gradient is used rather than an SVG arc so the
          trailing fade is smooth without stacking a dozen paths. */}
      {found ? null : (
        <motion.div
          className="absolute inset-0 rounded-full"
          style={{
            background:
              'conic-gradient(from 0deg, transparent 0deg, transparent 300deg, rgba(255,255,255,0.16) 350deg, rgba(255,255,255,0.34) 360deg)',
            maskImage: 'radial-gradient(circle, #000 0%, #000 97%, transparent 100%)',
            WebkitMaskImage: 'radial-gradient(circle, #000 0%, #000 97%, transparent 100%)',
          }}
          animate={{ rotate: 360 }}
          transition={{ duration: 3.4, repeat: Infinity, ease: 'linear' }}
        />
      )}

      <div className="absolute inset-0 flex items-center justify-center">
        <AdamFaceMark expression={found ? 'happy' : 'thinking'} size="lg" />
      </div>
    </div>
  );
}

/**
 * Indeterminate progress for the OTA and payment screens: a hairline track with
 * a travelling white segment. No spinner glyph — the system has none.
 */
export function ProgressTrack({
  value,
  className,
}: {
  /** 0–100. Omit for indeterminate. */
  value?: number;
  className?: string;
}) {
  return (
    <div
      className={cn('h-0.5 w-full overflow-hidden rounded-full bg-border-strong', className)}
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      {...(value === undefined ? {} : { 'aria-valuenow': Math.round(value) })}
    >
      {value === undefined ? (
        <motion.span
          className="block h-full w-1/3 rounded-full bg-fg"
          animate={{ x: ['-100%', '300%'] }}
          transition={{ duration: 1.6, repeat: Infinity, ease: 'easeInOut' }}
        />
      ) : (
        <motion.span
          className="block h-full rounded-full bg-fg"
          animate={{ width: `${Math.min(100, Math.max(0, value))}%` }}
          transition={{ duration: 0.4, ease: 'easeOut' }}
        />
      )}
    </div>
  );
}
