'use client';

import { useEffect, useRef, useState } from 'react';
import { cn } from '../lib/cn';

/**
 * The ADAM face mark: two rounded-rect eyes, nothing else. It is the product's
 * only mark, so it lives here rather than being re-drawn per screen.
 *
 * Expressions are carried by geometry and motion only — never colour.
 */
export type FaceExpression = 'idle' | 'listening' | 'thinking' | 'happy' | 'asleep';

export type FaceSize = 'sm' | 'md' | 'lg' | 'xl';

/** Eye width / height / gap per size, in px, kept on the 8px grid. */
const SIZES: Record<FaceSize, { w: number; h: number; gap: number }> = {
  sm: { w: 24, h: 8, gap: 8 },
  md: { w: 32, h: 10, gap: 12 },
  lg: { w: 48, h: 16, gap: 16 },
  xl: { w: 72, h: 22, gap: 24 },
};

export interface AdamFaceMarkProps {
  expression?: FaceExpression;
  size?: FaceSize;
  /** Soft white bloom behind the eyes. Off for small inline marks. */
  bloom?: boolean;
  /** Enable dynamic glance (left/right gaze shifts) and lifelike blinking. Defaults to true. */
  animated?: boolean;
  /** Enable left/right glance scanning. Defaults to true ONLY for size="xl" (hero face on /welcome). */
  glance?: boolean;
  /** Enable interactive cursor/touch tracking. Defaults to true for xl size. */
  interactive?: boolean;
  className?: string;
}

export function AdamFaceMark({
  expression = 'idle',
  size = 'lg',
  bloom = true,
  animated = true,
  glance,
  interactive,
  className,
}: AdamFaceMarkProps) {
  const { w, h, gap } = SIZES[size];
  const containerRef = useRef<HTMLDivElement>(null);
  const [pointerOffset, setPointerOffset] = useState<{ x: number; y: number } | null>(null);

  // asleep = eyes squeezed to a hairline; happy = a shorter, taller arc.
  const closed = expression === 'asleep';
  const eyeHeight = closed ? 2 : expression === 'happy' ? h + 4 : h;
  const eyeWidth = expression === 'happy' ? w - 8 : w;
  const shouldGlance = glance ?? (size === 'xl');
  const isInteractive = interactive ?? (size === 'xl');

  useEffect(() => {
    if (!isInteractive || !animated || closed || expression !== 'idle') return;

    let timeoutId: ReturnType<typeof setTimeout>;

    const handlePointerMove = (e: MouseEvent | TouchEvent) => {
      const clientX = 'touches' in e ? e.touches[0]?.clientX : (e as MouseEvent).clientX;
      const clientY = 'touches' in e ? e.touches[0]?.clientY : (e as MouseEvent).clientY;
      if (clientX === undefined || clientY === undefined) return;

      if (!containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;

      // Distance normalized (-1 to +1)
      const nx = Math.max(-1, Math.min(1, (clientX - centerX) / (window.innerWidth * 0.4)));
      const ny = Math.max(-1, Math.min(1, (clientY - centerY) / (window.innerHeight * 0.4)));

      const maxShiftX = w * 0.3;
      const maxShiftY = h * 0.35;

      setPointerOffset({
        x: Math.round(nx * maxShiftX * 10) / 10,
        y: Math.round(ny * maxShiftY * 10) / 10,
      });

      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => {
        setPointerOffset(null);
      }, 2200);
    };

    window.addEventListener('pointermove', handlePointerMove, { passive: true });
    window.addEventListener('touchmove', handlePointerMove, { passive: true });

    return () => {
      clearTimeout(timeoutId);
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('touchmove', handlePointerMove);
    };
  }, [isInteractive, animated, closed, expression, w, h]);

  return (
    <div
      ref={containerRef}
      className={cn(
        'relative flex items-center justify-center select-none',
        animated && shouldGlance && expression === 'idle' && 'animate-adam-float',
        className,
      )}
      style={{ height: `${SIZES[size].w}px` }}
      role="img"
      aria-label={`ADAM is ${expression}`}
    >
      {/* Moving eye pair wrapper (glance / gaze tracking) */}
      <div
        className={cn(
          'flex items-center justify-center will-change-transform',
          animated && shouldGlance && expression === 'idle' && !pointerOffset && 'animate-adam-glance',
          pointerOffset && 'transition-transform duration-100 ease-out',
        )}
        style={{
          gap: `${gap}px`,
          transform: pointerOffset
            ? `translate3d(${pointerOffset.x}px, ${pointerOffset.y}px, 0)`
            : undefined,
        }}
      >
        {[0, 1].map((eye) => (
          <span
            key={eye}
            className={cn(
              'block rounded-full bg-fg transition-all duration-base ease-standard will-change-transform',
              bloom && !closed && 'bloom',
              animated && expression === 'idle' && 'animate-adam-blink',
              expression === 'thinking' && 'animate-breathe',
              expression === 'listening' && 'animate-blink',
              // The second eye trails the first by a beat so the pair reads as alive.
              expression === 'thinking' && eye === 1 && '[animation-delay:300ms]',
            )}
            style={{
              width: `${eyeWidth}px`,
              height: `${eyeHeight}px`,
              transformOrigin: 'center center',
            }}
          />
        ))}
      </div>
    </div>
  );
}

