"use client";

/**
 * AdamSearchingDotPulse
 * ─────────────────────────────────────────────────────────────────────────
 * Clean, minimal, high-speed dot-matrix pulse animation for /discover.
 *
 * Visual Characteristics:
 * - Pure crisp dots (NO blur/halo).
 * - 4 pulsating rings expanding from ADAM's eyes at 1500ms cycle (375ms stagger).
 * - Smooth fade-away starting at 55% radius and reaching 0 at max radius.
 * - Ultra-high performance: batched vector path drawing at locked 60/120 FPS.
 * - Eyes mathematically locked at 50% / 50% center.
 * ─────────────────────────────────────────────────────────────────────────
 */

import { useEffect, useRef } from "react";

const PULSE_COUNT = 4;
const PULSE_DURATION_MS = 1500;
const STAGGER_MS = PULSE_DURATION_MS / PULSE_COUNT;
const START_RADIUS = 26;
const DOT_SPACING = 15; // px between dots along circumference
const DOT_SIZE = 1.6; // px radius for crisp micro-dots
const STATIC_GRAIN_SPACING = 28;
const STATIC_GRAIN_OPACITY = 0.03;

function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

interface AdamSearchingDotPulseProps {
  className?: string;
}

export function AdamSearchingDotPulse({ className }: AdamSearchingDotPulseProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) return;

    let dpr = Math.min(window.devicePixelRatio || 1, 2);
    let W = 0;
    let H = 0;
    let cx = 0;
    let cy = 0;
    let rafId = 0;
    let startTs: number | null = null;
    let grainCanvas: HTMLCanvasElement | null = null;

    function resize() {
      W = window.innerWidth || 390;
      H = window.innerHeight || 844;
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas!.width = W * dpr;
      canvas!.height = H * dpr;
      canvas!.style.width = W + "px";
      canvas!.style.height = H + "px";
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);

      cx = W / 2;
      cy = H / 2;

      // Draw faint background matrix grain once per resize
      grainCanvas = document.createElement("canvas");
      grainCanvas.width = W * dpr;
      grainCanvas.height = H * dpr;
      const gCtx = grainCanvas.getContext("2d");
      if (gCtx) {
        gCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
        gCtx.fillStyle = `rgba(255, 255, 255, ${STATIC_GRAIN_OPACITY})`;
        for (let y = STATIC_GRAIN_SPACING / 2; y < H; y += STATIC_GRAIN_SPACING) {
          for (let x = STATIC_GRAIN_SPACING / 2; x < W; x += STATIC_GRAIN_SPACING) {
            gCtx.beginPath();
            gCtx.arc(x, y, 1, 0, Math.PI * 2);
            gCtx.fill();
          }
        }
      }
    }

    function maxRadius() {
      return Math.min(W, H) * 0.40;
    }

    function drawRing(elapsedMs: number) {
      const t = (elapsedMs % PULSE_DURATION_MS) / PULSE_DURATION_MS;
      const eased = easeOutCubic(t);
      const maxR = maxRadius();
      const r = START_RADIUS + eased * (maxR - START_RADIUS);

      // Distance-based fade out:
      // Crisp from center, starts fading at fadeStartR, reaches 0 at maxR
      const fadeStartR = START_RADIUS + (maxR - START_RADIUS) * 0.52;
      const birthFade = Math.min(1, (r - START_RADIUS) / 14);

      let distanceFade = 1.0;
      if (r > fadeStartR) {
        const p = (r - fadeStartR) / (maxR - fadeStartR);
        distanceFade = Math.max(0, 1 - p);
      }

      const opacity = 0.85 * birthFade * Math.pow(distanceFade, 1.5);
      if (opacity <= 0.005) return;

      const circumference = 2 * Math.PI * r;
      const dotCount = Math.max(16, Math.floor(circumference / DOT_SPACING));
      const angleStep = (Math.PI * 2) / dotCount;

      // Batch draw crisp dots with a single path per ring (zero blur, maximum performance)
      ctx!.fillStyle = `rgba(255, 255, 255, ${opacity.toFixed(3)})`;
      ctx!.beginPath();
      for (let i = 0; i < dotCount; i++) {
        const angle = i * angleStep;
        const px = cx + Math.cos(angle) * r;
        const py = cy + Math.sin(angle) * r;
        ctx!.moveTo(px + DOT_SIZE, py);
        ctx!.arc(px, py, DOT_SIZE, 0, Math.PI * 2);
      }
      ctx!.fill();
    }

    function frame(ts: number) {
      if (startTs === null) startTs = ts;
      const elapsed = ts - startTs;

      // Pure black canvas
      ctx!.fillStyle = "#000000";
      ctx!.fillRect(0, 0, W, H);

      // Faint ambient background grain
      if (grainCanvas) {
        ctx!.drawImage(grainCanvas, 0, 0, W, H);
      }

      // Draw 4 crisp pulsating rings fading with radius
      for (let i = 0; i < PULSE_COUNT; i++) {
        drawRing(elapsed + i * STAGGER_MS);
      }

      rafId = requestAnimationFrame(frame);
    }

    resize();
    window.addEventListener("resize", resize);
    rafId = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <div
      id="adam-searching-dot-pulse"
      className={className}
      style={{
        position: "fixed",
        inset: 0,
        width: "100vw",
        height: "100dvh",
        zIndex: 0,
        overflow: "hidden",
        pointerEvents: "none",
        background: "#000000",
      }}
    >
      <canvas
        ref={canvasRef}
        id="adam-searching-canvas"
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
        }}
      />

      {/* Eyes locked at exact 50% 50% center of the pulse rings */}
      <div
        style={{
          position: "absolute",
          top: "50%",
          left: "50%",
          transform: "translate(-50%, -50%)",
          display: "flex",
          gap: 18,
          pointerEvents: "none",
          zIndex: 10,
        }}
      >
        <div
          id="adam-eyes"
          style={{ display: "flex", gap: 18 }}
        >
          <div
            className="animate-adam-blink bloom"
            style={eyeStyle}
          />
          <div
            className="animate-adam-blink bloom"
            style={eyeStyle}
          />
        </div>
      </div>
    </div>
  );
}

const eyeStyle: React.CSSProperties = {
  width: 34,
  height: 7,
  borderRadius: 4,
  background: "#FFFFFF",
  boxShadow: "0 0 16px rgba(255, 255, 255, 0.6), 0 0 32px rgba(255, 255, 255, 0.3)",
};

export default AdamSearchingDotPulse;
