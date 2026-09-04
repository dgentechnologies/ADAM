'use client';

import { Button, Screen, ScreenActions } from '@adam/ui';
import { ArrowRight, CheckCircle2, Pencil, RotateCcw, RotateCw, Camera } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useCallback, useEffect, useRef, useState } from 'react';

import { useSetupStore } from '@/stores/setup-store';
import { useFaceCamera } from './use-face-camera';

// ─── Types ─────────────────────────────────────────────────────────────────
type ViewStep = 'front' | 'left' | 'right';
type Phase =
  | 'starting'
  | 'capture-front'
  | 'confirm-front'
  | 'capture-left'
  | 'confirm-left'
  | 'capture-right'
  | 'confirm-right'
  | 'processing'
  | 'done'
  | 'error';

interface ViewConfig {
  step: ViewStep;
  label: string;
  instruction: string;
  hint: string;
  Icon: React.ElementType;
}

const VIEWS: ViewConfig[] = [
  {
    step: 'front',
    label: 'Look straight ahead',
    instruction: 'Face forward',
    hint: 'Keep your face centred and look directly at the camera.',
    Icon: Camera,
  },
  {
    step: 'left',
    label: 'Turn your head left',
    instruction: 'Turn left',
    hint: 'Slowly rotate your head about 45° to the left.',
    Icon: RotateCcw,
  },
  {
    step: 'right',
    label: 'Turn your head right',
    instruction: 'Turn right',
    hint: 'Slowly rotate your head about 45° to the right.',
    Icon: RotateCw,
  },
];

const VIEW_PHASES: Record<ViewStep, { capture: Phase; confirm: Phase; next: Phase }> = {
  front: { capture: 'capture-front', confirm: 'confirm-front', next: 'capture-left' },
  left: { capture: 'capture-left', confirm: 'confirm-left', next: 'capture-right' },
  right: { capture: 'capture-right', confirm: 'confirm-right', next: 'processing' },
};

const COUNTDOWN_MS = 1000;
const CONFIRM_FLASH_MS = 600;

// ─── Countdown ring component ───────────────────────────────────────────────
function CountdownRing({
  progress,
  size,
  isAligned,
}: {
  progress: number; // 0 → 1
  size: number;
  isAligned: boolean;
}) {
  const r = (size - 6) / 2;
  const circ = 2 * Math.PI * r;
  const dash = circ * progress;
  const strokeColor = isAligned ? '#10b981' : 'rgba(255,255,255,0.9)';
  const glow = isAligned
    ? 'drop-shadow(0 0 10px rgba(16, 185, 129, 0.9))'
    : 'drop-shadow(0 0 6px rgba(255,255,255,0.7))';

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      style={{
        transform: 'rotate(-90deg)',
        position: 'absolute',
        inset: 0,
        pointerEvents: 'none',
        zIndex: 15,
      }}
    >
      {/* Track */}
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={isAligned ? 'rgba(16, 185, 129, 0.2)' : 'rgba(255,255,255,0.1)'}
        strokeWidth={3}
      />
      {/* Progress */}
      {progress > 0 && (
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={strokeColor}
          strokeWidth={3.5}
          strokeLinecap="round"
          strokeDasharray={`${dash} ${circ}`}
          style={{ filter: glow, transition: 'stroke 0.2s ease' }}
        />
      )}
    </svg>
  );
}

// ─── Human Face Outline Guide ────────────────────────────────────────────────
function FaceOutline({
  size,
  isAligned,
  step,
}: {
  size: number;
  isAligned: boolean;
  step: ViewStep;
}) {
  const strokeColor = isAligned ? '#10b981' : 'rgba(255, 255, 255, 0.38)';
  const strokeWidth = isAligned ? 2.5 : 1.75;
  const glow = isAligned ? 'drop-shadow(0 0 10px rgba(16, 185, 129, 0.8))' : 'none';

  // Front, left, right head contour adjustments
  const headPath =
    step === 'left'
      ? 'M 132,44 C 164,44 186,70 184,112 C 182,152 160,182 132,182 C 104,182 82,152 84,112 C 86,70 100,44 132,44 Z'
      : step === 'right'
      ? 'M 148,44 C 180,44 196,70 196,112 C 196,152 176,182 148,182 C 120,182 98,152 96,112 C 94,70 116,44 148,44 Z'
      : 'M 140,44 C 172,44 194,70 194,112 C 194,152 170,182 140,182 C 110,182 86,152 86,112 C 86,70 108,44 140,44 Z';

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      style={{
        position: 'absolute',
        inset: 0,
        pointerEvents: 'none',
        zIndex: 10,
        filter: glow,
        transition: 'filter 0.3s ease',
      }}
    >
      {/* Head contour */}
      <path
        d={headPath}
        fill="none"
        stroke={strokeColor}
        strokeWidth={strokeWidth}
        strokeDasharray={isAligned ? undefined : '5 4'}
        style={{ transition: 'stroke 0.25s ease, stroke-width 0.25s ease' }}
      />

      {/* Neck and shoulders */}
      <path
        d="M 38,272 C 68,242 96,214 114,202 L 116,180"
        fill="none"
        stroke={strokeColor}
        strokeWidth={strokeWidth}
        strokeDasharray={isAligned ? undefined : '5 4'}
        style={{ transition: 'stroke 0.25s ease, stroke-width 0.25s ease' }}
      />
      <path
        d="M 242,272 C 212,242 184,214 166,202 L 164,180"
        fill="none"
        stroke={strokeColor}
        strokeWidth={strokeWidth}
        strokeDasharray={isAligned ? undefined : '5 4'}
        style={{ transition: 'stroke 0.25s ease, stroke-width 0.25s ease' }}
      />

      {/* Eye-level alignment tick guides */}
      <line
        x1={step === 'left' ? '76' : '78'}
        y1="112"
        x2={step === 'left' ? '88' : '90'}
        y2="112"
        stroke={strokeColor}
        strokeWidth={2}
        strokeLinecap="round"
      />
      <line
        x1={step === 'right' ? '192' : '190'}
        y1="112"
        x2={step === 'right' ? '204' : '202'}
        y2="112"
        stroke={strokeColor}
        strokeWidth={2}
        strokeLinecap="round"
      />

      {/* Forehead center tick */}
      <line
        x1={step === 'left' ? '132' : step === 'right' ? '148' : '140'}
        y1="48"
        x2={step === 'left' ? '132' : step === 'right' ? '148' : '140'}
        y2="58"
        stroke={strokeColor}
        strokeWidth={2}
        strokeLinecap="round"
      />

      {/* Chin guide tick */}
      <line
        x1={step === 'left' ? '126' : step === 'right' ? '142' : '134'}
        y1="182"
        x2={step === 'left' ? '138' : step === 'right' ? '154' : '146'}
        y2="182"
        stroke={strokeColor}
        strokeWidth={2}
        strokeLinecap="round"
      />

      {/* Directional rotation cues */}
      {step === 'left' && (
        <g opacity={isAligned ? 0.95 : 0.65} transform="translate(36, 110)">
          <path
            d="M 22,0 C 12,-2 4,6 0,14"
            fill="none"
            stroke={strokeColor}
            strokeWidth={2.5}
            strokeLinecap="round"
          />
          <polyline
            points="0,6 0,14 8,14"
            fill="none"
            stroke={strokeColor}
            strokeWidth={2.5}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </g>
      )}
      {step === 'right' && (
        <g opacity={isAligned ? 0.95 : 0.65} transform="translate(220, 110)">
          <path
            d="M 0,0 C 10,-2 18,6 22,14"
            fill="none"
            stroke={strokeColor}
            strokeWidth={2.5}
            strokeLinecap="round"
          />
          <polyline
            points="22,6 22,14 14,14"
            fill="none"
            stroke={strokeColor}
            strokeWidth={2.5}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </g>
      )}
    </svg>
  );
}

// ─── Step dots ──────────────────────────────────────────────────────────────
function StepDots({ phase }: { phase: Phase }) {
  const doneSteps: ViewStep[] = [];
  if (['confirm-front', 'capture-left', 'confirm-left', 'capture-right', 'confirm-right', 'processing', 'done'].includes(phase)) doneSteps.push('front');
  if (['confirm-left', 'capture-right', 'confirm-right', 'processing', 'done'].includes(phase)) doneSteps.push('left');
  if (['confirm-right', 'processing', 'done'].includes(phase)) doneSteps.push('right');

  const activeStep: ViewStep | null =
    phase === 'capture-front' || phase === 'confirm-front' ? 'front' :
    phase === 'capture-left' || phase === 'confirm-left' ? 'left' :
    phase === 'capture-right' || phase === 'confirm-right' ? 'right' : null;

  return (
    <div style={{ display: 'flex', gap: 8, alignItems: 'center', justifyContent: 'center' }}>
      {VIEWS.map(({ step }) => {
        const done = doneSteps.includes(step);
        const active = activeStep === step;
        return (
          <div key={step} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
            <div style={{
              width: active ? 28 : 8,
              height: 8,
              borderRadius: 4,
              background: done ? '#ffffff' : active ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.2)',
              transition: 'all 0.3s ease',
              boxShadow: active ? '0 0 8px rgba(255,255,255,0.6)' : 'none',
            }} />
          </div>
        );
      })}
    </div>
  );
}

// ─── Main page ──────────────────────────────────────────────────────────────
export default function FaceCapturePage() {
  const router = useRouter();
  const setUserNameForFace = useSetupStore((s) => s.setUserNameForFace);
  const complete = useSetupStore((s) => s.complete);
  const finish = useSetupStore((s) => s.finish);

  const [phase, setPhase] = useState<Phase>('starting');
  const [countdown, setCountdown] = useState(0); // 0 → 1
  const [name, setName] = useState('');
  const [capturedFrames, setCapturedFrames] = useState<Partial<Record<ViewStep, string>>>({});
  const [isAligned, setIsAligned] = useState(false);
  const [guidanceText, setGuidanceText] = useState('Fit face inside outline');

  // Keep phaseRef in sync with phase on every render to eliminate stale closure bugs
  const phaseRef = useRef<Phase>(phase);
  phaseRef.current = phase;

  const { videoRef, canvasRef, status, errorMessage, startCamera, stopCamera, captureFrame } = useFaceCamera();
  const countdownTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const confirmTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const detectorCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const startedRef = useRef(false);

  // Store metrics for differential pose verification
  const lastMetricsRef = useRef<{ centroidX: number; centroidY: number; cheekBalance: number; faceWidth: number } | null>(null);
  const baselineRef = useRef<{ centroidX: number; centroidY: number; cheekBalance: number; faceWidth: number } | null>(null);

  // Synchronously derive current view directly from phase — no useEffect lag or stale state
  const currentStep: ViewStep =
    phase === 'capture-left' || phase === 'confirm-left'
      ? 'left'
      : phase === 'capture-right' || phase === 'confirm-right'
      ? 'right'
      : 'front';

  const currentView = VIEWS.find((v) => v.step === currentStep) ?? VIEWS[0]!;

  const doCapture = useCallback(() => {
    if (countdownTimerRef.current) {
      clearInterval(countdownTimerRef.current);
      countdownTimerRef.current = null;
    }

    const curPhase = phaseRef.current;
    let step: ViewStep;
    if (curPhase === 'capture-front' || curPhase === 'confirm-front') {
      step = 'front';
    } else if (curPhase === 'capture-left' || curPhase === 'confirm-left') {
      step = 'left';
    } else if (curPhase === 'capture-right' || curPhase === 'confirm-right') {
      step = 'right';
    } else {
      return;
    }

    const frame = captureFrame();
    if (frame) {
      setCapturedFrames((prev) => ({ ...prev, [step]: frame }));
    }

    // Save baseline profile when front view is captured
    if (step === 'front' && lastMetricsRef.current) {
      baselineRef.current = { ...lastMetricsRef.current };
    }

    const { confirm, next } = VIEW_PHASES[step];
    setPhase(confirm);
    phaseRef.current = confirm;
    setIsAligned(false); // Reset alignment for the next phase

    if (confirmTimerRef.current) clearTimeout(confirmTimerRef.current);
    confirmTimerRef.current = setTimeout(() => {
      if (next === 'processing') {
        setPhase('processing');
        phaseRef.current = 'processing';
        stopCamera();
        // Brief on-device processing
        setTimeout(() => {
          setPhase('done');
          phaseRef.current = 'done';
        }, 1200);
      } else {
        setPhase(next);
        phaseRef.current = next;
      }
    }, CONFIRM_FLASH_MS);
  }, [captureFrame, stopCamera]);

  // Boot camera on mount
  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;
    startCamera().then(() => {
      setPhase('capture-front');
    });
    return () => {
      stopCamera();
      if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
      if (confirmTimerRef.current) clearTimeout(confirmTimerRef.current);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Handle camera errors
  useEffect(() => {
    if (status === 'denied' || status === 'unavailable' || status === 'error') {
      setPhase('error');
    }
  }, [status]);

  // Real-time Face Alignment & Pose Verification:
  // 1. Enforces that face fits the human head outline (rejects far-away or misaligned faces)
  // 2. Verifies head pose for each view:
  //    - Front: user must face directly forward
  //    - Left: user MUST physically turn head left (~45°)
  //    - Right: user MUST physically turn head right (~45°)
  useEffect(() => {
    const isCapturePhase =
      phase === 'capture-front' || phase === 'capture-left' || phase === 'capture-right';
    if (!isCapturePhase || status !== 'active') {
      setIsAligned(false);
      setGuidanceText('Fit face inside outline');
      return;
    }

    let active = true;
    let consecutiveHits = 0;
    let consecutiveMisses = 0;

    const interval = setInterval(async () => {
      const video = videoRef.current;
      if (!video || video.readyState < 2 || !active) return;

      const vw = video.videoWidth || 640;
      const vh = video.videoHeight || 640;
      const side = Math.min(vw, vh);
      const sx = (vw - side) / 2;
      const sy = (vh - side) / 2;

      let faceDetectorDirection: 'front' | 'left' | 'right' | null = null;
      let faceDetectorRelW: number | null = null;

      // 1. Native Shape Detection API (Chromium / Android Chrome)
      if (typeof window !== 'undefined' && 'FaceDetector' in window) {
        try {
          const detector = new (window as any).FaceDetector({ fastMode: true, maxDetectedFaces: 1 });
          const faces = await detector.detect(video);
          if (faces && faces.length > 0) {
            const f = faces[0];
            const box = f.boundingBox;
            faceDetectorRelW = box.width / side;

            if (f.landmarks && f.landmarks.length > 0) {
              const eyes = f.landmarks.filter((l: any) => l.type === 'eye');
              const nose = f.landmarks.find((l: any) => l.type === 'nose');
              if (eyes.length >= 2) {
                const eyeMidX = (eyes[0].locations[0].x + eyes[1].locations[0].x) / 2;
                const boxMidX = box.x + box.width / 2;
                const offset = (eyeMidX - boxMidX) / box.width;
                if (offset > 0.04) faceDetectorDirection = 'left';
                else if (offset < -0.04) faceDetectorDirection = 'right';
                else if (Math.abs(offset) < 0.035) faceDetectorDirection = 'front';
              } else if (eyes.length === 1 && nose && nose.locations && nose.locations.length > 0) {
                const eyeX = eyes[0].locations[0].x;
                const noseX = nose.locations[0].x;
                if (noseX > eyeX + 4) faceDetectorDirection = 'left';
                else if (noseX < eyeX - 4) faceDetectorDirection = 'right';
              }
            }
          }
        } catch (_) {}
      }

      // 2. High-speed Canvas Pixel & Geometry Analysis (Universal across iOS, Safari, Android, Desktop)
      let canvas = detectorCanvasRef.current;
      if (!canvas) {
        canvas = document.createElement('canvas');
        canvas.width = 80;
        canvas.height = 80;
        detectorCanvasRef.current = canvas;
      }
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) return;

      // Draw square center crop with mirror reflection to match preview
      ctx.save();
      ctx.translate(80, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, sx, sy, side, side, 0, 0, 80, 80);
      ctx.restore();

      const img = ctx.getImageData(0, 0, 80, 80).data;

      let insideOutlineCount = 0;
      let sumX = 0;
      let sumY = 0;
      let leftZoneSkin = 0;
      let rightZoneSkin = 0;
      let overflowCount = 0;

      // Target head outline oval: cx = 40, cy = 32, rx = 16, ry = 20
      for (let y = 8; y < 58; y++) {
        for (let x = 16; x < 64; x++) {
          const idx = (y * 80 + x) * 4;
          const r = img[idx]!;
          const g = img[idx + 1]!;
          const b = img[idx + 2]!;

          // Robust human skin chromaticity that rejects off-white/cream walls, door frames, and ambient beige
          const diffRG = r - g;
          const diffRB = r - b;
          const isSkin =
            r > 45 && g > 30 && b > 20 &&
            diffRG >= 12 && diffRB >= 18 &&
            (r / g) >= 1.12 && r < 248;

          if (isSkin) {
            const d = ((x - 40) ** 2) / (16 ** 2) + ((y - 32) ** 2) / (20 ** 2);
            if (d <= 1.0) {
              insideOutlineCount++;
              sumX += x;
              sumY += y;
              if (x < 39) leftZoneSkin++;
              else if (x > 41) rightZoneSkin++;
            } else if (d > 1.35 && y < 50) {
              overflowCount++;
            }
          }
        }
      }

      let aligned = false;
      let nextGuidance = 'Fit face inside outline';

      // Outline area = ~1005 pixels. A properly fitted face covers 300 to 860 pixels.
      if (insideOutlineCount < 120) {
        aligned = false;
        nextGuidance = 'Position face inside outline';
      } else if (insideOutlineCount < 300) {
        aligned = false;
        nextGuidance = 'Move closer to fit outline';
      } else if (insideOutlineCount > 860 && overflowCount > 400) {
        aligned = false;
        nextGuidance = 'Move back slightly';
      } else {
        const centroidX = sumX / (insideOutlineCount * 80);
        const centroidY = sumY / (insideOutlineCount * 80);

        if (centroidY < 0.31) {
          aligned = false;
          nextGuidance = 'Lower camera / head';
        } else if (centroidY > 0.54) {
          aligned = false;
          nextGuidance = 'Raise camera / head';
        } else if (centroidX < 0.38) {
          aligned = false;
          nextGuidance = 'Move slightly right';
        } else if (centroidX > 0.62) {
          aligned = false;
          nextGuidance = 'Move slightly left';
        } else {
          // Centered and well-scaled inside the outline!
          // Now verify POSE for currentStep
          const cheekBalance = (leftZoneSkin - rightZoneSkin) / (leftZoneSkin + rightZoneSkin + 1);
          const currentMetrics = { centroidX, centroidY, cheekBalance, faceWidth: insideOutlineCount / 1005 };
          lastMetricsRef.current = currentMetrics;

          const baseBalance = baselineRef.current?.cheekBalance ?? 0;
          const baseCentroidX = baselineRef.current?.centroidX ?? 0.5;
          const deltaBalance = cheekBalance - baseBalance;
          const deltaX = centroidX - baseCentroidX;

          if (currentStep === 'front') {
            if (Math.abs(centroidX - 0.5) > 0.10 || Math.abs(cheekBalance) > 0.28) {
              aligned = false;
              nextGuidance = 'Look straight ahead';
            } else {
              aligned = true;
              nextGuidance = 'Perfect • Hold still';
            }
          } else if (currentStep === 'left') {
            // User must physically turn head left
            // Must show positive shift and MUST NOT be turned right
            const isTurnedLeft = deltaBalance >= 0.12 || deltaX <= -0.03 || faceDetectorDirection === 'left';
            const isNotRight = deltaBalance > -0.06 && faceDetectorDirection !== 'right';

            if (isTurnedLeft && isNotRight) {
              aligned = true;
              nextGuidance = 'Left Angle Aligned • Hold Still';
            } else {
              aligned = false;
              nextGuidance = 'Turn your head left (~45°)';
            }
          } else if (currentStep === 'right') {
            // User must physically turn head right
            // Must show negative shift and MUST NOT be in left pose!
            const isTurnedRight = deltaBalance <= -0.12 || deltaX >= 0.03 || faceDetectorDirection === 'right';
            const isNotLeft = deltaBalance < 0.06 && faceDetectorDirection !== 'left';

            if (isTurnedRight && isNotLeft) {
              aligned = true;
              nextGuidance = 'Right Angle Aligned • Hold Still';
            } else {
              aligned = false;
              nextGuidance = 'Turn your head right (~45°)';
            }
          }
        }
      }

      setGuidanceText(nextGuidance);

      if (aligned) {
        consecutiveHits++;
        consecutiveMisses = 0;
        if (consecutiveHits >= 2) {
          setIsAligned(true);
        }
      } else {
        consecutiveMisses++;
        consecutiveHits = 0;
        if (consecutiveMisses >= 2) {
          setIsAligned(false);
        }
      }
    }, 100);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [phase, status, currentStep, videoRef]);

  // Auto-countdown: ONLY runs when user's face is aligned inside the outline!
  useEffect(() => {
    const isCapturePhase =
      phase === 'capture-front' || phase === 'capture-left' || phase === 'capture-right';
    if (!isCapturePhase || status !== 'active' || !isAligned) {
      setCountdown(0);
      if (countdownTimerRef.current) {
        clearInterval(countdownTimerRef.current);
        countdownTimerRef.current = null;
      }
      return;
    }

    setCountdown(0);
    const start = Date.now();

    if (countdownTimerRef.current) {
      clearInterval(countdownTimerRef.current);
      countdownTimerRef.current = null;
    }

    countdownTimerRef.current = setInterval(() => {
      const elapsed = Date.now() - start;
      const progress = Math.min(elapsed / COUNTDOWN_MS, 1);
      setCountdown(progress);
      if (progress >= 1) {
        if (countdownTimerRef.current) {
          clearInterval(countdownTimerRef.current);
          countdownTimerRef.current = null;
        }
        doCapture();
      }
    }, 16);

    return () => {
      if (countdownTimerRef.current) {
        clearInterval(countdownTimerRef.current);
        countdownTimerRef.current = null;
      }
    };
  }, [phase, status, isAligned, doCapture]);

  function save() {
    if (name.trim()) setUserNameForFace(name.trim());
    complete('face-capture');
    finish();
    router.push('/home');
  }

  function skip() {
    stopCamera();
    complete('face-capture');
    finish();
    router.push('/home');
  }

  // ── Error screen ───────────────────────────────────────────────────────────
  if (phase === 'error') {
    return (
      <Screen className="pt-safe">
        <div className="flex flex-1 flex-col items-center justify-center gap-8 px-6 text-center">
          <div style={{ fontSize: 48 }}>📷</div>
          <div className="flex flex-col gap-3">
            <p className="text-headline-sm text-fg">Camera unavailable</p>
            <p className="text-body-md text-fg-muted">{errorMessage ?? 'Could not access the camera.'}</p>
          </div>
          <Button variant="ghost" size="md" onClick={skip}>Skip for now</Button>
        </div>
      </Screen>
    );
  }

  // ── Done / name entry screen ───────────────────────────────────────────────
  if (phase === 'done') {
    const captureCount = Object.keys(capturedFrames).length;
    return (
      <Screen className="pt-safe" texture>
        <div className="flex flex-1 flex-col justify-center gap-8">
          {/* Preview thumbnails */}
          <div className="flex justify-center gap-3">
            {VIEWS.map(({ step, instruction }) => {
              const frame = capturedFrames[step];
              return (
                <div key={step} className="flex flex-col items-center gap-1.5">
                  <div style={{
                    width: 64, height: 64, borderRadius: 16,
                    background: frame ? 'transparent' : 'rgba(255,255,255,0.05)',
                    border: frame ? '1.5px solid rgba(255,255,255,0.3)' : '1.5px solid rgba(255,255,255,0.1)',
                    overflow: 'hidden',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                  }}>
                    {frame
                      ? <img src={frame} alt={instruction} style={{ width: '100%', height: '100%', objectFit: 'cover', transform: 'scaleX(-1)' }} />
                      : <CheckCircle2 size={20} style={{ color: 'rgba(255,255,255,0.3)' }} />
                    }
                  </div>
                  <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.45)', fontFamily: 'monospace', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                    {instruction}
                  </span>
                </div>
              );
            })}
          </div>

          {/* Confirmation */}
          <div className="flex flex-col items-center gap-4 text-center px-6">
            <span className="flex items-center gap-3">
              <CheckCircle2 className="h-7 w-7 text-fg" strokeWidth={1.5} />
              <span className="text-headline-sm text-fg">Got it.</span>
            </span>
            <p className="max-w-sm text-body-md text-fg-muted">
              {captureCount === 3
                ? 'All 3 views captured. Your face signature is stored on ADAM only — nothing left the device.'
                : 'Your face is stored on ADAM only — nothing left the device.'}
            </p>
          </div>

          {/* Name input */}
          <div className="flex flex-col gap-3 px-6">
            <span className="flex items-center gap-2 text-label-xs uppercase tracking-wide text-fg-muted">
              <span className="h-1.5 w-1.5 rounded-full bg-fg" />
              What should ADAM call you?
            </span>
            <div className="relative flex items-center">
              <input
                type="text"
                placeholder="Your name"
                maxLength={32}
                value={name}
                onChange={(e) => setName(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && save()}
                style={{
                  width: '100%',
                  background: 'rgba(255,255,255,0.06)',
                  border: '1px solid rgba(255,255,255,0.12)',
                  borderRadius: 12,
                  padding: '14px 44px 14px 16px',
                  color: '#fff',
                  fontSize: 16,
                  outline: 'none',
                  fontFamily: 'inherit',
                }}
              />
              <Pencil size={16} style={{ position: 'absolute', right: 16, color: 'rgba(255,255,255,0.35)', pointerEvents: 'none' }} />
            </div>
          </div>
        </div>

        <ScreenActions>
          <Button block variant="primary" onClick={save}>
            <span className="uppercase tracking-wide">Save</span>
            <ArrowRight className="h-5 w-5" strokeWidth={1.5} />
          </Button>
        </ScreenActions>
      </Screen>
    );
  }

  // ── Processing ─────────────────────────────────────────────────────────────
  if (phase === 'processing') {
    return (
      <Screen className="pt-safe">
        <div className="flex flex-1 flex-col items-center justify-center gap-6">
          <div style={{
            width: 72, height: 72,
            border: '2px solid rgba(255,255,255,0.15)',
            borderTop: '2px solid rgba(255,255,255,0.9)',
            borderRadius: '50%',
            animation: 'spin 0.9s linear infinite',
          }} />
          <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
          <p className="text-body-lg text-fg-muted">Mapping your face on device…</p>
        </div>
      </Screen>
    );
  }

  // ── Capture / confirm phases ───────────────────────────────────────────────
  const isConfirm = phase === 'confirm-front' || phase === 'confirm-left' || phase === 'confirm-right';
  const viewIndex = VIEWS.findIndex((v) => v.step === currentView.step);
  const FRAME_SIZE = 280;

  return (
    <Screen className="pt-safe">
      <div className="flex flex-1 flex-col items-center justify-between py-6">
        {/* Top: Step dots + instruction */}
        <div className="flex flex-col items-center gap-4 px-6 w-full">
          <StepDots phase={phase} />
          <div className="flex flex-col items-center gap-1.5 text-center">
            <span style={{
              fontSize: 11, fontFamily: 'monospace', textTransform: 'uppercase',
              letterSpacing: '0.1em', color: 'rgba(255,255,255,0.45)',
            }}>
              Step {viewIndex + 1} of 3
            </span>
            <p className="text-headline-sm text-fg flex items-center justify-center gap-2">
              {currentView.step !== 'front' && <currentView.Icon size={20} className="text-emerald-400" />}
              <span>{currentView.label}</span>
            </p>
            <p className="text-body-md text-fg-muted max-w-xs">{currentView.hint}</p>
          </div>
        </div>

        {/* Camera frame */}
        <div style={{ position: 'relative', width: FRAME_SIZE, height: FRAME_SIZE, flexShrink: 0 }}>
          {/* Confirm flash overlay */}
          {isConfirm && (
            <div style={{
              position: 'absolute', inset: 0, zIndex: 25, borderRadius: '50%',
              background: 'rgba(16,185,129,0.3)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              animation: 'fadeOut 0.8s ease forwards',
            }}>
              <style>{`@keyframes fadeOut { from { opacity:1 } to { opacity:0 } }`}</style>
              <CheckCircle2 size={56} color="#34d399" strokeWidth={1.5} />
            </div>
          )}

          {/* Countdown ring */}
          {!isConfirm && status === 'active' && (
            <CountdownRing progress={countdown} size={FRAME_SIZE} isAligned={isAligned} />
          )}

          {/* Camera video — mirrored (selfie view) */}
          <div style={{
            position: 'relative',
            width: FRAME_SIZE, height: FRAME_SIZE, borderRadius: '50%', overflow: 'hidden',
            background: '#0a0a0a',
            border: isConfirm
              ? '3.5px solid #10b981'
              : isAligned
              ? '3.5px solid #10b981'
              : '2px solid rgba(255,255,255,0.2)',
            boxShadow: isConfirm
              ? '0 0 35px rgba(16, 185, 129, 0.5), inset 0 0 20px rgba(16, 185, 129, 0.2)'
              : isAligned
              ? '0 0 35px rgba(16, 185, 129, 0.5), inset 0 0 20px rgba(16, 185, 129, 0.2)'
              : '0 0 60px rgba(0,0,0,0.8)',
            transition: 'border 0.25s ease, box-shadow 0.25s ease',
          }}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{
                width: '100%', height: '100%',
                objectFit: 'cover',
                transform: 'scaleX(-1)', // selfie mirror flip
                display: status === 'active' ? 'block' : 'none',
              }}
            />

            {/* Human face outline guide overlay — always visible in capture phases */}
            {!isConfirm && (
              <FaceOutline size={FRAME_SIZE} isAligned={isAligned} step={currentStep} />
            )}

            {/* Loading spinner over outline while camera starts */}
            {status !== 'active' && (
              <div style={{
                position: 'absolute',
                inset: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                pointerEvents: 'none',
              }}>
                <div style={{
                  width: 36, height: 36,
                  border: '2px solid rgba(255,255,255,0.1)',
                  borderTop: '2px solid rgba(255,255,255,0.6)',
                  borderRadius: '50%',
                  animation: 'spin 0.9s linear infinite',
                }} />
                <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
              </div>
            )}
          </div>

          {/* Alignment status indicator pill */}
          {!isConfirm && (
            <div style={{
              position: 'absolute',
              bottom: -36,
              left: '50%',
              transform: 'translateX(-50%)',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              whiteSpace: 'nowrap',
              padding: '6px 14px',
              borderRadius: 9999,
              background: isAligned ? 'rgba(16, 185, 129, 0.12)' : 'rgba(255, 255, 255, 0.05)',
              border: isAligned ? '1px solid rgba(16, 185, 129, 0.4)' : '1px solid rgba(255, 255, 255, 0.12)',
              boxShadow: isAligned ? '0 0 16px rgba(16, 185, 129, 0.25)' : 'none',
              transition: 'all 0.25s ease',
            }}>
              <span style={{
                width: 6, height: 6, borderRadius: '50%',
                background: isAligned ? '#10b981' : status !== 'active' ? 'rgba(255, 255, 255, 0.3)' : 'rgba(255, 255, 255, 0.4)',
                boxShadow: isAligned ? '0 0 8px #10b981' : 'none',
              }} className={isAligned ? 'animate-pulse' : ''} />
              <span style={{
                fontSize: 10,
                fontFamily: 'monospace',
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                color: isAligned ? '#34d399' : 'rgba(255, 255, 255, 0.75)',
                fontWeight: isAligned ? 600 : 500,
              }}>
                {status !== 'active'
                  ? 'Connecting Camera...'
                  : guidanceText}
              </span>
            </div>
          )}

          {/* Hidden capture canvas */}
          <canvas ref={canvasRef} style={{ display: 'none' }} />
        </div>

        {/* Bottom actions */}
        <div className="flex flex-col items-center gap-4 px-6 w-full">
          {status === 'active' && !isConfirm && (
            <button
              onClick={doCapture}
              style={{
                width: 64, height: 64, borderRadius: '50%',
                background: 'rgba(255,255,255,0.9)',
                border: '3px solid rgba(255,255,255,0.3)',
                boxShadow: '0 0 20px rgba(255,255,255,0.3)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                cursor: 'pointer',
                transition: 'transform 0.1s ease',
              }}
              onPointerDown={(e) => { (e.currentTarget as HTMLButtonElement).style.transform = 'scale(0.93)'; }}
              onPointerUp={(e) => { (e.currentTarget as HTMLButtonElement).style.transform = 'scale(1)'; }}
            >
              <Camera size={24} color="#000" />
            </button>
          )}
          <button
            onClick={skip}
            style={{
              background: 'none', border: 'none', cursor: 'pointer',
              color: 'rgba(255,255,255,0.35)', fontSize: 13,
              fontFamily: 'inherit', padding: '8px 16px',
            }}
          >
            Skip face ID
          </button>
        </div>
      </div>
    </Screen>
  );
}
