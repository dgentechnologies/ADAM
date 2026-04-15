'use client';

import styles from './AdamFace.module.css';

type Emotion        = 'idle' | 'happy' | 'thinking' | 'surprised' | 'sad' | 'excited' | 'confused' | 'sarcastic';
type FaceState      = 'idle' | 'listening' | 'speaking';
type MouthIntensity = 'closed' | 'low' | 'medium' | 'high';

interface AdamFaceProps {
  emotion?:        Emotion;
  faceState?:      FaceState;
  mouthIntensity?: MouthIntensity;
  size?:           number;
}

const EMOTION_EYE_CLASS: Record<Emotion, string> = {
  idle:      styles.eyeIdle,
  happy:     styles.eyeHappy,
  thinking:  styles.eyeThinking,
  surprised: styles.eyeSurprised,
  sad:       styles.eyeSad,
  excited:   styles.eyeExcited,
  confused:  styles.eyeConfused,
  sarcastic: styles.eyeSarcastic,
};

const MOUTH_CLASS: Record<MouthIntensity, string> = {
  closed: styles.mouthClosed,
  low:    styles.mouthLow,
  medium: styles.mouthMedium,
  high:   styles.mouthHigh,
};

export function AdamFace({
  emotion       = 'idle',
  faceState     = 'idle',
  mouthIntensity = 'closed',
  size          = 220,
}: AdamFaceProps) {
  const isListening = faceState === 'listening';
  const isSpeaking  = faceState === 'speaking';

  return (
    <div
      className={`${styles.face} ${isListening ? styles.listening : ''} ${isSpeaking ? styles.speaking : ''}`}
      style={{ width: size, height: size }}
      aria-label={`ADAM face — ${emotion}, ${faceState}`}
    >
      <div className={styles.glowRing} />
      <div className={styles.eyes}>
        <div className={`${styles.eye} ${styles.eyeLeft} ${EMOTION_EYE_CLASS[emotion]}`}>
          <div className={styles.pupil} />
        </div>
        <div className={`${styles.eye} ${styles.eyeRight} ${EMOTION_EYE_CLASS[emotion]}`}>
          <div className={styles.pupil} />
        </div>
      </div>
      <div className={styles.nose} />
      <div className={`${styles.mouth} ${MOUTH_CLASS[mouthIntensity]}`} />

      {isListening && (
        <>
          <div className={styles.pulseRing1} />
          <div className={styles.pulseRing2} />
        </>
      )}

      {isSpeaking && (
        <div className={styles.waveform}>
          {[...Array(5)].map((_, i) => (
            <div key={i} className={styles.waveBar} style={{ animationDelay: `${i * 0.1}s` }} />
          ))}
        </div>
      )}
    </div>
  );
}
