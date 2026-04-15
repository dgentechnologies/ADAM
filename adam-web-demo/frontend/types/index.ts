// Shared TypeScript types across the ADAM web demo

export type Emotion        = 'idle' | 'happy' | 'thinking' | 'surprised' | 'sad' | 'excited' | 'confused' | 'sarcastic';
export type FaceState      = 'idle' | 'listening' | 'speaking';
export type MouthIntensity = 'closed' | 'low' | 'medium' | 'high';
export type SessionState   = 'connecting' | 'active' | 'ended' | 'error';

// ── WebSocket protocol ────────────────────────────────────────────────────────

export type ClientMessage =
  | { type: 'auth';       token: string }
  | { type: 'audio';      data: string }   // base64 PCM 16kHz
  | { type: 'text';       text: string }
  | { type: 'end_turn' }
  | { type: 'disconnect' };

export type ServerMessage =
  | { type: 'session_ready';  sessionId: string; turnsAllowed: number; durationMs: number }
  | { type: 'audio';          data: string }   // base64 PCM 24kHz
  | { type: 'transcript';     text: string; role: 'user' | 'adam' }
  | { type: 'face_state';     state: FaceState }
  | { type: 'emotion';        emotion: Emotion; head?: string }
  | { type: 'mouth_sync';     intensity: MouthIntensity }
  | { type: 'turn_complete' }
  | { type: 'session_end';    reason: string }
  | { type: 'error';          code: string; message: string };

// ── Domain types ──────────────────────────────────────────────────────────────

export interface TranscriptEntry {
  role: 'user' | 'adam';
  text: string;
  ts:   number;
}

export interface WaitlistEntry {
  name?:     string;
  email:     string;
  company?:  string;
  use_case?: string;
}
