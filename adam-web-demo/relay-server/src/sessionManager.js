// sessionManager.js — tracks active sessions in memory, enforces per-user caps

import { SESSION_CAPS, isTester } from './config.js';

// Map<uid, { sessionId, startedAt, turnCount, timerHandle }>
const activeSessions = new Map();

// Map<uid, cooldownEndTimestamp>
const cooldownMap = new Map();

/**
 * Check whether a user can start a new session.
 * @param {string} uid
 * @param {number} dailySessions - count from Firestore for today
 * @returns {{ allowed: boolean, reason?: string }}
 */
export function canStartSession(uid, dailySessions) {
  // Testers are always allowed — no cooldown, no daily cap, no active-session block.
  if (isTester(uid)) return { allowed: true };

  if (activeSessions.has(uid)) {
    return { allowed: false, reason: 'session_active' };
  }

  const cooldownEnd = cooldownMap.get(uid);
  if (cooldownEnd && Date.now() < cooldownEnd) {
    const waitSec = Math.ceil((cooldownEnd - Date.now()) / 1000);
    return { allowed: false, reason: `cooldown_${waitSec}s` };
  }

  if (dailySessions >= SESSION_CAPS.MAX_SESSIONS_PER_DAY) {
    return { allowed: false, reason: 'daily_cap_reached' };
  }

  return { allowed: true };
}

/**
 * Register an active session and start the duration timer.
 */
export function registerSession(uid, sessionId, onExpire) {
  const startedAt = Date.now();
  const timerHandle = setTimeout(() => onExpire('timeout'), SESSION_CAPS.MAX_DURATION_MS);
  activeSessions.set(uid, { sessionId, startedAt, turnCount: 0, timerHandle });
}

/**
 * Increment turn counter.
 * @returns {{ turnCount: number, capReached: boolean }}
 */
export function incrementTurn(uid) {
  const session = activeSessions.get(uid);
  if (!session) return { turnCount: 0, capReached: false };
  session.turnCount += 1;
  // Testers never hit the turn cap.
  const capReached = !isTester(uid) && session.turnCount >= SESSION_CAPS.MAX_TURNS;
  return { turnCount: session.turnCount, capReached };
}

export function remainingMs(uid) {
  const session = activeSessions.get(uid);
  if (!session) return 0;
  return Math.max(0, SESSION_CAPS.MAX_DURATION_MS - (Date.now() - session.startedAt));
}

export function getSession(uid) {
  return activeSessions.get(uid) ?? null;
}

export function removeSession(uid) {
  const session = activeSessions.get(uid);
  if (session) {
    clearTimeout(session.timerHandle);
    activeSessions.delete(uid);
  }
  // Testers have no cooldown.
  if (!isTester(uid)) {
    cooldownMap.set(uid, Date.now() + SESSION_CAPS.COOLDOWN_MS);
  }
}

export function activeSessionCount() {
  return activeSessions.size;
}
