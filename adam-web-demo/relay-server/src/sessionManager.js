// sessionManager.js — tracks active sessions in memory, enforces per-user caps

import { SESSION_CAPS, isTester } from './config.js';

// Map<uid, { sessionId, startedAt, turnCount, timerHandle }>
const activeSessions = new Map();

/**
 * Check whether a user can start a new session.
 * @param {string} uid
 * @param {number} totalSessions - lifetime session count from Firestore
 * @returns {{ allowed: boolean, reason?: string }}
 */
export function canStartSession(uid, totalSessions) {
  // Testers bypass every cap — no limit of any kind.
  if (isTester(uid)) return { allowed: true };

  // Block if a session is already in progress for this account.
  if (activeSessions.has(uid)) {
    return { allowed: false, reason: 'session_active' };
  }

  // Each non-tester account gets exactly one lifetime session.
  if (totalSessions >= SESSION_CAPS.MAX_SESSIONS_LIFETIME) {
    return { allowed: false, reason: 'lifetime_cap_reached' };
  }

  return { allowed: true };
}

/**
 * Register an active session and start the duration timer.
 */
export function registerSession(uid, sessionId, onExpire) {
  const startedAt = Date.now();
  // Testers have no time limit — their session runs until they disconnect.
  const timerHandle = isTester(uid)
    ? null
    : setTimeout(() => onExpire('timeout'), SESSION_CAPS.MAX_DURATION_MS);
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
  // No cooldown for anyone — regular users are blocked by the lifetime cap instead.
}

export function activeSessionCount() {
  return activeSessions.size;
}
