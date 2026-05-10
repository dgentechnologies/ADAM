// firestoreClient.js — Firestore helpers for session & user tracking

import { getFirestore } from './firebaseAdmin.js';
import { FieldValue } from 'firebase-admin/firestore';

function db() {
  return getFirestore();
}

// ── User helpers ──────────────────────────────────────────────────────────────

/**
 * Upsert a user document keyed by Firebase UID.
 * Returns the user data (post-upsert).
 */
export async function upsertUser({ uid, email, name }) {
  const ref  = db().collection('adamUsers').doc(uid);
  const snap = await ref.get();

  if (snap.exists) {
    await ref.update({
      lastSeenAt: FieldValue.serverTimestamp(),
      ...(name && { name }),
    });
  } else {
    await ref.set({
      uid,
      email,
      name:                name ?? '',
      createdAt:           FieldValue.serverTimestamp(),
      lastSeenAt:          FieldValue.serverTimestamp(),
      demoSessionsToday:   0,
      totalDemoSessions:   0,
      lastSessionDate:     null,
      waitlisted:          false,
    });
  }

  return (await ref.get()).data();
}

/**
 * Get user doc by UID.
 */
export async function getUserByUid(uid) {
  const snap = await db().collection('adamUsers').doc(uid).get();
  return snap.exists ? snap.data() : null;
}

// ── Session helpers ───────────────────────────────────────────────────────────

/**
 * Create a session document and return its auto-generated ID.
 */
export async function createSession({ uid, userAgent, countryCode }) {
  const ref = await db().collection('demoSessions').add({
    uid,
    startedAt:   FieldValue.serverTimestamp(),
    endedAt:     null,
    durationMs:  null,
    turnCount:   0,
    endReason:   null,
    userAgent:   userAgent ?? '',
    countryCode: countryCode ?? '',
  });
  return ref.id;
}

/**
 * Mark a session as ended with final stats.
 */
export async function endSession({ sessionId, durationMs, turnCount, endReason }) {
  await db().collection('demoSessions').doc(sessionId).update({
    endedAt:    FieldValue.serverTimestamp(),
    durationMs,
    turnCount,
    endReason,
  });
}

/**
 * Atomically increment today's session count for a user.
 * Resets counter if it's a new day.
 */
export async function incrementSessionsToday(uid) {
  const today = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
  const ref   = db().collection('adamUsers').doc(uid);

  await db().runTransaction(async (tx) => {
    const snap = await tx.get(ref);
    const data = snap.data() ?? {};
    const isSameDay = data.lastSessionDate === today;

    tx.update(ref, {
      demoSessionsToday:  isSameDay ? FieldValue.increment(1) : 1,
      totalDemoSessions:  FieldValue.increment(1),
      lastSessionDate:    today,
    });
  });
}

// ── Waitlist helpers ──────────────────────────────────────────────────────────

export async function addToWaitlist({ email, name, company, useCase, referral }) {
  const col = db().collection('waitlist');

  // Check if already on waitlist
  const existing = await col.where('email', '==', email.toLowerCase().trim()).limit(1).get();
  if (!existing.empty) return { alreadyExists: true };

  await col.add({
    email:      email.toLowerCase().trim(),
    name:       name ?? '',
    company:    company ?? '',
    useCase:    useCase ?? '',
    referral:   referral ?? '',
    signedUpAt: FieldValue.serverTimestamp(),
    confirmed:  false,
  });

  return { alreadyExists: false };
}
