// app/api/relay-token/route.ts
// Verifies a Firebase ID token, checks session caps, and mints a short-lived relay JWT.

export const dynamic = 'force-dynamic';

import { NextRequest } from 'next/server';
import { adminAuth, adminDb } from '@/lib/firebaseAdmin';
import { SignJWT } from 'jose';
import { FieldValue } from 'firebase-admin/firestore';
import { randomUUID } from 'crypto';

const MAX_SESSIONS_PER_DAY = 3;
const MAX_ID_TOKEN_LENGTH = 8192;

function getRelaySecret(): Uint8Array {
  const value = process.env.RELAY_JWT_SECRET;
  if (!value) {
    throw new Error('RELAY_JWT_SECRET is not configured');
  }
  return new TextEncoder().encode(value);
}

function normalizeDisplayName(name: unknown): string {
  if (typeof name !== 'string') return 'User';
  const trimmed = name.trim();
  if (!trimmed) return 'User';
  return trimmed.slice(0, 120);
}

export async function POST(req: NextRequest) {
  try {
    const contentType = req.headers.get('content-type') ?? '';
    if (!contentType.includes('application/json')) {
      return Response.json({ error: 'Content-Type must be application/json' }, { status: 415 });
    }

    const { idToken } = await req.json() as { idToken?: string };

    if (!idToken || typeof idToken !== 'string' || idToken.length > MAX_ID_TOKEN_LENGTH) {
      return Response.json({ error: 'idToken is required' }, { status: 400 });
    }

    // Verify the Firebase ID token
    let decoded;
    try {
      decoded = await adminAuth.verifyIdToken(idToken);
    } catch {
      return Response.json({ error: 'Invalid or expired Firebase token' }, { status: 401 });
    }

    const { uid, email, name } = decoded;
    const safeName = normalizeDisplayName(name);

    // Check daily session cap in Firestore
    const userRef  = adminDb.collection('adamUsers').doc(uid);
    const userSnap = await userRef.get();
    const today    = new Date().toISOString().slice(0, 10);

    if (userSnap.exists) {
      const data = userSnap.data()!;
      const sessionsToday = data.lastSessionDate === today ? (data.demoSessionsToday ?? 0) : 0;

      if (sessionsToday >= MAX_SESSIONS_PER_DAY) {
        return Response.json(
          { error: 'Daily session limit reached. Come back tomorrow.' },
          { status: 429 },
        );
      }
    } else {
      // First-time user — create doc
      await userRef.set({
        uid,
        email:               email ?? '',
        name:                safeName,
        createdAt:           FieldValue.serverTimestamp(),
        lastSeenAt:          FieldValue.serverTimestamp(),
        demoSessionsToday:   0,
        lastSessionDate:     null,
        waitlisted:          false,
      });
    }

    // Mint short-lived relay JWT (60 seconds — just enough to open the WebSocket)
    const secret = getRelaySecret();

    const relayToken = await new SignJWT({ uid, email: email ?? '', name: safeName })
      .setProtectedHeader({ alg: 'HS256' })
      .setIssuer('dgentechnologies.com/adam')
      .setAudience('adam-relay')
      .setJti(randomUUID())
      .setIssuedAt()
      .setExpirationTime('60s')
      .sign(secret);

    return Response.json({ token: relayToken });
  } catch (err) {
    console.error('[relay-token] Unexpected error:', err);
    return Response.json({ error: 'Server error' }, { status: 500 });
  }
}
