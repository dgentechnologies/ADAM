// app/api/relay-token/route.ts
// Verifies a Firebase ID token, checks session caps, and mints a short-lived relay JWT.

import { NextRequest } from 'next/server';
import { adminAuth, adminDb } from '@/lib/firebaseAdmin';
import { SignJWT } from 'jose';
import { FieldValue } from 'firebase-admin/firestore';

const MAX_SESSIONS_PER_DAY = 3;

export async function POST(req: NextRequest) {
  try {
    const { idToken } = await req.json() as { idToken?: string };

    if (!idToken || typeof idToken !== 'string') {
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
        name:                name  ?? '',
        createdAt:           FieldValue.serverTimestamp(),
        lastSeenAt:          FieldValue.serverTimestamp(),
        demoSessionsToday:   0,
        lastSessionDate:     null,
        waitlisted:          false,
      });
    }

    // Mint short-lived relay JWT (60 seconds — just enough to open the WebSocket)
    const secret = new TextEncoder().encode(process.env.RELAY_JWT_SECRET!);

    const relayToken = await new SignJWT({ uid, email: email ?? '', name: name ?? 'User' })
      .setProtectedHeader({ alg: 'HS256' })
      .setIssuedAt()
      .setExpirationTime('60s')
      .sign(secret);

    return Response.json({ token: relayToken });
  } catch (err) {
    console.error('[relay-token] Unexpected error:', err);
    return Response.json({ error: 'Server error' }, { status: 500 });
  }
}
