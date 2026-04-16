// lib/firebaseAdmin.ts — Firebase Admin SDK (server-side only)
// NEVER import this in client components or NEXT_PUBLIC_ code paths.
// Initialization is lazy: Firebase is NOT touched at module load time so that
// Next.js build-phase static analysis never sees missing env vars.

import { initializeApp, getApps, cert, App } from 'firebase-admin/app';
import { getAuth, Auth }           from 'firebase-admin/auth';
import { getFirestore, Firestore } from 'firebase-admin/firestore';

function getAdminApp(): App {
  if (getApps().length) return getApps()[0];

  const privateKey = process.env.FIREBASE_ADMIN_PRIVATE_KEY?.replace(/\\n/g, '\n');

  return initializeApp({
    credential: cert({
      projectId:   process.env.FIREBASE_ADMIN_PROJECT_ID!,
      clientEmail: process.env.FIREBASE_ADMIN_CLIENT_EMAIL!,
      privateKey,
    }),
  });
}

// Lazy proxy — Firebase is only initialised on the first property access,
// which only happens inside a request handler, never at import/build time.
export const adminAuth = new Proxy({} as Auth, {
  get(_target, prop: string) {
    const auth = getAuth(getAdminApp());
    const value = (auth as unknown as Record<string, unknown>)[prop];
    return typeof value === 'function' ? (value as Function).bind(auth) : value;
  },
});

export const adminDb = new Proxy({} as Firestore, {
  get(_target, prop: string) {
    const db = getFirestore(getAdminApp());
    const value = (db as unknown as Record<string, unknown>)[prop];
    return typeof value === 'function' ? (value as Function).bind(db) : value;
  },
});
