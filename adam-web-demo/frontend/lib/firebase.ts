// lib/firebase.ts — Firebase client SDK (browser-side only)
// Initialization is LAZY: Firebase is NOT touched at module-load / SSR time.
// If the NEXT_PUBLIC_FIREBASE_API_KEY env var is absent (e.g. Vercel project
// settings not yet configured), Firebase is simply skipped — no crash.

import { initializeApp, getApps, getApp, type FirebaseApp } from 'firebase/app';
import { getAuth as _getAuth, GoogleAuthProvider, type Auth } from 'firebase/auth';
import { getFirestore as _getFirestore, type Firestore } from 'firebase/firestore';

/** Returns true only when all required Firebase client config vars are present. */
export function isFirebaseConfigured(): boolean {
  const key = process.env.NEXT_PUBLIC_FIREBASE_API_KEY;
  return !!(key && key.length > 0 && key !== 'your_firebase_api_key');
}

const firebaseConfig = {
  apiKey:            process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain:        process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId:         process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
  storageBucket:     process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
  appId:             process.env.NEXT_PUBLIC_FIREBASE_APP_ID,
};

let _app:  FirebaseApp | undefined;
let _auth: Auth        | undefined;
let _db:   Firestore   | undefined;

function getFirebaseApp(): FirebaseApp {
  if (!_app) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    _app = getApps().length ? getApp() : initializeApp(firebaseConfig as any);
  }
  return _app;
}

/** Returns a Proxy<T> that defers construction until the first property access. */
function makeLazyProxy<T extends object>(factory: () => T): T {
  let instance: T | undefined;
  return new Proxy({} as T, {
    get(_, prop) {
      if (!instance) instance = factory();
      const val = Reflect.get(instance, prop, instance);
      return typeof val === 'function' ? (val as (...a: unknown[]) => unknown).bind(instance) : val;
    },
  });
}

export const auth = makeLazyProxy<Auth>(     () => { if (!_auth) _auth = _getAuth(getFirebaseApp());      return _auth; });
export const db   = makeLazyProxy<Firestore>(() => { if (!_db)   _db   = _getFirestore(getFirebaseApp()); return _db;   });

// GoogleAuthProvider is a plain class — safe to construct without Firebase app.
export const googleProvider = new GoogleAuthProvider();
googleProvider.addScope('profile');
googleProvider.addScope('email');
