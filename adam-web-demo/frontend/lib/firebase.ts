// lib/firebase.ts — Firebase client SDK (browser-side only)
// getClientAuth() and getClientDb() return the ACTUAL Firebase instances so
// that Firebase's internal instanceof checks pass correctly.
// Call these only from client-side code (useEffect, event handlers).
// Do NOT call at module-load time — they are safe to call lazily.

import { initializeApp, getApps, getApp, type FirebaseApp } from 'firebase/app';
import { getAuth as _firebaseGetAuth, GoogleAuthProvider, type Auth } from 'firebase/auth';
import { getFirestore as _firebaseGetFirestore, type Firestore } from 'firebase/firestore';

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

/**
 * Returns the real Firebase Auth instance.
 * Must only be called from client-side code (inside useEffect / event handlers).
 * Passing the return value to Firebase SDK functions (signInWithPopup, etc.)
 * works correctly because this returns an actual Auth instance, not a proxy.
 */
export function getClientAuth(): Auth {
  if (!_auth) _auth = _firebaseGetAuth(getFirebaseApp());
  return _auth;
}

/**
 * Returns the real Firestore instance.
 * Must only be called from client-side code (inside useEffect / event handlers).
 * Passing the return value to doc(), collection(), etc. works correctly
 * because this returns an actual Firestore instance, not a proxy.
 */
export function getClientDb(): Firestore {
  if (!_db) _db = _firebaseGetFirestore(getFirebaseApp());
  return _db;
}

// GoogleAuthProvider is a plain class — safe to construct without Firebase app.
export const googleProvider = new GoogleAuthProvider();
googleProvider.addScope('profile');
googleProvider.addScope('email');
