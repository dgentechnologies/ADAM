import { Capacitor } from '@capacitor/core';
import { NativeAuthPersistence } from '../native/auth-persistence';
import { getApp, getApps, initializeApp, type FirebaseApp } from 'firebase/app';
import { getAuth, initializeAuth, GoogleAuthProvider, type Auth } from 'firebase/auth';
import { getFirestore, initializeFirestore, type Firestore } from 'firebase/firestore';

/**
 * Firebase Client Configuration for ADAM Companion App.
 *
 * Config values are sourced from NEXT_PUBLIC_* environment variables,
 * with fallbacks directly mapped from adam-mobile/google-services.json.
 */
export const firebaseConfig = {
  apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY || 'AIzaSyDiOlcOPRv-oxBCkHSpneSQCelKYEe8Z8k',
  authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN || 'adam-ai1.firebaseapp.com',
  projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID || 'adam-ai1',
  storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET || 'adam-ai1.firebasestorage.app',
  messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID || '759320300226',
  appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID || '1:759320300226:android:b09edb15d499270693c95b',
};

export const GOOGLE_WEB_CLIENT_ID =
  process.env.NEXT_PUBLIC_GOOGLE_WEB_CLIENT_ID ||
  '759320300226-spjdvgmqm9ccm622v6l80slbusokvcsp.apps.googleusercontent.com';

let firebaseApp: FirebaseApp | undefined;
let authInstance: Auth | undefined;
let firestoreInstance: Firestore | undefined;

export function getFirebaseApp(): FirebaseApp {
  if (!firebaseApp) {
    firebaseApp = getApps().length > 0 ? getApp() : initializeApp(firebaseConfig);
  }
  return firebaseApp;
}

export function getFirebaseAuth(): Auth {
  if (!authInstance) {
    authInstance = Capacitor.isNativePlatform()
      ? initializeAuth(getFirebaseApp(), { persistence: NativeAuthPersistence })
      : getAuth(getFirebaseApp());
  }
  return authInstance;
}

export function getFirebaseFirestore(): Firestore {
  if (!firestoreInstance) {
    firestoreInstance = Capacitor.isNativePlatform()
      ? initializeFirestore(getFirebaseApp(), { experimentalForceLongPolling: true })
      : getFirestore(getFirebaseApp());
  }
  return firestoreInstance;
}

export const googleProvider = new GoogleAuthProvider();
googleProvider.addScope('profile');
googleProvider.addScope('email');
