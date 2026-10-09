import { Capacitor } from '@capacitor/core';
import { GoogleAuth } from '@codetrix-studio/capacitor-google-auth';
import { signInWithPopup } from 'firebase/auth';
import { getFirebaseAuth, googleProvider, GOOGLE_WEB_CLIENT_ID } from '../firebase/config';

let initialization: Promise<void> | undefined;
export function initializeGoogleAuth(): Promise<void> {
  if (!initialization) initialization = GoogleAuth.initialize({
    clientId: GOOGLE_WEB_CLIENT_ID,
    scopes: ['profile', 'email'],
    grantOfflineAccess: false,
  }).catch((error) => { initialization = undefined; throw error; });
  return initialization;
}

export async function signOutNativeGoogle() {
  if (!Capacitor.isNativePlatform()) return;
  await initializeGoogleAuth();
  await GoogleAuth.signOut();
}
export async function performNativeGoogleSignIn(): Promise<{
  idToken?: string;
  isWebPopupFallback?: boolean;
}> {
  if (!Capacitor.isNativePlatform()) {
    await signInWithPopup(getFirebaseAuth(), googleProvider);
    return { isWebPopupFallback: true };
  }
  await initializeGoogleAuth();
  const result = await GoogleAuth.signIn();
  if (!result.authentication?.idToken)
    throw new Error('Google did not return an account token. Please try again.');
  return { idToken: result.authentication.idToken };
}
