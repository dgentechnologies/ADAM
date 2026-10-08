import { Capacitor } from '@capacitor/core';
import { GoogleAuth } from '@codetrix-studio/capacitor-google-auth';
import { signInWithPopup } from 'firebase/auth';
import { getFirebaseAuth, googleProvider } from '../firebase/config';
export async function performNativeGoogleSignIn(): Promise<{
  idToken?: string;
  isWebPopupFallback?: boolean;
}> {
  if (!Capacitor.isNativePlatform()) {
    await signInWithPopup(getFirebaseAuth(), googleProvider);
    return { isWebPopupFallback: true };
  }
  const result = await GoogleAuth.signIn();
  if (!result.authentication?.idToken)
    throw new Error('Google did not return an account token. Please try again.');
  return { idToken: result.authentication.idToken };
}
