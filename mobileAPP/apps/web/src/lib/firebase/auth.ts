import type { FirestoreUser } from '@adam/types';
import {
  GoogleAuthProvider,
  onAuthStateChanged,
  signInWithCredential,
  signOut,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  sendPasswordResetEmail,
  sendEmailVerification,
  updateProfile,
  type User,
} from 'firebase/auth';
import { doc, runTransaction, serverTimestamp } from 'firebase/firestore';
import { performNativeGoogleSignIn } from '../native/google-auth';
import { getFirebaseAuth, getFirebaseFirestore } from './config';
export interface EnsureUserResult {
  isNewUser: boolean;
  userDoc: FirestoreUser;
}
function profile(user: User): FirestoreUser {
  return {
    email: user.email ?? '',
    displayName: user.displayName ?? '',
    photoUrl: user.photoURL,
    createdAt: new Date().toISOString(),
    linkedDeviceIds: [],
  };
}
export async function ensureUserDocument(user: User): Promise<EnsureUserResult> {
  const db = getFirebaseFirestore();
  const ref = doc(db, 'users', user.uid);
  return runTransaction(db, async (transaction) => {
    const snapshot = await transaction.get(ref);
    if (snapshot.exists()) {
      const data = snapshot.data();
      return {
        isNewUser: false,
        userDoc: {
          ...profile(user), ...data,
          linkedDeviceIds: Array.isArray(data.linkedDeviceIds) ? data.linkedDeviceIds : [],
        } as FirestoreUser,
      };
    }
    const userDoc = profile(user);
    // A concurrent desktop sync may create this document. Retry the read on
    // contention and never replace its companion field with a profile object.
    transaction.set(ref, { ...userDoc, createdAt: serverTimestamp() }, { merge: true });
    return { isNewUser: true, userDoc };
  });
}
async function result(user: User) {
  // Authentication is usable even if the optional profile database is offline.
  const fallback = { isNewUser: false, userDoc: profile(user) };
  let timer: ReturnType<typeof setTimeout> | undefined;
  const saved = await Promise.race([
    ensureUserDocument(user).catch(() => fallback),
    new Promise<EnsureUserResult>((resolve) => {
      timer = setTimeout(() => resolve(fallback), 5000);
    }),
  ]);
  if (timer) clearTimeout(timer);
  return { user, ...saved };
}
export async function signInWithGoogle() {
  const auth = getFirebaseAuth();
  const native = await performNativeGoogleSignIn();
  const user = native.isWebPopupFallback
    ? auth.currentUser
    : (await signInWithCredential(auth, GoogleAuthProvider.credential(native.idToken))).user;
  if (!user) throw new Error('Sign-in did not complete. Please try again.');
  return result(user);
}
export async function signInWithEmail(email: string, password: string, create = false, name = '') {
  const auth = getFirebaseAuth();
  const credential = await (create
    ? createUserWithEmailAndPassword(auth, email, password)
    : signInWithEmailAndPassword(auth, email, password));
  if (create) {
    await updateProfile(credential.user, { displayName: name.trim() });
    await sendEmailVerification(credential.user).catch(() => undefined);
  }
  return result(credential.user);
}
export async function resetPassword(email: string) {
  await sendPasswordResetEmail(getFirebaseAuth(), email.trim());
}
export async function checkAuthStateAndLinkedDevice(
  timeoutMs = 3000,
): Promise<{
  signedIn: boolean;
  user: User | null;
  userDoc: FirestoreUser | null;
  hasLinkedDevice: boolean;
}> {
  const auth = getFirebaseAuth();
  const user = await new Promise<User | null>((resolve) => {
    let done = false;
    let unsubscribe = () => {};
    const finish = (u: User | null) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      unsubscribe();
      resolve(u);
    };
    const timer = setTimeout(() => finish(auth.currentUser), timeoutMs);
    unsubscribe = onAuthStateChanged(auth, finish, () => finish(null));
  });
  if (!user) return { signedIn: false, user: null, userDoc: null, hasLinkedDevice: false };
  return { signedIn: true, user, userDoc: profile(user), hasLinkedDevice: false };
}
export async function signOutUser() {
  await signOut(getFirebaseAuth());
}
export function subscribeToAuthState(callback: (user: User | null) => void) {
  return onAuthStateChanged(getFirebaseAuth(), callback);
}
export function authError(error: unknown): string {
  const code = String((error as { code?: string })?.code ?? '');
  if (/invalid-credential|wrong-password|user-not-found/.test(code))
    return 'The email or password is incorrect.';
  if (code.includes('email-already-in-use'))
    return 'An account with this email already exists. Sign in instead.';
  if (code.includes('weak-password')) return 'Use a password with at least 8 characters.';
  if (code.includes('invalid-email')) return 'Enter a valid email address.';
  if (code.includes('too-many-requests'))
    return 'Too many attempts. Please wait a moment and try again.';
  if (code.includes('network'))
    return 'Could not connect. Check your internet connection and try again.';
  if (code.includes('operation-not-allowed'))
    return 'This sign-in method is not available yet. Try Google or continue on this phone.';
  if (code.includes('popup-closed') || /cancel/i.test(String(error))) return 'Sign-in cancelled.';
  return 'Sign-in could not finish. Please try again, or continue on this phone.';
}
