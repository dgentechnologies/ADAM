'use client';

import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react';
import {
  onAuthStateChanged,
  signInWithPopup,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  updateProfile,
  signOut as firebaseSignOut,
  type User,
} from 'firebase/auth';
import { getClientAuth, googleProvider, isFirebaseConfigured } from '@/lib/firebase';

interface AuthContextValue {
  user:    User | null;
  loading: boolean;
  signInWithGoogle:        () => Promise<void>;
  signInWithEmail:         (email: string, password: string) => Promise<void>;
  signUpWithEmail:         (email: string, password: string, name?: string) => Promise<void>;
  signOut:                 () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue>({
  user:             null,
  loading:          true,
  signInWithGoogle: async () => {},
  signInWithEmail:  async () => {},
  signUpWithEmail:  async () => {},
  signOut:          async () => {},
});

export function FirebaseAuthProvider({ children }: { children: ReactNode }) {
  const [user,    setUser]    = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!isFirebaseConfigured()) {
      setLoading(false);
      return;
    }
    const unsubscribe = onAuthStateChanged(getClientAuth(), (firebaseUser) => {
      setUser(firebaseUser);
      setLoading(false);
    });
    return unsubscribe;
  }, []);

  const signInWithGoogle = async () => {
    await signInWithPopup(getClientAuth(), googleProvider);
  };

  const signInWithEmail = async (email: string, password: string) => {
    await signInWithEmailAndPassword(getClientAuth(), email, password);
  };

  const signUpWithEmail = async (email: string, password: string, name?: string) => {
    const cred = await createUserWithEmailAndPassword(getClientAuth(), email, password);
    if (name?.trim()) await updateProfile(cred.user, { displayName: name.trim() });
  };

  const signOut = async () => {
    await firebaseSignOut(getClientAuth());
  };

  return (
    <AuthContext.Provider value={{ user, loading, signInWithGoogle, signInWithEmail, signUpWithEmail, signOut }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
