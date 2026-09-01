import { isNative } from './platform';

/**
 * Auth-token storage.
 *
 * SECURITY (tech spec §7): session tokens must live in the platform keystore —
 * Keychain on iOS, EncryptedSharedPreferences on Android — never in plain
 * `localStorage`. The real implementation will be a secure-storage Capacitor
 * plugin; it is intentionally not installed in this pass.
 *
 * Until then this module refuses to persist anything on device and keeps tokens
 * in memory only. That is the safe failure mode: a lost session on relaunch is a
 * nuisance, a token sitting in a WebView's localStorage is a vulnerability.
 */
const memory = new Map<string, string>();

export async function getSecret(key: string): Promise<string | null> {
  return memory.get(key) ?? null;
}

export async function setSecret(key: string, value: string): Promise<void> {
  memory.set(key, value);

  if (isNative() && process.env.NODE_ENV !== 'production') {
    // Loud in dev so this stub cannot ship silently to a real device build.
    console.warn(
      '[adam] secure-storage is a memory-only stub; wire the keystore plugin before shipping auth.',
    );
  }
}

export async function clearSecrets(): Promise<void> {
  memory.clear();
}
