import { Companion } from './companion';
/** Firebase's persistence adapter keeps refresh credentials in Android Keystore-backed storage. */
export class NativeAuthPersistence {
  static type = 'LOCAL' as const;
  readonly type = 'LOCAL' as const;
  readonly _shouldAllowMigration = true;
  async _isAvailable() {
    try {
      await Companion.getSecret({ key: 'auth-storage-probe' });
      return true;
    } catch {
      return false;
    }
  }
  async _set(key: string, value: unknown) {
    await Companion.setSecret({ key, value: JSON.stringify(value) });
  }
  async _get<T>(key: string): Promise<T | null> {
    try {
      const { value } = await Companion.getSecret({ key });
      return value ? (JSON.parse(value) as T) : null;
    } catch {
      // A stale Keystore/session record must permit a new sign-in. Keep the old
      // ciphertext until a successful login replaces it; other secrets stay intact.
      return null;
    }
  }
  async _remove(key: string) {
    await Companion.removeSecret({ key });
  }
  _addListener(_key: string, _listener: unknown) {
    /* A single app WebView owns this store. */
  }
  _removeListener(_key: string, _listener: unknown) {
    /* No cross-tab native listeners. */
  }
}
