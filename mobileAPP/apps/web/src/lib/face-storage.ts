import { getItem, removeItem, setItem } from './native/preferences';
export interface UserFaceProfile {
  name: string;
  photoDataUrl: string;
  capturedAt: string;
  views?: { front?: string; left?: string; right?: string };
  deviceId?: string;
}
const KEY = 'adam.face.user_profile';
export async function saveUserFaceProfile(profile: UserFaceProfile) {
  await setItem(KEY, JSON.stringify(profile));
  window.dispatchEvent(new Event('adam:data'));
}
export async function getUserFaceProfile(): Promise<UserFaceProfile | null> {
  const raw = await getItem(KEY);
  return raw ? (JSON.parse(raw) as UserFaceProfile) : null;
}
export async function deleteUserFaceProfile() {
  await removeItem(KEY);
  if (typeof window !== 'undefined') {
    window.localStorage.removeItem(KEY);
    window.dispatchEvent(new Event('adam:data'));
  }
}
