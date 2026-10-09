import { Capacitor, CapacitorHttp } from '@capacitor/core';
import { setSecret } from './native/secure-storage';
import { updateLocalData } from './local-data';

export async function verifyAndSaveGeminiKey(key: string) {
  const value = key.trim();
  if (value.length < 20) throw new Error('Paste a valid Gemini API key.');
  const url = 'https://generativelanguage.googleapis.com/v1beta/models';
  let status: number;
  if (Capacitor.isNativePlatform()) status = (await CapacitorHttp.get({ url, headers: { 'x-goog-api-key': value }, connectTimeout: 10000, readTimeout: 10000 })).status;
  else {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 10000);
    try { status = (await fetch(url, { headers: { 'x-goog-api-key': value }, signal: controller.signal })).status; }
    finally { clearTimeout(timer); }
  }
  if (status !== 200) throw new Error('Google could not verify this key. Check its API access and try again.');
  await setSecret('gemini-api-key', value);
  await updateLocalData((data) => ({ ...data, brain: 'byok' }));
}
