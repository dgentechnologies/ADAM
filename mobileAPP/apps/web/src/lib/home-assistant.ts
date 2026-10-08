import { Capacitor, CapacitorHttp } from '@capacitor/core';
import { z } from 'zod';
import { getSecret } from './native/secure-storage';

export const Entity = z.object({
  entity_id: z.string(),
  state: z.string(),
  attributes: z.object({ friendly_name: z.string().optional() }).passthrough(),
});
export type Entity = z.infer<typeof Entity>;
export async function savedHomeToken(baseUrl: string): Promise<string | null> {
  const raw = await getSecret('home-assistant');
  if (!raw) return null;
  const saved = z.object({ url: z.string(), token: z.string() }).safeParse(JSON.parse(raw));
  return saved.success && saved.data.url === validateHomeUrl(baseUrl) ? saved.data.token : null;
}
export function validateHomeUrl(value: string): string {
  const url = new URL(value.trim());
  if (
    !['https:', 'http:'].includes(url.protocol) ||
    url.username ||
    url.password ||
    url.search ||
    url.hash
  )
    throw new Error('Enter your Home Assistant HTTP or HTTPS address without credentials.');
  // HTTP is supported for a user's own local home server. Public servers must
  // use HTTPS to protect the long-lived access token.
  const host = url.hostname;
  const octets = host.split('.').map(Number);
  const ipv4 =
    /^\d+\.\d+\.\d+\.\d+$/.test(host) &&
    octets.every((n) => Number.isInteger(n) && n >= 0 && n <= 255);
  const local =
    host === 'localhost' ||
    host === '[::1]' ||
    host.endsWith('.local') ||
    (!host.includes('.') && !host.includes(':')) ||
    (ipv4 &&
      (octets[0] === 10 ||
        octets[0] === 127 ||
        (octets[0] === 192 && octets[1] === 168) ||
        (octets[0] === 172 && (octets[1] ?? 0) >= 16 && (octets[1] ?? 0) <= 31)));
  if (url.protocol === 'http:' && !local)
    throw new Error('Use HTTPS for a Home Assistant server outside your local network.');
  return url.toString().replace(/\/$/, '');
}
async function request(
  baseUrl: string,
  path: string,
  method: 'GET' | 'POST',
  token: string,
  body?: object,
): Promise<unknown> {
  const url = `${validateHomeUrl(baseUrl)}/api/${path}`;
  const headers = { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' };
  if (Capacitor.isNativePlatform()) {
    const response = await CapacitorHttp.request({
      url,
      method,
      headers,
      data: body,
      connectTimeout: 8000,
      readTimeout: 8000,
      disableRedirects: true,
    });
    if (response.status < 200 || response.status >= 300)
      throw new Error(
        response.status === 401
          ? 'This access token was not accepted. Check it in Home Assistant.'
          : `Home Assistant returned ${response.status}.`,
      );
    return response.data;
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch(url, {
      method,
      headers,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
      redirect: 'error',
    });
    if (!response.ok)
      throw new Error(
        response.status === 401
          ? 'This access token was not accepted.'
          : `Home Assistant returned ${response.status}.`,
      );
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
}
export async function readHome(baseUrl: string, suppliedToken?: string): Promise<Entity[]> {
  const token = suppliedToken ?? (await savedHomeToken(baseUrl));
  if (!token) throw new Error('Enter a Home Assistant access token to connect.');
  const data = z.array(Entity).parse(await request(baseUrl, 'states', 'GET', token));
  return data.filter((e) =>
    /^(light|switch|scene|sensor|binary_sensor|climate|cover|fan)\./.test(e.entity_id),
  );
}
export async function setHomePower(baseUrl: string, entity: Entity): Promise<void> {
  const domain = entity.entity_id.split('.')[0];
  if (!domain || !['light', 'switch', 'fan', 'scene'].includes(domain))
    throw new Error('This entity is read-only here.');
  if (['unavailable', 'unknown'].includes(entity.state))
    throw new Error('This device is unavailable.');
  const token = await savedHomeToken(baseUrl);
  if (!token) throw new Error('Reconnect Home Assistant to control this device.');
  await request(
    baseUrl,
    `${domain === 'scene' ? 'services/scene/turn_on' : `services/${domain}/${entity.state === 'on' ? 'turn_off' : 'turn_on'}`}`,
    'POST',
    token,
    { entity_id: entity.entity_id },
  );
}
