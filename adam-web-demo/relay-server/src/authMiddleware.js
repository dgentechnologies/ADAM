// authMiddleware.js — validates short-lived relay JWT minted by /api/relay-token

import { jwtVerify } from 'jose';
import { CONFIG } from './config.js';

const secret = new TextEncoder().encode(CONFIG.RELAY_JWT_SECRET);

/**
 * Validates the JWT token sent in the first 'auth' message.
 * @param {string} token - The JWT string from the browser.
 * @returns {{ uid: string, email: string, name: string }} - Decoded payload.
 * @throws Error if token is invalid or expired.
 */
export async function validateToken(token) {
  if (typeof token !== 'string' || token.length < 20 || token.length > 4096) {
    throw new Error('auth_failed: Invalid token format');
  }

  try {
    const strict = await jwtVerify(token, secret, {
      algorithms: ['HS256'],
      issuer: 'dgentechnologies.com/adam',
      audience: 'adam-relay',
    });

    return mapPayload(strict.payload);
  } catch (err) {
    const message = err?.message ?? 'Token verification failed';
    const isMissingClaim = message.includes('missing required "iss" claim') || message.includes('missing required "aud" claim');

    // During rollout, old frontend deployments may still mint relay tokens
    // without iss/aud. We accept those if signature+exp+payload are valid.
    const canTryLegacy = CONFIG.ALLOW_LEGACY_RELAY_TOKENS || isMissingClaim;

    if (!canTryLegacy) {
      throw new Error(`auth_failed: ${message}`);
    }

    if (isMissingClaim) {
      console.warn(`[RELAY][auth] Legacy token detected (missing iss/aud). Accepting temporarily for migration.`);
    }

    const legacy = await jwtVerify(token, secret, {
      algorithms: ['HS256'],
    });

    return mapPayload(legacy.payload);
  }
}

function mapPayload(payload) {
  if (!payload.uid || typeof payload.uid !== 'string') {
    throw new Error('auth_failed: Token missing uid claim');
  }

  if (payload.email && typeof payload.email !== 'string') {
    throw new Error('auth_failed: Token email claim is invalid');
  }

  if (payload.name && typeof payload.name !== 'string') {
    throw new Error('auth_failed: Token name claim is invalid');
  }

  return {
    uid:   payload.uid,
    email: payload.email ?? '',
    name:  payload.name  ?? 'User',
  };
}
