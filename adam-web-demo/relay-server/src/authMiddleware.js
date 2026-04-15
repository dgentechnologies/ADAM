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
  try {
    const { payload } = await jwtVerify(token, secret, {
      algorithms: ['HS256'],
    });

    if (!payload.uid || typeof payload.uid !== 'string') {
      throw new Error('Token missing uid claim');
    }

    return {
      uid:   payload.uid,
      email: payload.email ?? '',
      name:  payload.name  ?? 'User',
    };
  } catch (err) {
    throw new Error(`auth_failed: ${err.message}`);
  }
}
