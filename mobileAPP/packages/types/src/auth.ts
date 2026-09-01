import { z } from 'zod';
import { IsoDateTime, Uuid } from './common.js';

/** POST /auth/google — the app exchanges a Google OAuth id_token for a session. */
export const GoogleAuthRequest = z.object({
  idToken: z.string().min(16),
  /** Set when the sign-in happened inside the Capacitor shell. */
  platform: z.enum(['web', 'android', 'ios']).default('web'),
});
export type GoogleAuthRequest = z.infer<typeof GoogleAuthRequest>;

export const User = z.object({
  id: Uuid,
  email: z.string().email(),
  displayName: z.string(),
  /** Remote avatar URL from the identity provider; null when unavailable. */
  avatarUrl: z.string().url().nullable(),
  createdAt: IsoDateTime,
});
export type User = z.infer<typeof User>;

/**
 * Session tokens.
 *
 * SECURITY (tech spec §7): the client must persist these in Capacitor secure
 * storage — never plain localStorage. See apps/web/src/lib/native/secure-store.
 */
export const Session = z.object({
  accessToken: z.string(),
  refreshToken: z.string(),
  expiresAt: IsoDateTime,
});
export type Session = z.infer<typeof Session>;

export const AuthResponse = z.object({
  user: User,
  session: Session,
});
export type AuthResponse = z.infer<typeof AuthResponse>;

/** Secondary path for users who decline Google (spec §2.2). */
export const EmailOtpStartRequest = z.object({ email: z.string().email() });
export const EmailOtpVerifyRequest = z.object({
  email: z.string().email(),
  code: z.string().length(6),
});
export type EmailOtpStartRequest = z.infer<typeof EmailOtpStartRequest>;
export type EmailOtpVerifyRequest = z.infer<typeof EmailOtpVerifyRequest>;
