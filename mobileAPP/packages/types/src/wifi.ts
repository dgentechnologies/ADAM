import { z } from 'zod';

/**
 * Wi-Fi handoff types (spec §2.4).
 *
 * The Pi Zero 2W radio is 2.4GHz-only, so the band is part of the model and the
 * UI must warn *before* attempting a 5GHz-only SSID rather than failing later.
 */
export const WifiBand = z.enum(['2.4GHz', '5GHz', 'dual']);
export type WifiBand = z.infer<typeof WifiBand>;

export const WifiSecurity = z.enum(['open', 'wep', 'wpa2', 'wpa3', 'enterprise']);
export type WifiSecurity = z.infer<typeof WifiSecurity>;

export const WifiNetwork = z.object({
  ssid: z.string().min(1).max(32),
  /** 0–4 bars, already bucketed by the native scan adapter. */
  signalBars: z.number().int().min(0).max(4),
  security: WifiSecurity,
  band: WifiBand,
  /** True when the SSID is 5GHz-only and therefore unusable by ADAM. */
  unsupported: z.boolean().default(false),
});
export type WifiNetwork = z.infer<typeof WifiNetwork>;

export const WifiCredentials = z.object({
  ssid: z.string().min(1).max(32),
  password: z.string().max(63),
});
export type WifiCredentials = z.infer<typeof WifiCredentials>;

/** The three rows of the Connecting screen, in order. */
export const HandoffStep = z.enum(['sending-credentials', 'device-connecting', 'confirming-online']);
export type HandoffStep = z.infer<typeof HandoffStep>;

export const HandoffStepState = z.enum(['pending', 'active', 'complete', 'failed']);
export type HandoffStepState = z.infer<typeof HandoffStepState>;

/** Every failure the Connecting screen must render explicitly (spec §2.4, §7). */
export const HandoffFailure = z.enum([
  'wrong-password',
  'ssid-not-found',
  'band-unsupported',
  'captive-portal',
  'timeout',
  'already-claimed',
  'backend-unreachable',
  'unknown',
]);
export type HandoffFailure = z.infer<typeof HandoffFailure>;

export const HandoffProgress = z.object({
  steps: z.array(z.object({ step: HandoffStep, state: HandoffStepState })),
  failure: HandoffFailure.nullable(),
  /** Handoff aborts at 60s and offers retry or start-over. */
  elapsedMs: z.number().int().nonnegative(),
});
export type HandoffProgress = z.infer<typeof HandoffProgress>;
