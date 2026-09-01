import { z } from 'zod';
import { IsoDateTime, Paise, Uuid } from './common.js';

/** The five packs from spec §2.6 / §13 of the Stitch brief. */
export const CreditPackId = z.enum(['trial', 'starter', 'standard', 'value', 'pro']);
export type CreditPackId = z.infer<typeof CreditPackId>;

export const CreditPack = z.object({
  id: CreditPackId,
  name: z.string(),
  /** Price in paise — ₹599 is 59900. */
  pricePaise: Paise,
  /** Display string the UI renders verbatim, e.g. "₹599". */
  priceLabel: z.string(),
  /** Estimated active processing minutes, shown as "approx. N active minutes". */
  estimatedMinutes: z.number().int().positive(),
  isMostPopular: z.boolean().default(false),
});
export type CreditPack = z.infer<typeof CreditPack>;

/** GET /credits/packs */
export const CreditPacksResponse = z.object({ packs: z.array(CreditPack) });
export type CreditPacksResponse = z.infer<typeof CreditPacksResponse>;

/** POST /credits/purchase — returns a Razorpay order for the client checkout. */
export const PurchaseCreditsRequest = z.object({
  packId: CreditPackId,
  deviceId: Uuid,
});
export type PurchaseCreditsRequest = z.infer<typeof PurchaseCreditsRequest>;

export const RazorpayOrder = z.object({
  orderId: z.string(),
  amountPaise: Paise,
  currency: z.literal('INR'),
  /** Public key id — safe to send to the client. */
  razorpayKeyId: z.string(),
});
export type RazorpayOrder = z.infer<typeof RazorpayOrder>;

/** GET /credits/balance/:deviceId */
export const CreditBalance = z.object({
  deviceId: Uuid,
  remainingMinutes: z.number().nonnegative(),
  totalPurchasedMinutes: z.number().nonnegative(),
  updatedAt: IsoDateTime,
});
export type CreditBalance = z.infer<typeof CreditBalance>;

/**
 * POST /tokens/ephemeral — managed mode only. Short-lived Gemini Live token
 * minted server-side so the raw provider key never reaches the device.
 */
export const EphemeralTokenResponse = z.object({
  token: z.string(),
  expiresAt: IsoDateTime,
});
export type EphemeralTokenResponse = z.infer<typeof EphemeralTokenResponse>;
