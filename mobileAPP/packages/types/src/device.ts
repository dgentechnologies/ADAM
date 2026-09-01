import { z } from 'zod';
import { DeviceSerial, DeviceShortId, IsoDateTime, Uuid } from './common.js';

/** How the app reached the unit during discovery (spec §2.3). */
export const DiscoveryTransport = z.enum(['ble', 'hotspot', 'mdns', 'manual']);
export type DiscoveryTransport = z.infer<typeof DiscoveryTransport>;

/** Which brain is powering the unit (spec §2.6). */
export const AiBrainMode = z.enum(['byok', 'managed', 'lite']);
export type AiBrainMode = z.infer<typeof AiBrainMode>;

export const DeviceStatus = z.enum(['online', 'offline', 'updating', 'sleeping']);
export type DeviceStatus = z.infer<typeof DeviceStatus>;

/**
 * Face-mark expression mirrored from the unit's TFT display on the dashboard.
 * Kept as a closed enum so <AdamFaceMark /> can exhaustively switch on it.
 */
export const FaceExpression = z.enum([
  'idle',
  'happy',
  'listening',
  'thinking',
  'speaking',
  'sleeping',
  'annoyed',
]);
export type FaceExpression = z.infer<typeof FaceExpression>;

/** A unit found over BLE/mDNS but not yet claimed — no backend record needed. */
export const DiscoveredDevice = z.object({
  shortId: DeviceShortId,
  serial: DeviceSerial,
  transport: DiscoveryTransport,
  /** BLE RSSI in dBm when available; used only to sort nearest-first. */
  signalStrength: z.number().int().optional(),
  isFounderEdition: z.boolean(),
  /** True when the backend already has an owner for this serial. */
  alreadyClaimed: z.boolean().default(false),
});
export type DiscoveredDevice = z.infer<typeof DiscoveredDevice>;

export const Device = z.object({
  id: Uuid,
  serial: DeviceSerial,
  shortId: DeviceShortId,
  /** User-chosen name; defaults to "ADAM" (spec §2.5). */
  name: z.string().min(1).max(32),
  ownerId: Uuid,
  status: DeviceStatus,
  expression: FaceExpression,
  aiBrainMode: AiBrainMode,
  firmwareVersion: z.string(),
  hardwareBatch: z.string(),
  isFounderEdition: z.boolean(),
  /** 1–10 for Founder Edition units, null otherwise. */
  founderNumber: z.number().int().min(1).max(10).nullable(),
  wifiSsid: z.string().nullable(),
  lastSeenAt: IsoDateTime.nullable(),
  claimedAt: IsoDateTime,
});
export type Device = z.infer<typeof Device>;

/** POST /devices/claim */
export const ClaimDeviceRequest = z.object({
  serial: DeviceSerial,
  /** Proof-of-possession nonce the unit emitted over the local channel. */
  pairingNonce: z.string().min(8),
  name: z.string().min(1).max(32).default('ADAM'),
});
export type ClaimDeviceRequest = z.infer<typeof ClaimDeviceRequest>;

/** PATCH /devices/:id */
export const UpdateDeviceRequest = z
  .object({
    name: z.string().min(1).max(32),
    expression: FaceExpression,
    status: DeviceStatus,
  })
  .partial();
export type UpdateDeviceRequest = z.infer<typeof UpdateDeviceRequest>;

/**
 * POST /devices/:id/ai-brain
 *
 * SECURITY (tech spec §7): a BYOK Gemini key MUST NOT appear in this payload.
 * The backend only records *which* mode is active. The key itself travels
 * phone → Pi over the local channel, encrypted with the Pi's public key, and is
 * stored encrypted on the Pi only. Enforced by the refinement below.
 */
export const SetAiBrainRequest = z
  .object({
    mode: AiBrainMode,
    /** Present for managed mode only, after a successful credit purchase. */
    creditPackId: z.string().optional(),
  })
  .strict();
export type SetAiBrainRequest = z.infer<typeof SetAiBrainRequest>;
