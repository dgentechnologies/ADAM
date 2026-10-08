import { z } from 'zod';

/**
 * Firestore Schema & Models for ADAM Companion App.
 *
 * Hard constraints from spec:
 * 1. Exact collection names & field names matching specification.
 * 2. photoUrl is strictly for display-only avatar in Settings, sourced from Google Auth profile.
 * 3. creditBalances is write-protected on client (allow write: if false;).
 */

export const FirestoreTimestamp = z.any();

/**
 * Top-level: users/{uid}
 * Document ID: uid
 */
export const FirestoreUser = z.object({
  email: z.string().email(),
  displayName: z.string(),
  /**
   * CRITICAL PRIVACY CONSTRAINT:
   * photoUrl is ONLY for displaying the user's profile avatar from Google Sign-In.
   * Never used for storing captured camera images or face scans.
   */
  photoUrl: z.string().nullable(),
  createdAt: FirestoreTimestamp,
  linkedDeviceIds: z.array(z.string()).default([]),
});
export type FirestoreUser = z.infer<typeof FirestoreUser>;

/**
 * Top-level: devices/{deviceId}
 * Document ID: deviceId (e.g., serial or generated UUID)
 */
export const FirestoreDevice = z.object({
  deviceId: z.string(),
  ownerUid: z.string(),
  name: z.string(),
  hardwareSerial: z.string(),
  bleAddress: z.string().nullable().default(null),
  wifiSsid: z.string().nullable().default(null),
  tailscaleIp: z.string().nullable().default(null),
  osVersion: z.string().default('1.0.0'),
  lastSeen: FirestoreTimestamp,
  createdAt: FirestoreTimestamp,
  status: z.enum(['online', 'offline', 'setup']).default('online'),
  // Optional alias for backwards compatibility
  ownerId: z.string().optional(),
  serial: z.string().optional(),
  isFounderEdition: z.boolean().optional(),
  founderNumber: z.number().int().nullable().optional(),
  aiBrainMode: z.enum(['byok', 'managed', 'lite']).optional(),
});
export type FirestoreDevice = z.infer<typeof FirestoreDevice>;

/**
 * Subcollection: devices/{deviceId}/memoryFacts/{factId}
 */
export const FirestoreMemoryFact = z.object({
  factId: z.string(),
  category: z.string(),
  content: z.string(),
  confidence: z.number().min(0).max(1).default(1.0),
  learnedAt: FirestoreTimestamp,
  source: z.enum(['conversation', 'vision', 'manual']).default('manual'),
  // Optional aliases for compatibility
  key: z.string().optional(),
  value: z.string().optional(),
  savedAt: FirestoreTimestamp.optional(),
});
export type FirestoreMemoryFact = z.infer<typeof FirestoreMemoryFact>;

/**
 * Subcollection: devices/{deviceId}/memoryPeople/{personId}
 */
export const FirestoreMemoryPerson = z.object({
  personId: z.string(),
  name: z.string(),
  relationship: z.string(),
  faceEncodingId: z.string().nullable().default(null),
  notes: z.string().default(''),
  firstSeen: FirestoreTimestamp,
  lastSeen: FirestoreTimestamp,
  // Optional alias for compatibility
  lastSeenAt: FirestoreTimestamp.optional(),
});
export type FirestoreMemoryPerson = z.infer<typeof FirestoreMemoryPerson>;

/**
 * Subcollection: devices/{deviceId}/laptopPairings/{pairingId}
 */
export const FirestoreLaptopPairing = z.object({
  pairingId: z.string(),
  hostname: z.string(),
  os: z.string(),
  tailscaleIp: z.string(),
  pairedAt: FirestoreTimestamp,
  lastActive: FirestoreTimestamp,
  // Optional aliases
  laptopName: z.string().optional(),
  lastActiveAt: FirestoreTimestamp.optional(),
  revoked: z.boolean().optional(),
});
export type FirestoreLaptopPairing = z.infer<typeof FirestoreLaptopPairing>;

/**
 * Top-level: creditBalances/{deviceId}
 * Document ID: deviceId
 *
 * NOTE: This collection is strictly WRITE-PROTECTED.
 * Companion app reads it for display, but never writes to it directly.
 */
export const FirestoreCreditBalance = z.object({
  deviceId: z.string(),
  balance: z.number().min(0),
  currency: z.literal('credits').default('credits'),
  lastUpdated: FirestoreTimestamp,
  autoRecharge: z.boolean().default(false),
  rechargeThreshold: z.number().nullable().default(null),
  // Optional alias
  balanceMinutes: z.number().optional(),
  lastUpdatedAt: FirestoreTimestamp.optional(),
});
export type FirestoreCreditBalance = z.infer<typeof FirestoreCreditBalance>;
