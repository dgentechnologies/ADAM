import { z } from 'zod';
import { IsoDateTime, Uuid } from './common.js';

/**
 * Mirrors what the unit keeps in adam_memory.json. The Memory tab exists so a
 * user can see and delete anything ADAM knows about them (spec §2.11).
 */
export const MemoryKind = z.enum(['person', 'fact']);
export type MemoryKind = z.infer<typeof MemoryKind>;

export const MemoryEntry = z.object({
  id: Uuid,
  deviceId: Uuid,
  kind: MemoryKind,
  /** Person name, or the fact's subject. */
  label: z.string(),
  /** The remembered content, rendered as the row's primary white text. */
  content: z.string(),
  /** True when a face embedding is attached to this person. */
  hasFaceProfile: z.boolean().default(false),
  createdAt: IsoDateTime,
});
export type MemoryEntry = z.infer<typeof MemoryEntry>;

export const MemoryListResponse = z.object({
  entries: z.array(MemoryEntry),
});
export type MemoryListResponse = z.infer<typeof MemoryListResponse>;

export const UpdateMemoryRequest = z.object({
  content: z.string().min(1).max(500),
});
export type UpdateMemoryRequest = z.infer<typeof UpdateMemoryRequest>;

/** A laptop paired to the unit — Settings → Connected Laptops. */
export const PairedLaptop = z.object({
  pairingId: Uuid,
  deviceId: Uuid,
  hostname: z.string(),
  os: z.enum(['windows', 'macos', 'linux']),
  lastSeenAt: IsoDateTime.nullable(),
  online: z.boolean(),
});
export type PairedLaptop = z.infer<typeof PairedLaptop>;

/** POST /laptops/pair */
export const PairLaptopRequest = z.object({
  deviceId: Uuid,
  /** Six-character code shown by the laptop agent. */
  code: z.string().length(6),
});
export type PairLaptopRequest = z.infer<typeof PairLaptopRequest>;
