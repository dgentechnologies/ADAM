import { z } from 'zod';
import { IsoDateTime } from './common.js';

/**
 * Stages surfaced in the Software Update screen while an OTA runs (spec §5).
 * The physical robot visibly reboots, so every stage needs its own copy.
 */
export const OtaStage = z.enum([
  'idle',
  'checking',
  'downloading',
  'verifying',
  'installing',
  'rebooting',
  'confirming',
  'complete',
  'failed',
  'rolled-back',
]);
export type OtaStage = z.infer<typeof OtaStage>;

export const OtaChangelogEntry = z.object({
  /** Rendered as a monochrome dash mark, never a coloured bullet. */
  text: z.string(),
});

/** GET /ota/manifest */
export const OtaManifest = z.object({
  latestVersion: z.string(),
  /** Bytes; the UI renders a human string such as "2.4 GB". */
  packageSizeBytes: z.number().int().positive(),
  changelog: z.array(OtaChangelogEntry),
  publishedAt: IsoDateTime,
  /** Signed package URL — the Pi verifies the signature before applying. */
  packageUrl: z.string().url(),
  signature: z.string(),
  mandatory: z.boolean().default(false),
});
export type OtaManifest = z.infer<typeof OtaManifest>;

/** Local view model for the two states of the Software Update screen. */
export const OtaState = z.object({
  currentVersion: z.string(),
  manifest: OtaManifest.nullable(),
  updateAvailable: z.boolean(),
  stage: OtaStage,
  /** 0–1 while downloading/installing. */
  progress: z.number().min(0).max(1),
  lastCheckedAt: IsoDateTime.nullable(),
  /** Default is notify-only — never silent auto-install (spec §5). */
  notifyOnUpdate: z.boolean().default(true),
});
export type OtaState = z.infer<typeof OtaState>;
