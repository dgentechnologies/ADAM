import { z } from 'zod';
import { IsoDateTime, Uuid } from './common.js';

/** Why the unit captured a frame (spec §3). */
export const CaptureReason = z.enum(['requested', 'face-recognised', 'moment', 'scheduled']);
export type CaptureReason = z.infer<typeof CaptureReason>;

export const GalleryItem = z.object({
  id: Uuid,
  deviceId: Uuid,
  /** LAN URL when on the same network, backend URL when cloud backup is on. */
  url: z.string(),
  thumbnailUrl: z.string(),
  width: z.number().int().positive(),
  height: z.number().int().positive(),
  capturedAt: IsoDateTime,
  reason: CaptureReason,
  starred: z.boolean().default(false),
  /** Set when the frame was matched to a saved person. */
  personName: z.string().nullable(),
});
export type GalleryItem = z.infer<typeof GalleryItem>;

/** Segmented filter on the Moments screen. */
export const GalleryFilter = z.enum(['all', 'starred', 'this-week']);
export type GalleryFilter = z.infer<typeof GalleryFilter>;

/** GET /gallery/:deviceId */
export const GalleryResponse = z.object({
  items: z.array(GalleryItem),
  nextCursor: z.string().nullable(),
  /** Surfaced in Settings so retention is never a surprise (spec §3). */
  retentionDays: z.number().int().positive(),
  cloudBackupEnabled: z.boolean(),
});
export type GalleryResponse = z.infer<typeof GalleryResponse>;

/** POST /gallery/upload-url — pre-signed target the Pi uploads to. */
export const GalleryUploadUrlRequest = z.object({
  deviceId: Uuid,
  contentType: z.enum(['image/jpeg', 'image/png']),
  contentLength: z.number().int().positive(),
});
export const GalleryUploadUrlResponse = z.object({
  uploadUrl: z.string().url(),
  expiresAt: IsoDateTime,
});
export type GalleryUploadUrlRequest = z.infer<typeof GalleryUploadUrlRequest>;
export type GalleryUploadUrlResponse = z.infer<typeof GalleryUploadUrlResponse>;
