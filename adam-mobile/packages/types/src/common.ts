import { z } from 'zod';

/** ISO-8601 timestamp, validated as a parseable date string. */
export const IsoDateTime = z
  .string()
  .refine((v) => !Number.isNaN(Date.parse(v)), { message: 'Expected an ISO-8601 date-time' });

export const Uuid = z.string().uuid();

/**
 * Printed on the ADAM base and the box: DGEN-ADAM-0007.
 * Founder Edition is 0001–0010 (spec §2.5).
 */
export const DeviceSerial = z
  .string()
  .regex(/^DGEN-ADAM-\d{4}$/, 'Expected a serial of the form DGEN-ADAM-0007');

/** Short BLE advertisement name, e.g. ADAM-3F2A. */
export const DeviceShortId = z.string().regex(/^ADAM-[0-9A-F]{4}$/);

/** Provisioning hotspot SSID fallback, e.g. ADAM-Setup-3F2A (spec §2.3). */
export const SetupSsid = z.string().regex(/^ADAM-Setup-[0-9A-F]{4}$/);

/** Envelope every API stub responds with, so the client has one shape to narrow. */
export const ApiError = z.object({
  error: z.object({
    code: z.string(),
    message: z.string(),
    details: z.record(z.unknown()).optional(),
  }),
});
export type ApiError = z.infer<typeof ApiError>;

export const Paginated = <T extends z.ZodTypeAny>(item: T) =>
  z.object({
    items: z.array(item),
    nextCursor: z.string().nullable(),
  });

/** Indian Rupee amounts cross the wire in paise to avoid float drift. */
export const Paise = z.number().int().nonnegative();
