import { z } from 'zod';

const utc = z.string().datetime({ precision: 3 });
const base = { id: z.string().uuid(), createdAt: utc, updatedAt: utc };
export const Todo = z.object({ ...base, text: z.string().trim().min(1).max(2000), done: z.boolean(), dueAt: z.union([utc, z.literal('')]), deviceId: z.union([z.string().uuid(), z.literal('')]) });
export const Clock = z.object({ ...base, kind: z.enum(['alarm', 'timer', 'reminder']), label: z.string().trim().max(80), when: utc, enabled: z.boolean(), deviceId: z.union([z.string().uuid(), z.literal('')]) });
export const SavedDevice = z.object({ ...base, name: z.string().trim().min(1).max(40), serial: z.string().trim().min(1).max(80), simulated: z.boolean() });
export type Todo = z.infer<typeof Todo>;
export type Clock = z.infer<typeof Clock>;
export type SavedDevice = z.infer<typeof SavedDevice>;
export const RECORD_SCHEMAS = { todos: Todo, clocks: Clock, devices: SavedDevice };
export type RecordKind = keyof typeof RECORD_SCHEMAS;
export type SharedRecord = Todo | Clock | SavedDevice;
export type Tombstone = { id: string; updatedAt: string; deleted: true };
export type SharedRecords = Record<string, (SharedRecord & { deleted: false }) | Tombstone>;

export function validateRecords(kind: RecordKind, value: unknown): SharedRecords {
  const entries = z.record(z.unknown()).parse(value);
  if (Object.keys(entries).length > (kind === 'devices' ? 100 : 1000)) throw new Error('Shared data has reached its supported limit.');
  const result: SharedRecords = {};
  for (const [id, raw] of Object.entries(entries)) {
    const item = z.object({ id: z.string().uuid(), updatedAt: utc, deleted: z.boolean() }).parse(raw);
    if (id !== item.id || id !== id.toLowerCase() || new Date(item.updatedAt).toISOString() !== item.updatedAt) throw new Error('Invalid shared record.');
    if (item.deleted) { result[id] = { ...item, deleted: true }; continue; }
    const parsed = RECORD_SCHEMAS[kind].parse(raw);
    if (parsed.createdAt > parsed.updatedAt || new Date(parsed.createdAt).toISOString() !== parsed.createdAt) throw new Error('Invalid shared record dates.');
    // Reject lone UTF-16 surrogates identically to the desktop validator.
    for (const text of Object.values(parsed).filter((value): value is string => typeof value === 'string')) {
      for (const char of text) { const point = char.codePointAt(0)!; if (point >= 0xd800 && point <= 0xdfff) throw new Error('Invalid shared text.'); }
    }
    result[id] = { ...parsed, deleted: false };
  }
  return result;
}
