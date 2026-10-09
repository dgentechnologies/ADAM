import { z } from 'zod';

import { getItem, setItem } from './native/preferences';
import { Clock, Todo, SavedDevice, RECORD_SCHEMAS, documentId, type RecordKind, type SharedRecord } from './companion-records';

export const Fact = z.object({
  id: z.string(),
  title: z.string().min(1).max(80),
  text: z.string().min(1).max(2000),
  kind: z.enum(['fact', 'person']),
  createdAt: z.string(),
  updatedAt: z.string(),
  deviceId: z.union([documentId, z.literal('')]).optional(),
  cloudId: documentId.optional(),
  confidence: z.number().min(0).max(1).optional(),
  source: z.enum(['conversation', 'vision', 'manual']).optional(),
  relationship: z.string().max(2000).optional(),
  faceEncodingId: z.string().max(128).nullable().optional(),
  firstSeen: z.string().optional(),
  lastSeen: z.string().optional(),
});
export type Fact = z.infer<typeof Fact>;
export const LocalData = z.object({
  version: z.literal(1),
  name: z.string().max(80).default(''),
  facts: z.array(Fact).max(1000).default([]),
  todos: z.array(Todo).max(1000).default([]),
  clocks: z.array(Clock).max(1000).default([]),
  devices: z.array(SavedDevice).max(100).default([]),
  voice: z.enum(['Charon', 'Aoede', 'Kore', 'Puck', 'Fenrir']).default('Charon'),
  wakeWord: z.enum(['Hey ADAM', 'ADAM']).default('Hey ADAM'),
  brain: z.enum(['lite', 'byok', 'managed']).default('lite'),
  homeAssistantUrl: z.string().default(''),
  onboardingComplete: z.boolean().default(false),
});
export type LocalData = z.infer<typeof LocalData>;
export const EMPTY_DATA: LocalData = LocalData.parse({ version: 1 });
const KEY = 'adam.companion.v1';
let pending: Promise<unknown> = Promise.resolve();

export async function readLocalData(): Promise<LocalData> {
  const raw = await getItem(KEY);
  if (!raw) return LocalData.parse({ version: 1 });
  // Never overwrite damaged storage with an empty record: surface the error so
  // the user can export or recover their data instead of silently losing it.
  return LocalData.parse(JSON.parse(raw));
}

export function updateLocalData(update: (current: LocalData) => LocalData): Promise<LocalData> {
  const operation = pending
    .catch(() => undefined)
    .then(async () => {
      const next = LocalData.parse(update(await readLocalData()));
      await setItem(KEY, JSON.stringify(next));
      if (typeof window !== 'undefined') window.dispatchEvent(new Event('adam:data'));
      return next;
    });
  pending = operation;
  return operation;
}

/** Called only after the user confirms a validated backup restore. */
export function restoreLocalData(backup: LocalData): Promise<LocalData> {
  const validated = LocalData.parse(backup);
  const operation = pending
    .catch(() => undefined)
    .then(async () => {
      // Recovery must also work when the old record cannot be parsed. Do not
      // inherit a server address from an imported backup or corrupt old record.
      const current = await readLocalData().catch(() => null);
      const next = {
        ...validated,
        homeAssistantUrl: current?.homeAssistantUrl ?? '',
        onboardingComplete: true,
      };
      await setItem(KEY, JSON.stringify(next));
      if (typeof window !== 'undefined') window.dispatchEvent(new Event('adam:data'));
      return next;
    });
  pending = operation;
  return operation;
}

export function saveFact(input: { id?: string; title: string; text: string; kind: Fact['kind']; deviceId?: string }) {
  return updateLocalData((data) => {
    const now = new Date().toISOString();
    const existing = data.facts.find((fact) => fact.id === input.id);
    if (existing?.deviceId && ((input.deviceId !== undefined && input.deviceId !== existing.deviceId) || input.kind !== existing.kind))
      throw new Error('Create a new memory to use a different ADAM or memory type.');
    const fact = Fact.parse({
      ...existing, ...input,
      title: input.title.trim(),
      text: input.text.trim(),
      id: existing?.id ?? crypto.randomUUID(),
      createdAt: existing?.createdAt ?? now,
      updatedAt: now,
    });
    return {
      ...data,
      facts: existing
        ? data.facts.map((item) => (item.id === fact.id ? fact : item))
        : [fact, ...data.facts],
    };
  });
}

export function deleteFact(id: string) {
  return updateLocalData((data) => ({
    ...data,
    facts: data.facts.filter((fact) => fact.id !== id),
  }));
}

export function saveRecord(kind: RecordKind, input: Record<string, unknown>) {
  return updateLocalData((data) => {
    const existing = data[kind].find((item) => item.id === input.id);
    const now = new Date(Math.max(Date.now(), existing ? Date.parse(existing.updatedAt) + 1 : 0)).toISOString();
    const id = existing?.id ?? crypto.randomUUID();
    const record = RECORD_SCHEMAS[kind].parse({ ...existing, ...input, id, ...(kind === 'devices' && !input.deviceId && !existing ? { deviceId: `ADAM-SIM-${id.toUpperCase()}` } : {}), createdAt: existing?.createdAt ?? now, updatedAt: now });
    return { ...data, [kind]: [...data[kind].filter((item: SharedRecord) => item.id !== record.id), record] };
  });
}

export function deleteRecord(kind: RecordKind, id: string) {
  return updateLocalData((data) => ({ ...data, [kind]: data[kind].filter((item: SharedRecord) => item.id !== id) }));
}

export function errorMessage(error: unknown): string {
  if (error instanceof DOMException && error.name === 'QuotaExceededError')
    return 'Storage is full. Free some space and try again.';
  return error instanceof Error ? error.message : 'Something went wrong. Please try again.';
}
