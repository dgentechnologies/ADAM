'use client';
import { useEffect, useState } from 'react';
import { z } from 'zod';
import { getItem, setItem } from './native/preferences';
import { errorMessage } from './local-data';

const Schema = z.object({
  connected: z.boolean().default(false),
  name: z.string().trim().min(1).max(40).default('Demo ADAM'),
  network: z.enum(['Demo Home', 'Demo Studio']).default('Demo Home'),
  volume: z.number().int().min(0).max(100).default(50),
  brightness: z.number().int().min(10).max(100).default(80),
  expression: z.enum(['idle', 'happy', 'listening', 'asleep']).default('idle'),
  laptop: z.boolean().default(false),
  laptopVolume: z.number().int().min(0).max(100).default(45),
  playing: z.boolean().default(false),
});
export type DemoDevice = z.infer<typeof Schema>;
const KEY = 'adam.demo.v1';
const initial = () => Schema.parse({});
let pending: Promise<unknown> = Promise.resolve();
async function read(): Promise<DemoDevice> {
  const raw = await getItem(KEY);
  if (!raw) return initial();
  try {
    return Schema.parse(JSON.parse(raw));
  } catch {
    return initial();
  }
}
export function updateDemoDevice(patch: Partial<DemoDevice>) {
  const operation = pending
    .catch(() => undefined)
    .then(async () => {
      const next = Schema.parse({ ...(await read()), ...patch });
      await setItem(KEY, JSON.stringify(next));
      if (typeof window !== 'undefined') window.dispatchEvent(new Event('adam:demo'));
      return next;
    });
  pending = operation;
  return operation;
}
export function useDemoDevice() {
  const [device, setDevice] = useState(initial);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  useEffect(() => {
    let active = true;
    let revision = 0;
    const refresh = () => {
      const request = ++revision;
      void read()
        .then((value) => {
          if (active && request === revision) {
            setDevice(value);
            setError('');
          }
        })
        .catch((e) => {
          if (active && request === revision) setError(errorMessage(e));
        })
        .finally(() => {
          if (active && request === revision) setLoading(false);
        });
    };
    refresh();
    window.addEventListener('adam:demo', refresh);
    return () => {
      active = false;
      window.removeEventListener('adam:demo', refresh);
    };
  }, []);
  return { device, loading, error };
}
