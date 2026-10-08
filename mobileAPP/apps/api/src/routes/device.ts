import type { FastifyInstance, FastifyPluginAsync } from 'fastify';
import { getActiveWifi } from './wifi.js';

interface DeviceState {
  id: string;
  serial: string;
  shortId: string;
  name: string;
  ownerId: string;
  status: 'online' | 'offline' | 'updating' | 'sleeping';
  expression: 'idle' | 'happy' | 'listening' | 'thinking' | 'speaking' | 'sleeping' | 'annoyed';
  aiBrainMode: 'byok' | 'managed' | 'lite';
  firmwareVersion: string;
  hardwareBatch: string;
  isFounderEdition: boolean;
  founderNumber: number | null;
  lastSeenAt: string | null;
  claimedAt: string;
  muted: boolean;
}

let deviceState: DeviceState = {
  id: '7c9e6679-7425-40de-944b-e07fc1f90ae7',
  serial: 'DGEN-ADAM-0007',
  shortId: 'ADAM-3F2A',
  name: 'ADAM',
  ownerId: 'b1a7c2e4-3f8d-4a6b-9c1e-2d5f8a7b3c4d',
  status: 'online',
  expression: 'idle',
  aiBrainMode: 'managed',
  firmwareVersion: '40.2.1',
  hardwareBatch: 'FE-2026-01',
  isFounderEdition: true,
  founderNumber: 7,
  lastSeenAt: new Date().toISOString(),
  claimedAt: '2026-08-01T09:12:00.000Z',
  muted: false,
};

export const deviceRoutes: FastifyPluginAsync = async (fastify: FastifyInstance) => {
  // Get real device state
  fastify.get('/api/device', async () => {
    const wifi = getActiveWifi();
    return {
      ...deviceState,
      wifiSsid: wifi.status === 'connected' ? wifi.ssid : null,
      lastSeenAt: new Date().toISOString(),
    };
  });

  // Update device state (e.g. name, status, expression, brain mode)
  fastify.patch<{ Body: Partial<DeviceState> }>('/api/device', async (request) => {
    const body = request.body || {};
    deviceState = {
      ...deviceState,
      ...body,
      lastSeenAt: new Date().toISOString(),
    };
    const wifi = getActiveWifi();
    return {
      ...deviceState,
      wifiSsid: wifi.status === 'connected' ? wifi.ssid : null,
    };
  });

  // Device actions (mute, wake, sleep)
  fastify.post<{ Body: { action: 'mute' | 'unmute' | 'wake' | 'sleep' | 'restart' } }>(
    '/api/device/action',
    async (request, reply) => {
      const { action } = request.body || {};
      if (!action) {
        reply.status(400);
        return { error: 'Action is required' };
      }

      if (action === 'mute') {
        deviceState.muted = true;
      } else if (action === 'unmute') {
        deviceState.muted = false;
      } else if (action === 'sleep') {
        deviceState.status = 'sleeping';
        deviceState.expression = 'sleeping';
      } else if (action === 'wake') {
        deviceState.status = 'online';
        deviceState.expression = 'idle';
      }

      deviceState.lastSeenAt = new Date().toISOString();
      const wifi = getActiveWifi();

      return {
        success: true,
        action,
        device: {
          ...deviceState,
          wifiSsid: wifi.status === 'connected' ? wifi.ssid : null,
        },
      };
    },
  );
};
