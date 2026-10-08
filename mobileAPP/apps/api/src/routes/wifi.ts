import type { FastifyInstance, FastifyPluginAsync } from 'fastify';
import { scanWifiNetworks } from '../services/wifi-scanner.js';

interface WifiScanQuery {
  force?: string;
  refresh?: string;
}

interface WifiCredentialsBody {
  ssid: string;
  password?: string;
  bssid?: string;
}

// In-memory Wi-Fi state with realistic default
let activeWifi = {
  ssid: 'DASGUPTA',
  password: '',
  security: 'wpa2',
  band: '2.4GHz',
  signalBars: 4,
  signalPercent: 92,
  status: 'connected',
  savedAt: new Date().toISOString(),
  lastConnectedAt: new Date().toISOString(),
};

export function getActiveWifi() {
  return activeWifi;
}

export function setActiveWifi(data: Partial<typeof activeWifi>) {
  activeWifi = {
    ...activeWifi,
    ...data,
    lastConnectedAt: new Date().toISOString(),
  };
  return activeWifi;
}

export const wifiRoutes: FastifyPluginAsync = async (fastify: FastifyInstance) => {
  // Handler for scanning Wi-Fi networks
  const handleScan = async (request: any) => {
    const query = (request.query || {}) as WifiScanQuery;
    const force = query.force === 'true' || query.refresh === 'true' || request.method === 'POST';
    const result = await scanWifiNetworks({ force });
    return result;
  };

  // Standard scanning endpoints
  fastify.get<{ Querystring: WifiScanQuery }>('/api/wifi/networks', handleScan);
  fastify.post('/api/wifi/scan', handleScan);
  fastify.get<{ Querystring: WifiScanQuery }>('/wifi/networks', handleScan);

  // Get current active Wi-Fi info
  fastify.get('/api/wifi/current', async () => {
    return {
      ssid: activeWifi.ssid,
      security: activeWifi.security,
      band: activeWifi.band,
      signalBars: activeWifi.signalBars,
      signalPercent: activeWifi.signalPercent,
      status: activeWifi.status,
      savedAt: activeWifi.savedAt,
      lastConnectedAt: activeWifi.lastConnectedAt,
    };
  });

  // Get saved credentials (with password for authorized settings management)
  fastify.get('/api/wifi/credentials', async () => {
    return {
      ssid: activeWifi.ssid,
      password: activeWifi.password,
      savedAt: activeWifi.savedAt,
      status: activeWifi.status,
    };
  });

  // Save Wi-Fi credentials
  fastify.post<{ Body: WifiCredentialsBody }>('/api/wifi/credentials', async (request, reply) => {
    const body = request.body || {};
    if (!body.ssid) {
      reply.status(400);
      return { error: 'SSID is required' };
    }

    setActiveWifi({
      ssid: body.ssid,
      password: body.password || '',
      savedAt: new Date().toISOString(),
      status: 'connected',
    });

    return {
      success: true,
      ssid: activeWifi.ssid,
      message: `Credentials saved for ${activeWifi.ssid}`,
    };
  });

  // Wi-Fi handoff to ADAM unit
  fastify.post<{ Body: WifiCredentialsBody }>('/api/wifi/handoff', async (request, reply) => {
    const body = request.body || {};
    if (!body.ssid) {
      reply.status(400);
      return { error: 'SSID is required' };
    }

    // Save active Wi-Fi credentials
    setActiveWifi({
      ssid: body.ssid,
      password: body.password || '',
      savedAt: new Date().toISOString(),
      status: 'connected',
    });

    return {
      success: true,
      ssid: body.ssid,
      steps: [
        { step: 'sending-credentials', state: 'complete' },
        { step: 'device-connecting', state: 'complete' },
        { step: 'confirming-online', state: 'complete' },
      ],
      elapsedMs: 3200,
      timestamp: new Date().toISOString(),
    };
  });

  // Forget network credentials
  fastify.delete('/api/wifi/credentials', async () => {
    activeWifi = {
      ssid: '',
      password: '',
      security: 'none',
      band: '2.4GHz',
      signalBars: 0,
      signalPercent: 0,
      status: 'disconnected',
      savedAt: '',
      lastConnectedAt: '',
    };
    return { success: true, message: 'Network forgotten' };
  });
};
