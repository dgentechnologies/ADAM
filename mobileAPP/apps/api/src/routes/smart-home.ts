import type { FastifyInstance, FastifyPluginAsync } from 'fastify';
import * as dgram from 'node:dgram';
import * as net from 'node:net';
import { exec } from 'node:child_process';
import { promisify } from 'node:util';

const execAsync = promisify(exec);

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export type SmartDeviceType =
  | 'hub'
  | 'light'
  | 'bulb'
  | 'strip'
  | 'switch'
  | 'plug'
  | 'thermostat'
  | 'speaker'
  | 'tv'
  | 'camera'
  | 'lock'
  | 'sensor'
  | 'router'
  | 'media_player'
  | 'unknown';

export interface SmartDevice {
  id: string;
  name: string;
  type: SmartDeviceType;
  brand: string;
  ip: string;
  mac?: string;
  model?: string;
  online: boolean;
  /** Protocol used to discover this device */
  protocol: 'ssdp' | 'mdns' | 'arp' | 'static';
  /** Raw USN / UDN from SSDP or mDNS service name */
  uid?: string;
  /** Extra info from SSDP location header or mDNS TXT record */
  location?: string;
  detectedAt: string;
}

export interface SmartHomeState {
  devices: SmartDevice[];
  count: number;
  scannedAt: string;
  source: 'live' | 'cached' | 'fallback';
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers — SSDP via UDP multicast
// ─────────────────────────────────────────────────────────────────────────────

const SSDP_ADDR = '239.255.255.250';
const SSDP_PORT = 1900;
const SSDP_TIMEOUT_MS = 3500;

const SSDP_SEARCH = [
  'M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1900\r\nMAN: "ssdp:discover"\r\nMX: 3\r\nST: ssdp:all\r\n\r\n',
  'M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1900\r\nMAN: "ssdp:discover"\r\nMX: 2\r\nST: urn:schemas-upnp-org:device:basic:1\r\n\r\n',
  'M-SEARCH * HTTP/1.1\r\nHOST: 239.255.255.250:1900\r\nMAN: "ssdp:discover"\r\nMX: 2\r\nST: urn:dial-multiscreen-org:service:dial:1\r\n\r\n',
];

interface SsdpResponse {
  ip: string;
  usn?: string;
  location?: string;
  server?: string;
  st?: string;
  raw: string;
}

async function ssdpDiscover(): Promise<SsdpResponse[]> {
  return new Promise((resolve) => {
    const responses: SsdpResponse[] = [];
    const seen = new Set<string>();

    const socket = dgram.createSocket({ type: 'udp4', reuseAddr: true });

    socket.on('error', () => {
      try { socket.close(); } catch {}
      resolve(responses);
    });

    socket.on('message', (msg, rinfo) => {
      const text = msg.toString('utf8');
      const ip = rinfo.address;
      const key = ip;
      if (seen.has(key)) return;
      seen.add(key);

      const header = (name: string): string | undefined => {
        const m = text.match(new RegExp(`^${name}:\\s*(.+)$`, 'im'));
        return m?.[1]?.trim();
      };

      responses.push({
        ip,
        usn: header('USN'),
        location: header('LOCATION'),
        server: header('SERVER'),
        st: header('ST'),
        raw: text,
      });
    });

    socket.bind(0, () => {
      for (const msg of SSDP_SEARCH) {
        const buf = Buffer.from(msg);
        socket.send(buf, 0, buf.length, SSDP_PORT, SSDP_ADDR, () => {});
      }
    });

    setTimeout(() => {
      try { socket.close(); } catch {}
      resolve(responses);
    }, SSDP_TIMEOUT_MS);
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Brand / type classification from SSDP metadata
// ─────────────────────────────────────────────────────────────────────────────

interface Classification {
  brand: string;
  type: SmartDeviceType;
  name?: string;
}

function classifySsdp(r: SsdpResponse): Classification {
  const text = `${r.server ?? ''} ${r.usn ?? ''} ${r.location ?? ''} ${r.st ?? ''}`.toLowerCase();

  if (text.includes('philips') || text.includes('hue') || text.includes('signify')) {
    return { brand: 'Philips Hue', type: 'hub', name: 'Philips Hue Bridge' };
  }
  if (text.includes('nanoleaf')) {
    return { brand: 'Nanoleaf', type: 'light', name: 'Nanoleaf Panel' };
  }
  if (text.includes('govee')) {
    return { brand: 'Govee', type: 'bulb', name: 'Govee Smart Bulb' };
  }
  if (text.includes('tp-link') || text.includes('tplink') || text.includes('kasa') || text.includes('tapo')) {
    if (text.includes('plug') || text.includes('mini')) return { brand: 'TP-Link', type: 'plug', name: 'Kasa Smart Plug' };
    if (text.includes('bulb') || text.includes('light') || text.includes('l5')) return { brand: 'TP-Link', type: 'bulb', name: 'Kasa Smart Bulb' };
    if (text.includes('strip') || text.includes('l9')) return { brand: 'TP-Link', type: 'strip', name: 'Kasa Light Strip' };
    if (text.includes('cam') || text.includes('c2')) return { brand: 'TP-Link', type: 'camera', name: 'Tapo Camera' };
    return { brand: 'TP-Link Kasa', type: 'switch', name: 'Kasa Smart Switch' };
  }
  if (text.includes('chromecast') || text.includes('cast')) {
    return { brand: 'Google', type: 'media_player', name: 'Chromecast' };
  }
  if (text.includes('google home') || text.includes('googlehome') || text.includes('nest hub')) {
    return { brand: 'Google', type: 'speaker', name: 'Google Home' };
  }
  if (text.includes('amazon') || text.includes('alexa') || text.includes('echo')) {
    return { brand: 'Amazon', type: 'speaker', name: 'Amazon Echo' };
  }
  if (text.includes('ring')) {
    return { brand: 'Ring', type: 'camera', name: 'Ring Device' };
  }
  if (text.includes('sonos')) {
    return { brand: 'Sonos', type: 'speaker', name: 'Sonos Speaker' };
  }
  if (text.includes('xiaomi') || text.includes('mi home') || text.includes('miio')) {
    return { brand: 'Xiaomi', type: 'bulb', name: 'Xiaomi Smart Device' };
  }
  if (text.includes('tuya') || text.includes('smart life') || text.includes('beken')) {
    return { brand: 'Tuya', type: 'switch', name: 'Tuya Smart Device' };
  }
  if (text.includes('yeelight')) {
    return { brand: 'Xiaomi', type: 'bulb', name: 'Yeelight Bulb' };
  }
  if (text.includes('shelly')) {
    return { brand: 'Shelly', type: 'switch', name: 'Shelly Smart Relay' };
  }
  if (text.includes('wemo') || text.includes('belkin')) {
    return { brand: 'Belkin', type: 'plug', name: 'Wemo Smart Plug' };
  }
  if (text.includes('smartthings') || text.includes('samsung')) {
    return { brand: 'Samsung', type: 'hub', name: 'SmartThings Hub' };
  }
  if (text.includes('homey') || text.includes('athom')) {
    return { brand: 'Athom', type: 'hub', name: 'Homey Hub' };
  }
  if (text.includes('roku')) {
    return { brand: 'Roku', type: 'tv', name: 'Roku Device' };
  }
  if (text.includes('apple-tv') || text.includes('appletv') || text.includes('airport')) {
    return { brand: 'Apple', type: 'media_player', name: 'Apple TV' };
  }
  if (text.includes('dial:1') || text.includes('dial-multiscreen')) {
    return { brand: 'Unknown', type: 'tv', name: 'Smart TV' };
  }
  if (text.includes('upnp') || text.includes('mediaserver') || text.includes('contentdirectory')) {
    return { brand: 'UPnP', type: 'media_player', name: 'UPnP Media Server' };
  }
  if (text.includes('router') || text.includes('gateway') || text.includes('igdv') || text.includes('wanip')) {
    return { brand: 'Router', type: 'router', name: 'Network Gateway' };
  }
  return { brand: 'Unknown', type: 'unknown' };
}

// ─────────────────────────────────────────────────────────────────────────────
// ARP scan fallback — reads OS ARP cache to find LAN hosts
// ─────────────────────────────────────────────────────────────────────────────

interface ArpEntry {
  ip: string;
  mac: string;
}

function parseMacVendor(mac: string): string {
  const prefix = mac.replace(/[:-]/g, '').substring(0, 6).toUpperCase();
  // Only well-known IoT / smart-home vendors
  const VENDORS: Record<string, string> = {
    '001788': 'Philips',
    '002272': 'Philips Hue',
    'A4DA22': 'TP-Link',
    'B0BE76': 'TP-Link',
    '50C7BF': 'TP-Link Kasa',
    '1C61B4': 'TP-Link',
    'E8DE27': 'TP-Link',
    '84D81B': 'TP-Link Tapo',
    '40A3CC': 'Google',
    '54600A': 'Google Nest',
    '7C2E0D': 'Google Home',
    'F4F5D8': 'Google',
    '30FD38': 'Amazon Echo',
    'FC65DE': 'Amazon Echo',
    '44650D': 'Amazon Echo',
    'F0272D': 'Samsung SmartThings',
    'AC5A14': 'Xiaomi',
    '28EF01': 'Xiaomi Mi',
    '6C5CF5': 'Govee',
    'C8B21E': 'Shelly',
    'E8DB84': 'Espressif/Tuya',
    'A4CF12': 'Espressif/Tuya',
  };
  return VENDORS[prefix] ?? '';
}

async function arpScan(): Promise<ArpEntry[]> {
  const entries: ArpEntry[] = [];
  try {
    const platform = process.platform;
    let stdout = '';
    if (platform === 'win32') {
      ({ stdout } = await execAsync('arp -a', { timeout: 4000 }));
      const lines = stdout.split(/\r?\n/);
      for (const line of lines) {
        const m = line.trim().match(/^(\d{1,3}(?:\.\d{1,3}){3})\s+([\da-fA-F-]{17})/);
        if (m) entries.push({ ip: m[1]!, mac: m[2]!.replace(/-/g, ':').toLowerCase() });
      }
    } else if (platform === 'linux') {
      ({ stdout } = await execAsync('arp -an', { timeout: 4000 }));
      const lines = stdout.split(/\r?\n/);
      for (const line of lines) {
        const m = line.match(/\((\d{1,3}(?:\.\d{1,3}){3})\)\s+at\s+([\da-fA-F:]{17})/i);
        if (m) entries.push({ ip: m[1]!, mac: m[2]!.toLowerCase() });
      }
    } else if (platform === 'darwin') {
      ({ stdout } = await execAsync('arp -an', { timeout: 4000 }));
      const lines = stdout.split(/\r?\n/);
      for (const line of lines) {
        const m = line.match(/\((\d{1,3}(?:\.\d{1,3}){3})\)\s+at\s+([\da-fA-F:]{17})/i);
        if (m) entries.push({ ip: m[1]!, mac: m[2]!.toLowerCase() });
      }
    }
  } catch {
    // ARP not available or failed; return empty
  }
  return entries;
}

// ─────────────────────────────────────────────────────────────────────────────
// Assemble SmartDevice list from discovered data
// ─────────────────────────────────────────────────────────────────────────────

function makeId(ip: string): string {
  return `smdev-${ip.replace(/\./g, '-')}`;
}

function dedupeByIp(devices: SmartDevice[]): SmartDevice[] {
  const map = new Map<string, SmartDevice>();
  for (const d of devices) {
    if (!map.has(d.ip)) {
      map.set(d.ip, d);
    }
  }
  return Array.from(map.values());
}

// ─────────────────────────────────────────────────────────────────────────────
// In-memory cache
// ─────────────────────────────────────────────────────────────────────────────

let cachedState: SmartHomeState | null = null;
const CACHE_TTL_MS = 20_000;

// ─────────────────────────────────────────────────────────────────────────────
// Main discovery function
// ─────────────────────────────────────────────────────────────────────────────

export async function discoverSmartDevices(force = false): Promise<SmartHomeState> {
  const now = Date.now();

  if (!force && cachedState && now - new Date(cachedState.scannedAt).getTime() < CACHE_TTL_MS) {
    return { ...cachedState, source: 'cached' };
  }

  const devices: SmartDevice[] = [];
  const ts = new Date().toISOString();

  // ── SSDP discovery ────────────────────────────────────────────────────────
  try {
    const ssdpResponses = await ssdpDiscover();
    for (const r of ssdpResponses) {
      const cls = classifySsdp(r);
      devices.push({
        id: makeId(r.ip),
        name: cls.name ?? `Device @ ${r.ip}`,
        type: cls.type,
        brand: cls.brand,
        ip: r.ip,
        online: true,
        protocol: 'ssdp',
        uid: r.usn,
        location: r.location,
        detectedAt: ts,
      });
    }
  } catch {
    // SSDP not available (e.g. blocked firewall)
  }

  // ── ARP fallback / supplemental ─────────────────────────────────────────
  try {
    const arpEntries = await arpScan();
    const existingIps = new Set(devices.map((d) => d.ip));

    for (const entry of arpEntries) {
      if (existingIps.has(entry.ip)) {
        // Add MAC to existing device
        const existing = devices.find((d) => d.ip === entry.ip);
        if (existing) existing.mac = entry.mac;
        continue;
      }

      const vendor = parseMacVendor(entry.mac);
      if (!vendor) continue; // Skip unknown non-IoT hosts

      // Classify by vendor name
      let type: SmartDeviceType = 'unknown';
      let brand = vendor;
      let name = `${vendor} Device`;

      if (vendor.includes('Philips')) {
        type = 'hub'; brand = 'Philips Hue'; name = 'Philips Hue Hub';
      } else if (vendor.includes('TP-Link') || vendor.includes('Kasa') || vendor.includes('Tapo')) {
        type = 'switch'; brand = 'TP-Link'; name = 'Kasa Smart Device';
      } else if (vendor.includes('Google')) {
        type = 'speaker'; brand = 'Google'; name = 'Google Home';
      } else if (vendor.includes('Amazon')) {
        type = 'speaker'; brand = 'Amazon'; name = 'Amazon Echo';
      } else if (vendor.includes('Samsung')) {
        type = 'hub'; brand = 'Samsung'; name = 'SmartThings Hub';
      } else if (vendor.includes('Xiaomi')) {
        type = 'bulb'; brand = 'Xiaomi'; name = 'Xiaomi Smart Device';
      } else if (vendor.includes('Govee')) {
        type = 'light'; brand = 'Govee'; name = 'Govee Smart Light';
      } else if (vendor.includes('Shelly')) {
        type = 'switch'; brand = 'Shelly'; name = 'Shelly Relay';
      } else if (vendor.includes('Tuya') || vendor.includes('Espressif')) {
        type = 'switch'; brand = 'Tuya'; name = 'Tuya Smart Device';
      }

      devices.push({
        id: makeId(entry.ip),
        name,
        type,
        brand,
        ip: entry.ip,
        mac: entry.mac,
        online: true,
        protocol: 'arp',
        detectedAt: ts,
      });
    }
  } catch {
    // ARP scan failed
  }

  const deduplicated = dedupeByIp(devices);
  // Sort: known brands first, then by IP
  deduplicated.sort((a, b) => {
    if (a.type === 'unknown' && b.type !== 'unknown') return 1;
    if (a.type !== 'unknown' && b.type === 'unknown') return -1;
    return a.ip.localeCompare(b.ip);
  });

  const state: SmartHomeState = {
    devices: deduplicated,
    count: deduplicated.length,
    scannedAt: ts,
    source: 'live',
  };

  cachedState = state;
  return state;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fastify plugin
// ─────────────────────────────────────────────────────────────────────────────

export const smartHomeRoutes: FastifyPluginAsync = async (fastify: FastifyInstance) => {
  // GET /api/smart-home/devices — discover all smart devices on LAN
  fastify.get<{ Querystring: { force?: string } }>('/api/smart-home/devices', async (request) => {
    const force = request.query?.force === 'true';
    return discoverSmartDevices(force);
  });

  // POST /api/smart-home/scan — alias for force-scan
  fastify.post('/api/smart-home/scan', async () => {
    return discoverSmartDevices(true);
  });

  // GET /api/smart-home/device/:ip — get a single device by IP
  fastify.get<{ Params: { ip: string } }>('/api/smart-home/device/:ip', async (request, reply) => {
    const { ip } = request.params;
    if (!net.isIP(ip)) {
      reply.status(400);
      return { error: 'Invalid IP address' };
    }
    // Return cached device if available
    if (cachedState) {
      const device = cachedState.devices.find((d) => d.ip === ip);
      if (device) return device;
    }
    // Otherwise, do a quick ARP to check if it's alive
    reply.status(404);
    return { error: 'Device not found. Run a scan first.' };
  });
};
