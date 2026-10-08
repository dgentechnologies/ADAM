import { exec } from 'node:child_process';
import { promisify } from 'node:util';
import { type WifiBand, WifiNetwork, type WifiSecurity } from '@adam/types';

const execAsync = promisify(exec);

export const FALLBACK_NETWORKS: WifiNetwork[] = [
  { ssid: 'DASGUPTA', signalBars: 4, security: 'wpa2', band: 'dual', unsupported: false, signalPercent: 90 },
  { ssid: 'DASGUPTA_5G', signalBars: 4, security: 'wpa2', band: '5GHz', unsupported: true, signalPercent: 95 },
  { ssid: 'ADAM_SETUP', signalBars: 3, security: 'wpa2', band: '2.4GHz', unsupported: false, signalPercent: 70 },
];

export interface ScanOptions {
  force?: boolean;
}

export interface ScanResult {
  networks: WifiNetwork[];
  count: number;
  source: 'system' | 'fallback' | 'mock';
  scannedAt: string;
}

interface RawBssidInfo {
  bssid: string;
  signalPercent: number;
  band?: '2.4GHz' | '5GHz';
  channel?: number;
}

interface RawSsidGroup {
  ssid: string;
  auth: string;
  encryption: string;
  bssids: RawBssidInfo[];
}

// In-memory cache for rapid sequential calls
let cachedResult: { timestamp: number; data: ScanResult } | null = null;
const CACHE_TTL_MS = 3500;

export function signalPercentToBars(percent: number): 0 | 1 | 2 | 3 | 4 {
  if (percent >= 75) return 4;
  if (percent >= 50) return 3;
  if (percent >= 25) return 2;
  if (percent > 0) return 1;
  return 0;
}

export function rssiToPercent(rssi: number): number {
  if (rssi <= -100) return 0;
  if (rssi >= -50) return 100;
  return Math.round(2 * (rssi + 100));
}

export function mapAuthToSecurity(auth: string): WifiSecurity {
  const lower = auth.toLowerCase();
  if (lower.includes('wpa3') || lower.includes('sae')) return 'wpa3';
  if (lower.includes('enterprise') || lower.includes('802.1x') || lower.includes('eap')) return 'enterprise';
  if (lower.includes('wpa2') || lower.includes('rsn') || lower.includes('wpa')) return 'wpa2';
  if (lower.includes('wep')) return 'wep';
  if (lower.includes('open') || lower.includes('none')) return 'open';
  return 'wpa2';
}

export function channelToBand(channel: number): '2.4GHz' | '5GHz' {
  return channel <= 14 ? '2.4GHz' : '5GHz';
}

/**
 * Consolidate raw SSID entries (handling multiple BSSIDs and duplicate SSIDs).
 */
function consolidateRawNetworks(groups: RawSsidGroup[]): WifiNetwork[] {
  const bySsid = new Map<string, {
    ssid: string;
    auth: string;
    bands: Set<'2.4GHz' | '5GHz'>;
    maxSignal: number;
    channel?: number;
    bssid?: string;
  }>();

  for (const group of groups) {
    const cleanSsid = group.ssid.replace(/^"(.*)"$/, '$1').trim().slice(0, 32);
    if (!cleanSsid) continue;

    const existing = bySsid.get(cleanSsid) || {
      ssid: cleanSsid,
      auth: group.auth,
      bands: new Set<'2.4GHz' | '5GHz'>(),
      maxSignal: 0,
      channel: undefined,
      bssid: undefined,
    };

    if (group.auth && !existing.auth) {
      existing.auth = group.auth;
    }

    if (group.bssids.length === 0) {
      // Default to 2.4GHz if no BSSID info was captured
      existing.bands.add('2.4GHz');
      existing.maxSignal = Math.max(existing.maxSignal, 50);
    } else {
      for (const b of group.bssids) {
        if (b.band) {
          existing.bands.add(b.band);
        } else if (b.channel) {
          existing.bands.add(channelToBand(b.channel));
        } else {
          existing.bands.add('2.4GHz');
        }

        if (b.signalPercent > existing.maxSignal) {
          existing.maxSignal = b.signalPercent;
          existing.channel = b.channel;
          existing.bssid = b.bssid;
        }
      }
    }

    bySsid.set(cleanSsid, existing);
  }

  const result: WifiNetwork[] = [];

  for (const item of bySsid.values()) {
    const has24 = item.bands.has('2.4GHz');
    const has5 = item.bands.has('5GHz');

    let band: WifiBand;
    let unsupported = false;

    const isExplicitly5GOnly = /[-_ :.]?5g(?:hz)?\b/i.test(item.ssid);
    if (has5 && !has24 && isExplicitly5GOnly) {
      band = '5GHz';
      unsupported = true;
    } else if (has24 && has5) {
      band = 'dual';
      unsupported = false;
    } else if (has5 && !has24) {
      // Modern dual-band routers broadcast on both bands; if single 5GHz beacon caught, treat as dual-band
      band = 'dual';
      unsupported = false;
    } else {
      band = '2.4GHz';
      unsupported = false;
    }

    const signalPercent = Math.min(100, Math.max(0, item.maxSignal));
    const signalBars = signalPercentToBars(signalPercent);
    const security = mapAuthToSecurity(item.auth);

    const parsed = WifiNetwork.safeParse({
      ssid: item.ssid,
      signalBars,
      security,
      band,
      unsupported,
      signalPercent,
      channel: item.channel,
      bssid: item.bssid,
    });

    if (parsed.success) {
      result.push(parsed.data);
    }
  }

  // Sort: supported networks first, then descending by signalPercent / signalBars
  result.sort((a, b) => {
    if (a.unsupported !== b.unsupported) {
      return a.unsupported ? 1 : -1;
    }
    const sigA = a.signalPercent ?? a.signalBars * 25;
    const sigB = b.signalPercent ?? b.signalBars * 25;
    if (sigB !== sigA) {
      return sigB - sigA;
    }
    return a.ssid.localeCompare(b.ssid);
  });

  return result;
}

/**
 * Parses Windows `netsh wlan show networks mode=bssid` output.
 */
export function parseNetshOutput(output: string): WifiNetwork[] {
  const lines = output.split(/\r?\n/);
  const groups: RawSsidGroup[] = [];
  let currentGroup: RawSsidGroup | null = null;
  let currentBssid: RawBssidInfo | null = null;

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    // SSID 1 : DASGUPTA
    const ssidMatch = trimmed.match(/^SSID\s+\d+\s*:\s*(.*)$/i);
    if (ssidMatch && ssidMatch[1] !== undefined) {
      if (currentGroup && currentGroup.ssid) {
        groups.push(currentGroup);
      }
      const rawSsid = ssidMatch[1].trim();
      currentGroup = rawSsid
        ? {
            ssid: rawSsid,
            auth: '',
            encryption: '',
            bssids: [],
          }
        : null;
      currentBssid = null;
      continue;
    }

    if (!currentGroup) continue;

    // Authentication : WPA2-Personal
    const authMatch = trimmed.match(/^Authentication\s*:\s*(.*)$/i);
    if (authMatch && authMatch[1] !== undefined) {
      currentGroup.auth = authMatch[1].trim();
      continue;
    }

    // Encryption : CCMP
    const encMatch = trimmed.match(/^Encryption\s*:\s*(.*)$/i);
    if (encMatch && encMatch[1] !== undefined) {
      currentGroup.encryption = encMatch[1].trim();
      continue;
    }

    // BSSID 1 : 30:68:93:87:a0:74
    const bssidMatch = trimmed.match(/^BSSID\s+\d+\s*:\s*([0-9a-fA-F:]{17})/i);
    if (bssidMatch && bssidMatch[1] !== undefined) {
      currentBssid = {
        bssid: bssidMatch[1],
        signalPercent: 50,
        band: undefined,
        channel: undefined,
      };
      currentGroup.bssids.push(currentBssid);
      continue;
    }

    if (currentBssid) {
      // Signal : 69%
      const sigMatch = trimmed.match(/^Signal\s*:\s*(\d+)%/i);
      if (sigMatch && sigMatch[1] !== undefined) {
        currentBssid.signalPercent = parseInt(sigMatch[1], 10);
        continue;
      }

      // Band : 5 GHz or 2.4 GHz
      const bandMatch = trimmed.match(/^Band\s*:\s*(.*)$/i);
      if (bandMatch && bandMatch[1] !== undefined) {
        const val = bandMatch[1].trim();
        if (val.includes('5')) currentBssid.band = '5GHz';
        else if (val.includes('2.4')) currentBssid.band = '2.4GHz';
        continue;
      }

      // Channel : 149
      const chanMatch = trimmed.match(/^Channel\s*:\s*(\d+)/i);
      if (chanMatch && chanMatch[1] !== undefined) {
        currentBssid.channel = parseInt(chanMatch[1], 10);
        if (!currentBssid.band) {
          currentBssid.band = channelToBand(currentBssid.channel);
        }
        continue;
      }
    }
  }

  if (currentGroup && currentGroup.ssid) {
    groups.push(currentGroup);
  }

  return consolidateRawNetworks(groups);
}

/**
 * Parses Linux `nmcli dev wifi list` or `nmcli -t -f SSID,BSSID,SIGNAL,SECURITY,CHAN,FREQ dev wifi list` output.
 */
export function parseNmcliOutput(output: string): WifiNetwork[] {
  const lines = output.split(/\r?\n/);
  const groups: RawSsidGroup[] = [];

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('IN-USE') || trimmed.startsWith('SSID')) continue;

    // Handles nmcli -t format: SSID:BSSID:SIGNAL:SECURITY:CHAN:FREQ
    // Note colons in BSSID can be escaped as \:
    const parts = trimmed.split(/(?<!\\):/);
    if (parts.length >= 4) {
      const p0 = parts[0];
      const p1 = parts[1];
      const p2 = parts[2];
      const p3 = parts[3];
      const p4 = parts[4];
      const p5 = parts[5];
      if (p0 === undefined || p1 === undefined || p2 === undefined || p3 === undefined) continue;

      const ssid = p0.replace(/\\:/g, ':').trim();
      if (!ssid || ssid === '--') continue;

      const signal = parseInt(p2, 10) || 50;
      const security = p3.replace(/\\:/g, ':').trim();
      const channel = p4 ? parseInt(p4, 10) : undefined;
      const freqNum = p5 ? parseInt(p5, 10) : NaN;

      let band: '2.4GHz' | '5GHz' | undefined;
      if (!isNaN(freqNum)) {
        band = freqNum >= 4900 ? '5GHz' : '2.4GHz';
      } else if (channel) {
        band = channelToBand(channel);
      }

      groups.push({
        ssid,
        auth: security,
        encryption: '',
        bssids: [{
          bssid: p1.replace(/\\:/g, ':').trim(),
          signalPercent: signal,
          band,
          channel,
        }],
      });
    }
  }

  return consolidateRawNetworks(groups);
}

/**
 * Parses macOS `airport -s` output.
 */
export function parseAirportOutput(output: string): WifiNetwork[] {
  const lines = output.split(/\r?\n/);
  const groups: RawSsidGroup[] = [];

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('SSID BSSID')) continue;

    const match = trimmed.match(/^(.*?)\s+([0-9a-fA-F:]{17})\s+(-?\d+)\s+([\d,+]+)\s+[yYnN]\s+\w+\s*(.*)$/);
    if (match && match[1] !== undefined && match[2] !== undefined && match[3] !== undefined && match[4] !== undefined) {
      const ssid = match[1].trim();
      const bssid = match[2].trim();
      const rssi = parseInt(match[3], 10);
      const channelStr = match[4].split(',')[0] ?? '';
      const channel = parseInt(channelStr, 10);
      const security = (match[5] ?? '').trim();

      if (!ssid) continue;

      const signalPercent = rssiToPercent(rssi);
      const band = !isNaN(channel) ? channelToBand(channel) : '2.4GHz';

      groups.push({
        ssid,
        auth: security,
        encryption: '',
        bssids: [{
          bssid,
          signalPercent,
          band,
          channel,
        }],
      });
    }
  }

  return consolidateRawNetworks(groups);
}

/**
 * Performs scanning using native OS tool with timeout.
 */
async function runSystemScan(): Promise<WifiNetwork[]> {
  const platform = process.platform;
  let rawOutput = '';

  if (platform === 'win32') {
    const { stdout } = await execAsync('netsh wlan show networks mode=bssid', {
      timeout: 8000,
      encoding: 'utf8',
      windowsHide: true,
    });
    rawOutput = stdout;
    return parseNetshOutput(rawOutput);
  }

  if (platform === 'linux') {
    try {
      const { stdout } = await execAsync('nmcli -t -f SSID,BSSID,SIGNAL,SECURITY,CHAN,FREQ dev wifi list', {
        timeout: 8000,
        encoding: 'utf8',
      });
      rawOutput = stdout;
      return parseNmcliOutput(rawOutput);
    } catch {
      // Fallback to iwlist if nmcli is missing
      const { stdout } = await execAsync('iwlist scan', { timeout: 8000, encoding: 'utf8' });
      return parseNetshOutput(stdout); // or basic parse
    }
  }

  if (platform === 'darwin') {
    const airportPath = '/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport';
    const { stdout } = await execAsync(`${airportPath} -s`, {
      timeout: 8000,
      encoding: 'utf8',
    });
    rawOutput = stdout;
    return parseAirportOutput(rawOutput);
  }

  throw new Error(`Unsupported OS platform for native Wi-Fi scan: ${platform}`);
}

const RECENT_NETWORK_TTL_MS = 120_000;
const rollingNetworks = new Map<string, { network: WifiNetwork; lastSeen: number }>();

/**
 * Main scan function. Handles caching, platform detection, and graceful fallback.
 */
export async function scanWifiNetworks(options: ScanOptions = {}): Promise<ScanResult> {
  const now = Date.now();

  if (!options.force && cachedResult && now - cachedResult.timestamp < CACHE_TTL_MS) {
    return cachedResult.data;
  }

  if (process.env.MOCK_WIFI === 'true') {
    const result: ScanResult = {
      networks: FALLBACK_NETWORKS,
      count: FALLBACK_NETWORKS.length,
      source: 'mock',
      scannedAt: new Date().toISOString(),
    };
    cachedResult = { timestamp: now, data: result };
    return result;
  }

  try {
    const freshNetworks = await runSystemScan();

    for (const net of freshNetworks) {
      const existing = rollingNetworks.get(net.ssid);
      if (existing) {
        const isDual =
          net.band === 'dual' ||
          existing.network.band === 'dual' ||
          (net.band === '2.4GHz' && existing.network.band === '5GHz') ||
          (net.band === '5GHz' && existing.network.band === '2.4GHz');

        const mergedBand: WifiBand = isDual ? 'dual' : net.band;
        const unsupported = mergedBand === '5GHz';

        rollingNetworks.set(net.ssid, {
          network: {
            ...net,
            band: mergedBand,
            unsupported,
            signalPercent: Math.max(net.signalPercent ?? 0, existing.network.signalPercent ?? 0),
            signalBars: Math.max(net.signalBars, existing.network.signalBars),
          },
          lastSeen: now,
        });
      } else {
        rollingNetworks.set(net.ssid, {
          network: net,
          lastSeen: now,
        });
      }
    }

    // Prune stale entries
    for (const [ssid, entry] of rollingNetworks.entries()) {
      if (now - entry.lastSeen > RECENT_NETWORK_TTL_MS) {
        rollingNetworks.delete(ssid);
      }
    }

    const consolidated = Array.from(rollingNetworks.values()).map((e) => e.network);
    consolidated.sort((a, b) => {
      if (a.unsupported !== b.unsupported) {
        return a.unsupported ? 1 : -1;
      }
      const sigA = a.signalPercent ?? a.signalBars * 25;
      const sigB = b.signalPercent ?? b.signalBars * 25;
      if (sigB !== sigA) return sigB - sigA;
      return a.ssid.localeCompare(b.ssid);
    });

    const result: ScanResult = {
      networks: consolidated.length > 0 ? consolidated : freshNetworks,
      count: consolidated.length > 0 ? consolidated.length : freshNetworks.length,
      source: 'system',
      scannedAt: new Date().toISOString(),
    };

    cachedResult = { timestamp: now, data: result };
    return result;
  } catch (err: any) {
    console.warn(`[wifi-scanner] Native Wi-Fi scan failed: ${err?.message || err}. Using fallback networks.`);
    const result: ScanResult = {
      networks: FALLBACK_NETWORKS,
      count: FALLBACK_NETWORKS.length,
      source: 'fallback',
      scannedAt: new Date().toISOString(),
    };
    return result;
  }
}
