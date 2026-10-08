import test from 'node:test';
import assert from 'node:assert/strict';
import {
  parseNetshOutput,
  parseNmcliOutput,
  parseAirportOutput,
  signalPercentToBars,
  rssiToPercent,
  mapAuthToSecurity,
  channelToBand,
  scanWifiNetworks,
} from './wifi-scanner.js';
import { app } from '../server.js';

const SAMPLE_NETSH_OUTPUT = `
Interface name : Wi-Fi 
There are 4 networks currently visible. 

SSID 1 : DASGUPTA
    Network type            : Infrastructure
    Authentication          : WPA2-Personal
    Encryption              : CCMP 
    BSSID 1                 : 30:68:93:87:a0:74
         Signal             : 69%  
         Radio type         : 802.11ac
         Band               : 5 GHz
         Channel            : 149 
    BSSID 2                 : 30:68:93:87:a0:72
         Signal             : 93%  
         Radio type         : 802.11ac
         Band               : 2.4 GHz
         Channel            : 3 

SSID 2 : 
    Network type            : Infrastructure
    Authentication          : WPA2-Personal
    Encryption              : CCMP 
    BSSID 1                 : 11:22:33:44:55:66
         Signal             : 40%  
         Radio type         : 802.11n
         Band               : 2.4 GHz
         Channel            : 6 

SSID 3 : sarkar house 5g
    Network type            : Infrastructure
    Authentication          : WPA3-Personal
    Encryption              : CCMP 
    BSSID 1                 : a0:4c:0c:9f:09:bb
         Signal             : 18%  
         Radio type         : 802.11ac
         Band               : 5 GHz
         Channel            : 36 

SSID 4 : Public Free Wifi
    Network type            : Infrastructure
    Authentication          : Open
    Encryption              : None 
    BSSID 1                 : 50:91:e3:e0:73:0c
         Signal             : 85%  
         Radio type         : 802.11n
         Band               : 2.4 GHz
         Channel            : 9 
`;

test('signalPercentToBars correctly maps percentage buckets', () => {
  assert.equal(signalPercentToBars(0), 0);
  assert.equal(signalPercentToBars(10), 1);
  assert.equal(signalPercentToBars(24), 1);
  assert.equal(signalPercentToBars(25), 2);
  assert.equal(signalPercentToBars(49), 2);
  assert.equal(signalPercentToBars(50), 3);
  assert.equal(signalPercentToBars(74), 3);
  assert.equal(signalPercentToBars(75), 4);
  assert.equal(signalPercentToBars(100), 4);
});

test('rssiToPercent converts dBm to percentage correctly', () => {
  assert.equal(rssiToPercent(-110), 0);
  assert.equal(rssiToPercent(-100), 0);
  assert.equal(rssiToPercent(-50), 100);
  assert.equal(rssiToPercent(-40), 100);
  assert.equal(rssiToPercent(-75), 50);
});

test('mapAuthToSecurity correctly identifies security types', () => {
  assert.equal(mapAuthToSecurity('WPA2-Personal'), 'wpa2');
  assert.equal(mapAuthToSecurity('WPA3-Personal'), 'wpa3');
  assert.equal(mapAuthToSecurity('WPA3-SAE'), 'wpa3');
  assert.equal(mapAuthToSecurity('Open'), 'open');
  assert.equal(mapAuthToSecurity('None'), 'open');
  assert.equal(mapAuthToSecurity('WEP-128'), 'wep');
  assert.equal(mapAuthToSecurity('WPA2-Enterprise (802.1X)'), 'enterprise');
});

test('channelToBand maps channels correctly', () => {
  assert.equal(channelToBand(1), '2.4GHz');
  assert.equal(channelToBand(6), '2.4GHz');
  assert.equal(channelToBand(14), '2.4GHz');
  assert.equal(channelToBand(36), '5GHz');
  assert.equal(channelToBand(149), '5GHz');
});

test('parseNetshOutput parses networks, dual band, and filters hidden SSID', () => {
  const networks = parseNetshOutput(SAMPLE_NETSH_OUTPUT);

  // Hidden SSID 2 should be skipped
  assert.equal(networks.length, 3);

  // Supported networks appear first
  assert.equal(networks[0]?.ssid, 'DASGUPTA');
  assert.equal(networks[0]?.band, 'dual');
  assert.equal(networks[0]?.unsupported, false);
  assert.equal(networks[0]?.signalPercent, 93);
  assert.equal(networks[0]?.signalBars, 4);
  assert.equal(networks[0]?.security, 'wpa2');

  assert.equal(networks[1]?.ssid, 'Public Free Wifi');
  assert.equal(networks[1]?.band, '2.4GHz');
  assert.equal(networks[1]?.unsupported, false);
  assert.equal(networks[1]?.security, 'open');
  assert.equal(networks[1]?.signalBars, 4);

  // 5GHz-only network appears last as unsupported
  assert.equal(networks[2]?.ssid, 'sarkar house 5g');
  assert.equal(networks[2]?.band, '5GHz');
  assert.equal(networks[2]?.unsupported, true);
  assert.equal(networks[2]?.security, 'wpa3');
});

test('parseNmcliOutput parses Linux formatted lines', () => {
  const sampleNmcli = `
OfficeNet\\:5G:00\\:11\\:22\\:33\\:44\\:55:82:WPA2:36:5180 MHz
Home-2G:66\\:77\\:88\\:99\\:aa\\:bb:95:WPA2:6:2437 MHz
CoffeeFree:aa\\:bb\\:cc\\:dd\\:ee\\:ff:45:Open:1:2412 MHz
`;
  const networks = parseNmcliOutput(sampleNmcli);
  assert.equal(networks.length, 3);

  const home = networks.find((n) => n.ssid === 'Home-2G');
  assert.ok(home);
  assert.equal(home?.band, '2.4GHz');
  assert.equal(home?.unsupported, false);
  assert.equal(home?.signalBars, 4);

  const office = networks.find((n) => n.ssid === 'OfficeNet:5G');
  assert.ok(office);
  assert.equal(office?.band, '5GHz');
  assert.equal(office?.unsupported, true);
});

test('parseAirportOutput parses macOS output', () => {
  const sampleAirport = `
                            SSID BSSID             RSSI CHANNEL HT CC SECURITY (auth/unicast/group)
                         MacWifi 12:34:56:78:9a:bc -60  6       Y  US WPA2(PSK/AES/AES)
                      MacWifi-5G 12:34:56:78:9a:bd -85  149     Y  US WPA2(PSK/AES/AES)
`;
  const networks = parseAirportOutput(sampleAirport);
  assert.equal(networks.length, 2);

  const mac2g = networks.find((n) => n.ssid === 'MacWifi');
  assert.ok(mac2g);
  assert.equal(mac2g?.band, '2.4GHz');
  assert.equal(mac2g?.unsupported, false);

  const mac5g = networks.find((n) => n.ssid === 'MacWifi-5G');
  assert.ok(mac5g);
  assert.equal(mac5g?.band, '5GHz');
  assert.equal(mac5g?.unsupported, true);
});

test('scanWifiNetworks returns valid result with network array', async () => {
  const result = await scanWifiNetworks({ force: true });
  assert.ok(Array.isArray(result.networks));
  assert.ok(typeof result.count === 'number');
  assert.ok(['system', 'fallback', 'mock'].includes(result.source));
  assert.ok(result.scannedAt);

  // If running on this dev system with real Wi-Fi, it should have discovered networks
  if (result.source === 'system') {
    assert.ok(result.networks.length > 0);
    // Every network satisfies the schema
    for (const net of result.networks) {
      assert.ok(net.ssid.length >= 1 && net.ssid.length <= 32);
      assert.ok(net.signalBars >= 0 && net.signalBars <= 4);
      assert.ok(['2.4GHz', '5GHz', 'dual'].includes(net.band));
      assert.ok(typeof net.unsupported === 'boolean');
    }
  }
});

test('Fastify server endpoints: /health and /api/wifi/networks', async () => {
  const healthRes = await app.inject({
    method: 'GET',
    url: '/health',
  });
  assert.equal(healthRes.statusCode, 200);
  const healthData = JSON.parse(healthRes.payload);
  assert.equal(healthData.status, 'ok');

  const wifiRes = await app.inject({
    method: 'GET',
    url: '/api/wifi/networks',
  });
  assert.equal(wifiRes.statusCode, 200);
  const wifiData = JSON.parse(wifiRes.payload);
  assert.ok(Array.isArray(wifiData.networks));
  assert.ok(typeof wifiData.count === 'number');

  // Also test alias /wifi/networks
  const aliasRes = await app.inject({
    method: 'GET',
    url: '/wifi/networks',
  });
  assert.equal(aliasRes.statusCode, 200);

  // Also test POST /api/wifi/scan
  const scanRes = await app.inject({
    method: 'POST',
    url: '/api/wifi/scan',
  });
  assert.equal(scanRes.statusCode, 200);
});
