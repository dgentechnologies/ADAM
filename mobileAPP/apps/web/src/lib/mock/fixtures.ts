import type {
  CreditBalance,
  CreditPack,
  Device,
  DiscoveredDevice,
  GalleryItem,
  MemoryEntry,
  OtaState,
  PairedLaptop,
  WifiNetwork,
} from '@adam/types';

/**
 * Mock fixtures. This pass ships no backend, so every screen reads from here
 * through `lib/mock/api.ts`. Values satisfy the Zod schemas in @adam/types, which
 * means swapping in real endpoints later is a transport change only.
 *
 * Copy is taken from the Stitch screens where it exists, normalised where the
 * export was inconsistent (the network list and the password headline disagreed
 * on the mock SSID; both now use `Home_Wifi_2.4`).
 */
export const MOCK_DEVICE_ID = '7c9e6679-7425-40de-944b-e07fc1f90ae7';
const OWNER_ID = 'b1a7c2e4-3f8d-4a6b-9c1e-2d5f8a7b3c4d';

export const MOCK_DISCOVERED: DiscoveredDevice[] = [
  {
    shortId: 'ADAM-3F2A',
    serial: 'DGEN-ADAM-0007',
    transport: 'ble',
    signalStrength: -52,
    isFounderEdition: true,
    alreadyClaimed: false,
  },
];

export const MOCK_DEVICE: Device = {
  id: MOCK_DEVICE_ID,
  serial: 'DGEN-ADAM-0007',
  shortId: 'ADAM-3F2A',
  name: 'ADAM',
  ownerId: OWNER_ID,
  status: 'online',
  expression: 'idle',
  aiBrainMode: 'byok',
  firmwareVersion: '40.2.1',
  hardwareBatch: 'FE-2026-01',
  isFounderEdition: true,
  founderNumber: 7,
  wifiSsid: 'Home_Wifi_2.4',
  lastSeenAt: '2026-08-30T02:14:00.000Z',
  claimedAt: '2026-08-01T09:12:00.000Z',
};

/**
 * The scan list from `connecting_to_wi_fi`. `DGEN_STUDIO_5G` is kept 5GHz-only so
 * the "ADAM only supports 2.4GHz networks" disabled state is exercised.
 */
export const MOCK_NETWORKS: WifiNetwork[] = [
  { ssid: 'DGEN_STUDIO_5G', signalBars: 4, security: 'wpa2', band: '5GHz', unsupported: true },
  { ssid: 'ADAM_GUEST_NET', signalBars: 4, security: 'wpa2', band: '2.4GHz', unsupported: false },
  { ssid: 'Starlink_42', signalBars: 1, security: 'wpa3', band: 'dual', unsupported: false },
  { ssid: 'Home_Wifi_2.4', signalBars: 2, security: 'wpa2', band: '2.4GHz', unsupported: false },
  { ssid: 'Coffee_Shop_Free', signalBars: 3, security: 'open', band: '2.4GHz', unsupported: false },
];

/** Prices from `choose_a_credit_pack`, stored in paise per the schema. */
export const MOCK_CREDIT_PACKS: CreditPack[] = [
  { id: 'trial', name: 'Trial', pricePaise: 59900, priceLabel: '₹599', estimatedMinutes: 60, isMostPopular: false },
  { id: 'starter', name: 'Starter', pricePaise: 149900, priceLabel: '₹1,499', estimatedMinutes: 180, isMostPopular: false },
  { id: 'standard', name: 'Standard', pricePaise: 299900, priceLabel: '₹2,999', estimatedMinutes: 400, isMostPopular: true },
  { id: 'value', name: 'Value', pricePaise: 549900, priceLabel: '₹5,499', estimatedMinutes: 850, isMostPopular: false },
  { id: 'pro', name: 'Pro', pricePaise: 1199900, priceLabel: '₹11,999', estimatedMinutes: 2000, isMostPopular: false },
];

export const MOCK_BALANCE: CreditBalance = {
  deviceId: MOCK_DEVICE_ID,
  remainingMinutes: 312,
  totalPurchasedMinutes: 400,
  updatedAt: '2026-08-30T01:40:00.000Z',
};

export const MOCK_MEMORY: MemoryEntry[] = [
  {
    id: 'c1f3a5b7-1111-4a2b-8c3d-4e5f6a7b8c90',
    deviceId: MOCK_DEVICE_ID,
    kind: 'person',
    label: 'Sarah',
    content: 'Prefers coffee in the morning.',
    hasFaceProfile: true,
    createdAt: '2026-08-12T07:30:00.000Z',
  },
  {
    id: 'c1f3a5b7-2222-4a2b-8c3d-4e5f6a7b8c90',
    deviceId: MOCK_DEVICE_ID,
    kind: 'person',
    label: 'James',
    content: 'Enjoys classical music.',
    hasFaceProfile: false,
    createdAt: '2026-08-14T18:05:00.000Z',
  },
  {
    id: 'c1f3a5b7-3333-4a2b-8c3d-4e5f6a7b8c90',
    deviceId: MOCK_DEVICE_ID,
    kind: 'fact',
    label: 'Office',
    content: 'The office is kept at 22°C.',
    hasFaceProfile: false,
    createdAt: '2026-08-20T11:00:00.000Z',
  },
  {
    id: 'c1f3a5b7-4444-4a2b-8c3d-4e5f6a7b8c90',
    deviceId: MOCK_DEVICE_ID,
    kind: 'fact',
    label: 'Calendar',
    content: 'Meeting scheduled for 3 PM tomorrow.',
    hasFaceProfile: false,
    createdAt: '2026-08-29T15:20:00.000Z',
  },
];

/**
 * Gallery placeholders carry no `url` that resolves — every Stitch image was a
 * signed `lh3.googleusercontent.com` link that will expire, so the grid renders a
 * generated texture tile instead of fetching anything.
 */
export const MOCK_GALLERY: GalleryItem[] = Array.from({ length: 9 }, (_, index) => ({
  id: `a2b4c6d8-${String(index).padStart(4, '0')}-4e2f-9a1b-3c5d7e9f0a1b`,
  deviceId: MOCK_DEVICE_ID,
  url: '',
  thumbnailUrl: '',
  width: 1080,
  height: 1080,
  capturedAt: new Date(Date.UTC(2026, 7, 29 - index, 9 + (index % 8))).toISOString(),
  reason: index % 3 === 0 ? 'face-recognised' : index % 3 === 1 ? 'moment' : 'requested',
  starred: index % 4 === 0,
  personName: index % 3 === 0 ? (index % 2 === 0 ? 'Sarah' : 'James') : null,
}));

export const MOCK_LAPTOPS: PairedLaptop[] = [
  {
    pairingId: 'd3e5f7a9-1111-4b3c-8d4e-5f6a7b8c9d01',
    deviceId: MOCK_DEVICE_ID,
    hostname: 'DGEN-STUDIO-01',
    os: 'windows',
    lastSeenAt: '2026-08-30T01:55:00.000Z',
    online: true,
  },
  {
    pairingId: 'd3e5f7a9-2222-4b3c-8d4e-5f6a7b8c9d01',
    deviceId: MOCK_DEVICE_ID,
    hostname: 'tirthankar-mbp',
    os: 'macos',
    lastSeenAt: '2026-08-27T21:10:00.000Z',
    online: false,
  },
];

/**
 * Both Software Update states behind one flag, as agreed. Flip
 * `MOCK_UPDATE_AVAILABLE` to see the other half of the screen.
 *
 * The Stitch export disagreed with itself on version (v40.2 in one state, v40.2.1
 * in the other); v40.2.1 is treated as current and v41.0 as the available update.
 */
export const MOCK_UPDATE_AVAILABLE = true;

export const MOCK_OTA: OtaState = {
  currentVersion: '40.2.1',
  manifest: MOCK_UPDATE_AVAILABLE
    ? {
        latestVersion: '41.0',
        packageSizeBytes: 2_576_980_378,
        changelog: [
          { text: 'Improved spatial awareness in crowded environments.' },
          { text: 'Faster response times for complex cognitive tasks.' },
          { text: 'Minor bug fixes and security enhancements.' },
        ],
        publishedAt: '2026-08-28T00:00:00.000Z',
        packageUrl: 'https://updates.dgen.tech/adam/41.0/adam-os-41.0.tar.zst',
        signature: 'mock-signature',
        mandatory: false,
      }
    : null,
  updateAvailable: MOCK_UPDATE_AVAILABLE,
  stage: 'idle',
  progress: 0,
  lastCheckedAt: '2026-08-30T02:00:00.000Z',
  notifyOnUpdate: true,
};

export const MOCK_USER = {
  name: 'Tirthankar Dasgupta',
  email: 'tirthankar@dgen.tech',
} as const;
