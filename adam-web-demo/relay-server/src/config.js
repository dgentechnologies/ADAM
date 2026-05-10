// config.js — environment validation and session caps

const REQUIRED_ENV = [
  'GOOGLE_API_KEY',
  'FIREBASE_ADMIN_PROJECT_ID',
  'FIREBASE_ADMIN_CLIENT_EMAIL',
  'FIREBASE_ADMIN_PRIVATE_KEY',
  'FIREBASE_WEBSITE_DATABASE_ID',
  'RELAY_JWT_SECRET',
  'ALLOWED_ORIGIN',
];

for (const key of REQUIRED_ENV) {
  if (!process.env[key]) {
    console.error(`[CONFIG] Missing required environment variable: ${key}`);
    process.exit(1);
  }
}

function parsePort(raw) {
  const parsed = Number.parseInt(raw ?? '8080', 10);
  if (Number.isNaN(parsed) || parsed <= 0 || parsed > 65535) {
    console.error('[CONFIG] PORT must be a valid TCP port (1-65535)');
    process.exit(1);
  }
  return parsed;
}

function parseAllowedOrigins(raw) {
  const origins = String(raw)
    .split(',')
    .map((origin) => origin.trim())
    .filter(Boolean);

  if (!origins.length) {
    console.error('[CONFIG] ALLOWED_ORIGIN must contain at least one origin');
    process.exit(1);
  }

  for (const origin of origins) {
    try {
      const parsed = new URL(origin);
      if (!['http:', 'https:'].includes(parsed.protocol)) {
        throw new Error('Unsupported protocol');
      }
    } catch {
      console.error(`[CONFIG] Invalid origin in ALLOWED_ORIGIN: ${origin}`);
      process.exit(1);
    }
  }

  return new Set(origins);
}

function parseBoolean(raw, fallback = false) {
  if (raw === undefined || raw === null || String(raw).trim() === '') return fallback;
  return ['true', '1', 'yes', 'on'].includes(String(raw).trim().toLowerCase());
}

export const CONFIG = {
  PORT:            parsePort(process.env.PORT),
  NODE_ENV:        process.env.NODE_ENV ?? 'development',
  ALLOWED_ORIGINS: parseAllowedOrigins(process.env.ALLOWED_ORIGIN),
  GOOGLE_API_KEY:  process.env.GOOGLE_API_KEY,
  GEMINI_LIVE_MODEL: process.env.GEMINI_LIVE_MODEL ?? 'gemini-3.1-flash-live-preview',
  RELAY_JWT_SECRET: process.env.RELAY_JWT_SECRET,
  ALLOW_LEGACY_RELAY_TOKENS: parseBoolean(process.env.ALLOW_LEGACY_RELAY_TOKENS),

  FIREBASE: {
    PROJECT_ID:    process.env.FIREBASE_ADMIN_PROJECT_ID,
    CLIENT_EMAIL:  process.env.FIREBASE_ADMIN_CLIENT_EMAIL,
    DATABASE_ID:   process.env.FIREBASE_WEBSITE_DATABASE_ID,
    // Railway stores the key with literal \n — replace them with real newlines
    PRIVATE_KEY:   process.env.FIREBASE_ADMIN_PRIVATE_KEY.replace(/\\n/g, '\n'),
  },
};

export const SESSION_CAPS = {
  MAX_DURATION_MS:         5 * 60 * 1000,  // 5 minutes per session
  MAX_TURNS:               20,             // 20 conversation turns
  // Regular users get exactly ONE lifetime session (no repeats, no daily resets).
  MAX_SESSIONS_LIFETIME:   1,
};

// UIDs that bypass all session caps (internal testers / devs).
// Add UIDs as a comma-separated TESTER_UIDS env var, or hardcode here.
const envTesters = (process.env.TESTER_UIDS ?? '')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean);

export const TESTER_UIDS = new Set([
  'DUHBEYpqD1W0oInXMvVIrgN5pfG2',
  'J18eb5xtHMVGTTOHoAguFfOgU7p2', // internal tester
  ...envTesters,
]);

/** Returns true if the uid is a privileged tester — all caps are skipped. */
export function isTester(uid) {
  return TESTER_UIDS.has(uid);
}
