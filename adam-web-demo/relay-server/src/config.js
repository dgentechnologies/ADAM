// config.js — environment validation and session caps

const REQUIRED_ENV = [
  'GOOGLE_API_KEY',
  'FIREBASE_ADMIN_PROJECT_ID',
  'FIREBASE_ADMIN_CLIENT_EMAIL',
  'FIREBASE_ADMIN_PRIVATE_KEY',
  'RELAY_JWT_SECRET',
  'ALLOWED_ORIGIN',
];

for (const key of REQUIRED_ENV) {
  if (!process.env[key]) {
    console.error(`[CONFIG] Missing required environment variable: ${key}`);
    process.exit(1);
  }
}

export const CONFIG = {
  PORT:            parseInt(process.env.PORT ?? '8080', 10),
  NODE_ENV:        process.env.NODE_ENV ?? 'development',
  ALLOWED_ORIGIN:  process.env.ALLOWED_ORIGIN,
  GOOGLE_API_KEY:  process.env.GOOGLE_API_KEY,
  RELAY_JWT_SECRET: process.env.RELAY_JWT_SECRET,

  FIREBASE: {
    PROJECT_ID:    process.env.FIREBASE_ADMIN_PROJECT_ID,
    CLIENT_EMAIL:  process.env.FIREBASE_ADMIN_CLIENT_EMAIL,
    // Railway stores the key with literal \n — replace them with real newlines
    PRIVATE_KEY:   process.env.FIREBASE_ADMIN_PRIVATE_KEY.replace(/\\n/g, '\n'),
  },
};

export const SESSION_CAPS = {
  MAX_DURATION_MS:       5 * 60 * 1000,  // 5 minutes per session
  MAX_TURNS:             20,              // 20 conversation turns
  MAX_SESSIONS_PER_DAY:  3,              // per Firebase UID
  COOLDOWN_MS:           10 * 60 * 1000, // 10 min between sessions
};
