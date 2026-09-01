/**
 * @adam/types — the single API contract.
 *
 * Both apps/web and apps/api import from here; a route stub and its client
 * caller cannot drift because they validate against the same Zod schema.
 */
export * from './common.js';
export * from './auth.js';
export * from './device.js';
export * from './wifi.js';
export * from './credits.js';
export * from './ota.js';
export * from './gallery.js';
export * from './memory.js';
export * from './setup.js';
export * from './preferences.js';
