/**
 * @adam/types — the single API contract.
 *
 * Both apps/web and apps/api import from here; a route stub and its client
 * caller cannot drift because they validate against the same Zod schema.
 */
export * from './common';
export * from './auth';
export * from './device';
export * from './wifi';
export * from './credits';
export * from './ota';
export * from './gallery';
export * from './memory';
export * from './setup';
export * from './preferences';
export * from './firestore';

