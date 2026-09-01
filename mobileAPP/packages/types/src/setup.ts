import { z } from 'zod';
import { AiBrainMode } from './device.js';
import { DeviceSerial } from './common.js';

/**
 * Setup wizard steps, ordered. The slug matches the route segment under
 * app/(setup)/ exactly, so navigation is derived from this array rather than
 * hard-coded per screen.
 */
export const SetupStep = z.enum([
  'splash',
  'welcome',
  'sign-in',
  'discover',
  'device-found',
  'wifi-select',
  'wifi-password',
  'connecting',
  'name-device',
  'founder-reveal',
  'ai-brain',
  'byok',
  'credits',
  'camera-permission',
  'face-capture',
]);
export type SetupStep = z.infer<typeof SetupStep>;

export const SETUP_STEP_ORDER = SetupStep.options;

/**
 * Steps that count toward the "Step X of Y" indicator. Splash and welcome are
 * pre-wizard; founder-reveal is conditional; byok/credits are branches of
 * ai-brain, so none of them advance the counter.
 */
export const SETUP_PROGRESS_STEPS = [
  'sign-in',
  'discover',
  'wifi-select',
  'name-device',
  'ai-brain',
  'camera-permission',
] as const satisfies readonly SetupStep[];

/**
 * The only steps a user cannot skip (spec §1): Account → Pairing → Wi-Fi →
 * Naming → AI Brain choice. Lite Mode must always remain reachable, so nothing
 * downstream of ai-brain is ever required.
 */
export const SETUP_REQUIRED_STEPS = [
  'sign-in',
  'device-found',
  'connecting',
  'name-device',
  'ai-brain',
] as const satisfies readonly SetupStep[];

/**
 * Persisted wizard state. Spec §7 requires resuming exactly where the user left
 * off after the app is closed mid-setup, so this is written to storage on every
 * transition — not held in memory only.
 */
export const SetupState = z.object({
  /** Schema version so a stored state from an older build can be discarded. */
  version: z.literal(1),
  currentStep: SetupStep,
  completedSteps: z.array(SetupStep),
  signedIn: z.boolean(),
  selectedSerial: DeviceSerial.nullable(),
  selectedSsid: z.string().nullable(),
  deviceName: z.string().nullable(),
  aiBrainMode: AiBrainMode.nullable(),
  isFounderEdition: z.boolean(),
  founderNumber: z.number().int().min(1).max(10).nullable(),
  cameraPermissionGranted: z.boolean(),
  userNameForFace: z.string().nullable(),
  completedAt: z.string().nullable(),
});
export type SetupState = z.infer<typeof SetupState>;

export const INITIAL_SETUP_STATE: SetupState = {
  version: 1,
  currentStep: 'splash',
  completedSteps: [],
  signedIn: false,
  selectedSerial: null,
  selectedSsid: null,
  deviceName: null,
  aiBrainMode: null,
  isFounderEdition: false,
  founderNumber: null,
  cameraPermissionGranted: false,
  userNameForFace: null,
  completedAt: null,
};
