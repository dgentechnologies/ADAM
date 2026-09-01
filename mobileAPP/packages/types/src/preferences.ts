import { z } from 'zod';

/**
 * Personality & voice preferences (spec §2.7). The UI slots exist now even where
 * the firmware ships a single option, so adding voices later is data, not UI work.
 */
export const VoiceId = z.enum(['charon']);
export type VoiceId = z.infer<typeof VoiceId>;

export const LanguagePreference = z.enum(['en', 'hi', 'bn', 'hinglish-auto']);
export type LanguagePreference = z.infer<typeof LanguagePreference>;

/** Maps to the attention/VAD threshold on the unit. */
export const WakeSensitivity = z.enum(['less-sensitive', 'default', 'more-sensitive']);
export type WakeSensitivity = z.infer<typeof WakeSensitivity>;

export const DevicePreferences = z.object({
  voiceId: VoiceId,
  language: LanguagePreference,
  /** 0 = dry professional, 100 = full roast. Ships at a sensible default. */
  sarcasmLevel: z.number().int().min(0).max(100),
  wakeSensitivity: WakeSensitivity,
  muted: z.boolean(),
  /** Opt-in, off by default — nothing leaves the device unless chosen (spec §6). */
  cloudPhotoBackup: z.boolean(),
  notifyOnUpdate: z.boolean(),
});
export type DevicePreferences = z.infer<typeof DevicePreferences>;

export const DEFAULT_DEVICE_PREFERENCES: DevicePreferences = {
  voiceId: 'charon',
  language: 'hinglish-auto',
  sarcasmLevel: 45,
  wakeSensitivity: 'default',
  muted: false,
  cloudPhotoBackup: false,
  notifyOnUpdate: true,
};

export const UpdatePreferencesRequest = DevicePreferences.partial();
export type UpdatePreferencesRequest = z.infer<typeof UpdatePreferencesRequest>;
