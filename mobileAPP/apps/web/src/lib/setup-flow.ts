import type { SetupStep } from '@adam/types';

/**
 * The wizard's linear order, and the mapping between a `SetupStep` and its route.
 *
 * Kept in one place so "next"/"back" are derived rather than hard-coded per page —
 * the previous hazard being fifteen screens each with their own opinion about
 * where they lead.
 */
export const SETUP_ORDER: readonly SetupStep[] = [
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
];

export function setupHref(step: SetupStep): string {
  return `/${step}`;
}

/** Reverse lookup used by the layout to record the current route in the store. */
export function stepFromPathname(pathname: string): SetupStep | null {
  const slug = pathname.replace(/^\/+|\/+$/g, '');
  return SETUP_ORDER.find((step) => step === slug) ?? null;
}

/**
 * Branch points the linear order cannot express:
 *  - a non-Founder unit skips the reveal;
 *  - the brain choice forks to BYOK, credits, or straight past both (Lite);
 *  - BYOK and credits both rejoin at the camera step.
 */
export function nextStep(
  step: SetupStep,
  context: { isFounderEdition: boolean; aiBrainMode: string | null },
): SetupStep | 'done' {
  switch (step) {
    case 'name-device':
      return context.isFounderEdition ? 'founder-reveal' : 'ai-brain';
    case 'ai-brain':
      if (context.aiBrainMode === 'byok') return 'byok';
      if (context.aiBrainMode === 'managed') return 'credits';
      return 'camera-permission';
    case 'byok':
    case 'credits':
      return 'camera-permission';
    case 'face-capture':
      return 'done';
    default: {
      const index = SETUP_ORDER.indexOf(step);
      const next = SETUP_ORDER[index + 1];
      return next ?? 'done';
    }
  }
}
