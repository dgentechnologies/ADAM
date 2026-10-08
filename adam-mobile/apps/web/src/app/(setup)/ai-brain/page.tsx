'use client';

import type { AiBrainMode } from '@adam/types';
import { OptionCard, Screen, ScreenHeader } from '@adam/ui';
import { useRouter } from 'next/navigation';

import { nextStep, setupHref } from '@/lib/setup-flow';
import { useSetupStore } from '@/stores/setup-store';

/**
 * `choose_adam_s_brain` — the only branch point in the wizard.
 *
 * Each card navigates on tap (chevron affordance, no bottom CTA), matching the
 * Stitch screen: the choice *is* the action, so a confirm button would add a step
 * that decides nothing. Copy is Stitch's verbatim.
 *
 * Lite Mode stays reachable here forever (spec §1): it is a listed option, not a
 * "skip" link, because a user who never buys credits still owns a working device.
 */
const MODES: ReadonlyArray<{
  mode: AiBrainMode;
  title: string;
  description: string;
  badge?: string;
}> = [
  {
    mode: 'byok',
    title: 'Bring Your Own Key',
    description: 'Free. Your own Google API key. Your data, your quota.',
    badge: 'Recommended',
  },
  {
    mode: 'managed',
    title: 'DGEN Managed Credits',
    description: 'We handle it. One-time credit packs from ₹599.',
  },
  {
    mode: 'lite',
    title: 'Skip for now (Lite Mode)',
    description: 'Clock, alarms, smart home — no live AI conversation yet.',
  },
];

export default function AiBrainPage() {
  const router = useRouter();
  const setAiBrainMode = useSetupStore((state) => state.setAiBrainMode);
  const complete = useSetupStore((state) => state.complete);
  const isFounderEdition = useSetupStore((state) => state.isFounderEdition);

  function choose(mode: AiBrainMode) {
    setAiBrainMode(mode);
    complete('ai-brain');
    const next = nextStep('ai-brain', { isFounderEdition, aiBrainMode: mode });
    router.push(next === 'done' ? '/home' : setupHref(next));
  }

  return (
    <Screen className="pt-stack-md">
      <ScreenHeader
        size="xs"
        title="Choose ADAM’s brain."
        subtitle="Select how you want to power the intelligence engine."
      />

      <div className="flex flex-col gap-stack-md pt-stack-lg">
        {MODES.map(({ mode, title, description, badge }) => (
          <OptionCard
            key={mode}
            title={title}
            description={description}
            badge={badge}
            affordance="chevron"
            onSelect={() => choose(mode)}
          />
        ))}
      </div>
    </Screen>
  );
}
