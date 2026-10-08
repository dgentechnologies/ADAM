'use client';

import type { CreditPackId } from '@adam/types';
import { Button, OptionCard, Screen, ScreenActions, ScreenHeader } from '@adam/ui';
import { useQuery } from '@tanstack/react-query';
import { ArrowRight } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { useState } from 'react';

import { fetchCreditPacks, queryKeys } from '@/lib/mock/api';
import { useSetupStore } from '@/stores/setup-store';

/**
 * `credit_selection` — managed mode only.
 *
 * Razorpay is explicitly out of scope for this pass, so "Continue to Payment"
 * advances the wizard without a checkout. The pack list comes from the mock API
 * rather than a local constant so the real endpoint is a one-line swap.
 */
export default function CreditsPage() {
  const router = useRouter();
  const complete = useSetupStore((state) => state.complete);
  const [selected, setSelected] = useState<CreditPackId | null>(null);

  const { data: packs = [], isPending } = useQuery({
    queryKey: queryKeys.creditPacks,
    queryFn: fetchCreditPacks,
  });

  function submit() {
    if (!selected) return;
    complete('credits');
    router.push('/camera-permission');
  }

  return (
    <Screen className="pt-stack-md">
      <ScreenHeader
        size="md"
        title="Pick a credit pack."
        subtitle="Credits are active processing minutes. They never expire."
      />

      <div className="flex flex-col gap-stack-md pt-stack-lg">
        {isPending
          ? Array.from({ length: 5 }, (_, index) => (
              <div
                key={index}
                className="h-20 animate-breathe rounded-card border border-border bg-surface-raised"
              />
            ))
          : packs.map((pack) => (
              <OptionCard
                key={pack.id}
                layout="stacked"
                lead={pack.priceLabel}
                title={pack.name}
                description={`approx. ${pack.estimatedMinutes} active minutes`}
                badge={pack.isMostPopular ? 'Most Popular' : undefined}
                selected={selected === pack.id}
                onSelect={() => setSelected(pack.id)}
              />
            ))}
      </div>

      <ScreenActions>
        <Button block variant="primary" disabled={!selected} onClick={submit}>
          Continue to Payment
          <ArrowRight className="h-5 w-5" strokeWidth={1.5} aria-hidden />
        </Button>
        <p className="text-center text-label-md text-fg-faint">
          Payment is not wired up in this build.
        </p>
      </ScreenActions>
    </Screen>
  );
}
