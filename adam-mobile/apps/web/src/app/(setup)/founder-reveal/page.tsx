'use client';

import { Button, Card, Screen, ScreenActions } from '@adam/ui';
import { ArrowRight, Gem, MessagesSquare, Zap } from 'lucide-react';
import { useRouter } from 'next/navigation';

import { useSetupStore } from '@/stores/setup-store';

/**
 * `founder_edition` — reached only for units 1–10.
 *
 * This is the one screen licensed to use the coarse diagonal hatch instead of the
 * dot matrix, which is what makes it feel like a different surface without
 * introducing a colour.
 */
const PERKS = [
  { Icon: Zap, title: 'Lifetime Priority Credits', body: 'Never wait in queue.' },
  { Icon: MessagesSquare, title: 'Founder Discord Access', body: 'Direct line to the team.' },
] as const;

export default function FounderRevealPage() {
  const router = useRouter();
  const founderNumber = useSetupStore((state) => state.founderNumber);
  const complete = useSetupStore((state) => state.complete);

  function next() {
    complete('founder-reveal');
    router.push('/ai-brain');
  }

  return (
    <Screen className="pt-safe">
      <div className="digital-skin-hatch pointer-events-none fixed inset-0" aria-hidden />

      <div className="relative flex flex-1 flex-col justify-center gap-stack-lg">
        <div className="flex flex-col items-center gap-stack-md text-center">
          {/* The outlined disc + gem glyph the Stitch screen opens with. */}
          <span className="flex h-20 w-20 items-center justify-center rounded-full border border-border-strong">
            <Gem className="h-8 w-8 text-fg" strokeWidth={1.5} aria-hidden />
          </span>

          {/* Michroma per the locked type split — the reveal is the one screen
              besides the wordmark licensed to use the display face at size. */}
          <h1 className="font-display text-title-md leading-tight text-fg">
            Founder Edition № {String(founderNumber ?? 1).padStart(3, '0')}
          </h1>
          <p className="text-body-lg text-fg-muted">You’re one of the first ten.</p>
        </div>

        <Card surface="raised" padding="none">
          <ul className="[&>*+*]:hairline-t">
            {PERKS.map(({ Icon, title, body }) => (
              <li key={title} className="flex items-center gap-gutter p-stack-md">
                <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full border border-border bg-surface">
                  <Icon className="h-5 w-5 text-fg" strokeWidth={1.5} aria-hidden />
                </span>
                <span className="flex min-w-0 flex-col">
                  <span className="text-body-md text-fg">{title}</span>
                  <span className="text-label-md text-fg-muted">{body}</span>
                </span>
              </li>
            ))}
          </ul>
        </Card>
      </div>

      <ScreenActions className="relative">
        <Button block variant="primary" onClick={next}>
          Continue
          <ArrowRight className="h-5 w-5" strokeWidth={1.5} aria-hidden />
        </Button>
      </ScreenActions>
    </Screen>
  );
}
