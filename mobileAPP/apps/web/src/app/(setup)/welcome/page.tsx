'use client';

import { AdamFaceMark, Screen, ScreenActions, ScreenHeader } from '@adam/ui';
import Link from 'next/link';

import { LinkButton } from '@/components/link-button';
import { CanvasRevealEffect } from '@/components/canvas-reveal-effect';

/**
 * `set_up_adam` — first real screen. Copy is Stitch's verbatim ("Let's wake him
 * up." / "Set up your ADAM in a few minutes.").
 *
 * Layout follows the screenshot: the mark and both lines are centred, and the
 * secondary action is a plain grey text link rather than a second pill — two
 * stacked pills read as two equally-weighted choices, which this is not.
 */
export default function WelcomePage() {
  return (
    <Screen id="welcome-screen" data-page="welcome" className="relative pt-stack-lg min-h-0 flex-1 justify-between pb-safe pb-stack-md">
      {/* Dynamic CanvasRevealEffect Dot Matrix Background */}
      <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden [mask-image:radial-gradient(ellipse_80%_55%_at_50%_42%,black_20%,transparent_75%)]" aria-hidden="true">
        <CanvasRevealEffect
          animationSpeed={3}
          colors={[
            [255, 255, 255],
            [255, 255, 255],
          ]}
          dotSize={4}
          showGradient={true}
        />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_rgba(0,0,0,0.85)_0%,_rgba(0,0,0,0)_100%)] pointer-events-none" />
        <div className="absolute top-0 left-0 right-0 h-1/3 bg-gradient-to-b from-black to-transparent pointer-events-none" />
      </div>

      <div className="relative z-10 flex flex-1 flex-col items-center justify-center gap-stack-lg">
        <AdamFaceMark expression="idle" size="xl" />
        <ScreenHeader
          align="center"
          size="lg"
          title="Let’s wake him up."
          subtitle="Set up your ADAM in a few minutes."
        />
      </div>

      <ScreenActions className="relative z-10 items-center">
        <LinkButton href="/sign-in" block size="lg" variant="primary">
          Set up my ADAM.
        </LinkButton>
        <Link
          href="/sign-in"
          className="pt-stack-sm text-label-md text-fg-muted transition-colors duration-fast ease-standard hover:text-fg"
        >
          I already have an ADAM set up.
        </Link>
      </ScreenActions>
    </Screen>
  );
}
