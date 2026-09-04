'use client';

import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { usePathname } from 'next/navigation';
import { useEffect, useRef, type ReactNode } from 'react';

import { SETUP_ORDER, stepFromPathname } from '../lib/setup-flow';

/**
 * Step-to-step transition for the `(setup)` wizard.
 *
 * Static jumps read as a reload rather than a wizard, so each step slides 12px in
 * the direction of travel while fading. Direction comes from the step's index in
 * `SETUP_ORDER` versus the previous one, so `router.back()` mirrors the animation
 * without the screens knowing anything about it.
 *
 * `mode="wait"` is required: two absolutely-positioned steps overlapping would
 * double the fixed CTA rail and let the user tap the outgoing screen's button.
 *
 * Reduced motion drops the translate and keeps only opacity — the plugin already
 * neutralises CSS animation duration, but Framer Motion runs off the main thread
 * and has to be told separately.
 */
const DISTANCE = 12;
const DURATION = 0.22;
const EASE = [0.22, 1, 0.36, 1] as const;

export function SetupTransition({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const reduceMotion = useReducedMotion();

  const step = stepFromPathname(pathname);
  const index = step ? SETUP_ORDER.indexOf(step) : -1;
  const previousIndex = useRef(index);

  const back = index >= 0 && previousIndex.current >= 0 && index < previousIndex.current;

  useEffect(() => {
    previousIndex.current = index;
  }, [index]);

  const offset = reduceMotion ? 0 : back ? -DISTANCE : DISTANCE;

  return (
    <AnimatePresence mode="wait" initial={false}>
      <motion.div
        key={pathname}
        className="flex flex-1 min-h-0 flex-col overflow-hidden"
        initial={{ opacity: 0, x: offset }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: -offset }}
        transition={{ duration: DURATION, ease: EASE }}
      >
        {children}
      </motion.div>
    </AnimatePresence>
  );
}
