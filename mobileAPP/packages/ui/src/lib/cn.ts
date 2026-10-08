import { clsx, type ClassValue } from 'clsx';
import { extendTailwindMerge } from 'tailwind-merge';

const customTwMerge = extendTailwindMerge({
  extend: {
    classGroups: {
      'font-size': [
        'text-display-lg',
        'text-headline-md',
        'text-headline-sm',
        'text-title-md',
        'text-body-lg',
        'text-body-md',
        'text-label-md',
        'text-label-sm',
        'text-label-xs',
      ],
    },
  },
});

/** Tailwind-aware class joiner used by every component in this package. */
export function cn(...inputs: ClassValue[]): string {
  return customTwMerge(clsx(inputs));
}
