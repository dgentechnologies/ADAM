import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

/** Tailwind-aware class joiner used by every component in this package. */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
