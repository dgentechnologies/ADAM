import { Inter, Michroma } from 'next/font/google';

/**
 * Typography split (confirmed decision):
 *   Inter    — all UI type. Matches every rendered Stitch screen.
 *   Michroma — the ADAM wordmark and eyebrow labels only. DESIGN.md asks for it
 *              at every level, but it is a display face; at 16px it forces
 *              three-line wraps at 375px and matches no screenshot.
 *
 * next/font self-hosts both at build time, so the static export has no runtime
 * dependency on fonts.googleapis.com — required, since the phone is joined to
 * ADAM's offline provisioning hotspot during setup.
 */
export const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-inter',
  weight: ['400', '500', '600', '700'],
  fallback: ['system-ui', 'sans-serif'],
});

export const michroma = Michroma({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-michroma',
  weight: '400',
  fallback: ['var(--font-inter)', 'system-ui', 'sans-serif'],
});

export const fontVariables = `${inter.variable} ${michroma.variable}`;
