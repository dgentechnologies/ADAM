// Explicit .js extension: tailwindcss ships no exports map, so bare
// 'tailwindcss/plugin' is unresolvable from an ESM module.
import plugin from 'tailwindcss/plugin.js';
import { palette, texture } from './tokens.js';

/**
 * ADAM base + utilities, registered through the Tailwind plugin API rather than
 * a plain CSS file.
 *
 * Why not a `.css` file in @adam/ui: Next's CSS pipeline hands every
 * `@import`-ed stylesheet to PostCSS as its own entry, so an imported file
 * containing `@layer base` fails with "no matching @tailwind base directive".
 * Registering here also guarantees these rules land *after* preflight, which is
 * what makes the body/html resets below actually win.
 */
const darkVars = {
  '--adam-black': palette.black,
  '--adam-near-black': palette['near-black'],
  '--adam-charcoal': palette.charcoal,
  '--adam-charcoal-raised': palette['charcoal-raised'],
  '--adam-grey-mid': palette['grey-mid'],
  '--adam-grey': palette.grey,
  '--adam-grey-light': palette['grey-light'],
  '--adam-grey-lighter': palette['grey-lighter'],
  '--adam-off-white': palette['off-white'],
  '--adam-white': palette.white,

  '--adam-background': 'var(--adam-black)',
  '--adam-surface': 'var(--adam-near-black)',
  '--adam-surface-raised': 'var(--adam-charcoal)',
  '--adam-surface-pressed': 'var(--adam-charcoal-raised)',
  '--adam-border': 'var(--adam-charcoal-raised)',
  '--adam-border-strong': 'var(--adam-grey-mid)',
  '--adam-fg': 'var(--adam-white)',
  '--adam-fg-muted': 'var(--adam-grey-light)',
  '--adam-fg-subtle': 'var(--adam-grey)',
  '--adam-fg-faint': 'var(--adam-grey-mid)',
  '--adam-fg-inverse': 'var(--adam-black)',

  '--adam-shadow-soft': '0px 10px 30px rgba(0, 0, 0, 0.5)',
  '--adam-bloom': '0 0 15px rgba(255, 255, 255, 0.3)',
  '--adam-chrome-bg': 'rgba(0, 0, 0, 0.6)',
  '--adam-texture-dot': 'var(--adam-white)',
  '--adam-texture-opacity': '0.14',
};

/** Light mode is a full inverse of the same ramp, not an afterthought. */
const lightVars = {
  '--adam-background': 'var(--adam-white)',
  '--adam-surface': 'var(--adam-off-white)',
  '--adam-surface-raised': 'var(--adam-white)',
  '--adam-surface-pressed': 'var(--adam-grey-lighter)',
  '--adam-border': 'var(--adam-grey-lighter)',
  '--adam-border-strong': 'var(--adam-grey-light)',
  '--adam-fg': 'var(--adam-black)',
  '--adam-fg-muted': 'var(--adam-grey)',
  '--adam-fg-subtle': 'var(--adam-grey-light)',
  '--adam-fg-faint': 'var(--adam-grey-lighter)',
  '--adam-fg-inverse': 'var(--adam-white)',

  '--adam-shadow-soft': '0px 10px 30px rgba(0, 0, 0, 0.12)',
  '--adam-bloom': '0 0 15px rgba(0, 0, 0, 0.16)',
  '--adam-chrome-bg': 'rgba(255, 255, 255, 0.72)',
  '--adam-texture-dot': 'var(--adam-black)',
  '--adam-texture-opacity': '0.12',
};

export default plugin(({ addBase, addUtilities }) => {
  addBase({
    ':root': darkVars,
    '[data-theme="dark"]': darkVars,
    '[data-theme="light"]': lightVars,

    html: {
      height: '100%',
      backgroundColor: 'var(--adam-background)',
      // Overscroll in a wrapped WebView must never reveal a white gutter.
      overscrollBehavior: 'none',
      WebkitTextSizeAdjust: '100%',
    },
    body: {
      minHeight: '100%',
      margin: '0',
      padding: '0',
      backgroundColor: 'var(--adam-background)',
      color: 'var(--adam-fg)',
      WebkitFontSmoothing: 'antialiased',
      MozOsxFontSmoothing: 'grayscale',
      overscrollBehavior: 'none',
      // No blue flash on tap inside the Capacitor shell.
      WebkitTapHighlightColor: 'transparent',
    },

    // Native-feeling scrolling: no visible scrollbars anywhere.
    '::-webkit-scrollbar': { display: 'none' },
    '*': { scrollbarWidth: 'none' },

    '::selection': {
      backgroundColor: 'var(--adam-fg)',
      color: 'var(--adam-fg-inverse)',
    },

    // Focus is a white hairline ring — never a coloured outline.
    ':focus-visible': {
      outline: '1px solid var(--adam-fg)',
      outlineOffset: '2px',
    },

    'input, textarea, select, button': {
      font: 'inherit',
      color: 'inherit',
      background: 'none',
    },
    'input::placeholder, textarea::placeholder': { color: 'var(--adam-fg-subtle)' },

    // Kill the UA autofill wash, which is always a colour.
    'input:-webkit-autofill, input:-webkit-autofill:focus': {
      WebkitTextFillColor: 'var(--adam-fg)',
      WebkitBoxShadow: '0 0 0 1000px var(--adam-surface) inset',
      caretColor: 'var(--adam-fg)',
    },

    // Ambient loops are decorative, never informational.
    '@media (prefers-reduced-motion: reduce)': {
      '*, *::before, *::after': {
        animationDuration: '0.001ms !important',
        animationIterationCount: '1 !important',
        transitionDuration: '0.001ms !important',
        scrollBehavior: 'auto !important',
      },
    },
  });

  addUtilities(UTILITIES);
});

/**
 * Declared after the plugin call intentionally — the handler runs when Tailwind
 * invokes it, well after module evaluation, so the reference above resolves.
 */
const UTILITIES = {
  // Safe-area helpers, matching the Stitch export's .pt-safe / .pb-safe.
  '.pt-safe': { paddingTop: 'env(safe-area-inset-top, 0px)' },
  '.pb-safe': { paddingBottom: 'env(safe-area-inset-bottom, 0px)' },
  '.mt-safe': { marginTop: 'env(safe-area-inset-top, 0px)' },
  '.mb-safe': { marginBottom: 'env(safe-area-inset-bottom, 0px)' },
  '.h-safe-top': { height: 'env(safe-area-inset-top, 0px)' },

  /**
   * Digital Skin — the signature dot-matrix texture. Apply to an absolutely
   * positioned overlay so its opacity never touches foreground content.
   */
  '.digital-skin': {
    backgroundImage: 'radial-gradient(var(--adam-texture-dot) 1px, transparent 0)',
    backgroundSize: `${texture.dotSize} ${texture.dotSize}`,
    opacity: 'var(--adam-texture-opacity)',
  },
  '.digital-skin-coarse': {
    backgroundImage: 'radial-gradient(var(--adam-texture-dot) 1px, transparent 0)',
    backgroundSize: `${texture.dotSizeCoarse} ${texture.dotSizeCoarse}`,
    opacity: 'var(--adam-texture-opacity)',
  },
  // Diagonal crosshatch variant — Founder Edition reveal only.
  '.digital-skin-hatch': {
    backgroundImage: `repeating-linear-gradient(45deg, var(--adam-texture-dot) 0px, var(--adam-texture-dot) 1px, transparent 1px, transparent ${texture.hatchSize})`,
    opacity: String(texture.opacity.strong),
  },

  // Frosted chrome used by the app bar and tab bar.
  '.chrome-blur': {
    backgroundColor: 'var(--adam-chrome-bg)',
    backdropFilter: 'blur(24px) saturate(100%)',
    WebkitBackdropFilter: 'blur(24px) saturate(100%)',
  },

  '.shadow-soft': { boxShadow: 'var(--adam-shadow-soft)' },
  '.bloom': { boxShadow: 'var(--adam-bloom)' },

  '.hairline-t': { borderTop: '1px solid var(--adam-border)' },
  '.hairline-b': { borderBottom: '1px solid var(--adam-border)' },
};
