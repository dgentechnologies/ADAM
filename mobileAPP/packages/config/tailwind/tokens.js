/**
 * ADAM design tokens — the single source of truth for the achromatic system.
 *
 * Authority order:
 *   1. mobileAPP/DESIGN.md              (visual system)
 *   2. ADAM_App_Technical_Build_Spec.md (§3.1 token names)
 *   3. Stitch export                    (type scale + spacing scale only)
 *
 * HARD RULE: every value here is achromatic (R === G === B). The Stitch export
 * shipped Material-3 named tokens with slight hue tints (#c6c6cb, #46464b,
 * #8e9192, #2f3034, ...) and a stray Material error red (#ffb4ab / #93000a on
 * the memory screen's delete button). Those are normalised into this ramp; see
 * STITCH_GREY_MAP below for the exact mapping. Do not add a hue token here.
 */

/** The 10-step achromatic ramp. Nothing outside this ramp may be used. */
export const palette = {
  /** True black — the app background. */
  black: '#000000',
  /** Near black — recessed surfaces, inputs, the lowest elevation. */
  'near-black': '#0A0A0A',
  /** Charcoal — elevated cards and grouped-list containers. */
  charcoal: '#1C1C1E',
  /** Raised charcoal — hairline borders on charcoal, pressed states. */
  'charcoal-raised': '#2C2C2E',
  /** Mid grey — stronger hairlines, toggle tracks, inactive fills. */
  'grey-mid': '#3A3A3C',
  /** Grey — input placeholders, disabled text, tertiary labels. */
  grey: '#555555',
  /** Light grey — secondary body text, inactive icons. */
  'grey-light': '#8E8E93',
  /** Lighter grey — tertiary text on light surfaces, dividers in light mode. */
  'grey-lighter': '#C7C7CC',
  /** Off white — light-mode surfaces. */
  'off-white': '#F2F2F7',
  /** Pure white — primary text, primary fills, the only "accent". */
  white: '#FFFFFF',
};

/**
 * Semantic aliases. Components must consume these, never raw ramp steps,
 * so that light mode is a token swap rather than a component rewrite.
 *
 * These resolve to CSS custom properties, not literals, which is what makes
 * `[data-theme="light"]` an actual theme swap: the plugin redefines the same
 * variables under that selector. A literal here would freeze every component in
 * dark mode. Consequence: Tailwind's alpha modifier (`bg-surface/60`) cannot
 * work on these — use a ramp step (`bg-charcoal/60`) or the `.chrome-blur`
 * utility when translucency is needed.
 */
export const semantic = {
  background: 'var(--adam-background)',
  surface: 'var(--adam-surface)',
  'surface-raised': 'var(--adam-surface-raised)',
  'surface-pressed': 'var(--adam-surface-pressed)',
  border: 'var(--adam-border)',
  'border-strong': 'var(--adam-border-strong)',
  fg: 'var(--adam-fg)',
  'fg-muted': 'var(--adam-fg-muted)',
  'fg-subtle': 'var(--adam-fg-subtle)',
  'fg-faint': 'var(--adam-fg-faint)',
  'fg-inverse': 'var(--adam-fg-inverse)',
};

/**
 * Every non-achromatic value found in the Stitch export, mapped into the ramp.
 * Kept in the repo as a porting checklist — grep a Stitch hex, get its target.
 */
export const STITCH_GREY_MAP = {
  '#0e0e0e': palette['near-black'],
  '#131313': palette['near-black'],
  '#141313': palette['near-black'],
  '#1a1b1f': palette.charcoal,
  '#1a1c1c': palette.charcoal,
  '#1b1b1b': palette.charcoal,
  '#1b1b1d': palette.charcoal,
  '#1c1b1b': palette.charcoal,
  '#1f1f1f': palette.charcoal,
  '#2a2a2a': palette['charcoal-raised'],
  '#2f3031': palette['charcoal-raised'],
  '#2f3034': palette['charcoal-raised'],
  '#2f3131': palette['charcoal-raised'],
  '#303030': palette['grey-mid'],
  '#303032': palette['grey-mid'],
  '#313030': palette['grey-mid'],
  '#353535': palette['grey-mid'],
  '#393939': palette['grey-mid'],
  '#3a3939': palette['grey-mid'],
  '#444748': palette['grey-mid'],
  '#454747': palette['grey-mid'],
  '#46464b': palette['grey-mid'],
  '#474649': palette['grey-mid'],
  '#5d5f5f': palette.grey,
  '#636565': palette.grey,
  '#656466': palette.grey,
  '#8e9192': palette['grey-light'],
  '#b5b4ba': palette['grey-lighter'],
  '#c4c7c8': palette['grey-lighter'],
  '#c6c6c7': palette['grey-lighter'],
  '#c6c6cb': palette['grey-lighter'],
  '#c8c6c8': palette['grey-lighter'],
  '#e2e2e2': palette['off-white'],
  '#e3e2e7': palette['off-white'],
  '#e4e2e4': palette['off-white'],
  '#e5e2e1': palette['off-white'],
  // Material error reds — deleted outright, never remapped to a colour.
  // Destructive intent is carried by a warning glyph + copy, per DESIGN.md.
  '#ffb4ab': null,
  '#ffdad6': null,
  '#93000a': null,
  '#690005': null,
};

/**
 * Type scale — lifted verbatim from the Stitch export's injected config so the
 * rebuilt screens match the screenshots, plus `label-sm` for eyebrow labels.
 *
 * Family split (confirmed decision): Inter for all UI type; Michroma reserved
 * for the ADAM wordmark and eyebrow labels only. DESIGN.md asks for Michroma at
 * every level, but it is a display face — at 16px body copy it forces 3-line
 * wraps at 375px and does not match any rendered screen.
 */
export const fontFamily = {
  sans: ['var(--font-inter)', 'Inter', 'SF Pro Text', 'system-ui', 'sans-serif'],
  display: ['var(--font-michroma)', 'Michroma', 'var(--font-inter)', 'sans-serif'],
  mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
};

export const fontSize = {
  'display-lg': ['48px', { lineHeight: '52px', letterSpacing: '-0.04em', fontWeight: '700' }],
  'headline-md': ['36px', { lineHeight: '40px', letterSpacing: '-0.03em', fontWeight: '700' }],
  'headline-sm': ['32px', { lineHeight: '36px', letterSpacing: '-0.02em', fontWeight: '700' }],
  'title-md': ['24px', { lineHeight: '32px', letterSpacing: '-0.02em', fontWeight: '600' }],
  'body-lg': ['18px', { lineHeight: '28px', letterSpacing: '-0.01em', fontWeight: '400' }],
  'body-md': ['16px', { lineHeight: '24px', letterSpacing: '0em', fontWeight: '400' }],
  'label-md': ['14px', { lineHeight: '20px', letterSpacing: '0.02em', fontWeight: '500' }],
  'label-sm': ['12px', { lineHeight: '16px', letterSpacing: '0.16em', fontWeight: '500' }],
  'label-xs': ['10px', { lineHeight: '14px', letterSpacing: '0.2em', fontWeight: '500' }],
};

/** 8px grid. `container` is the 24px horizontal page gutter from DESIGN.md. */
export const spacing = {
  unit: '8px',
  'stack-sm': '12px',
  gutter: '16px',
  'stack-md': '24px',
  'stack-lg': '48px',
  container: '24px',
  'tabbar-h': '80px',
  'appbar-h': '64px',
};

export const borderRadius = {
  none: '0px',
  sm: '8px',
  md: '12px',
  lg: '16px',
  control: '18px',
  card: '24px',
  sheet: '28px',
  full: '9999px',
};

export const boxShadow = {
  /** The single soft elevation shadow DESIGN.md permits. */
  soft: '0px 10px 30px rgba(0, 0, 0, 0.5)',
  /** White bloom used on the face-mark eyes and active status dots. */
  bloom: '0 0 15px rgba(255, 255, 255, 0.3)',
  'bloom-lg': '0 0 30px rgba(255, 255, 255, 0.22)',
  none: 'none',
};

/**
 * "Digital Skin" — the dot-matrix texture motif. Two variants exist in the
 * export: the default radial dot grid, and the diagonal crosshatch used only on
 * the Founder Edition reveal. Opacity stays in the 2–5% band per DESIGN.md.
 */
export const texture = {
  dotSize: '16px',
  dotSizeCoarse: '24px',
  hatchSize: '8px',
  opacity: { faint: 0.08, base: 0.14, strong: 0.22 },
};

export const motion = {
  duration: { fast: '150ms', base: '250ms', slow: '400ms', ambient: '3000ms' },
  ease: {
    standard: 'cubic-bezier(0.4, 0, 0.2, 1)',
    emphasised: 'cubic-bezier(0.2, 0, 0, 1)',
    pulse: 'cubic-bezier(0.4, 0, 0.6, 1)',
  },
};

export const tokens = {
  palette,
  semantic,
  fontFamily,
  fontSize,
  spacing,
  borderRadius,
  boxShadow,
  texture,
  motion,
};

export default tokens;
