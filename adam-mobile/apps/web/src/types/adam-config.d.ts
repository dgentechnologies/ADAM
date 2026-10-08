/**
 * @adam/config ships plain JS (it is consumed by Tailwind's config loader, not
 * by the app bundle), so its shape is declared here for the type checker.
 */
declare module '@adam/config/tailwind-preset' {
  import type { Config } from 'tailwindcss';
  const preset: Partial<Config>;
  export default preset;
}

declare module '@adam/config/tokens' {
  /**
   * Keys are enumerated rather than widened to `Record<string, string>`: under
   * `noUncheckedIndexedAccess` an index signature makes every lookup
   * `string | undefined`, which would force non-null assertions at every use.
   */
  export type PaletteToken =
    | 'black'
    | 'near-black'
    | 'charcoal'
    | 'charcoal-raised'
    | 'grey-mid'
    | 'grey'
    | 'grey-light'
    | 'grey-lighter'
    | 'off-white'
    | 'white';

  export type SemanticToken =
    | 'background'
    | 'surface'
    | 'surface-raised'
    | 'surface-pressed'
    | 'border'
    | 'border-strong'
    | 'fg'
    | 'fg-muted'
    | 'fg-subtle'
    | 'fg-faint'
    | 'fg-inverse';

  export const palette: { [K in PaletteToken]: string };
  export const semantic: { [K in SemanticToken]: string };
  export const STITCH_GREY_MAP: Record<string, string | null>;
  export const fontFamily: Record<string, string[]>;
  export const fontSize: Record<string, unknown>;
  export const spacing: Record<string, string>;
  export const borderRadius: Record<string, string>;
  export const boxShadow: Record<string, string>;
  export const texture: {
    dotSize: string;
    dotSizeCoarse: string;
    hatchSize: string;
    opacity: { faint: number; base: number; strong: number };
  };
  export const motion: {
    duration: Record<string, string>;
    ease: Record<string, string>;
  };
}
