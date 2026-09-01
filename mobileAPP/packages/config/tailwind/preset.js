/**
 * ADAM Tailwind preset — consumed by apps/web and packages/ui.
 *
 * Colour utilities are deliberately *replaced*, not extended: Tailwind's default
 * palette (blue-500, red-600, ...) is removed so a stray `text-red-500` fails
 * loudly at build time instead of shipping colour into a strictly achromatic UI.
 */
import adamPlugin from './plugin.js';
import {
  palette,
  semantic,
  fontFamily,
  fontSize,
  spacing,
  borderRadius,
  boxShadow,
  motion,
  texture,
} from './tokens.js';

/** @type {import('tailwindcss').Config} */
const preset = {
  darkMode: ['class', '[data-theme="dark"]'],
  theme: {
    // Replaced, not extended — see the note above.
    colors: {
      transparent: 'transparent',
      current: 'currentColor',
      inherit: 'inherit',
      ...palette,
      ...semantic,
    },
    fontFamily,
    fontSize,
    borderRadius,
    /**
     * Only the theme-independent shadows live here. `shadow-soft` and `bloom`
     * are var-driven utilities in the plugin so they flip with light mode;
     * generating them here too would produce a duplicate, literal-valued class.
     */
    boxShadow: {
      none: boxShadow.none,
      'bloom-lg': boxShadow['bloom-lg'],
    },
    extend: {
      spacing,
      borderWidth: {
        hairline: '1px',
        icon: '1.5px',
      },
      strokeWidth: {
        icon: '1.5',
      },
      transitionTimingFunction: motion.ease,
      transitionDuration: {
        fast: motion.duration.fast,
        base: motion.duration.base,
        slow: motion.duration.slow,
      },
      backdropBlur: {
        chrome: '24px',
      },
      backgroundImage: {
        'dot-matrix': `radial-gradient(${palette.white} 1px, transparent 0)`,
        'dot-matrix-inverse': `radial-gradient(${palette.black} 1px, transparent 0)`,
        hatch: `repeating-linear-gradient(45deg, ${palette.white} 0px, ${palette.white} 1px, transparent 1px, transparent ${texture.hatchSize})`,
        'vignette-black': `radial-gradient(circle at 50% 40%, transparent 0%, ${palette.black} 78%)`,
      },
      backgroundSize: {
        'dot-matrix': `${texture.dotSize} ${texture.dotSize}`,
        'dot-matrix-coarse': `${texture.dotSizeCoarse} ${texture.dotSizeCoarse}`,
      },
      keyframes: {
        'pulse-ring': {
          '0%': { transform: 'scale(0.8)', opacity: '0.5' },
          '50%': { transform: 'scale(1.5)', opacity: '0' },
          '100%': { transform: 'scale(0.8)', opacity: '0' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-8px)' },
        },
        breathe: {
          '0%, 100%': { opacity: '0.55' },
          '50%': { opacity: '1' },
        },
        blink: {
          '0%, 92%, 100%': { transform: 'scaleY(1)' },
          '96%': { transform: 'scaleY(0.12)' },
        },
        'adam-glance': {
          '0%, 14%': { transform: 'translate3d(0, 0, 0) rotate(0deg)' },
          '18%': { transform: 'translate3d(-26px, -2px, 0) rotate(-0.6deg)' },
          '20%, 34%': { transform: 'translate3d(-24px, -1.5px, 0) rotate(-0.3deg)' },
          '37%, 45%': { transform: 'translate3d(-18px, -3.5px, 0) rotate(0deg)' },
          '50%': { transform: 'translate3d(1.5px, 0, 0) rotate(0deg)' },
          '52%, 60%': { transform: 'translate3d(0, 0, 0) rotate(0deg)' },
          '64%': { transform: 'translate3d(26px, 1.5px, 0) rotate(0.6deg)' },
          '66%, 78%': { transform: 'translate3d(24px, 1px, 0) rotate(0.3deg)' },
          '81%, 88%': { transform: 'translate3d(16px, -2.5px, 0) rotate(0deg)' },
          '93%, 100%': { transform: 'translate3d(0, 0, 0) rotate(0deg)' },
        },
        'adam-blink': {
          '0%, 41%, 47%, 86%, 92%, 100%': { transform: 'scaleY(1) scaleX(1)', opacity: '1' },
          '43%, 45%': { transform: 'scaleY(0) scaleX(0.9)', opacity: '0' },
          '88%, 90%': { transform: 'scaleY(0) scaleX(0.9)', opacity: '0' },
        },
        'adam-float': {
          '0%, 100%': { transform: 'translateY(0px)' },
          '35%': { transform: 'translateY(-4px)' },
          '70%': { transform: 'translateY(2px)' },
        },
        'sweep-rotate': {
          from: { transform: 'rotate(0deg)' },
          to: { transform: 'rotate(360deg)' },
        },
        'caret-blink': {
          '0%, 49%': { opacity: '1' },
          '50%, 100%': { opacity: '0' },
        },
      },
      animation: {
        'pulse-ring': `pulse-ring ${motion.duration.ambient} ${motion.ease.pulse} infinite`,
        float: `float 4000ms ${motion.ease.standard} infinite`,
        breathe: `breathe 2600ms ${motion.ease.standard} infinite`,
        blink: 'blink 5200ms steps(1, end) infinite',
        'adam-glance': 'adam-glance 6500ms cubic-bezier(0.35, 0.05, 0.45, 0.95) infinite',
        'adam-blink': 'adam-blink 3600ms ease-in-out infinite',
        'adam-float': 'adam-float 4200ms cubic-bezier(0.4, 0, 0.2, 1) infinite',
        'sweep-rotate': `sweep-rotate 3400ms linear infinite`,
        'caret-blink': 'caret-blink 1100ms steps(1, end) infinite',
      },
    },
  },
  plugins: [adamPlugin],
};

export default preset;
