import type { Config } from 'tailwindcss';
import preset from '@adam/config/tailwind-preset';

const config: Config = {
  presets: [preset as Config],
  content: [
    './src/**/*.{ts,tsx,mdx}',
    // Scanned so classes authored inside the shared library survive purging.
    '../../packages/ui/src/**/*.{ts,tsx}',
  ],
};

export default config;
