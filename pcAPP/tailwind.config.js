/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0a0a0a',
        surface: '#131313',
        'surface-dim': '#0e0e0e',
        'surface-container-low': '#181818',
        'surface-container': '#1f1f1f',
        'surface-container-high': '#262626',
        'surface-container-highest': '#333333',
        charcoal: '#1c1c1e',
        'near-black': '#0e0e10',
        hairline: '#2c2c2e',
        'hairline-bright': '#3a3a3c',
        'on-surface': '#e2e2e2',
        'on-surface-variant': '#a0a0a5',
        muted: '#8e8e93',
        dim: '#555555',
      },
      borderRadius: {
        'card': '24px',
        'panel': '20px',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
      boxShadow: {
        'ambient': '0 10px 30px rgba(0, 0, 0, 0.65)',
        'subtle': '0 4px 20px rgba(0, 0, 0, 0.4)',
        'glow-white': '0 0 20px rgba(255, 255, 255, 0.15)',
      },
    },
  },
  plugins: [],
}
