# ADAM PC Desktop Companion App

Lightweight desktop companion app for ADAM (Windows / macOS / Linux) providing local control (volume, brightness, screen lock, media controls) and real-time status telemetry over the LAN.

## Design System
Implements the **Achromatic Intelligence** design language defined in `DESIGN.md`:
- True Black (`#000000`/`#0A0A0A`), Charcoal (`#1C1C1E`), Pure White accents, 1px hairlines (`#2C2C2E`), 24px corner radii, and digital skin dot-matrix background.

## Running Locally

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the interactive preview server:
   ```bash
   npm run dev
   ```
   Open `http://localhost:5174` in your browser.

3. Build for production:
   ```bash
   npm run build
   ```
