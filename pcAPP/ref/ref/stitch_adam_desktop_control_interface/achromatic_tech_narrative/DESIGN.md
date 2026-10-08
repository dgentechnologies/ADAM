---
name: Achromatic Tech Narrative
colors:
  surface: '#141313'
  surface-dim: '#141313'
  surface-bright: '#3a3939'
  surface-container-lowest: '#0e0e0e'
  surface-container-low: '#1c1b1b'
  surface-container: '#201f1f'
  surface-container-high: '#2a2a2a'
  surface-container-highest: '#353434'
  on-surface: '#e5e2e1'
  on-surface-variant: '#c4c7c8'
  inverse-surface: '#e5e2e1'
  inverse-on-surface: '#313030'
  outline: '#8e9192'
  outline-variant: '#444748'
  surface-tint: '#c6c6c7'
  primary: '#ffffff'
  on-primary: '#2f3131'
  primary-container: '#e2e2e2'
  on-primary-container: '#636565'
  inverse-primary: '#5d5f5f'
  secondary: '#c8c6c8'
  on-secondary: '#303032'
  secondary-container: '#474649'
  on-secondary-container: '#b6b4b7'
  tertiary: '#ffffff'
  on-tertiary: '#2f3131'
  tertiary-container: '#e2e2e2'
  on-tertiary-container: '#636565'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#e2e2e2'
  primary-fixed-dim: '#c6c6c7'
  on-primary-fixed: '#1a1c1c'
  on-primary-fixed-variant: '#454747'
  secondary-fixed: '#e4e2e4'
  secondary-fixed-dim: '#c8c6c8'
  on-secondary-fixed: '#1b1b1d'
  on-secondary-fixed-variant: '#474649'
  tertiary-fixed: '#e2e2e2'
  tertiary-fixed-dim: '#c6c6c7'
  on-tertiary-fixed: '#1a1c1c'
  on-tertiary-fixed-variant: '#454747'
  background: '#141313'
  on-background: '#e5e2e1'
  surface-variant: '#353434'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-md-mobile:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
  technical-data:
    fontFamily: Space Mono
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: 0.05em
  technical-label:
    fontFamily: Space Mono
    fontSize: 10px
    fontWeight: '700'
    lineHeight: 12px
    letterSpacing: 0.1em
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-sm:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  label-caps:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.05em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  unit: 4px
  container-margin: 24px
  sidebar-width: 280px
  gutter: 16px
  card-padding: 20px
  element-gap: 8px
---

## Brand & Style

This design system is built on a philosophy of "Exposed Utility." It draws heavy inspiration from high-end industrial design and "Nothing Technology" aesthetics, emphasizing transparency, hardware-centric layouts, and a zero-hue palette. The brand personality is technical, premium, and sophisticated, designed to feel like an interface running directly on the robot's internal hardware.

The visual style is a blend of **Minimalism** and **Technical Brutalism**. It utilizes deep blacks to create infinite depth, contrasted by razor-sharp 1px hairlines and dot-matrix textures. The emotional response should be one of precision and reliability—an "unfiltered" look into the machine's soul. Key elements include visible grid structures, technical labels in monospace, and a strict adherence to grayscale hierarchy.

## Colors

The palette is strictly achromatic (Zero Hue). It relies on luminance levels rather than color to define importance and state.

- **Foundational Black (#000000):** Used for the primary background to maximize OLED contrast and depth.
- **Layered Darks (#0A0A0A - #1C1C1E):** Used for card surfaces and sidebars to create subtle separation.
- **Structural Greys (#2C2C2E - #555555):** Reserved for 1px hairline borders, dividers, and inactive states.
- **High Contrast White (#FFFFFF):** Reserved for primary actions, headlines, and active status indicators.
- **Functional Greys (#8E8E93 - #C7C7CC):** Used for secondary text, metadata, and technical labels to ensure readability without competing with primary content.

## Typography

The typography system uses a dual-font approach to balance human readability with a technical "machine-interface" feel.

- **Primary (Inter):** A clean, geometric sans-serif used for the majority of the UI. It provides high legibility for body text and a modern feel for headlines.
- **Technical Accent (Space Mono):** A monospaced font used for data readouts, counters, and small utility labels. It mimics a command-line or dot-matrix printer output.

**Scaling & Hierarchy:** Use `display-lg` sparingly for major dashboard headers. All labels for buttons and inputs should use `label-caps` or `technical-label` to maintain the industrial aesthetic. Technical data (like counters) should always use `technical-data` to distinguish dynamic numbers from static text.

## Layout & Spacing

The layout follows a **Fixed-Fluid Hybrid** model. 

1. **Sidebar:** A fixed 280px left panel houses navigation and core status. It is separated from the main content by a 1px vertical hairline border (#2C2C2E).
2. **Main Content:** A fluid area that utilizes a strict 8px/4px grid system. 
3. **Internal Grids:** Within cards, elements should align to a visible or implied grid. Use 1px borders to "box" different sections of a card (e.g., header vs. content).

**Breakpoints:**
- **Desktop (>1024px):** Full two-panel layout with fixed sidebar.
- **Tablet (768px - 1024px):** Sidebar collapses into a narrow icon-only rail (64px) or a hidden drawer.
- **Mobile (<768px):** Single column. Navigation moves to a bottom "dock" or a top-level menu button. Card padding reduces to 16px.

## Elevation & Depth

This design system avoids traditional drop shadows in favor of **Tonal Layering** and **Hairline Outlines**.

- **Level 0 (Background):** Pure Black (#000000). Contains the "Dot-Matrix" texture (a repeating 1px white dot at 5-8% opacity).
- **Level 1 (Panels/Sidebar):** Near-Black (#0A0A0A). These are flat surfaces defined by 1px borders.
- **Level 2 (Cards/Modals):** Dark Charcoal (#1C1C1E). These surfaces use a subtle inner glow (1px top-border white at 10% opacity) to simulate a physical edge, combined with a #2C2C2E external hairline.
- **Active State:** Focus is drawn via "Luminance Elevation"—moving from grey text to pure white text, or from a hollow border to a solid white fill.

## Shapes

The shape language combines the "organic" feel of rounded corners with the "mechanical" precision of hard lines.

- **Cards:** Use `rounded-xl` (1.5rem / 24px) or `rounded-lg` (1rem / 16px) for main containers to create the signature "Nothing" aesthetic.
- **Buttons:** Primary buttons are pill-shaped (fully rounded) to contrast against the rectangular grid.
- **Inputs:** Use `rounded-lg` (16px) to match card containers.
- **Status Indicators:** Small geometric primitives (squares/circles) are used for "active/inactive" indicators.

## Components

- **Buttons:** 
  - *Primary:* Solid White (#FFFFFF) pill with Black (#000000) text.
  - *Secondary:* 1px White hairline outline pill with White text. No fill.
  - *Tertiary:* Ghost style, White text only, 10% white background on hover.
- **Cards:** Dark Charcoal (#1C1C1E) fill, 1px #2C2C2E border. Content is padded by 20px. Often features a "Technical Label" in the top-left corner.
- **Status Indicators:** 
  - *Active:* 8px solid white circle or square. 
  - *Inactive:* 8px hollow circle or square with a 1px #555555 border.
- **Input Fields:** Near-Black (#0A0A0A) background, 1px #2C2C2E border. Text is #FFFFFF. Focus state changes border to #8E8E93.
- **Lists:** Clean rows separated by 1px #1C1C1E horizontal lines. High contrast between primary list item (White) and metadata (Mid Grey).
- **Dot-Matrix Background:** A global overlay texture of repeating dots. It must remain static during scrolling to feel like a "screen filter."
- **Data Readouts:** Large monospaced numbers (Space Mono) paired with tiny uppercase labels (e.g., "CPU LOAD").