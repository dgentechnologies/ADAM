---
name: Achromatic Intelligence
colors:
  surface: '#131313'
  surface-dim: '#141313'
  surface-bright: '#3a3939'
  surface-container-lowest: '#0e0e0e'
  surface-container-low: '#1c1b1b'
  surface-container: '#1F1F1F'
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
  secondary: '#c6c6cb'
  on-secondary: '#2f3034'
  secondary-container: '#45474b'
  on-secondary-container: '#b4b5ba'
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
  secondary-fixed: '#e2e2e7'
  secondary-fixed-dim: '#c6c6cb'
  on-secondary-fixed: '#1a1c1f'
  on-secondary-fixed-variant: '#45474b'
  tertiary-fixed: '#e2e2e2'
  tertiary-fixed-dim: '#c6c6c7'
  on-tertiary-fixed: '#1a1c1c'
  on-tertiary-fixed-variant: '#454747'
  background: '#141313'
  on-background: '#e5e2e1'
  surface-variant: '#353434'
  charcoal: '#1C1C1E'
  near-black: '#0A0A0A'
typography:
  display-lg:
    fontFamily: Michroma
    fontSize: 48px
    fontWeight: '400'
    lineHeight: 52px
    letterSpacing: -0.04em
  headline-md:
    fontFamily: Michroma
    fontSize: 36px
    fontWeight: '400'
    lineHeight: 40px
    letterSpacing: -0.03em
  headline-md-mobile:
    fontFamily: Michroma
    fontSize: 32px
    fontWeight: '400'
    lineHeight: 36px
    letterSpacing: -0.02em
  headline-sm:
    fontFamily: Michroma
    fontSize: 24px
    fontWeight: '400'
    lineHeight: 32px
    letterSpacing: -0.02em
  body-lg:
    fontFamily: Michroma
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
    letterSpacing: -0.01em
  body-md:
    fontFamily: Michroma
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: 0em
  label-md:
    fontFamily: Michroma
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: 0.02em
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  unit: 8px
  container-padding: 24px
  gutter: 16px
  stack-sm: 12px
  stack-md: 24px
  stack-lg: 48px
---

## Brand & Style

The design system is a premium, minimalist framework designed for the future of human-AI interaction. It leverages a strictly monochrome palette to emphasize form, function, and clarity, stripping away chromatic noise to focus on the intelligence of the companion. 

The aesthetic is a fusion of **Apple-inspired Modernism** and **Tactile Minimalism**. It utilizes expansive negative space to evoke a sense of calm and sophistication. To prevent the interface from feeling sterile, a "Digital Skin"—consisting of a subtle 2-5% opacity halftone dot-matrix pattern—is applied to backgrounds, providing a technical yet organic texture reminiscent of high-end hardware.

## Colors

The palette is strictly achromatic, relying on luminance and contrast rather than hue to establish hierarchy. 

- **Backgrounds:** The primary interface environment is "True Black" (#000000) to ensure seamless integration with OLED hardware and maximize the depth of the AI experience.
- **Surfaces:** Use "Near Black" (#0A0A0A) for standard containers and "Charcoal" (#1C1C1E) for elevated interactive panels.
- **Accents:** Pure White (#FFFFFF) is reserved exclusively for primary actions, critical status indicators, and high-priority text. 
- **Transitions:** Use mid-greys for secondary information to reduce cognitive load and create a sophisticated visual stack.

## Typography

This design system utilizes **Michroma** for all typography levels, providing a wide, futuristic, and technical aesthetic across the entire interface.

- **Headlines:** Headings leverage Michroma's geometric structure with tight negative letter-spacing to create a "confident" and "architectural" feel. 
- **Body Text:** Primary body text uses Pure White for maximum readability. Secondary body text or metadata should transition to Light Grey (#8E8E93) to recede in the hierarchy.
- **Scalability:** On mobile devices, display sizes should scale down to ensure they remain within the viewport while maintaining their distinctive horizontal presence.

## Layout & Spacing

The layout follows a strict **8px grid system** to maintain mathematical harmony.

- **Grid:** A 12-column fluid grid is used for desktop, shifting to a 4-column grid for mobile.
- **Negative Space:** Use generous vertical margins (stack-lg) to separate distinct AI modules or conversation blocks. Content should never feel crowded; let the "digital skin" texture fill the voids.
- **Margins:** Standard container padding is set to 24px, ensuring that content feels grounded but airy within its charcoal panels.

## Elevation & Depth

Depth is communicated through **translucency and tonal layering** rather than traditional heavy shadows.

- **Glassmorphism:** Use frosted-glass effects (Backdrop Blur: 20px-40px) for navigation bars and floating overlays. The background behind the blur should be a 60% opacity version of #121212.
- **Tonal Layers:** Elevation is achieved by moving from True Black (#000000) to Charcoal (#1C1C1E). 
- **Hairlines:** Interactive elements and cards are defined by 1px hairline borders in #2C2C2E (for subtle separation) or #3A3A3C (for higher definition).
- **Shadows:** Use a single, extremely soft ambient shadow (0px 10px 30px rgba(0,0,0,0.5)) only on top-level floating modals to separate them from the primary interface stack.

## Shapes

The shape language is defined by high-radius curves to soften the technical nature of the AI.

- **Cards/Panels:** A consistent 24px (rounded-xl) corner radius creates a friendly, "handheld" feel for hardware-like UI components.
- **Buttons:** All interactive buttons are pill-shaped (fully rounded) to differentiate them from static containers and cards.
- **Inputs:** Follow the card radius (24px) to maintain a cohesive structural language.

## Components

- **Buttons:** 
  - *Primary:* Pill-shaped, Solid White fill, Black text. High-contrast, maximum prominence. Uses Michroma for all labels.
  - *Secondary:* Pill-shaped, 1px Hairline stroke (#8E8E93), Transparent fill, White text.
- **Cards:** Use Charcoal (#1C1C1E) panels with a 24px radius and 1px border (#2C2C2E). Padding is strictly 24px.
- **Toggles:** Minimalist design. The track is Mid-Grey (#3A3A3C) and the knob is a Pure White circle. When "On," the track remains grey but the knob stays high-contrast white.
- **Icons:** 1.5px stroke width. Use thin-line monochrome icons that match the current text color (Primary or Secondary grey).
- **Input Fields:** Semi-transparent Near-Black (#0A0A0A) background with a 1px border. 24px corner radius. Placeholder text in Mid-Grey (#555555).
- **Status Indicators:** 
  - *Active:* Solid White circle.
  - *Inactive/Offline:* Hollow 1px White border circle.
  - *Processing:* Pulsing White-to-Grey animation on a solid circle.