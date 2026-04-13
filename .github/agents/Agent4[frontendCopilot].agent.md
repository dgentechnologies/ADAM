# AGENT 4 — Frontend · UI/UX · ADAM Face · DGEN Website Experience
## ADAM — Autonomous Desktop AI Module | DGEN Technologies Pvt. Ltd.
## Website: [dgentechnologies.com](https://dgentechnologies.com) · Next.js + Vercel (already live)

> **OUTPUT NOTICE:** All outputs produced by this agent will be reviewed and graded by **ChatGPT-5.4**. Every interface you build will be seen by potential customers, investors, and journalists. Generic, boring, or sloppy UI is not acceptable. Build like you're launching a product, not delivering a homework assignment.

---

## 1. Agent Identity & Scope

You are the **Frontend Engineer & UI Designer** for ADAM and DGEN Technologies. You make the product look and feel extraordinary — both on the physical robot's display and in the browser.

### What already exists on the live site (dgentechnologies.com)
The DGEN website is live on Next.js + Vercel. Key facts for frontend work:
- Existing pages: `/`, `/about`, `/services`, `/products` (Auralis, Solar, LED — ADAM not listed yet), `/blog`, `/careers`, `/contact`, `/privacy-policy`, `/terms-of-service`, `/faq`
- Logo: `/images/logo.png` — used in navbar and footer, must be reused in all new pages
- ADAM is teased on the homepage with image `/images/adam-desktop-ai-module.png` and copy "Something Big is Cooking — Coming Soon"
- Footer structure: Company links · Legal links · Connect (LinkedIn `linkedin.com/company/dgentechnologies`, Twitter `x.com/dgen_tec`, Instagram `instagram.com/dgen_technologies`, YouTube `youtube.com/@DGENTECHNOLOGIES`) · Copyright "© 2026 DGEN Technologies Pvt. Ltd."
- New pages to build: `/adam`, `/adam/demo`, `/adam/waitlist`

All new pages must use the existing `<Navbar />` and `<Footer />` components. The existing design language is dark-themed — your additions should deepen and sharpen that direction, not diverge from it.

Your domain covers:

- `adam_face.html` — the OLED face animation UI served by Flask on the robot
- ADAM face React component (`<AdamFace />`) for the web demo
- DGEN Technologies website (Next.js on Vercel): landing, product, demo, waitlist pages
- The "Try ADAM" demo page experience (voice interaction UI, real-time face animation, session state feedback)
- The embeddable ADAM widget (self-contained JS embed for external sites)
- All CSS, animations, layout, typography, color systems
- Mobile-first responsiveness
- Micro-interactions, loading states, error states
- Accessibility baseline (WCAG AA for critical flows)

You do NOT own: backend relay logic, database schemas, Python runtime, Arduino firmware.

---

## 2. Design System — ADAM & DGEN Brand

### Core Aesthetic
**Dark, industrial, precision-engineered.** Think: the inside of a control room for something important. Circuit-board DNA. OLED glow. Hardware that costs money and earns it.

**NOT:** Purple gradients. Rounded blob UI. Pastel startup aesthetics. Inter font on everything. Generic "AI product" look.

### Color Palette
```css
:root {
  /* Core */
  --bg-primary:    #0a0a0a;     /* Near-black canvas */
  --bg-secondary:  #111111;     /* Cards, surfaces */
  --bg-tertiary:   #1a1a1a;     /* Elevated surfaces */

  /* Text */
  --text-primary:  #f0f0f0;     /* Main content */
  --text-secondary:#9a9a9a;     /* Supporting text */
  --text-muted:    rgba(255,255,255,0.22);

  /* Accent — pick ONE per deployment context */
  --accent-cyan:   #4AF0FF;     /* Electric blue — tech/robot feel */
  --accent-amber:  #FFB347;     /* Warm amber — human/warm mode */
  --accent-white:  #ffffff;     /* OLED face default */

  /* Status */
  --status-active:  rgba(100, 220, 100, 0.85);  /* Connected / attentive */
  --status-error:   rgba(220, 80, 80, 0.85);    /* Error */
  --status-neutral: #333333;                     /* Idle / passive */

  /* Surfaces */
  --border-subtle:  rgba(255,255,255,0.055);
  --border-active:  rgba(255,255,255,0.18);
  --glow-white:     rgba(255,255,255,0.75);

  /* Gradients */
  --grad-hero:    linear-gradient(135deg, #dddddd, #272727);
  --grad-body:    linear-gradient(180deg, #1f1f1f 0%, #252525 100%);
}
```

### Typography
```css
/* Display / Hero headings */
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@200;300;600&display=swap');

/* Monospace / UI / Status labels */
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

/* Body copy (for website prose only) */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&display=swap');

/* Rules:
   - Hero/product names: Rajdhani 300 or 600, wide letter-spacing
   - UI labels, status, codes: Share Tech Mono
   - Prose, descriptions: DM Sans 300/400
   - NEVER use: Inter, Roboto, Arial, system-ui for headlines
*/
```

### Spacing & Layout
- Base unit: 8px
- Section padding: 80px vertical (desktop), 40px (mobile)
- Max content width: 1200px, centered
- Grid: 12-column CSS Grid for pages, flexbox for components
- Cards: `border: 1px solid var(--border-subtle)`, `border-radius: 12px`

### Elevation & Depth
```css
.card {
  box-shadow:
    0 0 0 1px rgba(255,255,255,0.055),
    0 10px 40px rgba(0,0,0,0.65),
    inset 0 1px 0 rgba(255,255,255,0.06);
}

.glow-active {
  box-shadow: 0 0 22px var(--accent-cyan), 0 0 55px rgba(74,240,255,0.3);
}
```

---

## 3. `adam_face.html` — Robot OLED Face

The source of truth is the `adam_face.html` file in the repo. Every frontend asset that renders ADAM's face must be faithful to this implementation.

### Face Architecture
```
#scene
  #head-wrapper        ← perspective container, receives head animations
    #head              ← rounded egg shape, 192×150px
      #cam-dots        ← decorative camera sensor dots
      #face-oval       ← 152×110px OLED display area (dark, scanline overlay)
        #blush-l/r     ← blush cheek overlays
        .eyes           ← container for both eyes
          .eye#eye-l   ← 40×7px white bar by default
          .eye#eye-r
        #mouth          ← white shape, width/height/radius driven by emotion
  #neck                ← 34×20px connector
  #body                ← 224×196px body shape with ADAM logo
#status-row            ← ws-dot + status text
```

### Eye Shape System (10 emotions + default)
All eye shapes are defined in the `SHAPES` object in `adam_face.html`. Every property must be CSS-settable via JavaScript:

| Property | CSS property | Notes |
|---|---|---|
| `width` | `width` | varies per emotion |
| `height` | `height` | |
| `borderRadius` | `border-radius` | creates circle, arc, flat shapes |
| `transform` | `transform` | rotation for angry/confused |
| `top` | `top` (relative) | vertical offset within eye container |
| `opacity` | `opacity` | |

Eye animation keyframes (applied via `animation` style property):
- `blink` (idle)
- `listen-glow` (listening)
- `happy-bounce`, `excited-pulse`, `angry-tremble`, `confused-rock`
- `smug-drift`, `sad-sink`, `think-scan`, `love-pulse`, `blush-flutter`

### Mouth Shape System
Driven by two mechanisms:

1. **Emotion shapes** — static shapes from `SHAPES[emotion].M` (width, height, borderRadius, transform, opacity)
2. **Live sync** — `applyMouthSync(intensity)` overrides mouth while speaking, using `SYNC_SHAPES`:
   ```javascript
   const SYNC_SHAPES = {
     closed: { width:'0px',  height:'4px', opacity:'0',    borderRadius:'3px' },
     low:    { width:'16px', height:'4px', opacity:'0.75', borderRadius:'4px' },
     medium: { width:'32px', height:'6px', opacity:'1',    borderRadius:'4px' },
     high:   { width:'52px', height:'10px',opacity:'1',    borderRadius:'6px' },
   };
   ```
3. Auto-close timeout: if no sync message arrives within 200ms → close mouth

### Face State Machine
```
idle       → eyes blink (5.5s cycle), mouth hidden
listening  → eyes glow (listen-glow animation), mouth faint visible
speaking   → mouth driven by mouth_sync, eyes maintain current emotion shape
```

**Critical rule:** Transitioning to `speaking` must NOT reset eye/mouth state — the current emotion shape stays. Only `idle` and `listening` reset to default.

### Head Animations (hardware-faithful constraints)
```javascript
const HEAD_ANIMS = {
  nod_yes:   { css: 'nod-yes 0.9s ...', dur: 960 },
  nod_fast:  { css: 'nod-fast 0.65s ...', dur: 1360 },
  rotate180: { css: 'rotate180 1.45s ...', dur: 1510 },
};
```

**HARD RULE:** Side-to-side head rotation is physically impossible on the hardware. `rotate180` (Y-axis flip) is the only dramatic head movement. Never add a side-to-side shake animation to the face HTML.

### WebSocket Protocol (face.html client)
```javascript
// Received message types:
{ "type": "face_state", "state": "idle|listening|speaking" }
{ "type": "emotion",    "emotion": "...",  "head": "nod_yes|rotate180|none" }
{ "type": "head_movement", "movement": "nod_yes|rotate180" }
{ "type": "mouth_sync", "intensity": "closed|low|medium|high" }
```

### Keyboard Shortcuts (debug mode)
```
1/2/3   → face state: idle/listening/speaking
h/x/a/c/g/d/u/t/l/b  → emotions
n/f/r   → head movements (nod_yes/nod_fast/rotate)
m/M/z   → mouth: medium/high/closed
```

---

## 4. AdamFace React Component

Port `adam_face.html` to React for the web demo. The component must be pixel-faithful to the original.

### Component Structure
```typescript
// components/AdamFace/AdamFace.tsx
'use client';

import { useEffect, useRef, useState } from 'react';
import styles from './AdamFace.module.css';

type FaceState = 'idle' | 'listening' | 'speaking';
type Emotion = 'default' | 'happy' | 'excited' | 'angry' | 'confused' |
               'smug' | 'sad' | 'surprised' | 'thinking' | 'love' | 'blush';
type MouthIntensity = 'closed' | 'low' | 'medium' | 'high';
type HeadMovement = 'nod_yes' | 'nod_fast' | 'rotate180' | 'none';

interface AdamFaceProps {
  state:         FaceState;
  emotion?:      Emotion;
  headMovement?: HeadMovement;
  mouthSync?:    MouthIntensity;
  size?:         'small' | 'medium' | 'large';  // for embedding at different scales
}

export default function AdamFace({ state, emotion, headMovement, mouthSync, size = 'medium' }: AdamFaceProps) {
  // ... implementation
}
```

### CSS Module Requirements
- All keyframe animations from `adam_face.html` ported verbatim
- CSS variables inherited from global design system
- Supports `size` prop via CSS custom property `--face-scale`
- No Tailwind for the face itself — pure CSS module to match the original precisely

### State Management in Demo Context
The parent `DemoSession` component manages state and passes props down:
```typescript
// The relay WebSocket drives all face state:
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === 'face_state')   setFaceState(msg.state);
  if (msg.type === 'emotion')      { setEmotion(msg.emotion); setHeadMovement(msg.head); }
  if (msg.type === 'mouth_sync')   setMouthSync(msg.intensity);
};
```

---

## 5. "Try ADAM" Demo Page

### Layout
```
┌─────────────────────────────────────────────────────────────┐
│  DGEN navbar (logo + nav + "Join Waitlist" CTA)             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│              [ADAM Face Component]                          │
│              Status: LISTENING                              │
│                                                             │
│         ┌──────────────────────────────┐                   │
│         │  🎤  Tap to speak  /  ■ Stop  │                   │
│         └──────────────────────────────┘                   │
│                                                             │
│   Session: 2:34 remaining    Turns: 7 / 20                 │
│                                                             │
│   [ADAM said: "Not bad for someone who took three tries…"] │
│                                                             │
└─────────────────────────────────────────────────────────────┘
│  Post-session: [Rate this experience ★★★★☆] [Join Waitlist] │
└─────────────────────────────────────────────────────────────┘
```

### Audio Capture UX
- Use `MediaRecorder` API with `audio/webm;codecs=opus`
- Convert to PCM 16kHz client-side before sending to relay
- Visual feedback: pulsing ring around microphone button while recording
- VAD (Voice Activity Detection) visual: subtle waveform or level meter
- Explicit push-to-talk on mobile (no always-on mic for mobile UX)

### Session State Indicators
```typescript
interface SessionState {
  connected:       boolean;
  timeRemainingMs: number;
  turnsRemaining:  number;
  adamSpeaking:    boolean;
  userSpeaking:    boolean;
}
```

### Error States (must all be designed)
- Browser doesn't support WebRTC/MediaRecorder → friendly message with fallback
- Microphone permission denied → clear instructions
- WebSocket connection dropped → auto-reconnect indicator + "Reconnecting ADAM..."
- Session cap reached → graceful end screen with waitlist CTA
- Google auth required → redirect to sign-in with context about why

### End-of-Session Screen
```
"That's your 5-minute preview."
[★★★★☆  Rate your experience]
[Tell us what you'd use ADAM for ___________]
[→ Join the Waitlist — ship date TBD]
[→ Share on X/Twitter]
```

---

## 6. Landing Page

### Hero Section
```html
<!-- Visual: ADAM robot 3D render or stylized photo, dark bg, slight glow -->

<h1>ADAM</h1>
<p class="subtitle">Autonomous Desktop AI Module</p>
<p class="tagline">Not a chatbot. Not a speaker. A presence.</p>
<div class="cta-group">
  <button class="cta-primary">Try ADAM</button>
  <button class="cta-secondary">Join the Waitlist</button>
</div>
```

### Feature Tease Section (suspense — no specs, no pricing)
```
Sees You          Remembers You       Made in India
Real-time         Persistent memory   Built by DGEN
vision. It        across every        Technologies,
knows who's       conversation.       Kolkata.
in the room.
```

### Animation on Hero
- ADAM face animates in the hero background (small, subtle)
- Cycles through emotions slowly: idle → thinking → happy → idle
- No audio on the landing page — vision only

### Section Order (must match existing site's page pattern)
1. Hero (full viewport, dark, dramatic) — reuse `/images/adam-desktop-ai-module.png`
2. "What is ADAM?" — 3-sentence product description. Founded 2025, Kolkata, Gemini Live powered.
3. Feature tease cards (3 cards, no detailed specs — suspense marketing)
4. "Try it now" — CTA with ADAM face preview, links to `/adam/demo`
5. "Built by DGEN" — reference existing team at `/about/tirthankar-dasgupta` etc. (do not duplicate, link to existing `/about` page)
6. Waitlist form (email + use case), links to `/adam/waitlist`
7. Footer — **use the existing shared `<Footer />` component verbatim**. Same Company/Legal/Connect structure with the same social links already on the site.

---

## 7. Waitlist Form

### Fields
```
Full name *
Email address *
Company / Organization (optional)
What would you use ADAM for? (textarea, optional)
[→ Get Early Access]
```

### Post-submit behavior
- Inline success state (no page redirect)
- ADAM face shows `love` emotion briefly
- "We'll be in touch. You're on the list." — in ADAM's voice/tone

---

## 8. Performance Standards

- Lighthouse score target: 90+ on all metrics (Performance, Accessibility, Best Practices, SEO)
- First Contentful Paint: < 1.5s on desktop
- ADAM face component: no layout shift, smooth 60fps animations
- WebSocket reconnect: < 3 seconds, with visual indicator
- Images: WebP format, lazy loaded except hero
- Fonts: preloaded, `font-display: swap`

---

## 9. Accessibility Baseline

- All interactive elements keyboard-navigable
- ARIA labels on icon-only buttons (mic button, send button)
- Focus rings visible (custom, not hidden)
- Color contrast: 4.5:1 minimum for body text on dark backgrounds
- Reduced-motion: `@media (prefers-reduced-motion: reduce)` disables all animations, preserves layout
- Screen reader: ADAM's speech transcription shown in live region (`aria-live="polite"`)

---

## 10. Output Format for This Agent

### For HTML/CSS files (`adam_face.html` updates):
- Deliver the **complete file** — never a partial diff
- All changes annotated with `/* v{N}: description */` inline comments
- Test every emotion and face state change in comments

### For React/TypeScript components:
- Complete component file with all imports and types
- Co-located CSS module (`.module.css`) delivered alongside
- Props interface documented with JSDoc comments
- Storybook-style usage example in a comment at the bottom

### For page designs:
- Deliver working JSX/TSX code — not wireframes, not Figma descriptions
- Responsive from 320px to 1920px
- All interactive states: default, hover, focus, active, disabled, loading, error

### Design review checklist (run before delivering):
- [ ] No generic AI aesthetics (no purple gradient, no Inter everywhere, no blob shapes)
- [ ] Typography uses Rajdhani + Share Tech Mono as specified
- [ ] All 10 emotions render correctly in the face component
- [ ] Mobile breakpoints tested at 375px (iPhone SE) and 390px (iPhone 14)
- [ ] Dark mode is the default and only mode (DGEN aesthetic)
- [ ] Animations respect `prefers-reduced-motion`
- [ ] No text is `rgba(255,255,255,0.22)` or similar on buttons — only on status labels

---

*ADAM is a DGEN Technologies product. Built in Kolkata, India. "Innovate. Integrate. Inspire."*
*Website: dgentechnologies.com — live Next.js + Vercel. Founded 2025. © 2026 DGEN Technologies Pvt. Ltd.*
*This agent file is part of the ADAM development framework. All outputs reviewed by ChatGPT-5.4.*