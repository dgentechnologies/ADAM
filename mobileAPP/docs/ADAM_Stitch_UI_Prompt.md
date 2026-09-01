# ADAM Companion App — Google Stitch UI Prompt
**Use this as the design brief inside Stitch (stitch.withgoogle.com). Select "App" mode, paste the Master Style Prompt first to lock the design system, then generate screens one at a time using the individual screen prompts below, in order.**

---

## MASTER STYLE PROMPT (paste this first, every time, to keep every screen consistent)

```
Design a modern, minimalist, premium mobile app UI for "ADAM" — an AI desk
companion robot by DGEN Technologies. This app is the companion/setup app
for the physical robot.

STRICT COLOR RULE: Use ONLY black and white — no other hues, no accent
colors, no colored icons, no colored buttons. Build all depth and hierarchy
using shades and tints between pure black (#000000) and pure white
(#FFFFFF): true black, near-black (#0A0A0A, #121212), dark grey (#1C1C1E,
#2C2C2E), mid grey (#3A3A3C, #555555), light grey (#8E8E93, #C7C7CC),
off-white (#F2F2F7), pure white. Never introduce blue, green, red, or any
saturated color anywhere — not in charts, not in status dots, not in
toggles. Status/success/error states are communicated through shape,
icon, motion, and grey-scale contrast alone, never through color.

VISUAL STYLE: Apple-like design language — generous negative space, large
confident typography, soft depth via subtle shadows and translucency
(frosted-glass / blurred dark panels) rather than color, restrained
rounded corners (18-28px radius on cards, full-round on pills/buttons),
edge-to-edge full-bleed dark backgrounds, SF Pro / Inter-style geometric
sans-serif typography with tight letter-spacing on large headlines.

SIGNATURE TEXTURE MOTIF (use subtly, never overpowering): a fine ASCII /
dot-matrix texture pattern — think a grid of tiny monospace characters or
a halftone dot-screen, rendered in very low-opacity white-on-black (or
black-on-white on light screens) — used as a background texture behind
hero sections, empty states, and the ADAM face/avatar area. It should
feel like the visual signature of a robot's "digital skin" — subtle
noise/grain, not busy. Think: a dot-matrix printer aesthetic crossed with
a NASA mission-control screen, extremely restrained and elegant, never
distracting from content on top of it.

ADAM'S FACE / BRAND MARK: ADAM is represented by a minimal geometric face
— two simple rounded-rectangle "eyes" on a black circular/rounded-square
canvas, similar to the TFT face already used on the physical robot.
Render this mark in pure white linework on black, or black on white,
never in color. Use it as the loading/splash animation motif and as a
recurring small icon (status indicator) throughout the app.

TYPOGRAPHY: Large, bold, confident headlines (36-48px equivalent) in
pure white on black screens / pure black on white screens. Body text in
mid-grey for secondary information, high-contrast white/black reserved
for primary content and calls-to-action. Avoid all-caps except for tiny
label/eyebrow text above headlines (e.g. "STEP 2 OF 6").

COMPONENTS:
- Buttons: full-width or pill-shaped, solid white fill with black text
  (primary, on dark backgrounds) or solid black fill with white text
  (primary, on light backgrounds); secondary buttons are outlined,
  1px hairline stroke, transparent fill, white/black text.
- Cards: dark charcoal (#1C1C1E) panels with soft 1px lighter-grey
  hairline border, subtle inner glow/shadow, generous internal padding.
- Toggles/switches: monochrome — on-state is solid white knob on dark
  grey track, off-state is grey knob on darker track. No green/colored
  "on" states anywhere.
- Progress indicators (setup flow): thin horizontal line/dot steppers in
  white against dark grey, or a minimal numbered "Step X of Y" label —
  never a colored progress bar.
- Icons: thin-line (1.5px stroke) monochrome icon set, consistent
  weight throughout, no filled colored icon variants.
- Status dots (online/offline/connected): use filled white dot =
  active/connected, hollow/outline grey dot = inactive/offline — shape
  and fill state carries meaning instead of color.

OVERALL MOOD: feels like unboxing a premium Apple product crossed with a
sci-fi command console — quiet confidence, no clutter, no gradients
except very subtle black-to-near-black vignettes, no drop-shadows except
extremely soft ones for depth on cards, no skeuomorphism. Dark mode is
the default and primary experience; light mode (pure white background,
pure black text/icons) should be treated as a fully supported inverse of
the same system, not an afterthought.
```

---

## SCREEN-BY-SCREEN PROMPTS
*(Generate in this order. Reference "the ADAM app design system already established" in each prompt so Stitch stays consistent.)*

### 1. Splash Screen
```
Using the ADAM app design system: full-bleed pure black background with
the subtle ASCII/dot-matrix texture very faint in the corners. Center
screen: ADAM's minimal geometric face mark (two rounded-rectangle eyes)
in pure white linework, mid-blink animation implied by a subtle motion
blur or slightly asymmetric eye state to suggest it's "waking up." Below
the mark, small centered wordmark "ADAM" in bold white, with a tiny grey
subtitle "by DGEN Technologies" underneath in smaller light-grey text.
No buttons on this screen — it's a 2-second animated loading state.
```

### 2. Welcome Screen
```
Using the ADAM app design system: black background, ASCII dot-texture
subtly visible behind a large centered headline. Big bold white headline:
"Let's wake him up." Sub-line in grey below: "Set up your ADAM in a few
minutes." Large primary white pill button at the bottom: "Set up my
ADAM." Beneath it, a smaller ghost/text-only secondary link: "I already
have an ADAM set up." Minimal, huge whitespace, feels like an Apple
product first-unboxing screen.
```

### 3. Sign In Screen
```
Using the ADAM app design system: black background. Top third: small
ADAM face mark icon. Headline: "Who am I working for?" in bold white.
Below: one primary button styled as an outlined white pill with a
monochrome (white, not colored) Google "G" glyph and text "Continue with
Google." A secondary text link below: "Use email instead." At the very
bottom, small light-grey legal text: "By continuing you agree to DGEN's
Terms and Privacy Policy" with the two terms underlined, plus a small
unchecked/checked monochrome checkbox row.
```

### 4. Device Discovery — "Find My ADAM" Screen
```
Using the ADAM app design system: black background with a large centered
animated radar/scanning motif rendered entirely in monochrome — concentric
thin white circular rings pulsing outward from the ADAM face mark at the
center, over the faint ASCII dot texture. Headline above: "Looking for
ADAM…" Sub-text: "Make sure he's powered on and the eyes are open." Small
illustration/icon row beneath: a minimal line-art icon of the physical
ADAM robot silhouette. Bottom of screen: a subtle grey text link,
"Having trouble? Connect manually," in case auto-discovery fails.
```

### 5. Device Found / Confirm Screen
```
Using the ADAM app design system: dark card centered on a black
background, containing a small monochrome line-art icon of the ADAM
robot, the text "ADAM-3F2A found nearby" in bold white, and a smaller
grey line "Serial: DGEN-ADAM-0007 · Founder Edition." Two buttons below
the card: primary solid white pill "Yes, this is my ADAM," secondary
outlined pill "Not my device."
```

### 6. Wi-Fi Selection Screen
```
Using the ADAM app design system: black background, top headline "Get
him online." A vertically stacked list of Wi-Fi network rows inside dark
charcoal cards, each row showing a thin-line wifi-signal icon (monochrome,
no color bars), the network name in white, and a thin-line lock icon if
secured. One row is highlighted with a subtle lighter-grey hairline
border to indicate selection. Bottom: primary white pill button "Continue."
Small grey helper text under the list: "ADAM only supports 2.4GHz
networks."
```

### 7. Wi-Fi Password Entry Screen
```
Using the ADAM app design system: black background, headline "Enter
password for 'Home_5G_2.4'" in bold white. A single large minimal input
field below with a hairline underline (not a boxed input), placeholder
text "Wi-Fi password" in grey, and a small monochrome eye icon to toggle
password visibility on the right. Primary white pill button below:
"Connect." Keyboard shown at the bottom of frame in native iOS style,
all monochrome.
```

### 8. Connecting / Progress Screen
```
Using the ADAM app design system: black background, center: the ADAM
face mark with a subtle pulsing/breathing animation. Below it, a
sequential checklist rendered in monochrome: three rows, each with a
small circular status indicator (hollow grey circle = pending, filled
white circle with a thin checkmark = complete) next to text: "Sending
credentials," "ADAM connecting," "Confirming online." Minimal, calm,
centered, lots of black space around it.
```

### 9. Name Your ADAM Screen
```
Using the ADAM app design system: black background, headline "What
should we call him?" in bold white. Large centered text input styled as
a big underlined text field with placeholder "ADAM," monochrome cursor.
Below it, a subtle grey helper line: "You can change this anytime."
Primary white pill button at bottom: "Continue."
```

### 10. Founder Edition Reveal Screen (special case)
```
Using the ADAM app design system: full black background with the ASCII
dot-matrix texture more pronounced here, arranged to look like a subtle
certificate/engraving backdrop. Large bold centered text: "Founder
Edition № 007." Beneath it in grey: "You're one of the first ten." A
thin hairline-bordered card below showing small monochrome badge icons
for "Lifetime Priority Credits" and "Founder Discord Access." Primary
white pill button: "Continue." This screen should feel ceremonial and
premium, like an engraved plaque.
```

### 11. AI Brain Setup — Choice Screen
```
Using the ADAM app design system: black background, headline "Choose
ADAM's brain." Three vertically stacked selectable cards, each a dark
charcoal rounded rectangle with a hairline border:
Card 1 — bold white title "Bring Your Own Key" with a small "Recommended"
monochrome pill tag (outlined, not colored), grey description "Free.
Your own Google API key. Your data, your quota."
Card 2 — bold white title "DGEN Managed Credits," grey description "We
handle it. One-time credit packs from ₹599."
Card 3 — bold white title "Skip for now (Lite Mode)," grey description
"Clock, alarms, smart home — no live AI conversation yet."
Each card has a thin-line chevron arrow on the right indicating
tappability. No color differentiation between cards — only typography
weight and the tag on Card 1 differentiate "recommended."
```

### 12. BYOK — API Key Entry Screen
```
Using the ADAM app design system: black background, headline "Connect
your key." Numbered step list (monochrome circular numeral badges: 1, 2,
3) each with a short instruction line: "1. Tap below to create a free
key," "2. Copy it," "3. Paste it here." Below the steps, an outlined pill
button "Open Google AI Studio" with a small external-link glyph. Beneath
that, a text input field styled as a monospace-font box (suggesting a
code/key field) with placeholder "Paste your API key," and a small
"Paste" button aligned to its right, monochrome. Primary white pill
button at the bottom: "Connect ADAM."
```

### 13. Managed Credits — Pack Selection Screen
```
Using the ADAM app design system: black background, headline "Choose a
credit pack." A horizontally scrollable or vertically stacked set of
pricing cards (dark charcoal, hairline border), each showing a bold
white price ("₹599," "₹1,499," "₹2,999," "₹5,499," "₹11,999"), a pack
name in grey ("Trial," "Starter," "Standard," "Value," "Pro"), and
estimated active-minutes in smaller grey text. The "Standard" card has a
subtle outlined "Most Popular" tag (monochrome outline only). Primary
white pill button fixed at the bottom: "Continue to Payment."
```

### 14. Permission Priming — Camera/Face Screen
```
Using the ADAM app design system: black background with the ASCII dot
texture forming a soft circular vignette behind a large centered
monochrome line-art icon of an eye/camera lens. Headline: "Let ADAM see
you." Grey body text below: "He'll recognize your face and react to
your expressions. Everything is processed on-device." Primary white pill
button: "Let ADAM meet you." Secondary grey text link: "Not now."
```

### 15. Face Capture / Confirmation Screen
```
Using the ADAM app design system: black background, top area shows a
circular monochrome camera viewfinder frame with a thin pulsing white
ring around it (scanning animation, no color). Below it, headline "Got
it." with a small monochrome checkmark icon. A text input below asking
"What should ADAM call you?" with placeholder "Your name." Primary white
pill button: "Save."
```

### 16. Home Dashboard Screen
```
Using the ADAM app design system: black background, top area shows a
large card with the ASCII dot-texture background, containing ADAM's
live face-mark avatar centered (currently "happy" expression rendered in
minimal white linework), a status row beneath with a small filled white
dot + text "Online" in white, and below that a horizontal row of three
compact icon-buttons (monochrome, hairline-bordered circles): mute icon,
sleep/moon icon, wake/sun icon. Beneath the hero card, a 2x2 grid of
smaller navigation cards labeled "Gallery," "Smart Home," "Memory," and
"Settings," each with a small thin-line icon and label, dark charcoal
background with hairline borders. Bottom of screen: a fixed tab bar with
5 monochrome line icons (Home, Gallery, Smart Home, Memory, Settings),
the active tab (Home) shown filled/bold white, inactive tabs in grey.
```

### 17. Gallery Screen
```
Using the ADAM app design system: black background, headline "Moments"
at the top. A clean 3-column photo grid below, images shown with subtle
rounded corners and a thin hairline border/separator, some thumbnails
showing a small monochrome "starred" outline-star badge in the corner.
A segmented filter control at the top (monochrome pill-shaped segmented
control): "All / Starred / This Week." Bottom fixed tab bar as
established.
```

### 18. Memory Screen
```
Using the ADAM app design system: black background, headline "What
ADAM remembers." Below it, a vertically stacked list of rows inside
dark charcoal cards: each row shows a small monochrome line-icon (a
person icon for "People" entries, a small tag/note icon for "Facts"),
the memory content in white text, and a thin-line trash/delete icon on
the right for each row. A segmented control at top separates "People"
and "Facts." Small grey helper text under the headline: "You can edit or
delete anything here."
```

### 19. Settings Screen
```
Using the ADAM app design system: black background, headline "Settings."
A grouped list of rows in the classic iOS settings style — dark charcoal
rounded-rectangle groups containing rows like "Account," "AI Brain,"
"Wi-Fi," "Voice & Personality," "Connected Laptops," "Software Update,"
"Notifications," "About & Support," each row with a small monochrome
line-icon on the left, label in white, and a thin-line chevron on the
right. A final isolated card at the bottom in a slightly different
(more muted grey) tone containing "Factory Reset" in place of a
destructive-red color, using bold white text with a thin warning-triangle
line-icon instead of color to convey caution.
```

### 20. Software Update Screen
```
Using the ADAM app design system: black background, headline "Software
Update." A card showing current version "ADAM OS v40.2" in white,
beneath it in grey "Up to date" or, in the update-available state, a
second visual variant of this same screen showing "New update available
— v41.0" with a bullet-point monochrome changelog list below (small
dash-marks, not colored bullets), and a primary white pill button
"Update Now." Below that, a toggle row: "Notify me about updates" with a
monochrome switch in the "on" state.
```

---

## USAGE NOTES FOR STITCH

- Paste the **Master Style Prompt** as the first message in a new Stitch project so it establishes the design system before any screen is generated.
- Generate screens **in the numbered order above** — Stitch tends to stay more visually consistent when later prompts can reference "the same design system as the previous screens."
- After each generation, if Stitch drifts toward color or gradients, add this corrective line to the next prompt: *"Remember: strictly black, white, and greyscale only — no color anywhere, including in status indicators or charts."*
- Once all screens are generated, ask Stitch to "generate a connected prototype flow" linking Splash → Welcome → Sign In → Discovery → Found → Wi-Fi List → Wi-Fi Password → Connecting → Naming → (Founder reveal if applicable) → AI Brain Choice → BYOK or Credits → Camera Permission → Face Capture → Home Dashboard, so you get a clickable end-to-end prototype rather than 20 disconnected screens.
