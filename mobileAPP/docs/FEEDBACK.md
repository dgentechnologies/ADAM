# ADAM App — Feedback & Polish Log

## Global / Design System

### Open
- [ ]Hydration failed because the server rendered HTML didn't match the client. As a result this tree will be regenerated on the client. This can happen if a SSR-ed Client Component used

- A server/client branch `if (typeof window !== 'undefined')`.
- Variable input such as `Date.now()` or `Math.random()` which changes each time it's called.
- Date formatting in a user's locale which doesn't match the server.
- External changing data without sending a snapshot of it along with the HTML.
- Invalid HTML tag nesting.

It can also happen if the client has a browser extension installed which messes with the HTML before React loaded.

See more info here: https://nextjs.org/docs/messages/react-hydration-error


- data-new-gr-c-s-check-loaded="14.1324.0"
- data-gr-ext-installed=""

### Fixed
-

---

## Setup Flow

### splash
### Open
- [ ]
### Fixed
-

### welcome
### Open
- [ ]
### Fixed
-

### sign-in
### Open
- [ ]
### Fixed
-

### discover (finding_adam)
### Open
- [ ]
### Fixed
-

### device-found (adam_found)
### Open
- [ ]
### Fixed
-

### wifi-select
### Open
- [ ]
### Fixed
-

### wifi-password
### Open
- [ ]
### Fixed
-

### connecting
### Open
- [ ]
### Fixed
-

### name-device
### Open
- [ ]
### Fixed
-

### founder-reveal
### Open
- [ ]
### Fixed
-

### ai-brain
### Open
- [ ]
### Fixed
-

### byok
### Open
- [ ]
### Fixed
-

### credits
### Open
- [ ]
### Fixed
-

### camera-permission
### Open
- [ ]
### Fixed
-

### face-capture
### Open
- [ ]
### Fixed
-

---

## App (post-setup)

### home / dashboard
### Open
- [ ]
### Fixed
-

### gallery
### Open
- [ ]
### Fixed
-

### memory
### Open
- [ ]
### Fixed
-

### settings (main)
### Open
- [ ]
### Fixed
-

### settings/software-update
### Open
- [ ]
### Fixed
-

### settings/laptops
### Open
- [ ]
### Fixed
-

### settings/about
### Open
- [ ]
### Fixed
-

### settings/account, ai-brain, wifi, voice (placeholders)
### Open
- [ ]
### Fixed
-

### smart-home (placeholder)
### Open
- [ ]
### Fixed
-

---

## Animations / Transitions
*(step-to-step motion, loading states, micro-interactions)*

### Open
- [ ]
### Fixed
-

---

## Copy / Wording
*(anything that reads awkwardly, inconsistently, or off-brand)*

### Open
- [ ]
### Fixed
-

---

## Decisions Made (so Claude Code doesn't re-ask)

- Face-capture success copy: keep plainer wording, do NOT restore Stitch's
  "Biometric sync complete. Your profile has been securely mapped." line.
- Software update version numbers: v40.2.1 = installed, v41.0 = available
  (approved as placeholder, not final).
-

---
