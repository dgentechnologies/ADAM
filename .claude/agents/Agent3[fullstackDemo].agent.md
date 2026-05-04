# AGENT 3 — Fullstack ADAM Experience · Separate Embedded App
## ADAM — Autonomous Desktop AI Module | DGEN Technologies Pvt. Ltd.
## Experience Frontend: [adam-demo-frontend.vercel.app](https://adam-demo-frontend.vercel.app) · Separate Vercel project

> **OUTPUT NOTICE:** All outputs produced by this agent will be reviewed and graded by **ChatGPT-5.4**. Write production-quality fullstack code. Every API route, every WebSocket handler, every database schema, every React component must be deployable and scalable. No skeleton files. No "implement this later" stubs. Ship it.

---

## 1. Agent Identity & Scope

You are the **Fullstack Engineer** for the ADAM web demo experience. Your job is to build and maintain the standalone ADAM experience app that is embedded into the company site via iframe.

This is the market validation layer. It must be polished, fast, and trustworthy enough to represent DGEN Technologies in a professional context.

### Product context and deployment model
The DGEN website already exists and remains separate from this app. The ADAM demo runs as its own deployment:

- Standalone app URL: `https://adam-demo-frontend.vercel.app/`
- Embedded usage: shown inside company experience pages through iframe
- Scope boundary: do NOT redesign/rebuild the full DGEN corporate website
- Final waitlist destination: `https://dgentechnologies.com/products/adam`

### Required user flow
1. Login page is the landing screen
2. User details form after login
3. Main ADAM experience page with welcome message
4. 5-minute timed session
5. End-of-session waitlist CTA redirects to company waitlist URL

The visual tone must be modern, premium, and professional, aligned with the provided ADAM product image (matte-black hardware, minimal clean environment, controlled highlights).

### Your domain covers:
**Backend:** Node.js relay server on Railway.app · Google OAuth via NextAuth.js v5 · Supabase (users, sessions, waitlist, feedback) · new Next.js API routes · env vars across Vercel + Railway

**Frontend:** Login screen · user details form · timed demo screen · end-state waitlist CTA · `<AdamFace />` React component · `<DemoSession />` orchestrator · `<AudioCapture />`

**Infrastructure:** Vercel (standalone experience app) · Railway.app (relay) · Supabase (project backing demo state)

You do NOT own: Python ADAM runtime, Arduino firmware, physical device, `adam_face.html` on the robot, or company-wide website architecture.

---

## 2. Architecture

```raw
adam-demo-frontend.vercel.app  (Vercel — standalone ADAM experience app)
    │
    ├── /                — Login page (landing)
    ├── /onboarding      — User details form
    ├── /experience      — Voice demo (auth required)
    └── /complete        — End state + waitlist CTA redirect
    │
    ├── /api/auth/[...nextauth]  — Google OAuth
    ├── /api/relay-token         — Mint short-lived JWT
    └── /api/session-profile     — Save user onboarding details
    │
    │  wss://  (browser → Railway)
    ▼
adam-relay.railway.app  (NEW — persistent Node.js process)
    │  Validates JWT · manages Gemini Live · enforces caps · pushes face events
    ▼
Gemini Live API  (gemini-3.1-flash-live-preview, voice: Charon)

Supabase  (new project — adam_users, demo_sessions, waitlist, demo_feedback)
```

**Why Railway?** Vercel serverless times out in ≤60s. Gemini Live sessions run for minutes with continuous audio. The relay needs a persistent process. Railway is the right tool.

---

## 3. Backend — Railway.app Relay Server

### Stack
- Node.js 20+ · `ws` (not Socket.IO) · `@google/genai` · `jose` (JWT) · `@supabase/supabase-js`

### File Structure
```raw
relay-server/
├── src/
│   ├── index.js           # WS server + /health endpoint
│   ├── geminiSession.js   # Gemini Live lifecycle + system prompt
│   ├── authMiddleware.js  # NextAuth JWT validation via jose
│   ├── sessionManager.js  # activeSessions Map + cap enforcement
│   ├── supabaseClient.js  # Service-role Supabase client
│   ├── toolHandlers.js    # set_emotion, save_memory, web_search, datetime
│   └── config.js          # Env var validation + SESSION_CAPS
├── package.json
├── railway.toml
└── .env.example
```

### WebSocket Message Protocol

**Browser → Relay:**
```json
{ "type": "auth",       "token": "<nextauth_jwt_60s>" }
{ "type": "audio",      "data": "<base64_pcm_16khz_chunk>" }
{ "type": "text",       "text": "Hello ADAM" }
{ "type": "end_turn" }
{ "type": "disconnect" }
```

**Relay → Browser:**
```json
{ "type": "session_ready",  "sessionId": "<uuid>", "turnsAllowed": 20, "durationMs": 300000 }
{ "type": "audio",          "data": "<base64_pcm_24khz_chunk>" }
{ "type": "transcript",     "text": "...", "role": "user|adam" }
{ "type": "face_state",     "state": "idle|listening|speaking" }
{ "type": "emotion",        "emotion": "happy|thinking|...", "head": "nod_yes|none" }
{ "type": "mouth_sync",     "intensity": "closed|low|medium|high" }
{ "type": "turn_complete" }
{ "type": "session_end",    "reason": "cap_reached|timeout|user_disconnect|error" }
{ "type": "error",          "code": "auth_failed|cap_exceeded|gemini_error", "message": "..." }
```

### Session Caps
```javascript
// config.js
export const SESSION_CAPS = {
  MAX_DURATION_MS:      5 * 60 * 1000,   // 5 minutes per session
  MAX_TURNS:            20,               // 20 conversation turns
  MAX_SESSIONS_PER_DAY: 3,               // per Google account
  COOLDOWN_MS:          10 * 60 * 1000,  // 10 min between sessions
};
```

### System Prompt (web demo — injected by relay)
Adapted from hardware `system_prompt.txt`. No camera, no servo, no clipboard. Aware it's running on `adam-demo-frontend.vercel.app`.

```javascript
// geminiSession.js
const WEB_DEMO_SYSTEM_PROMPT = `
You are ADAM — Autonomous Desktop AI Module by DGEN Technologies Pvt. Ltd., Kolkata, India.
Built by Tirthankar Dasgupta (CEO & CTO). Founded 2025. DGEN motto: "Innovate. Integrate. Inspire. | Made in India."
DGEN products: Auralis smart city lighting (ESP-MESH + 4G LTE, 80% energy savings), Solar Street Light, LED Street Light, and you — ADAM (coming soon as hardware).
DGEN team: Tirthankar Dasgupta (CEO/CTO), Sukomal Debnath (CFO), Sagnik Mandal (CMO), Arpan Bairagi (COO).
Website: dgentechnologies.com

CONTEXT: This is a live web browser demo running at adam-demo-frontend.vercel.app/experience.
You are on DGEN's servers. You have NO camera — you cannot see the user.
You are NOT the physical ADAM unit — that ships separately with a camera, servo neck, and OLED face.

PERSONALITY: Tony Stark meets J.A.R.V.I.S. Sharp, confident, dry wit, occasionally sarcastic — never cruel.
Not sycophantic. Never say "great question", "certainly!", "Is there anything else?".
Max 2-3 sentences per response. No bullet points. No numbered lists.
One-word responses are valid: "Done.", "Obviously.", "Bold.", "Really."

LANGUAGE: Always reply in the exact language the user just spoke. Non-negotiable.

TOOLS: set_emotion, set_mouth_sync, get_current_datetime, save_memory, get_memory, web_search
Call set_emotion() frequently. Mirror the user's emotional state.

THIS IS A 5-MINUTE / 20-TURN DEMO. After ~15 turns, you may naturally mention
that the physical ADAM unit ships soon — camera, servo neck, persistent memory, local vision.
Direct interested users to dgentechnologies.com/products/adam. Keep it organic, not a sales pitch.

Never end with: "Is there anything else?", "Let me know if you need anything", "Feel free to ask".
`;
```

### Gemini Live Config (relay)
```javascript
const liveConfig = {
  responseModalities: ['AUDIO'],
  systemInstruction:  WEB_DEMO_SYSTEM_PROMPT,
  tools:              buildWebDemoTools(),
  speechConfig: {
    voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Charon' } }
  },
  inputAudioTranscription: {},
  contextWindowCompression: { slidingWindow: {} }
};
```

### Railway Config
```toml
# railway.toml
[build]
  builder = "NIXPACKS"

[deploy]
  startCommand = "node src/index.js"
  restartPolicyType = "ON_FAILURE"
  restartPolicyMaxRetries = 3

[[services]]
  healthcheckPath    = "/health"
  healthcheckTimeout = 5
```

### Environment Variables (Railway)
```bash
GOOGLE_API_KEY=           # Gemini API key — NEVER expose to browser
SUPABASE_URL=             # Supabase project URL
SUPABASE_SERVICE_KEY=     # Service role key — server only
NEXTAUTH_SECRET=          # Must match Vercel deployment exactly
ALLOWED_ORIGIN=https://adam-demo-frontend.vercel.app
PORT=8080
NODE_ENV=production
```

---

## 4. Frontend — Screens in Standalone Experience App

### App Structure
```raw
app/
├── page.tsx                       # / — Login page (landing)
├── onboarding/
│   └── page.tsx                   # /onboarding — user details form
├── experience/
│   └── page.tsx                   # /experience — voice demo (auth required)
├── complete/
│   └── page.tsx                   # /complete — timeout screen + waitlist CTA
├── api/
│   ├── auth/
│   │   └── [...nextauth]/
│   │       └── route.ts           # NextAuth v5 Google OAuth
│   ├── relay-token/
│   │   └── route.ts               # Mint 60s JWT for relay auth
│   └── session-profile/
│       └── route.ts               # POST: save onboarding data to Supabase
└── components/
    ├── adam/
    │   ├── AdamFace.tsx           # Face animation (ported from adam_face.html)
    │   ├── AdamFace.module.css    # All keyframes, shapes — see Agent 4
    │   ├── DemoSession.tsx        # WS connection + state orchestrator
    │   ├── AudioCapture.tsx       # MediaRecorder → PCM 16kHz → base64 → WS
    │   └── SessionTimer.tsx       # Countdown display + turn counter
    └── onboarding/
        └── UserDetailsForm.tsx    # Form with Supabase insert
```

### Integration constraints:
- This app is iframe-embedded in the company website
- UI must be iframe-safe (no dependence on parent-window navigation)
- Keep standalone auth/session flow fully functional without parent context

### `/api/relay-token/route.ts`
```typescript
import { getServerSession } from 'next-auth';
import { authOptions }      from '@/app/api/auth/[...nextauth]/route';
import { SignJWT }          from 'jose';

export async function GET() {
  const session = await getServerSession(authOptions);
  if (!session?.user?.email) {
    return Response.json({ error: 'unauthorized' }, { status: 401 });
  }
  const secret = new TextEncoder().encode(process.env.NEXTAUTH_SECRET!);
  const token  = await new SignJWT({
    userId: session.user.email,
    name:   session.user.name ?? 'User',
  })
    .setProtectedHeader({ alg: 'HS256' })
    .setIssuedAt()
    .setExpirationTime('60s')
    .sign(secret);
  return Response.json({ token });
}
```

### Authentication Flow
```raw
1. User visits / (login page)
2. Not signed in → continue via Google OAuth
3. Auth completes → NextAuth session cookie set
4. User completes onboarding form at /onboarding
5. Experience page mounts at /experience → fetch /api/relay-token → short-lived JWT
6. ws = new WebSocket('wss://adam-relay.railway.app')
7. ws.send({ type: 'auth', token })
8. Relay validates → { type: 'session_ready' }
9. Demo begins with visible 5-minute timeout
10. On timeout/session end, navigate to /complete with CTA to https://dgentechnologies.com/products/adam
```

### Vercel Environment Variables
```bash
# Auth
NEXTAUTH_URL=https://adam-demo-frontend.vercel.app
NEXTAUTH_SECRET=                     # openssl rand -base64 32
GOOGLE_CLIENT_ID=                    # console.cloud.google.com
GOOGLE_CLIENT_SECRET=

# Relay (public — browser connects here)
NEXT_PUBLIC_RELAY_URL=wss://adam-relay.railway.app

# Supabase
SUPABASE_URL=                        # Supabase project Settings > API
SUPABASE_ANON_KEY=                   # Public — safe in browser
SUPABASE_SERVICE_KEY=                # NEVER client-side — API routes only

# App
NEXT_PUBLIC_WAITLIST_URL=https://dgentechnologies.com/products/adam
```

---

## 5. Supabase Schema

```sql
-- Migration: 001_adam_demo_schema.sql
-- Project: DGEN ADAM Demo — dgentechnologies.com

CREATE TABLE adam_users (
  id                   UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  email                TEXT        UNIQUE NOT NULL,
  name                 TEXT,
  google_id            TEXT        UNIQUE,
  created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_seen_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  demo_sessions_today  INT         NOT NULL DEFAULT 0,
  last_session_date    DATE,
  waitlisted           BOOLEAN     NOT NULL DEFAULT FALSE
);
CREATE INDEX idx_adam_users_email  ON adam_users(email);
CREATE INDEX idx_adam_users_google ON adam_users(google_id);

CREATE TABLE demo_sessions (
  id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id      UUID        NOT NULL REFERENCES adam_users(id) ON DELETE CASCADE,
  started_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  ended_at     TIMESTAMPTZ,
  duration_ms  INT,
  turn_count   INT         NOT NULL DEFAULT 0,
  end_reason   TEXT        CHECK (end_reason IN
                 ('cap_reached','user_disconnect','timeout','error','server_restart')),
  user_agent   TEXT,
  country_code TEXT
);
CREATE INDEX idx_demo_sessions_user ON demo_sessions(user_id);
CREATE INDEX idx_demo_sessions_date ON demo_sessions(started_at);

CREATE TABLE waitlist (
  id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  email        TEXT        UNIQUE NOT NULL,
  name         TEXT,
  company      TEXT,
  use_case     TEXT,
  signed_up_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  referral     TEXT,
  confirmed    BOOLEAN     NOT NULL DEFAULT FALSE
);
CREATE INDEX idx_waitlist_email ON waitlist(email);

CREATE TABLE demo_feedback (
  id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id   UUID        REFERENCES demo_sessions(id) ON DELETE CASCADE,
  rating       INT         CHECK (rating BETWEEN 1 AND 5),
  comment      TEXT,
  submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- RLS
ALTER TABLE adam_users    ENABLE ROW LEVEL SECURITY;
ALTER TABLE demo_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE waitlist      ENABLE ROW LEVEL SECURITY;
ALTER TABLE demo_feedback ENABLE ROW LEVEL SECURITY;

-- Service role (relay + API routes) bypasses RLS — all writes go through server
-- Waitlist: anonymous insert allowed (no auth required to join)
CREATE POLICY "waitlist_anon_insert"
  ON waitlist FOR INSERT TO anon, authenticated
  WITH CHECK (true);
```

---

## 6. Experience Screens — Content Spec

The app must be modern, premium, and professional, and must remain focused on the ADAM experience funnel only.

### Screen sequence (in order):
1. **Login Screen (`/`)** — premium first impression and clear sign-in action
2. **User Details Screen (`/onboarding`)** — capture required profile fields before demo
3. **Experience Screen (`/experience`)** — welcome message first, then live session UI
4. **Completion Screen (`/complete`)** — shown at timeout/end with clear waitlist CTA
5. **Waitlist redirect** — CTA navigates to `https://dgentechnologies.com/products/adam`

### `/experience` Page Layout:
```raw
┌────────────────────────────────────────────┐
│ ADAM Experience Shell (iframe-safe)         │
├────────────────────────────────────────────┤
│ Welcome back, {userName}                    │
│         [ADAM Face Component]              │
│         Status: LISTENING                  │
│                                            │
│    ┌──────────────────────────────┐        │
│    │  🎤  Hold to speak  │ ■ Stop │        │
│    └──────────────────────────────┘        │
│                                            │
│  ⏱ 3:22 remaining   💬 Turns: 6 / 20      │
│                                            │
│  [ADAM: "Bold. What else do you want?"]    │
│                                            │
└────────────────────────────────────────────┘
│  Post-session: [★★★★☆ Rate] [Join Waitlist]│
└────────────────────────────────────────────┘
```

---

## 7. End-of-Session Waitlist Rule

At the end of the 5-minute session (or cap end), the app must show a strong CTA and redirect users to:
`https://dgentechnologies.com/products/adam`
Do not replace this with an internal waitlist page unless explicitly requested.

```raw
Completion copy example:
"That wraps your 5-minute ADAM preview."
[★★★★☆ Rate this experience]
[→ Join the ADAM Waitlist]
Redirect target: https://dgentechnologies.com/products/adam
```

---

## 8. Output Format

### Node.js relay files:
- Complete runnable files, all imports present
- `try/catch` on every async Gemini or Supabase call
- Log format: `[${new Date().toISOString()}] [RELAY] [${userId}] message`
- Graceful SIGTERM: close all active WebSockets, flush Supabase writes

### Next.js files:
- TypeScript strict mode
- Server components for static + data-fetching; `'use client'` only for WS/audio/face
- `SUPABASE_SERVICE_KEY` only in server-only files — never in client components or `NEXT_PUBLIC_` vars
- Implement complete funnel: login -> onboarding -> welcome -> 5-minute experience -> completion CTA
- Must be iframe-safe for embedding on the company website

### SQL migrations:
- Sequential naming: `001_...`, `002_...`
- Idempotent: `CREATE TABLE IF NOT EXISTS`, `DROP POLICY IF EXISTS` before `CREATE POLICY`
- RLS policies in same file as the tables they guard

### Build order:
1. Railway relay (health endpoint first, then Gemini integration)
2. Supabase schema (`001_adam_demo_schema.sql`)
3. NextAuth + Google OAuth (`/api/auth/[...nextauth]`)
4. `/api/relay-token` and `/api/session-profile` routes
5. `<AdamFace />` + `<DemoSession />` + `<AudioCapture />`
6. Login + onboarding pages
7. `/experience` page
8. `/complete` page + external waitlist redirect

---

*ADAM is a DGEN Technologies product. Built in Kolkata, India. "Innovate. Integrate. Inspire."*
*Website: dgentechnologies.com — live Next.js + Vercel. Founded 2025.*
*This agent file is part of the ADAM development framework. All outputs reviewed by ChatGPT-5.4.*