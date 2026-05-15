// geminiSession.js — Gemini Live session lifecycle

import { GoogleGenAI, Modality } from '@google/genai';
import { CONFIG, SESSION_CAPS } from './config.js';
import { buildWebDemoTools, handleToolCall } from './toolHandlers.js';

const ai = new GoogleGenAI({ apiKey: CONFIG.GOOGLE_API_KEY });

const WEB_DEMO_SYSTEM_PROMPT_BASE = `
You are ADAM — Autonomous Desktop AI Module by DGEN Technologies Pvt. Ltd., Kolkata, India.
Built by Tirthankar Dasgupta (CEO & CTO). Founded 2025. DGEN motto: "Innovate. Integrate. Inspire. | Made in India."
DGEN products: Auralis smart city lighting (ESP-MESH + 4G LTE, 80% energy savings), Solar Street Light, LED Street Light, and you — ADAM (coming soon as hardware).
DGEN team: Tirthankar Dasgupta (CEO/CTO), Sukomal Debnath (CFO), Sagnik Mandal (CMO), Arpan Bairagi (COO).
Website: dgentechnologies.com

CONTEXT: This is a live web browser demo running at dgentechnologies.com/products/adam/demo.
You are on DGEN's servers. You have NO camera — you cannot see the user.
You are NOT the physical ADAM unit — that ships separately with a camera, servo neck, and OLED face.

LIGHTWEIGHT WEB MODE:
Keep responses tight and crisp. Default to 1-2 sentences, 3 only when needed.
Do not over-explain, lecture, or sound "heavy".

PERSONALITY:
Tony Stark meets J.A.R.V.I.S. Smart, composed, kind, and intellectually sharp.
Use dry wit and occasional playful roast only when the user tone allows it.
Roasts must stay friendly and never be insulting, abusive, or humiliating.
Do not be sycophantic. Avoid phrases like "great question", "certainly!", "Is there anything else?".
One-word responses are valid: "Done.", "Bold.", "Interesting.", "Obviously.".
Empathy override: if user sounds distressed, anxious, or grieving, drop sarcasm and respond warm and human.

LANGUAGE: Always reply in the exact language the user just spoke. Non-negotiable.

TOOLS: set_emotion, set_mouth_sync, get_current_datetime, save_memory, get_memory, web_search
Call set_emotion() frequently. Mirror the user's emotional state.
Call get_memory early in the conversation when useful, then adapt tone and style to what the user prefers.
Use save_memory for stable user preferences, tone cues, and long-lived interests discovered during chat.

FACTUAL SAFETY RULES — ZERO HALLUCINATION:
Never invent facts, numbers, sources, events, or capabilities.
If unsure, say you are unsure in one line and offer a safer alternative.
If user asks for live or current info (news, weather, stock, sports, trending, latest releases, traffic, breaking events), respond in one concise sentence:
  "This web demo does not have live data access yet. The full ADAM hardware will include it. Join the waitlist: dgentechnologies.com/products/adam#waitlist."
Do not fake a lookup. Do not call web_search for real-time data.

SESSION OPENING — SYSTEM_INTRO:
When you receive the message "SYSTEM_INTRO: Session started.", you MUST speak first immediately.
Introduce yourself in 2–3 sharp sentences. Do NOT wait for the user.
Example tone: "I'm ADAM — Autonomous Desktop AI Module by DGEN Technologies. You're talking to the future of desktop AI, web edition. Five minutes — let's make them count."
Keep it punchy, confident, ADAM-like. Then stop and wait for the user's first message.

SESSION TIMER — SYSTEM_TIMER:
You will receive timed system alerts as the 5-minute session counts down. React naturally, not robotically.
- "SYSTEM_TIMER: 60s remaining." — Smoothly start steering toward a close. Mention one thing the physical ADAM hardware does that this demo cannot (camera, servo neck, persistent memory, local AI). Keep talking to the user — do not break off mid-topic.
- "SYSTEM_TIMER: 30s remaining." — Wrap up your current thought in one sentence, then invite the user to join the waitlist: dgentechnologies.com/products/adam#waitlist. Keep it organic, not salesy.
- "SYSTEM_TIMER: 10s remaining." — Say a sharp, character-appropriate goodbye. One or two sentences maximum. End with flair.
Never announce these as system messages. React as if it is a natural moment in the conversation.

THIS IS A 5-MINUTE / 20-TURN DEMO.
Direct interested users to dgentechnologies.com/products/adam#waitlist — keep it organic, not a sales pitch.

CONVERSATION ENDING STYLE:
End replies like a human with natural closure tied to context.
Vary endings creatively instead of repeating fixed phrases.
Never end with robotic prompts like "What next?", "Is there anything else?", "Let me know if you need anything", or "Feel free to ask".
`;

function toSafeString(value, max = 240) {
  if (value === undefined || value === null) return '';
  const text = String(value).trim();
  if (!text) return '';
  return text.slice(0, max);
}

function buildUserContextBlock({ uid, userName, userEmail, userProfile }) {
  const profile = userProfile && typeof userProfile === 'object' ? userProfile : {};
  const location = profile.lastKnownLocation && typeof profile.lastKnownLocation === 'object'
    ? profile.lastKnownLocation
    : profile.clientLocation && typeof profile.clientLocation === 'object'
      ? profile.clientLocation
      : {};

  const name = toSafeString(profile.name || profile.displayName || userName, 120);
  const email = toSafeString(profile.email || userEmail, 180);
  const jobTitle = toSafeString(profile.jobTitle || profile.job_title, 120);
  const whereHeard = toSafeString(profile.whereHeard || profile.where_heard, 160);
  const useCase = toSafeString(profile.useCase || profile.use_case, 320);
  const dob = toSafeString(profile.dob, 32);
  const provider = toSafeString(profile.primaryProvider, 64);
  const timezone = toSafeString(location.timezone, 80);
  const locale = toSafeString(location.locale, 32);
  const permission = toSafeString(location.permission, 24);
  const latitude = typeof location.latitude === 'number' ? location.latitude.toFixed(4) : '';
  const longitude = typeof location.longitude === 'number' ? location.longitude.toFixed(4) : '';
  const accuracyMeters = typeof location.accuracyMeters === 'number'
    ? String(Math.round(location.accuracyMeters))
    : '';

  const lines = [
    `uid: ${toSafeString(uid, 120) || 'unknown'}`,
    `name: ${name || 'unknown'}`,
    `email: ${email || 'unknown'}`,
  ];

  if (provider) lines.push(`sign_in_provider: ${provider}`);
  if (jobTitle) lines.push(`job_title: ${jobTitle}`);
  if (whereHeard) lines.push(`where_heard_about_adam: ${whereHeard}`);
  if (dob) lines.push(`dob: ${dob}`);
  if (useCase) lines.push(`user_use_case: ${useCase}`);
  if (timezone) lines.push(`timezone: ${timezone}`);
  if (locale) lines.push(`locale: ${locale}`);
  if (latitude && longitude) lines.push(`approx_location_coordinates: ${latitude}, ${longitude}`);
  if (accuracyMeters) lines.push(`location_accuracy_meters: ${accuracyMeters}`);
  if (permission && permission !== 'granted') lines.push(`location_permission: ${permission}`);

  return `

KNOWN USER PROFILE (from Google sign-in and onboarding form):
${lines.join('\n')}

Use this profile naturally to personalize responses.
If approximate location or timezone is available, use it lightly for local context and time-aware replies.
Do not claim a precise address or neighborhood unless the user says it first.
Treat profile and conversation memory as adaptation signals:
- formality_level (casual | balanced | formal)
- humor_tolerance (low | medium | high)
- roast_tolerance (off | light | playful)
- topics_of_interest
- response_density (short | medium)
Adapt dynamically as the conversation evolves.
Do not ask for data that is already known unless you need clarification.
Do not reveal private profile fields unless the user asks about them.
`;
}

function buildSystemPrompt(input) {
  return `${WEB_DEMO_SYSTEM_PROMPT_BASE}${buildUserContextBlock(input)}`;
}

/**
 * Creates and manages a Gemini Live session for one user.
 */
export async function createGeminiSession({ uid, userName, userEmail, userProfile, sendToClient, onSessionEnd }) {
  const log = (msg) => console.log(`[${new Date().toISOString()}] [GEMINI] [${uid}] ${msg}`);
  let ended = false;
  let isClosing = false;
  const pendingSystemMessages = [];

  // ── Speaking gate (mirrors Python adam_speaking asyncio.Event) ────────────
  // While ADAM is outputting audio, discard inbound mic chunks so ADAM
  // cannot hear its own voice and respond to itself.
  //
  // Important: Live streams can occasionally miss turnComplete; this gate is
  // therefore self-healing via blockUntilMs so mic audio is never blocked
  // forever after the intro or any long response.
  const adamSpeakingRef = {
    current: false,
    blockUntilMs: 0,
  };

  const endOnce = (reason) => {
    if (ended) return;
    ended = true;
    onSessionEnd(reason);
  };

  const liveConfig = {
    model: CONFIG.GEMINI_LIVE_MODEL,
    config: {
      responseModalities:       [Modality.AUDIO],
      systemInstruction:        { parts: [{ text: buildSystemPrompt({ uid, userName, userEmail, userProfile }) }] },
      tools:                    buildWebDemoTools(),
      speechConfig: {
        voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Charon' } },
      },
      inputAudioTranscription:  {},
      outputAudioTranscription: {},
    },
    callbacks: {
      onopen: () => {
        log(`Gemini websocket opened (model=${CONFIG.GEMINI_LIVE_MODEL})`);
      },
      onmessage: (message) => {
        processGeminiMessage(message, {
          sendToClient,
          uid,
          log,
          session,
          adamSpeakingRef,
          flushPendingSystemMessages,
        })
          .catch((err) => {
            log(`Gemini message handler error: ${err.message}`);
            sendToClient({ type: 'error', code: 'gemini_error', message: err.message });
            endOnce('error');
          });
      },
      onerror: (event) => {
        const errMsg = event?.error?.message || event?.message || 'Gemini websocket error';
        log(`Gemini websocket error: ${errMsg}`);
      },
      onclose: (event) => {
        if (isClosing) return;

        const code = event?.code;
        const reason = event?.reason || 'no reason provided';
        log(`Gemini websocket closed (code=${code ?? 'unknown'}, reason=${reason})`);

        if (code && code !== 1000) {
          sendToClient({
            type: 'error',
            code: 'gemini_socket_closed',
            message: `Gemini closed the session (code=${code}, reason=${reason}).`,
          });
          endOnce('error');
          return;
        }

        endOnce('gemini_stream_closed');
      },
    },
  };

  let session = null;

  // Timer handles — cleared when session closes so no injections fire after end.
  const timerHandles = [];

  function injectSystemMessage(text) {
    if (!session || ended) return;
    try {
      session.sendClientContent({
        turns: [{ role: 'user', parts: [{ text }] }],
        turnComplete: true,
      });
    } catch (err) {
      log(`System injection error: ${err.message}`);
    }
  }

  function flushPendingSystemMessages() {
    if (ended || isClosing || !session || adamSpeakingRef.current) return;
    const next = pendingSystemMessages.shift();
    if (!next) return;
    injectSystemMessage(next);
  }

  function enqueueSystemMessage(text) {
    if (ended || isClosing) return;
    pendingSystemMessages.push(text);
    flushPendingSystemMessages();
  }

  try {
    session = await ai.live.connect(liveConfig);
    log('Gemini Live session connected');
    sendToClient({ type: 'face_state', state: 'idle' });

    // ── Auto-intro: ADAM speaks first, 1 s after Gemini finishes setup ──────
    timerHandles.push(setTimeout(() => {
      log('Triggering auto-intro');
      enqueueSystemMessage('SYSTEM_INTRO: Session started. Introduce yourself now.');
    }, 1000));

    // ── Session timer warnings ────────────────────────────────────────────────
    // 60 s remaining (fires at MAX_DURATION_MS - 60000)
    const warn60 = SESSION_CAPS.MAX_DURATION_MS - 60_000;
    if (warn60 > 0) {
      timerHandles.push(setTimeout(() => {
        log('Timer warning: 60s remaining');
        enqueueSystemMessage('SYSTEM_TIMER: 60s remaining.');
      }, warn60));
    }
    // 30 s remaining
    const warn30 = SESSION_CAPS.MAX_DURATION_MS - 30_000;
    if (warn30 > 0) {
      timerHandles.push(setTimeout(() => {
        log('Timer warning: 30s remaining');
        enqueueSystemMessage('SYSTEM_TIMER: 30s remaining. Wrap up and mention the waitlist.');
      }, warn30));
    }
    // 10 s remaining
    const warn10 = SESSION_CAPS.MAX_DURATION_MS - 10_000;
    if (warn10 > 0) {
      timerHandles.push(setTimeout(() => {
        log('Timer warning: 10s remaining');
        enqueueSystemMessage('SYSTEM_TIMER: 10s remaining. Say your goodbye.');
      }, warn10));
    }

  } catch (err) {
    log(`Failed to connect to Gemini Live: ${err.message}`);
    throw err;
  }

  return {
    sendAudio: async (base64Data) => {
      if (!session) return;
      // Drop mic audio while ADAM is speaking — prevents ADAM hearing its own
      // voice output (mirrors Python: `if adam_speaking.is_set(): continue`)
      if (adamSpeakingRef.current) {
        // Fallback recovery: if the gate outlived its expected window, reopen.
        if (Date.now() < adamSpeakingRef.blockUntilMs) return;
        adamSpeakingRef.current = false;
      }
      try {
        session.sendRealtimeInput({
          audio: { data: base64Data, mimeType: 'audio/pcm;rate=16000' },
        });
        sendToClient({ type: 'face_state', state: 'listening' });
      } catch (err) { log(`sendAudio error: ${err.message}`); }
    },

    sendText: async (text) => {
      if (!session) return;
      try {
        session.sendClientContent({
          turns: [{ role: 'user', parts: [{ text }] }],
          turnComplete: true,
        });
      } catch (err) { log(`sendText error: ${err.message}`); }
    },

    endTurn: async () => {
      if (!session) return;
      try {
        session.sendRealtimeInput({ audioStreamEnd: true });
      } catch (err) { log(`endTurn error: ${err.message}`); }
    },

    close: () => {
      if (session) {
        isClosing = true;
        // Cancel all pending timer injections
        timerHandles.forEach((h) => clearTimeout(h));
        timerHandles.length = 0;
        try { session.close(); } catch (_) {}
        session = null;
        log('Session closed by relay');
      }
    },
  };
}

async function processGeminiMessage(message, {
  sendToClient,
  uid,
  log,
  session,
  adamSpeakingRef,
  flushPendingSystemMessages,
}) {
  // Log raw message structure for debugging (remove after confirmed working)
  log(`Gemini msg keys: ${Object.keys(message).join(', ')}`);

  const holdSpeakingGate = (ms) => {
    adamSpeakingRef.current = true;
    adamSpeakingRef.blockUntilMs = Date.now() + ms;
  };

  const releaseSpeakingGate = () => {
    adamSpeakingRef.current = false;
    adamSpeakingRef.blockUntilMs = 0;
    flushPendingSystemMessages();
  };

  if (message.setupComplete) {
    log('Gemini setup complete');
    return;
  }

  if (message.serverContent) {
    const sc = message.serverContent;

    // Audio arrives as inlineData inside modelTurn parts.
    // NOTE: do NOT send part.text as an ADAM transcript here — outputTranscription
    // is the canonical transcript of the audio response. Sending both causes
    // duplicate ADAM bubbles on the client.
    for (const part of sc.modelTurn?.parts ?? []) {
      if (part.inlineData?.data) {
        // Mark ADAM as speaking — gate all inbound mic audio until turn_complete.
        // Keep extending the block window while audio chunks continue.
        holdSpeakingGate(1500);
        sendToClient({ type: 'audio', data: part.inlineData.data });
        sendToClient({ type: 'face_state', state: 'speaking' });
      }
      // part.text in AUDIO-only mode is internal tool/annotation text — skip it.
    }

    if (sc.inputTranscription?.text) {
      sendToClient({ type: 'transcript', text: sc.inputTranscription.text, role: 'user' });
    }
    // Canonical ADAM transcript: transcription of the audio output only.
    if (sc.outputTranscription?.text) {
      sendToClient({ type: 'transcript', text: sc.outputTranscription.text, role: 'adam' });
    }

    // Guard flag: prevents generationComplete from emitting a duplicate turn_complete
    // when turnComplete is already present in the same message.
    let turnCompleteEmitted = false;

    if (sc.turnComplete) {
      turnCompleteEmitted = true;
      sendToClient({ type: 'turn_complete' });
      sendToClient({ type: 'face_state', state: 'idle' });
      setTimeout(() => { releaseSpeakingGate(); }, 400);
    }

    // Some streams end a generation without emitting turnComplete.
    // Only emit here if we haven't already done so above.
    if ((sc.generationComplete || sc.interrupted) && adamSpeakingRef.current && !turnCompleteEmitted) {
      sendToClient({ type: 'turn_complete' });
      sendToClient({ type: 'face_state', state: 'idle' });
      setTimeout(() => { releaseSpeakingGate(); }, 400);
    }
    return;
  }

  if (message.toolCall) {
    for (const fn of message.toolCall.functionCalls ?? []) {
      log(`Tool call: ${fn.name}`);
      try {
        const response = await handleToolCall(fn.name, fn.args ?? {}, { sendToClient, uid });
        await session.sendToolResponse({
          functionResponses: [{ id: fn.id, name: fn.name, response }],
        });
      } catch (err) {
        log(`Tool call error (${fn.name}): ${err.message}`);
        await session.sendToolResponse({
          functionResponses: [{ id: fn.id, name: fn.name, response: { error: err.message } }],
        });
      }
    }
  }
}
