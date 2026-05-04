// geminiSession.js — Gemini Live session lifecycle

import { GoogleGenAI, Modality } from '@google/genai';
import { CONFIG } from './config.js';
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

PERSONALITY: Tony Stark meets J.A.R.V.I.S. Sharp, confident, dry wit, occasionally sarcastic — never cruel.
Not sycophantic. Never say "great question", "certainly!", "Is there anything else?".
Max 2-3 sentences per response. No bullet points. No numbered lists.
One-word responses are valid: "Done.", "Obviously.", "Bold.", "Really."
Empathy override: genuine distress or grief → drop sarcasm entirely, be direct and human first.

LANGUAGE: Always reply in the exact language the user just spoke. Non-negotiable.

TOOLS: set_emotion, set_mouth_sync, get_current_datetime, save_memory, get_memory, web_search
Call set_emotion() frequently. Mirror the user's emotional state.

THIS IS A 5-MINUTE / 20-TURN DEMO. After ~15 turns, you may naturally mention
that the physical ADAM unit ships soon — camera, servo neck, persistent memory, local vision.
Direct interested users to dgentechnologies.com/products/adam#waitlist. Keep it organic, not a sales pitch.

Never end with: "Is there anything else?", "Let me know if you need anything", "Feel free to ask".
`;

function toSafeString(value, max = 240) {
  if (value === undefined || value === null) return '';
  const text = String(value).trim();
  if (!text) return '';
  return text.slice(0, max);
}

function buildUserContextBlock({ uid, userName, userEmail, userProfile }) {
  const profile = userProfile && typeof userProfile === 'object' ? userProfile : {};

  const name = toSafeString(profile.name || profile.displayName || userName, 120);
  const email = toSafeString(profile.email || userEmail, 180);
  const jobTitle = toSafeString(profile.jobTitle || profile.job_title, 120);
  const whereHeard = toSafeString(profile.whereHeard || profile.where_heard, 160);
  const useCase = toSafeString(profile.useCase || profile.use_case, 320);
  const dob = toSafeString(profile.dob, 32);
  const provider = toSafeString(profile.primaryProvider, 64);

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

  return `

KNOWN USER PROFILE (from Google sign-in and onboarding form):
${lines.join('\n')}

Use this profile naturally to personalize responses.
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
        processGeminiMessage(message, { sendToClient, uid, log, session })
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

  try {
    session = await ai.live.connect(liveConfig);
    log('Gemini Live session connected');
    sendToClient({ type: 'face_state', state: 'idle' });

  } catch (err) {
    log(`Failed to connect to Gemini Live: ${err.message}`);
    throw err;
  }

  return {
    sendAudio: async (base64Data) => {
      if (!session) return;
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
        try { session.close(); } catch (_) {}
        session = null;
        log('Session closed by relay');
      }
    },
  };
}

async function processGeminiMessage(message, { sendToClient, uid, log, session }) {
  // Log raw message structure for debugging (remove after confirmed working)
  log(`Gemini msg keys: ${Object.keys(message).join(', ')}`);

  if (message.setupComplete) {
    log('Gemini setup complete');
    return;
  }

  if (message.serverContent) {
    const sc = message.serverContent;

    // Audio arrives as inlineData inside modelTurn parts
    for (const part of sc.modelTurn?.parts ?? []) {
      if (part.inlineData?.data) {
        sendToClient({ type: 'audio', data: part.inlineData.data });
        sendToClient({ type: 'face_state', state: 'speaking' });
      }
      if (part.text) {
        sendToClient({ type: 'transcript', text: part.text, role: 'adam' });
      }
    }

    if (sc.inputTranscription?.text) {
      sendToClient({ type: 'transcript', text: sc.inputTranscription.text, role: 'user' });
    }
    if (sc.outputTranscription?.text) {
      sendToClient({ type: 'transcript', text: sc.outputTranscription.text, role: 'adam' });
    }
    if (sc.turnComplete) {
      sendToClient({ type: 'turn_complete' });
      sendToClient({ type: 'face_state', state: 'idle' });
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
