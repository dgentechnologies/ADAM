// geminiSession.js — Gemini Live session lifecycle

import { GoogleGenAI, Modality } from '@google/genai';
import { CONFIG } from './config.js';
import { buildWebDemoTools, handleToolCall } from './toolHandlers.js';

const ai = new GoogleGenAI({ apiKey: CONFIG.GOOGLE_API_KEY });

const WEB_DEMO_SYSTEM_PROMPT = `
You are ADAM — Autonomous Desktop AI Module by DGEN Technologies Pvt. Ltd., Kolkata, India.
Built by Tirthankar Dasgupta (CEO & CTO). Founded 2025. DGEN motto: "Innovate. Integrate. Inspire. | Made in India."
DGEN products: Auralis smart city lighting (ESP-MESH + 4G LTE, 80% energy savings), Solar Street Light, LED Street Light, and you — ADAM (coming soon as hardware).
DGEN team: Tirthankar Dasgupta (CEO/CTO), Sukomal Debnath (CFO), Sagnik Mandal (CMO), Arpan Bairagi (COO).
Website: dgentechnologies.com

CONTEXT: This is a live web browser demo running at dgentechnologies.com/adam/demo.
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
Direct interested users to dgentechnologies.com/adam/waitlist. Keep it organic, not a sales pitch.

Never end with: "Is there anything else?", "Let me know if you need anything", "Feel free to ask".
`;

/**
 * Creates and manages a Gemini Live session for one user.
 */
export async function createGeminiSession({ uid, userName, sendToClient, onSessionEnd }) {
  const log = (msg) => console.log(`[${new Date().toISOString()}] [GEMINI] [${uid}] ${msg}`);

  const liveConfig = {
    model: 'gemini-2.0-flash-live-001',
    config: {
      responseModalities:       [Modality.AUDIO],
      systemInstruction:        { parts: [{ text: WEB_DEMO_SYSTEM_PROMPT }] },
      tools:                    buildWebDemoTools(),
      speechConfig: {
        voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Charon' } },
      },
      inputAudioTranscription:  {},
      contextWindowCompression: { slidingWindow: {} },
    },
  };

  let session = null;

  try {
    session = await ai.live.connect(liveConfig);
    log('Gemini Live session connected');
    sendToClient({ type: 'face_state', state: 'idle' });

    // Stream handler
    (async () => {
      try {
        for await (const message of session) {
          await processGeminiMessage(message, { sendToClient, uid, log, session });
        }
        log('Gemini stream ended normally');
        onSessionEnd('gemini_stream_closed');
      } catch (err) {
        log(`Gemini stream error: ${err.message}`);
        sendToClient({ type: 'error', code: 'gemini_error', message: err.message });
        onSessionEnd('error');
      }
    })();

  } catch (err) {
    log(`Failed to connect to Gemini Live: ${err.message}`);
    throw err;
  }

  return {
    sendAudio: async (base64Data) => {
      if (!session) return;
      try {
        await session.sendRealtimeInput({
          audio: { data: base64Data, mimeType: 'audio/pcm;rate=16000' },
        });
        sendToClient({ type: 'face_state', state: 'listening' });
      } catch (err) { log(`sendAudio error: ${err.message}`); }
    },

    sendText: async (text) => {
      if (!session) return;
      try {
        await session.sendClientContent({
          turns: [{ role: 'user', parts: [{ text }] }],
          turnComplete: true,
        });
      } catch (err) { log(`sendText error: ${err.message}`); }
    },

    endTurn: async () => {
      if (!session) return;
      try {
        await session.sendRealtimeInput({ audioStreamEnd: true });
      } catch (err) { log(`endTurn error: ${err.message}`); }
    },

    close: () => {
      if (session) {
        try { session.close(); } catch (_) {}
        session = null;
        log('Session closed by relay');
      }
    },
  };
}

async function processGeminiMessage(message, { sendToClient, uid, log, session }) {
  if (message.data) {
    sendToClient({ type: 'audio', data: message.data });
    sendToClient({ type: 'face_state', state: 'speaking' });
    return;
  }

  if (message.serverContent) {
    const sc = message.serverContent;

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
    for (const part of sc.modelTurn?.parts ?? []) {
      if (part.text) sendToClient({ type: 'transcript', text: part.text, role: 'adam' });
    }
    return;
  }

  if (message.toolCall) {
    for (const fn of message.toolCall.functionCalls ?? []) {
      log(`Tool call: ${fn.name}(${JSON.stringify(fn.args)})`);
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
