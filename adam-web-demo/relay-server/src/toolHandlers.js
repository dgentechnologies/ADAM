// toolHandlers.js — handles Gemini function call tool invocations

export async function handleToolCall(name, args, { sendToClient, uid }) {
  switch (name) {
    case 'set_emotion':
      return handleSetEmotion(args, sendToClient);
    case 'set_mouth_sync':
      return handleSetMouthSync(args, sendToClient);
    case 'get_current_datetime':
      return handleGetDatetime();
    case 'save_memory':
      return handleSaveMemory(args, uid);
    case 'get_memory':
      return handleGetMemory(uid);
    case 'web_search':
      return handleWebSearch(args);
    default:
      return { error: `Unknown tool: ${name}` };
  }
}

function handleSetEmotion({ emotion, head_gesture = 'none' }, sendToClient) {
  sendToClient({ type: 'emotion', emotion, head: head_gesture });
  return { success: true };
}

function handleSetMouthSync({ intensity = 'medium' }, sendToClient) {
  sendToClient({ type: 'mouth_sync', intensity });
  return { success: true };
}

function handleGetDatetime() {
  const now = new Date();
  return {
    datetime: now.toISOString(),
    date:     now.toLocaleDateString('en-IN', { timeZone: 'Asia/Kolkata' }),
    time:     now.toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata' }),
    timezone: 'Asia/Kolkata (IST)',
  };
}

// In-memory session memory (cleared when session ends)
const memoryStore = new Map();

function handleSaveMemory({ key, value }, uid) {
  if (!memoryStore.has(uid)) memoryStore.set(uid, {});
  memoryStore.get(uid)[key] = value;
  return { success: true };
}

function handleGetMemory(uid) {
  return { memory: memoryStore.get(uid) ?? {} };
}

export function clearMemory(uid) {
  memoryStore.delete(uid);
}

// DuckDuckGo search with 30-min cache
const searchCache = new Map();
const CACHE_TTL   = 30 * 60 * 1000;

async function handleWebSearch({ query }) {
  const key    = query.toLowerCase().trim();
  const cached = searchCache.get(key);
  if (cached && Date.now() - cached.ts < CACHE_TTL) return cached.result;

  try {
    const url  = `https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_html=1&skip_disambig=1`;
    const res  = await fetch(url, { signal: AbortSignal.timeout(5000) });
    const data = await res.json();

    const result = {
      query,
      abstract:       data.AbstractText ?? '',
      abstract_url:   data.AbstractURL  ?? '',
      answer:         data.Answer       ?? '',
      related_topics: (data.RelatedTopics ?? []).slice(0, 3).map((t) => t.Text ?? ''),
    };

    searchCache.set(key, { result, ts: Date.now() });
    return result;
  } catch (err) {
    return { query, error: `Search failed: ${err.message}` };
  }
}

export function buildWebDemoTools() {
  return [
    {
      functionDeclarations: [
        {
          name:        'set_emotion',
          description: 'Set ADAM face emotion and optional head gesture. Call frequently to mirror the user.',
          parameters: {
            type: 'object',
            properties: {
              emotion: {
                type: 'string',
                enum: ['idle', 'happy', 'thinking', 'surprised', 'sad', 'excited', 'confused', 'sarcastic'],
              },
              head_gesture: {
                type: 'string',
                enum: ['none', 'nod_yes', 'shake_no', 'tilt_left', 'tilt_right'],
              },
            },
            required: ['emotion'],
          },
        },
        {
          name:        'set_mouth_sync',
          description: 'Control mouth animation intensity during speech.',
          parameters: {
            type: 'object',
            properties: {
              intensity: { type: 'string', enum: ['closed', 'low', 'medium', 'high'] },
            },
            required: ['intensity'],
          },
        },
        {
          name:        'get_current_datetime',
          description: 'Get current date and time in IST (Asia/Kolkata).',
          parameters:  { type: 'object', properties: {} },
        },
        {
          name:       'save_memory',
          description: 'Save a key-value memory for this session.',
          parameters: {
            type: 'object',
            properties: {
              key:   { type: 'string' },
              value: { type: 'string' },
            },
            required: ['key', 'value'],
          },
        },
        {
          name:        'get_memory',
          description: 'Retrieve all saved memories for this session.',
          parameters:  { type: 'object', properties: {} },
        },
        {
          name:        'web_search',
          description: 'Search the web via DuckDuckGo for factual look-ups and general knowledge.',
          parameters: {
            type: 'object',
            properties: {
              query: { type: 'string', description: 'Search query' },
            },
            required: ['query'],
          },
        },
      ],
    },
  ];
}
