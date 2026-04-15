// index.js — ADAM Relay Server entry point
// WebSocket server — bridges browser ↔ Gemini Live. Auth via Firebase-minted relay JWT.

import 'dotenv/config';
import http from 'http';
import { WebSocketServer } from 'ws';
import { v4 as uuidv4 } from 'uuid';

import { CONFIG } from './config.js';
import { validateToken } from './authMiddleware.js';
import { createGeminiSession } from './geminiSession.js';
import { clearMemory } from './toolHandlers.js';
import {
  canStartSession,
  registerSession,
  incrementTurn,
  remainingMs,
  getSession,
  removeSession,
  activeSessionCount,
} from './sessionManager.js';
import {
  upsertUser,
  createSession,
  endSession,
  incrementSessionsToday,
} from './firestoreClient.js';

// Initialise Firebase Admin (side-effect import)
import './firebaseAdmin.js';

const log = (uid, msg) =>
  console.log(`[${new Date().toISOString()}] [RELAY] [${uid ?? 'system'}] ${msg}`);

// ── HTTP server ───────────────────────────────────────────────────────────────

const httpServer = http.createServer((req, res) => {
  if (req.method === 'GET' && req.url === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      status:         'ok',
      activeSessions: activeSessionCount(),
      timestamp:      new Date().toISOString(),
    }));
    return;
  }
  res.writeHead(404);
  res.end();
});

// ── WebSocket server ──────────────────────────────────────────────────────────

const wss = new WebSocketServer({ server: httpServer });

wss.on('connection', (ws, req) => {
  const origin = req.headers.origin;
  if (CONFIG.NODE_ENV === 'production' && origin !== CONFIG.ALLOWED_ORIGIN) {
    log(null, `Rejected connection from origin: ${origin}`);
    ws.close(4001, 'Origin not allowed');
    return;
  }

  const connId = uuidv4().slice(0, 8);
  log(connId, `WebSocket connected from ${origin ?? 'unknown'}`);

  let authed      = false;
  let uid         = null;
  let userName    = null;
  let userEmail   = null;
  let dbSessionId = null;
  let sessionStart = null;
  let gemini      = null;

  const send = (obj) => {
    if (ws.readyState === ws.OPEN) ws.send(JSON.stringify(obj));
  };

  const closeSession = async (reason) => {
    log(uid, `Session ending — reason: ${reason}`);

    if (gemini) { gemini.close(); gemini = null; }
    if (uid) {
      clearMemory(uid);
      removeSession(uid);
    }

    if (dbSessionId && sessionStart) {
      try {
        await endSession({
          sessionId:  dbSessionId,
          durationMs: Date.now() - sessionStart,
          turnCount:  getSession(uid)?.turnCount ?? 0,
          endReason:  reason,
        });
      } catch (err) {
        log(uid, `endSession Firestore error: ${err.message}`);
      }
    }

    send({ type: 'session_end', reason });
    if (ws.readyState === ws.OPEN) ws.close();
  };

  // ── Message handler ─────────────────────────────────────────────────────────

  ws.on('message', async (raw) => {
    let msg;
    try { msg = JSON.parse(raw.toString()); }
    catch { send({ type: 'error', code: 'parse_error', message: 'Invalid JSON' }); return; }

    // ── AUTH ──────────────────────────────────────────────────────────────────
    if (msg.type === 'auth') {
      if (authed) return;
      try {
        const payload = await validateToken(msg.token);
        uid       = payload.uid;
        userName  = payload.name;
        userEmail = payload.email;

        log(uid, `Auth successful — ${userName} <${userEmail}>`);

        // Upsert user doc in Firestore
        const userDoc = await upsertUser({ uid, email: userEmail, name: userName });

        // Check capacity
        const today           = new Date().toISOString().slice(0, 10);
        const sessionsToday   = userDoc.lastSessionDate === today ? userDoc.demoSessionsToday : 0;
        const { allowed, reason } = canStartSession(uid, sessionsToday);

        if (!allowed) {
          send({ type: 'error', code: 'cap_exceeded', message: reason });
          ws.close();
          return;
        }

        authed = true;

        // Create Firestore session record
        dbSessionId = await createSession({
          uid,
          userAgent:   req.headers['user-agent'] ?? '',
          countryCode: req.headers['cf-ipcountry'] ?? '',
        });

        sessionStart = Date.now();
        registerSession(uid, dbSessionId, (reason) => closeSession(reason));
        await incrementSessionsToday(uid);

        send({
          type:         'session_ready',
          sessionId:    dbSessionId,
          turnsAllowed: SESSION_CAPS_TURNS,
          durationMs:   remainingMs(uid),
        });

        gemini = await createGeminiSession({
          uid,
          userName,
          sendToClient: send,
          onSessionEnd: closeSession,
        });

        log(uid, 'Session ready');
      } catch (err) {
        log(connId, `Auth error: ${err.message}`);
        send({ type: 'error', code: 'auth_failed', message: err.message });
        ws.close();
      }
      return;
    }

    if (!authed || !gemini) {
      send({ type: 'error', code: 'auth_failed', message: 'Not authenticated' });
      return;
    }

    if (msg.type === 'audio') {
      await gemini.sendAudio(msg.data);
      return;
    }

    if (msg.type === 'text') {
      const { capReached } = incrementTurn(uid);
      await gemini.sendText(msg.text);
      if (capReached) await closeSession('cap_reached');
      return;
    }

    if (msg.type === 'end_turn') {
      const { capReached } = incrementTurn(uid);
      await gemini.endTurn();
      if (capReached) await closeSession('cap_reached');
      return;
    }

    if (msg.type === 'disconnect') {
      await closeSession('user_disconnect');
      return;
    }
  });

  ws.on('close', async () => {
    log(uid ?? connId, 'WebSocket closed');
    if (authed && gemini) await closeSession('user_disconnect');
  });

  ws.on('error', (err) => {
    log(uid ?? connId, `WebSocket error: ${err.message}`);
  });
});

// Pull MAX_TURNS for session_ready message
const SESSION_CAPS_TURNS = 20;

// ── Graceful shutdown ─────────────────────────────────────────────────────────

process.on('SIGTERM', () => {
  log(null, 'SIGTERM — shutting down gracefully');
  wss.clients.forEach((client) => {
    if (client.readyState === client.OPEN) {
      client.send(JSON.stringify({ type: 'session_end', reason: 'server_restart' }));
      client.close();
    }
  });
  httpServer.close(() => { log(null, 'HTTP server closed'); process.exit(0); });
});

// ── Start ─────────────────────────────────────────────────────────────────────

httpServer.listen(CONFIG.PORT, () => {
  log(null, `ADAM Relay Server listening on port ${CONFIG.PORT}`);
  log(null, `Environment: ${CONFIG.NODE_ENV} | Origin: ${CONFIG.ALLOWED_ORIGIN}`);
});
