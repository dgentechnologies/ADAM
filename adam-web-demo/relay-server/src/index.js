// index.js — ADAM Relay Server entry point
// WebSocket server — bridges browser ↔ Gemini Live. Auth via Firebase-minted relay JWT.

import 'dotenv/config';
import http from 'http';
import { WebSocketServer, WebSocket } from 'ws';
import { v4 as uuidv4 } from 'uuid';

import { CONFIG, SESSION_CAPS, isTester } from './config.js';
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

const wss = new WebSocketServer({
  server: httpServer,
  maxPayload: 1 * 1024 * 1024,
  perMessageDeflate: false,
});

const heartbeatInterval = setInterval(() => {
  wss.clients.forEach((client) => {
    if (client.isAlive === false) {
      client.terminate();
      return;
    }

    client.isAlive = false;
    client.ping();
  });
}, 30000);

wss.on('connection', (ws, req) => {
  const origin = req.headers.origin;
  if (CONFIG.NODE_ENV === 'production' && (!origin || !CONFIG.ALLOWED_ORIGINS.has(origin))) {
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
  let closing     = false;
  let persistedSessionEnd = false;

  ws.isAlive = true;
  ws.on('pong', () => {
    ws.isAlive = true;
  });

  const send = (obj) => {
    if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  };

  const normalizeCapError = (reason) => {
    if (reason === 'session_active') return 'An active session is already running on this account.';
    if (reason === 'lifetime_cap_reached') return 'You have already used your ADAM demo session. Join the waitlist to get the full experience: dgentechnologies.com/products/adam#waitlist';
    return 'Session unavailable right now. Please try again shortly.';
  };

  const closeSession = async (reason) => {
    if (closing) return;
    closing = true;

    log(uid, `Session ending — reason: ${reason}`);

    if (gemini) { gemini.close(); gemini = null; }

    const sessionSnapshot = uid ? getSession(uid) : null;

    if (uid) {
      clearMemory(uid);
      removeSession(uid);
    }

    if (!persistedSessionEnd && dbSessionId && sessionStart) {
      persistedSessionEnd = true;
      try {
        await endSession({
          sessionId:  dbSessionId,
          durationMs: Date.now() - sessionStart,
          turnCount:  sessionSnapshot?.turnCount ?? 0,
          endReason:  reason,
        });
      } catch (err) {
        log(uid, `endSession Firestore error: ${err.message}`);
      }
    }

    send({ type: 'session_end', reason });
    if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
      ws.close();
    }
  };

  // ── Message handler ─────────────────────────────────────────────────────────

  ws.on('message', async (raw) => {
    let msg;
    try { msg = JSON.parse(raw.toString()); }
    catch { send({ type: 'error', code: 'parse_error', message: 'Invalid JSON' }); return; }

    if (!msg || typeof msg !== 'object' || typeof msg.type !== 'string') {
      send({ type: 'error', code: 'protocol_error', message: 'Invalid message format' });
      return;
    }

    // ── AUTH ──────────────────────────────────────────────────────────────────
    if (msg.type === 'auth') {
      if (authed) return;
      try {
        if (typeof msg.token !== 'string') {
          throw new Error('auth_failed: token is required');
        }

        const payload = await validateToken(msg.token);
        uid       = payload.uid;
        userName  = payload.name;
        userEmail = payload.email;

        log(uid, `Auth successful — ${userName} <${userEmail}>`);

        // Upsert user doc in Firestore
        const userDoc = await upsertUser({ uid, email: userEmail, name: userName });

        // Check capacity — testers are unlimited; regular users get one lifetime session.
        const totalSessions = Number(userDoc.totalDemoSessions ?? 0);
        const { allowed, reason } = canStartSession(uid, totalSessions);

        if (!allowed) {
          send({ type: 'error', code: 'cap_exceeded', message: normalizeCapError(reason) });
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

        const tester = isTester(uid);
        send({
          type:         'session_ready',
          sessionId:    dbSessionId,
          turnsAllowed: tester ? 9999 : SESSION_CAPS.MAX_TURNS,
          durationMs:   remainingMs(uid),
        });

        gemini = await createGeminiSession({
          uid,
          userName,
          userEmail,
          userProfile: userDoc,
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
      if (typeof msg.data !== 'string' || msg.data.length === 0 || msg.data.length > 700000) {
        send({ type: 'error', code: 'bad_request', message: 'Invalid audio payload' });
        return;
      }

      await gemini.sendAudio(msg.data);
      return;
    }

    if (msg.type === 'text') {
      if (typeof msg.text !== 'string' || msg.text.trim().length === 0 || msg.text.length > 2000) {
        send({ type: 'error', code: 'bad_request', message: 'Invalid text payload' });
        return;
      }

      const { capReached } = incrementTurn(uid);
      await gemini.sendText(msg.text.trim());
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

    send({ type: 'error', code: 'unknown_message_type', message: `Unsupported message type: ${msg.type}` });
  });

  ws.on('close', async () => {
    log(uid ?? connId, 'WebSocket closed');
    if (authed) await closeSession('user_disconnect');
  });

  ws.on('error', (err) => {
    log(uid ?? connId, `WebSocket error: ${err.message}`);
  });
});

// ── Graceful shutdown ─────────────────────────────────────────────────────────

const shutdown = (signal) => {
  if (shutdown.started) return;
  shutdown.started = true;

  clearInterval(heartbeatInterval);

  log(null, `${signal} — shutting down gracefully`);
  wss.clients.forEach((client) => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(JSON.stringify({ type: 'session_end', reason: 'server_restart' }));
      client.close();
    }
  });

  httpServer.close(() => {
    log(null, 'HTTP server closed');
    process.exit(0);
  });
};

shutdown.started = false;

process.on('SIGTERM', () => {
  shutdown('SIGTERM');
});

process.on('SIGINT', () => {
  shutdown('SIGINT');
});

process.on('uncaughtException', (err) => {
  log(null, `Uncaught exception: ${err.message}`);
});

process.on('unhandledRejection', (reason) => {
  log(null, `Unhandled rejection: ${String(reason)}`);
});

httpServer.on('error', (err) => {
  log(null, `HTTP server error: ${err.message}`);
});

wss.on('error', (err) => {
  log(null, `WebSocket server error: ${err.message}`);
});

// ── Start ─────────────────────────────────────────────────────────────────────

httpServer.listen(CONFIG.PORT, () => {
  log(null, `ADAM Relay Server listening on port ${CONFIG.PORT}`);
  log(null, `Environment: ${CONFIG.NODE_ENV} | Origin(s): ${Array.from(CONFIG.ALLOWED_ORIGINS).join(',')}`);
});
