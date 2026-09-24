import type { Server as HttpServer } from 'node:http';
import { Server, Socket } from 'socket.io';
import { randomUUID } from 'node:crypto';
import type { SessionStore } from './sessionStore.js';
import type { SessionBaseline } from './types.js';
import type { AppConfig } from '../config.js';
import { ownerTokenMatches } from './persistence.js';

interface JoinPayload {
  sessionId: string;
  name: string;
  role: 'edit' | 'view';
  ownerToken?: string;
}

interface ActionPayload {
  type: string;
  payload: unknown;
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

// The collab socket server (and its `io`) is created after the ACA1/ACA2
// routes that need it are built (see index.ts), so `io` can't be passed to
// them directly at construction time — a mutable ref, populated once it
// exists and read lazily on each call, sidesteps reordering startServer().
// `current` stays null (broadcasts are just skipped) for any caller — tests
// included — that has no socket server at all.
export interface IoRef { current: Server | null }

/**
 * Broadcast a full resync to everyone currently in a session's room — used
 * whenever a session's baseline was just replaced (locked-session update, or
 * a create-time overwrite of an existing session), so connected clients
 * apply the new data instead of trying to replay old actions against it. A
 * no-op if the session isn't loaded (nobody could be connected to it then).
 */
export function broadcastResync(io: Server, store: SessionStore, id: string): void {
  const s = store.getSession(id);
  if (!s) return;
  io.to(id).emit('sync-init', {
    ok: true, name: s.name, baseline: s.baseline, actions: s.actions, participants: s.participants, status: s.status,
  });
}

export function createCollabSocketServer(
  httpServer: HttpServer,
  store: SessionStore,
  config: AppConfig,
): Server {
  const io = new Server(httpServer, {
    path: '/collab/socket.io',
    // Reflect any origin in local/LAN mode (webOrigin null); pin to the known
    // web origin once we know it (aca2 cloud build).
    cors: { origin: config.webOrigin ?? true },
    // Big enough for a full baseline / sync-init (schedules are a few MB of
    // JSON); caps a hostile oversized frame. Configurable via MAX_SOCKET_MB.
    maxHttpBufferSize: config.limits.maxSocketMessageBytes,
  });

  io.on('connection', (socket: Socket) => {
    const participantId = randomUUID();
    let joinedSessionId: string | null = null;
    let joinedRole: 'edit' | 'view' = 'view';
    let isOwner = false;

    socket.on('join', async ({ sessionId, name, role, ownerToken }: JoinPayload) => {
      const ok = store.isLoaded(sessionId) || (await store.activateFromStorage(sessionId));
      if (!ok) {
        socket.emit('sync-init', { ok: false });
        return;
      }
      // Cap participants per session (owner-token holder is let past so a
      // creator can always reach their own full session).
      const ownerHash = store.ownerTokenHash(sessionId);
      isOwner = !!ownerToken && !!ownerHash && ownerTokenMatches(ownerToken, ownerHash);
      if (!isOwner && store.participantCount(sessionId) >= config.limits.maxParticipants) {
        socket.emit('sync-init', { ok: false, error: 'session is full' });
        return;
      }
      joinedSessionId = sessionId;
      joinedRole = role;
      void socket.join(sessionId);
      const participants = store.addParticipant(sessionId, participantId, name, role) ?? [];
      void store.markJoined(sessionId); // stamp status.json for the session-list sort
      const s = store.getSession(sessionId)!;
      socket.emit('sync-init', {
        ok: true,
        name: s.name,
        baseline: s.baseline,
        actions: s.actions,
        participants,
        status: s.status,
      });
      socket.to(sessionId).emit('presence', participants);
    });

    socket.on('action', ({ type, payload }: ActionPayload) => {
      if (!joinedSessionId || joinedRole !== 'edit') return;
      // A locked session is read-only for everyone.
      if (store.getSession(joinedSessionId)?.status === 'lock') return;
      const logged = store.appendAction(joinedSessionId, type, payload);
      if (!logged) return;
      socket.to(joinedSessionId).emit('action', { type: logged.type, payload: logged.payload });
    });

    // A best-effort final snapshot from the last connected editor before they
    // leave (see AppContext.leaveCollabSession) — this is how current.json
    // in storage picks up whatever was edited during the session, since
    // in-progress edits are never durably logged (see persistence.ts).
    // Edit-role only, same as 'action'; a malformed payload is dropped rather
    // than persisted. If this never fires (abrupt disconnect, or the last
    // participant left was view-only), current.json simply stays at
    // whatever it was as of the last checkpoint/creation/update.
    socket.on('checkpoint', (payload: SessionBaseline) => {
      if (!joinedSessionId || joinedRole !== 'edit') return;
      if (!isPlainObject(payload) || !payload.schedule || !payload.envConfig) return;
      void store.replaceBaseline(joinedSessionId, payload);
    });

    // Lock / unlock is open to any participant in the session — it's a shared
    // "freeze editing" toggle, not an ownership control.
    const setLock = async (locked: boolean): Promise<void> => {
      if (!joinedSessionId) return;
      const status = store.setLocked(joinedSessionId, locked);
      if (!status) return;
      await store.flush(joinedSessionId);
      io.to(joinedSessionId).emit('session-status', { status });
    };
    socket.on('lock', () => void setLock(true));
    socket.on('unlock', () => void setLock(false));

    // Explicit "replace this session's whole data" push, gated to locked
    // sessions only (so nobody's mid-edit when it happens) — open to any
    // participant, same as lock/unlock, since it's a session-wide admin
    // action rather than an incremental edit. Broadcasts a fresh sync-init to
    // the whole room (including the sender) so everyone's local editor
    // re-syncs from the new baseline instead of trying to replay old actions
    // against unrelated data. Stays locked afterward — someone still has to
    // explicitly unlock.
    socket.on('session-update', async (payload: SessionBaseline) => {
      if (!joinedSessionId) return;
      if (store.getSession(joinedSessionId)?.status !== 'lock') return;
      if (!isPlainObject(payload) || !payload.schedule || !payload.envConfig) return;
      const ok = await store.replaceBaseline(joinedSessionId, payload);
      if (!ok) return;
      broadcastResync(io, store, joinedSessionId);
    });

    const handleLeave = async (): Promise<void> => {
      if (!joinedSessionId) return;
      const sid = joinedSessionId;
      joinedSessionId = null; // guard against leave + disconnect double-firing
      const participants = store.removeParticipant(sid, participantId) ?? [];
      socket.to(sid).emit('presence', participants);
      if (store.participantCount(sid) === 0) {
        await store.evict(sid);      // flushes status.json (no durable action log)
        await store.markClosed(sid); // status.json → close, relay pointer cleared
      }
    };

    socket.on('leave', () => void handleLeave());
    socket.on('disconnect', () => void handleLeave());
  });

  const sweep = setInterval(() => {
    store.sweepIdleSessions(config.idleSessionTimeoutMs);
    void store.flushAll();
  }, config.idleSweepMs);
  sweep.unref();

  return io;
}
