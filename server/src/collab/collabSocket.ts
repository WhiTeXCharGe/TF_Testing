import type { Server as HttpServer } from 'node:http';
import { Server, Socket } from 'socket.io';
import { randomUUID } from 'node:crypto';
import type { SessionStore } from './sessionStore.js';
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

    const handleLeave = async (): Promise<void> => {
      if (!joinedSessionId) return;
      const sid = joinedSessionId;
      joinedSessionId = null; // guard against leave + disconnect double-firing
      const participants = store.removeParticipant(sid, participantId) ?? [];
      socket.to(sid).emit('presence', participants);
      if (store.participantCount(sid) === 0) {
        await store.evict(sid);      // flushes baseline+log to storage
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
