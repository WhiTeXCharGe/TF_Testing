import type { Server as HttpServer } from 'node:http';
import { Server, Socket } from 'socket.io';
import { randomUUID } from 'node:crypto';
import * as store from './sessionStore.js';

interface JoinPayload {
  sessionId: string;
  name: string;
  role: 'edit' | 'view';
}

interface ActionPayload {
  type: string;
  payload: unknown;
}

const IDLE_SWEEP_INTERVAL_MS = 5 * 60 * 1000;
const IDLE_SESSION_TIMEOUT_MS = 30 * 60 * 1000;

export function createCollabSocketServer(httpServer: HttpServer): Server {
  const io = new Server(httpServer, {
    path: '/collab/socket.io',
    cors: { origin: true },
    maxHttpBufferSize: 20 * 1024 * 1024, // schedules can be a few MB of JSON
  });

  io.on('connection', (socket: Socket) => {
    const participantId = randomUUID();
    let joinedSessionId: string | null = null;
    let joinedRole: 'edit' | 'view' = 'view';

    socket.on('join', ({ sessionId, name, role }: JoinPayload) => {
      const session = store.getSession(sessionId);
      if (!session) {
        socket.emit('sync-init', { ok: false });
        return;
      }
      joinedSessionId = sessionId;
      joinedRole = role;
      void socket.join(sessionId);
      const participants = store.addParticipant(sessionId, participantId, name, role) ?? [];
      socket.emit('sync-init', { ok: true, baseline: session.baseline, actions: session.actions, participants });
      socket.to(sessionId).emit('presence', participants);
    });

    socket.on('action', ({ type, payload }: ActionPayload) => {
      if (!joinedSessionId || joinedRole !== 'edit') return;
      const logged = store.appendAction(joinedSessionId, type, payload);
      if (!logged) return;
      socket.to(joinedSessionId).emit('action', { type: logged.type, payload: logged.payload });
    });

    const handleLeave = () => {
      if (!joinedSessionId) return;
      const participants = store.removeParticipant(joinedSessionId, participantId) ?? [];
      socket.to(joinedSessionId).emit('presence', participants);
      joinedSessionId = null;
    };

    socket.on('leave', handleLeave);
    socket.on('disconnect', handleLeave);
  });

  setInterval(() => store.sweepIdleSessions(IDLE_SESSION_TIMEOUT_MS), IDLE_SWEEP_INTERVAL_MS).unref();

  return io;
}
