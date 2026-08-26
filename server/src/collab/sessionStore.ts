import { randomUUID } from 'node:crypto';

// Pure, in-memory session/action-log store — no I/O, no Socket.IO here, so
// this can be unit-tested directly. Deliberately dumb: it stores and orders
// actions, it never interprets them — the reducer (shared, already tested,
// on every client) is the single source of truth for what an action does.

export interface SessionBaseline {
  schedule: unknown;
  envConfig: unknown;
  currentView: 'worker' | 'device';
}

export interface LoggedAction {
  seq: number;
  type: string;
  payload: unknown;
}

export interface SessionParticipant {
  id: string;
  name: string;
  role: 'edit' | 'view';
}

interface CollabSession {
  id: string;
  baseline: SessionBaseline;
  actions: LoggedAction[];
  participants: Map<string, SessionParticipant>;
  nextSeq: number;
  lastActivityAt: number;
}

const sessions = new Map<string, CollabSession>();

export function createSession(baseline: SessionBaseline): string {
  const id = randomUUID();
  sessions.set(id, {
    id,
    baseline,
    actions: [],
    participants: new Map(),
    nextSeq: 0,
    lastActivityAt: Date.now(),
  });
  return id;
}

export function getSession(
  id: string,
): { baseline: SessionBaseline; actions: LoggedAction[]; participants: SessionParticipant[] } | null {
  const session = sessions.get(id);
  if (!session) return null;
  return { baseline: session.baseline, actions: session.actions, participants: [...session.participants.values()] };
}

export function appendAction(sessionId: string, type: string, payload: unknown): LoggedAction | null {
  const session = sessions.get(sessionId);
  if (!session) return null;
  const action: LoggedAction = { seq: session.nextSeq, type, payload };
  session.nextSeq += 1;
  session.actions.push(action);
  session.lastActivityAt = Date.now();
  return action;
}

export function addParticipant(
  sessionId: string, participantId: string, name: string, role: 'edit' | 'view',
): SessionParticipant[] | null {
  const session = sessions.get(sessionId);
  if (!session) return null;
  session.participants.set(participantId, { id: participantId, name, role });
  session.lastActivityAt = Date.now();
  return [...session.participants.values()];
}

export function removeParticipant(sessionId: string, participantId: string): SessionParticipant[] | null {
  const session = sessions.get(sessionId);
  if (!session) return null;
  session.participants.delete(participantId);
  session.lastActivityAt = Date.now();
  return [...session.participants.values()];
}

export function sweepIdleSessions(maxIdleMs: number, now = Date.now()): number {
  let removed = 0;
  for (const [id, session] of sessions) {
    if (session.participants.size === 0 && now - session.lastActivityAt > maxIdleMs) {
      sessions.delete(id);
      removed += 1;
    }
  }
  return removed;
}

/** Test-only: clears all sessions so tests don't leak state into each other. */
export function _resetForTests(): void {
  sessions.clear();
}
