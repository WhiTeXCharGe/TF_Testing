import type { StorageClient } from './storage/storageClient.js';
import type { SessionBaseline, LoggedAction, SessionStatus, SessionStatusRecord } from './types.js';
import {
  loadSessionRecord, writeLog, writeStatus, statusKey,
} from './persistence.js';

// The relay's working set: sessions currently held in memory on this replica.
// Storage (StorageClient) is the durable record — a session is loaded from it
// on first join, flushed back periodically and on the last participant
// leaving, and dropped from memory ("evicted") when empty. The store still
// never interprets an action: it stores and orders them; the client reducer
// is the single source of truth.
//
// Session creation does NOT live here anymore — ACA1 writes the initial
// record via persistence.createSessionRecord. This store only ever *loads*
// existing records.

export type { SessionBaseline, LoggedAction } from './types.js';

export interface SessionParticipant {
  id: string;
  name: string;
  role: 'edit' | 'view';
}

/** What a joined client is handed for a session held in memory. */
export interface LoadedSessionView {
  name: string;
  baseline: SessionBaseline;
  actions: LoggedAction[];
  participants: SessionParticipant[];
  /** In memory a session is 'open' or 'lock'; 'close' only exists at rest. */
  status: Exclude<SessionStatus, 'close'>;
}

export interface SessionStoreDeps {
  storage: StorageClient;
}

export interface SessionStore {
  isLoaded(id: string): boolean;
  /** Load meta+baseline+log into memory if not already present. false = no such record. */
  activateFromStorage(id: string): Promise<boolean>;
  getSession(id: string): LoadedSessionView | null;
  appendAction(id: string, type: string, payload: unknown): LoggedAction | null;
  addParticipant(id: string, pid: string, name: string, role: 'edit' | 'view'): SessionParticipant[] | null;
  removeParticipant(id: string, pid: string): SessionParticipant[] | null;
  participantCount(id: string): number;
  /** Flip lock state of a loaded session. Returns the new status, or null if not loaded. */
  setLocked(id: string, locked: boolean): Exclude<SessionStatus, 'close'> | null;
  ownerTokenHash(id: string): string | null;
  /** Live view for ACA1's session list — reads storage when the session isn't loaded here. */
  getLive(id: string): Promise<{ active: boolean; participantCount: number; status: SessionStatus }>;
  /** Persist log + status for a loaded session if it has unsaved changes. */
  flush(id: string): Promise<void>;
  flushAll(): Promise<void>;
  /** Flush, then drop from memory. */
  evict(id: string): Promise<void>;
  /** Record this replica as the live host and set status open (or keep lock). */
  markActivated(id: string, relay: { relayInstance: string; relayUrl: string }): Promise<SessionStatus>;
  /** Set the at-rest status back to 'close' with no relay pointer. */
  markClosed(id: string): Promise<void>;
  /** Drop in-memory sessions that have had no participants past the idle window. */
  sweepIdleSessions(maxIdleMs: number, now?: number): number;
}

interface InMemSession {
  id: string;
  name: string;
  ownerTokenHash: string;
  baseline: SessionBaseline;
  actions: LoggedAction[];
  participants: Map<string, SessionParticipant>;
  nextSeq: number;
  status: Exclude<SessionStatus, 'close'>;
  dirty: boolean;
  lastActivityAt: number;
}

export function createSessionStore({ storage }: SessionStoreDeps): SessionStore {
  const sessions = new Map<string, InMemSession>();

  const isLoaded = (id: string): boolean => sessions.has(id);

  const activateFromStorage = async (id: string): Promise<boolean> => {
    if (sessions.has(id)) return true;
    const rec = await loadSessionRecord(storage, id);
    if (!rec) return false;
    sessions.set(id, {
      id,
      name: rec.meta.name,
      ownerTokenHash: rec.meta.ownerTokenHash,
      baseline: rec.baseline,
      actions: [...rec.log],
      participants: new Map(),
      nextSeq: (rec.log.at(-1)?.seq ?? -1) + 1,
      status: rec.status.status === 'lock' ? 'lock' : 'open',
      dirty: false,
      lastActivityAt: Date.now(),
    });
    return true;
  };

  const getSession = (id: string): LoadedSessionView | null => {
    const s = sessions.get(id);
    if (!s) return null;
    return {
      name: s.name,
      baseline: { ...s.baseline },
      actions: [...s.actions],
      participants: [...s.participants.values()],
      status: s.status,
    };
  };

  const appendAction = (id: string, type: string, payload: unknown): LoggedAction | null => {
    const s = sessions.get(id);
    if (!s) return null;
    const action: LoggedAction = { seq: s.nextSeq, type, payload };
    s.nextSeq += 1;
    s.actions.push(action);
    s.dirty = true;
    s.lastActivityAt = Date.now();
    return action;
  };

  const addParticipant = (
    id: string, pid: string, name: string, role: 'edit' | 'view',
  ): SessionParticipant[] | null => {
    const s = sessions.get(id);
    if (!s) return null;
    s.participants.set(pid, { id: pid, name, role });
    s.lastActivityAt = Date.now();
    return [...s.participants.values()];
  };

  const removeParticipant = (id: string, pid: string): SessionParticipant[] | null => {
    const s = sessions.get(id);
    if (!s) return null;
    s.participants.delete(pid);
    s.lastActivityAt = Date.now();
    return [...s.participants.values()];
  };

  const participantCount = (id: string): number => sessions.get(id)?.participants.size ?? 0;

  const setLocked = (id: string, locked: boolean): Exclude<SessionStatus, 'close'> | null => {
    const s = sessions.get(id);
    if (!s) return null;
    s.status = locked ? 'lock' : 'open';
    s.dirty = true;
    s.lastActivityAt = Date.now();
    return s.status;
  };

  const ownerTokenHash = (id: string): string | null => sessions.get(id)?.ownerTokenHash ?? null;

  const getLive = async (
    id: string,
  ): Promise<{ active: boolean; participantCount: number; status: SessionStatus }> => {
    const s = sessions.get(id);
    if (s) return { active: true, participantCount: s.participants.size, status: s.status };
    const rec = await storage.getJson<SessionStatusRecord>(statusKey(id));
    return { active: false, participantCount: 0, status: rec?.status ?? 'close' };
  };

  const flush = async (id: string): Promise<void> => {
    const s = sessions.get(id);
    if (!s || !s.dirty) return;
    await writeLog(storage, id, s.actions);
    await writeStatus(storage, id, { status: s.status, lastActivityAt: s.lastActivityAt });
    s.dirty = false;
  };

  const flushAll = async (): Promise<void> => {
    await Promise.all([...sessions.keys()].map((id) => flush(id)));
  };

  const evict = async (id: string): Promise<void> => {
    await flush(id);
    sessions.delete(id);
  };

  const markActivated = async (
    id: string, relay: { relayInstance: string; relayUrl: string },
  ): Promise<SessionStatus> => {
    const s = sessions.get(id);
    const status: SessionStatus = s?.status === 'lock' ? 'lock' : 'open';
    await writeStatus(storage, id, { status, relayInstance: relay.relayInstance, relayUrl: relay.relayUrl });
    return status;
  };

  const markClosed = async (id: string): Promise<void> => {
    await writeStatus(storage, id, { status: 'close', relayInstance: null, relayUrl: null });
  };

  const sweepIdleSessions = (maxIdleMs: number, now = Date.now()): number => {
    let removed = 0;
    for (const [id, s] of sessions) {
      if (s.participants.size === 0 && now - s.lastActivityAt > maxIdleMs) {
        sessions.delete(id);
        removed += 1;
      }
    }
    return removed;
  };

  return {
    isLoaded, activateFromStorage, getSession, appendAction,
    addParticipant, removeParticipant, participantCount, setLocked,
    ownerTokenHash, getLive, flush, flushAll, evict,
    markActivated, markClosed, sweepIdleSessions,
  };
}
