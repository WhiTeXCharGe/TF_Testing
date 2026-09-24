import { createHash, randomUUID, timingSafeEqual } from 'node:crypto';
import type { StorageClient } from './storage/storageClient.js';
import type {
  SessionBaseline, SessionMeta, SessionStatusRecord,
} from './types.js';

// Reads/writes the three files that make up a persisted session:
//   sessions/<id>/meta.json      (immutable)
//   sessions/<id>/status.json    (mutable: open|lock|close + relay pointer)
//   sessions/<id>/current.json   (the latest full SessionBaseline)
// No per-action log is persisted — a session's storage footprint is one
// snapshot, not a snapshot plus an ever-growing history, however many edits
// happen while it's live (the in-memory action log in sessionStore.ts still
// exists, but only to replay recent edits to a newly-joining client while
// the session is loaded; it's never written to storage). current.json is
// only ever refreshed wholesale — session creation, the last participant's
// leave-time checkpoint, an explicit session-data update, or a create-time
// overwrite (see sessionStore.replaceBaseline). Pure storage plumbing — no
// in-memory session state, no Socket.IO.

export const metaKey = (id: string): string => `sessions/${id}/meta.json`;
export const statusKey = (id: string): string => `sessions/${id}/status.json`;
export const currentKey = (id: string): string => `sessions/${id}/current.json`;

export function hashOwnerToken(token: string): string {
  return createHash('sha256').update(token).digest('hex');
}

/** Constant-time check that `presentedToken` hashes to `expectedHash`. */
export function ownerTokenMatches(presentedToken: string, expectedHash: string): boolean {
  const a = Buffer.from(hashOwnerToken(presentedToken), 'hex');
  const b = Buffer.from(expectedHash, 'hex');
  return a.length === b.length && a.length > 0 && timingSafeEqual(a, b);
}

export interface SessionRecord {
  meta: SessionMeta;
  baseline: SessionBaseline;
  status: SessionStatusRecord;
}

export async function createSessionRecord(
  s: StorageClient,
  args: { name: string; baseline: SessionBaseline; ownerTokenHash: string },
): Promise<string> {
  const id = randomUUID();
  const now = Date.now();
  const meta: SessionMeta = { id, name: args.name, createdAt: now, ownerTokenHash: args.ownerTokenHash };
  const status: SessionStatusRecord = {
    status: 'close', relayInstance: null, relayUrl: null, lastActivityAt: now, lastJoinAt: null,
  };
  await Promise.all([
    s.putJson(metaKey(id), meta),
    s.putJson(currentKey(id), args.baseline),
    s.putJson(statusKey(id), status),
  ]);
  return id;
}

export async function loadSessionRecord(s: StorageClient, id: string): Promise<SessionRecord | null> {
  const meta = await s.getJson<SessionMeta>(metaKey(id));
  if (!meta) return null;
  const [baseline, status] = await Promise.all([
    s.getJson<SessionBaseline>(currentKey(id)),
    s.getJson<SessionStatusRecord>(statusKey(id)),
  ]);
  if (!baseline) return null;
  return {
    meta,
    baseline,
    status: status ?? { status: 'close', relayInstance: null, relayUrl: null, lastActivityAt: meta.createdAt, lastJoinAt: null },
  };
}

// writeStatus is read-modify-write, so concurrent callers for the same
// session (e.g. markActivated from open racing markJoined from a socket join
// landing moments later) must not overlap — besides a lost-update race, two
// simultaneous renames onto the same status.json throw EPERM on Windows.
// A per-id promise chain serializes them without blocking other sessions.
const statusLocks = new Map<string, Promise<unknown>>();

function withStatusLock<T>(id: string, fn: () => Promise<T>): Promise<T> {
  const prior = statusLocks.get(id) ?? Promise.resolve();
  const run = prior.then(fn, fn);
  statusLocks.set(id, run.then(() => undefined, () => undefined));
  return run;
}

export async function writeStatus(
  s: StorageClient, id: string, patch: Partial<SessionStatusRecord>,
): Promise<SessionStatusRecord> {
  return withStatusLock(id, async () => {
    const current = (await s.getJson<SessionStatusRecord>(statusKey(id)))
      ?? { status: 'close' as const, relayInstance: null, relayUrl: null, lastActivityAt: Date.now(), lastJoinAt: null };
    const next: SessionStatusRecord = {
      ...current,
      ...patch,
      lastActivityAt: patch.lastActivityAt ?? Date.now(),
    };
    await s.putJson(statusKey(id), next);
    return next;
  });
}

export async function writeCurrentState(s: StorageClient, id: string, baseline: SessionBaseline): Promise<void> {
  await s.putJson(currentKey(id), baseline);
}

export async function listSessionIds(s: StorageClient): Promise<string[]> {
  const keys = await s.listPrefix('sessions/');
  const ids = new Set<string>();
  for (const k of keys) {
    const parts = k.split('/'); // sessions/<id>/<file>
    if (parts.length >= 3 && parts[0] === 'sessions') ids.add(parts[1]);
  }
  return [...ids];
}

export async function deleteSessionRecord(s: StorageClient, id: string): Promise<void> {
  await s.deletePrefix(`sessions/${id}`);
}
