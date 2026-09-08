import { createHash, randomUUID, timingSafeEqual } from 'node:crypto';
import type { StorageClient } from './storage/storageClient.js';
import type {
  SessionBaseline, LoggedAction, SessionMeta, SessionStatusRecord,
} from './types.js';

// Reads/writes the four files that make up a persisted session:
//   sessions/<id>/meta.json      (immutable)
//   sessions/<id>/status.json    (mutable: open|lock|close + relay pointer)
//   sessions/<id>/baseline.json
//   sessions/<id>/log.json       (ordered LoggedAction[])
// Pure storage plumbing — no in-memory session state, no Socket.IO.

export const metaKey = (id: string): string => `sessions/${id}/meta.json`;
export const statusKey = (id: string): string => `sessions/${id}/status.json`;
export const baselineKey = (id: string): string => `sessions/${id}/baseline.json`;
export const logKey = (id: string): string => `sessions/${id}/log.json`;

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
  log: LoggedAction[];
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
    status: 'close', relayInstance: null, relayUrl: null, lastActivityAt: now,
  };
  await Promise.all([
    s.putJson(metaKey(id), meta),
    s.putJson(baselineKey(id), args.baseline),
    s.putJson(logKey(id), [] satisfies LoggedAction[]),
    s.putJson(statusKey(id), status),
  ]);
  return id;
}

export async function loadSessionRecord(s: StorageClient, id: string): Promise<SessionRecord | null> {
  const meta = await s.getJson<SessionMeta>(metaKey(id));
  if (!meta) return null;
  const [baseline, log, status] = await Promise.all([
    s.getJson<SessionBaseline>(baselineKey(id)),
    s.getJson<LoggedAction[]>(logKey(id)),
    s.getJson<SessionStatusRecord>(statusKey(id)),
  ]);
  if (!baseline) return null;
  return {
    meta,
    baseline,
    log: log ?? [],
    status: status ?? { status: 'close', relayInstance: null, relayUrl: null, lastActivityAt: meta.createdAt },
  };
}

export async function writeStatus(
  s: StorageClient, id: string, patch: Partial<SessionStatusRecord>,
): Promise<SessionStatusRecord> {
  const current = (await s.getJson<SessionStatusRecord>(statusKey(id)))
    ?? { status: 'close' as const, relayInstance: null, relayUrl: null, lastActivityAt: Date.now() };
  const next: SessionStatusRecord = {
    ...current,
    ...patch,
    lastActivityAt: patch.lastActivityAt ?? Date.now(),
  };
  await s.putJson(statusKey(id), next);
  return next;
}

export async function writeLog(s: StorageClient, id: string, log: LoggedAction[]): Promise<void> {
  await s.putJson(logKey(id), log);
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
