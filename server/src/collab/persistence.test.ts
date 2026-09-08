import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createFsStorage } from './storage/fsStorage.js';
import {
  createSessionRecord, loadSessionRecord, writeStatus, writeLog,
  listSessionIds, deleteSessionRecord, hashOwnerToken,
} from './persistence.js';

let root: string;
let s: ReturnType<typeof createFsStorage>;

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), 'gantt-persist-'));
  s = createFsStorage(root);
});
afterEach(() => rm(root, { recursive: true, force: true }));

const baseline = { schedule: { t: 1 }, envConfig: { e: 2 }, currentView: 'worker' as const };

describe('persistence', () => {
  it('creates then loads a full record with status=close and empty log', async () => {
    const id = await createSessionRecord(s, { name: 'Plan A', baseline, ownerTokenHash: hashOwnerToken('secret') });
    const rec = await loadSessionRecord(s, id);
    expect(rec?.meta.name).toBe('Plan A');
    expect(rec?.meta.id).toBe(id);
    expect(rec?.meta.ownerTokenHash).toBe(hashOwnerToken('secret'));
    expect(rec?.status.status).toBe('close');
    expect(rec?.status.relayInstance).toBeNull();
    expect(rec?.baseline).toEqual(baseline);
    expect(rec?.log).toEqual([]);
  });

  it('loadSessionRecord returns null for unknown id', async () => {
    expect(await loadSessionRecord(s, 'nope')).toBeNull();
  });

  it('writeStatus merges and bumps lastActivityAt', async () => {
    const id = await createSessionRecord(s, { name: 'x', baseline, ownerTokenHash: 'h' });
    const before = (await loadSessionRecord(s, id))!.status.lastActivityAt;
    await new Promise((r) => setTimeout(r, 3));
    const next = await writeStatus(s, id, { status: 'open', relayInstance: 'r1', relayUrl: 'http://r' });
    expect(next.status).toBe('open');
    expect(next.relayInstance).toBe('r1');
    expect(next.relayUrl).toBe('http://r');
    expect(next.lastActivityAt).toBeGreaterThan(before);
    expect((await loadSessionRecord(s, id))!.status).toEqual(next);
  });

  it('writeLog persists and reloads; nextSeq derivable from it', async () => {
    const id = await createSessionRecord(s, { name: 'x', baseline, ownerTokenHash: 'h' });
    await writeLog(s, id, [
      { seq: 0, type: 'SET_SCHEDULE', payload: { a: 1 } },
      { seq: 1, type: 'UPDATE_PLAN_RANGE', payload: { startDate: 'x', endDate: 'y' } },
    ]);
    expect((await loadSessionRecord(s, id))!.log).toHaveLength(2);
  });

  it('listSessionIds returns created ids; deleteSessionRecord removes them', async () => {
    const a = await createSessionRecord(s, { name: 'a', baseline, ownerTokenHash: 'h' });
    const b = await createSessionRecord(s, { name: 'b', baseline, ownerTokenHash: 'h' });
    expect((await listSessionIds(s)).sort()).toEqual([a, b].sort());
    await deleteSessionRecord(s, a);
    expect(await listSessionIds(s)).toEqual([b]);
    expect(await loadSessionRecord(s, a)).toBeNull();
  });

  it('hashOwnerToken is stable and differs per token', () => {
    expect(hashOwnerToken('abc')).toBe(hashOwnerToken('abc'));
    expect(hashOwnerToken('abc')).not.toBe(hashOwnerToken('abd'));
    expect(hashOwnerToken('abc')).toMatch(/^[0-9a-f]{64}$/);
  });
});
