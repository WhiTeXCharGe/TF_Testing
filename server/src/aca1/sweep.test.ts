import { describe, it, expect, beforeEach, vi } from 'vitest';
import { createMemStorage } from '../collab/storage/memStorage.js';
import type { StorageClient } from '../collab/storage/storageClient.js';
import { createSessionRecord, writeStatus, loadSessionRecord } from '../collab/persistence.js';
import { runSweepOnce } from './sweep.js';
import type { Aca2Client } from './aca2Client.js';

const BASELINE = { schedule: {}, envConfig: {}, currentView: 'worker' as const };
const config = { absoluteSessionMaxMs: 8 * 3600_000, idleSessionTimeoutMs: 30 * 60_000, idleSweepMs: 60_000 };
const NOW = 1_000_000_000_000;

let storage: StorageClient;
let aca2: { activate: ReturnType<typeof vi.fn>; live: ReturnType<typeof vi.fn>; evict: ReturnType<typeof vi.fn> };

beforeEach(() => {
  storage = createMemStorage();
  aca2 = { activate: vi.fn(), live: vi.fn(), evict: vi.fn() };
});

const deps = () => ({ storage, aca2: aca2 as unknown as Aca2Client, config, now: NOW });

describe('runSweepOnce', () => {
  it('deletes a close session older than the absolute cap', async () => {
    const id = await createSessionRecord(storage, { name: 'old', baseline: BASELINE, ownerTokenHash: 'h' });
    await writeStatus(storage, id, { status: 'close', lastActivityAt: NOW - 9 * 3600_000 });
    const res = await runSweepOnce(deps());
    expect(res.deleted).toEqual([id]);
    expect(await loadSessionRecord(storage, id)).toBeNull();
  });

  it('keeps a recent close session', async () => {
    const id = await createSessionRecord(storage, { name: 'fresh', baseline: BASELINE, ownerTokenHash: 'h' });
    await writeStatus(storage, id, { status: 'close', lastActivityAt: NOW - 60_000 });
    const res = await runSweepOnce(deps());
    expect(res.deleted).toEqual([]);
    expect(await loadSessionRecord(storage, id)).not.toBeNull();
  });

  it('forces an idle open session to close when ACA2 says it is not active', async () => {
    const id = await createSessionRecord(storage, { name: 'orphan', baseline: BASELINE, ownerTokenHash: 'h' });
    await writeStatus(storage, id, { status: 'open', relayInstance: 'r9', relayUrl: 'http://r', lastActivityAt: NOW - 45 * 60_000 });
    aca2.live.mockResolvedValue({ active: false, participantCount: 0, status: 'open' });
    const res = await runSweepOnce(deps());
    expect(res.closed).toEqual([id]);
    const rec = await loadSessionRecord(storage, id);
    expect(rec?.status).toMatchObject({ status: 'close', relayInstance: null, relayUrl: null });
  });

  it('forces an idle open session to close when ACA2 is unreachable', async () => {
    const id = await createSessionRecord(storage, { name: 'orphan2', baseline: BASELINE, ownerTokenHash: 'h' });
    await writeStatus(storage, id, { status: 'lock', lastActivityAt: NOW - 45 * 60_000 });
    aca2.live.mockResolvedValue({ unreachable: true });
    const res = await runSweepOnce(deps());
    expect(res.closed).toEqual([id]);
  });

  it('leaves a live open session alone even when idle', async () => {
    const id = await createSessionRecord(storage, { name: 'live', baseline: BASELINE, ownerTokenHash: 'h' });
    await writeStatus(storage, id, { status: 'open', lastActivityAt: NOW - 45 * 60_000 });
    aca2.live.mockResolvedValue({ active: true, participantCount: 2, status: 'open' });
    const res = await runSweepOnce(deps());
    expect(res).toEqual({ deleted: [], closed: [] });
  });

  it('does not touch an open session that is not yet past the idle timeout', async () => {
    const id = await createSessionRecord(storage, { name: 'recent-open', baseline: BASELINE, ownerTokenHash: 'h' });
    await writeStatus(storage, id, { status: 'open', lastActivityAt: NOW - 5 * 60_000 });
    const res = await runSweepOnce(deps());
    expect(res).toEqual({ deleted: [], closed: [] });
    expect(aca2.live).not.toHaveBeenCalled();
  });
});
