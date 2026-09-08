import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { createServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { io as ioClient, type Socket as ClientSocket } from 'socket.io-client';
import { loadConfig } from '../config.js';
import { startServer, type RunningServer } from '../index.js';
import { createFsStorage } from '../collab/storage/fsStorage.js';
import { loadSessionRecord } from '../collab/persistence.js';

// A free TCP port, released before use (a small race, fine for a local test).
function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = createServer();
    srv.listen(0, () => {
      const { port } = srv.address() as AddressInfo;
      srv.close((err) => (err ? reject(err) : resolve(port)));
    });
  });
}

const INTERNAL_KEY = 'integration-internal-key';
const BASELINE_BODY = { name: 'Integration Plan', schedule: { tasks: [{ id: 't1' }] }, envConfig: { workers: [] }, currentView: 'worker' as const };

let blobDir: string;
let aca1: RunningServer;
let aca2: RunningServer;
let aca1Url: string;
let aca2Port: number;

beforeAll(async () => {
  blobDir = await mkdtemp(join(tmpdir(), 'gantt-localflow-'));
  aca2Port = await freePort();
  aca2 = await startServer(loadConfig({
    ROLE: 'aca2', PORT: String(aca2Port), STORAGE: 'fs', MOCK_BLOB_DIR: blobDir,
    INTERNAL_KEY, PUBLIC_RELAY_URL: `http://localhost:${aca2Port}`,
  }));
  aca1 = await startServer(loadConfig({
    ROLE: 'aca1', PORT: '0', STORAGE: 'fs', MOCK_BLOB_DIR: blobDir,
    INTERNAL_KEY, ACA2_URL: `http://localhost:${aca2Port}`,
  }));
  aca1Url = `http://localhost:${aca1.port}`;
});

afterAll(async () => {
  await aca1?.close();
  await aca2?.close();
  await rm(blobDir, { recursive: true, force: true });
});

const api = (path: string, init?: RequestInit) => fetch(`${aca1Url}${path}`, init);

function socket(): ClientSocket {
  return ioClient(`http://localhost:${aca2Port}`, { path: '/collab/socket.io', transports: ['websocket'] });
}

function joinAndSync(
  client: ClientSocket,
  payload: { sessionId: string; name: string; role: 'edit' | 'view'; ownerToken?: string },
): Promise<any> {
  return new Promise((resolve) => {
    client.on('connect', () => client.emit('join', payload));
    client.on('sync-init', resolve);
  });
}

const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

describe('local ACA1 + ACA2 + fs-blob end-to-end', () => {
  it('runs the full session lifecycle', async () => {
    // 1. create (desktop JSON path)
    const created = await api('/api/sessions', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(BASELINE_BODY),
    }).then((r) => r.json());
    expect(created.ok).toBe(true);
    const { sessionId, ownerToken } = created as { sessionId: string; ownerToken: string };

    // 2. list — one entry, not yet live
    const list1 = await api('/api/sessions').then((r) => r.json());
    expect(list1.sessions).toHaveLength(1);
    expect(list1.sessions[0]).toMatchObject({ id: sessionId, name: 'Integration Plan', status: 'close' });

    // 3. open — ACA1 wakes ACA2
    const opened = await api(`/api/sessions/${sessionId}/open`, { method: 'POST' }).then((r) => r.json());
    expect(opened).toMatchObject({ ok: true, sessionId, relayUrl: `http://localhost:${aca2Port}`, status: 'open' });

    // 4 + 5. two participants join the relay
    const a = socket();
    const b = socket();
    const aSync = joinAndSync(a, { sessionId, name: 'A', role: 'edit', ownerToken });
    const bSync = joinAndSync(b, { sessionId, name: 'B', role: 'edit' });
    expect(await aSync).toMatchObject({ ok: true, status: 'open', baseline: BASELINE_BODY.schedule ? expect.anything() : undefined });
    const bInit = await bSync;
    expect(bInit.baseline).toEqual({ schedule: BASELINE_BODY.schedule, envConfig: BASELINE_BODY.envConfig, currentView: 'worker' });

    // 6. edit propagates A -> B
    const bGotEdit = new Promise<any>((resolve) => b.on('action', resolve));
    a.emit('action', { type: 'SET_SCHEDULE', payload: { tasks: [{ id: 't1' }, { id: 't2' }] } });
    expect(await bGotEdit).toEqual({ type: 'SET_SCHEDULE', payload: { tasks: [{ id: 't1' }, { id: 't2' }] } });

    // 7. owner locks; both see it; ACA1 list reflects it
    const aLocked = new Promise<any>((resolve) => a.on('session-status', resolve));
    const bLocked = new Promise<any>((resolve) => b.on('session-status', resolve));
    a.emit('lock');
    expect(await aLocked).toEqual({ status: 'lock' });
    expect(await bLocked).toEqual({ status: 'lock' });
    const afterLock = await api(`/api/sessions/${sessionId}`).then((r) => r.json());
    expect(afterLock.session.status).toBe('lock');

    // 8. while locked, edits are dropped
    let aGotWhileLocked = false;
    a.on('action', () => { aGotWhileLocked = true; });
    b.emit('action', { type: 'SET_SCHEDULE', payload: { tasks: [] } });
    await wait(200);
    expect(aGotWhileLocked).toBe(false);

    // 9. unlock re-enables propagation
    await new Promise<void>((resolve) => { a.on('session-status', () => resolve()); a.emit('unlock'); });
    const bGotAfterUnlock = new Promise<any>((resolve) => b.on('action', resolve));
    a.emit('action', { type: 'UPDATE_PLAN_RANGE', payload: { startDate: '2026-01-01', endDate: '2026-03-31' } });
    expect(await bGotAfterUnlock).toEqual({ type: 'UPDATE_PLAN_RANGE', payload: { startDate: '2026-01-01', endDate: '2026-03-31' } });

    // 10. everyone leaves -> ACA2 flushes + closes
    a.disconnect();
    b.disconnect();
    let sess = await api(`/api/sessions/${sessionId}`).then((r) => r.json());
    for (let i = 0; i < 80 && sess.session.status !== 'close'; i++) {
      await wait(25);
      sess = await api(`/api/sessions/${sessionId}`).then((r) => r.json());
    }
    expect(sess.session.status).toBe('close');

    // 11. the log is on disk
    const rec = await loadSessionRecord(createFsStorage(blobDir), sessionId);
    expect(rec?.log.map((x) => x.type)).toEqual(['SET_SCHEDULE', 'UPDATE_PLAN_RANGE']);

    // 12. re-open replays from storage
    const reopened = await api(`/api/sessions/${sessionId}/open`, { method: 'POST' }).then((r) => r.json());
    expect(reopened.status).toBe('open');
    const c = socket();
    const cInit = await joinAndSync(c, { sessionId, name: 'C', role: 'edit' });
    expect(cInit.actions.map((x: any) => x.type)).toEqual(['SET_SCHEDULE', 'UPDATE_PLAN_RANGE']);
    c.disconnect();
    await wait(300);

    // 13. delete needs the owner token, then the list is empty
    await api(`/api/sessions/${sessionId}`, { method: 'DELETE' }).then((r) => expect(r.status).toBe(403));
    const del = await api(`/api/sessions/${sessionId}`, { method: 'DELETE', headers: { 'x-owner-token': ownerToken } });
    expect(del.status).toBe(200);
    const list2 = await api('/api/sessions').then((r) => r.json());
    expect(list2.sessions).toEqual([]);
  }, 30_000);
});
