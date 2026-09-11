import { describe, it, expect, afterEach } from 'vitest';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { loadConfig } from './config.js';
import { startServer, type RunningServer } from './index.js';

const servers: RunningServer[] = [];
const tmpDirs: string[] = [];

afterEach(async () => {
  await Promise.all(servers.splice(0).map((s) => s.close()));
  await Promise.all(tmpDirs.splice(0).map((d) => rm(d, { recursive: true, force: true })));
});

async function start(env: NodeJS.ProcessEnv): Promise<RunningServer> {
  const dir = await mkdtemp(join(tmpdir(), 'gantt-roleswitch-'));
  tmpDirs.push(dir);
  const srv = await startServer(loadConfig({ PORT: '0', STORAGE: 'fs', MOCK_BLOB_DIR: dir, ...env }));
  servers.push(srv);
  return srv;
}

describe('ROLE switch', () => {
  it('aca1 serves /api/health role aca1 and does NOT expose /internal/*', async () => {
    const srv = await start({ ROLE: 'aca1', INTERNAL_KEY: 'k', ACA2_URL: 'http://127.0.0.1:59999' });
    const health = await fetch(`http://localhost:${srv.port}/api/health`).then((r) => r.json());
    expect(health).toMatchObject({ ok: true, role: 'aca1' });
    const internal = await fetch(`http://localhost:${srv.port}/internal/sessions/x/live`);
    expect(internal.status).toBe(404);
  });

  it('aca2 serves /internal/* (401 without key) and /api/health role aca2', async () => {
    const srv = await start({ ROLE: 'aca2', INTERNAL_KEY: 'k' });
    const internal = await fetch(`http://localhost:${srv.port}/internal/sessions/x/live`);
    expect(internal.status).toBe(401);
    const health = await fetch(`http://localhost:${srv.port}/api/health`).then((r) => r.json());
    expect(health).toMatchObject({ ok: true, role: 'aca2' });
  });

  it('aca2 does NOT expose the local-only routes', async () => {
    const srv = await start({ ROLE: 'aca2', INTERNAL_KEY: 'k' });
    expect((await fetch(`http://localhost:${srv.port}/api/network-info`)).status).toBe(404);
  });

  it('local serves the local-file routes AND a self-contained session API', async () => {
    const srv = await start({ ROLE: 'local' });
    const health = await fetch(`http://localhost:${srv.port}/api/health`).then((r) => r.json());
    expect(health).toMatchObject({ ok: true, role: 'local' });
    expect((await fetch(`http://localhost:${srv.port}/api/network-info`)).status).toBe(200);

    // ACA1 session API is mounted here too (no separate ACA1/ACA2 processes).
    const listed = await fetch(`http://localhost:${srv.port}/api/sessions`).then((r) => r.json());
    expect(listed).toEqual({ ok: true, sessions: [] });

    const created = await fetch(`http://localhost:${srv.port}/api/sessions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ name: 'Local', schedule: { a: 1 }, envConfig: { b: 2 }, currentView: 'worker' }),
    }).then((r) => r.json());
    expect(created).toMatchObject({ ok: true });

    const opened = await fetch(`http://localhost:${srv.port}/api/sessions/${created.sessionId}/open`, { method: 'POST' })
      .then((r) => r.json());
    expect(opened).toMatchObject({ ok: true, status: 'open' });
  });

  it('local exposes /api/lan-hosts (empty, not an error, when discovery has nothing yet)', async () => {
    const srv = await start({ ROLE: 'local' });
    const res = await fetch(`http://localhost:${srv.port}/api/lan-hosts`);
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: true, hosts: [] });
  });
});
