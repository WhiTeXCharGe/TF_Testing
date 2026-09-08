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

  it('local serves the collab + network-info routes and role local', async () => {
    const srv = await start({ ROLE: 'local' });
    const health = await fetch(`http://localhost:${srv.port}/api/health`).then((r) => r.json());
    expect(health).toMatchObject({ ok: true, role: 'local' });
    expect((await fetch(`http://localhost:${srv.port}/api/network-info`)).status).toBe(200);
  });
});
