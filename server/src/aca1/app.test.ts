import { describe, it, expect, beforeEach, vi } from 'vitest';
import request from 'supertest';
import { createMemStorage } from '../collab/storage/memStorage.js';
import type { StorageClient } from '../collab/storage/storageClient.js';
import { loadSessionRecord } from '../collab/persistence.js';
import { createAca1App } from './app.js';
import type { Aca2Client } from './aca2Client.js';
import type { AppConfig } from '../config.js';

const config = { webOrigin: null, instanceId: 'r1', publicRelayUrl: 'http://relay:4010' } as AppConfig;

let storage: StorageClient;
let aca2: {
  activate: ReturnType<typeof vi.fn>;
  live: ReturnType<typeof vi.fn>;
  evict: ReturnType<typeof vi.fn>;
};
let app: ReturnType<typeof createAca1App>;

beforeEach(() => {
  storage = createMemStorage();
  aca2 = {
    activate: vi.fn(async () => ({ relayUrl: 'http://relay:4010', status: 'open' as const })),
    live: vi.fn(async () => ({ unreachable: true as const })),
    evict: vi.fn(async () => {}),
  };
  app = createAca1App({ storage, aca2: aca2 as unknown as Aca2Client, config });
});

async function createJsonSession(name = 'Plan') {
  const res = await request(app)
    .post('/api/sessions')
    .send({ name, schedule: { a: 1 }, envConfig: { b: 2 }, currentView: 'worker' })
    .expect(200);
  return res.body as { ok: true; sessionId: string; ownerToken: string };
}

describe('POST /api/sessions', () => {
  it('creates a session from a JSON body and returns sessionId + ownerToken', async () => {
    const body = await createJsonSession();
    expect(body.sessionId).toMatch(/[0-9a-f-]{36}/);
    expect(body.ownerToken).toMatch(/^[0-9a-f]{48}$/);
    const rec = await loadSessionRecord(storage, body.sessionId);
    expect(rec?.meta.name).toBe('Plan');
    expect(rec?.baseline).toEqual({ schedule: { a: 1 }, envConfig: { b: 2 }, currentView: 'worker' });
    expect(rec?.status.status).toBe('close');
  });

  it('creates a session from a multipart 2-YAML upload', async () => {
    const res = await request(app)
      .post('/api/sessions')
      .field('name', 'FromYaml')
      .attach('schedule', Buffer.from('a: 1\n'), 'Schedule.yaml')
      .attach('envConfig', Buffer.from('b: 2\n'), 'EnvConfig.yaml')
      .expect(200);
    const rec = await loadSessionRecord(storage, res.body.sessionId);
    expect(rec?.meta.name).toBe('FromYaml');
    expect(rec?.baseline.schedule).toEqual({ a: 1 });
  });

  it('rejects a bad body with 400', async () => {
    await request(app).post('/api/sessions').send({ name: '' }).expect(400);
  });
});

describe('GET /api/sessions', () => {
  it('lists created sessions, newest first, status close, participantCount null', async () => {
    await createJsonSession('A');
    await createJsonSession('B');
    const res = await request(app).get('/api/sessions').expect(200);
    expect(res.body.sessions).toHaveLength(2);
    expect(res.body.sessions[0]).toMatchObject({ name: 'B', status: 'close', participantCount: null });
    expect(aca2.live).not.toHaveBeenCalled(); // skipped for close sessions
  });

  it('404 for an unknown id', async () => {
    await request(app).get('/api/sessions/nope').expect(404);
  });
});

describe('POST /api/sessions/:id/open', () => {
  it('calls aca2.activate and returns its relayUrl + status', async () => {
    const { sessionId } = await createJsonSession();
    const res = await request(app).post(`/api/sessions/${sessionId}/open`).expect(200);
    expect(res.body).toEqual({ ok: true, sessionId, relayUrl: 'http://relay:4010', status: 'open' });
    expect(aca2.activate).toHaveBeenCalledWith(sessionId);
  });

  it('maps aca2 notFound to 404', async () => {
    aca2.activate.mockResolvedValueOnce({ notFound: true });
    await request(app).post('/api/sessions/ghost/open').expect(404);
  });
});

describe('DELETE /api/sessions/:id', () => {
  it('403 without the owner token', async () => {
    const { sessionId } = await createJsonSession();
    await request(app).delete(`/api/sessions/${sessionId}`).expect(403);
  });

  it('removes the record and evicts on ACA2 with the owner token', async () => {
    const { sessionId, ownerToken } = await createJsonSession();
    await request(app).delete(`/api/sessions/${sessionId}`).set('x-owner-token', ownerToken).expect(200);
    expect(aca2.evict).toHaveBeenCalledWith(sessionId);
    expect(await loadSessionRecord(storage, sessionId)).toBeNull();
  });

  it('404 for an unknown id', async () => {
    await request(app).delete('/api/sessions/nope').set('x-owner-token', 'x').expect(404);
  });
});

describe('GET /api/health', () => {
  it('reports role aca1', async () => {
    const res = await request(app).get('/api/health').expect(200);
    expect(res.body).toMatchObject({ ok: true, role: 'aca1' });
  });
});
