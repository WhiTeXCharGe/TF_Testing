import { describe, it, expect, beforeEach } from 'vitest';
import express from 'express';
import request from 'supertest';
import { createMemStorage } from '../collab/storage/memStorage.js';
import { createSessionStore } from '../collab/sessionStore.js';
import { createSessionRecord, hashOwnerToken } from '../collab/persistence.js';
import { createInternalRouter } from './internal.js';
import { internalAuth } from '../internalAuth.js';
import type { AppConfig } from '../config.js';

const config = { internalKey: 'topsecret', instanceId: 'replica-1', publicRelayUrl: 'http://relay:4010' } as AppConfig;
const BASELINE = { schedule: {}, envConfig: {}, currentView: 'worker' as const };

let app: express.Express;
let id: string;

beforeEach(async () => {
  const storage = createMemStorage();
  id = await createSessionRecord(storage, { name: 'S', baseline: BASELINE, ownerTokenHash: hashOwnerToken('o') });
  const store = createSessionStore({ storage });
  app = express();
  app.use(express.json());
  app.use(internalAuth(config.internalKey));
  app.use(createInternalRouter(store, config));
});

describe('internal routes', () => {
  it('401 without the internal key', async () => {
    await request(app).get(`/internal/sessions/${id}/live`).expect(401);
  });

  it('activate loads the session and writes status open + relay info', async () => {
    const res = await request(app)
      .post(`/internal/sessions/${id}/activate`)
      .set('x-internal-key', 'topsecret')
      .expect(200);
    expect(res.body).toMatchObject({ ok: true, relayUrl: 'http://relay:4010', status: 'open' });
  });

  it('live reports active after activate', async () => {
    await request(app).post(`/internal/sessions/${id}/activate`).set('x-internal-key', 'topsecret');
    const res = await request(app)
      .get(`/internal/sessions/${id}/live`)
      .set('x-internal-key', 'topsecret')
      .expect(200);
    expect(res.body).toMatchObject({ ok: true, active: true, participantCount: 0, status: 'open' });
  });

  it('activate on an unknown id → 404', async () => {
    await request(app)
      .post('/internal/sessions/does-not-exist/activate')
      .set('x-internal-key', 'topsecret')
      .expect(404);
  });

  it('evict sets status close', async () => {
    await request(app).post(`/internal/sessions/${id}/activate`).set('x-internal-key', 'topsecret');
    await request(app).post(`/internal/sessions/${id}/evict`).set('x-internal-key', 'topsecret').expect(200);
    const res = await request(app).get(`/internal/sessions/${id}/live`).set('x-internal-key', 'topsecret').expect(200);
    expect(res.body.status).toBe('close');
    expect(res.body.active).toBe(false);
  });
});
