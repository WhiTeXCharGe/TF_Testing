import { describe, it, expect, beforeEach, vi } from 'vitest';
import express from 'express';
import request from 'supertest';
import { createMemStorage } from '../collab/storage/memStorage.js';
import { createSessionStore, type SessionStore } from '../collab/sessionStore.js';
import { createSessionRecord, hashOwnerToken, loadSessionRecord } from '../collab/persistence.js';
import type { StorageClient } from '../collab/storage/storageClient.js';
import { createInternalRouter } from './internal.js';
import { internalAuth } from '../internalAuth.js';
import type { AppConfig } from '../config.js';
import type { IoRef } from '../collab/collabSocket.js';

const config = { internalKey: 'topsecret', instanceId: 'replica-1', publicRelayUrl: 'http://relay:4010' } as AppConfig;
const BASELINE = { schedule: {}, envConfig: {}, currentView: 'worker' as const };

let app: express.Express;
let id: string;
let storage: StorageClient;
let store: SessionStore;
let ioRef: IoRef;

beforeEach(async () => {
  storage = createMemStorage();
  id = await createSessionRecord(storage, { name: 'S', baseline: BASELINE, ownerTokenHash: hashOwnerToken('o') });
  store = createSessionStore({ storage });
  ioRef = { current: null };
  app = express();
  app.use(express.json());
  app.use(internalAuth(config.internalKey));
  app.use(createInternalRouter(store, config, ioRef));
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

  describe('replace', () => {
    const NEW_BASELINE = { schedule: { updated: true }, envConfig: { updated: true }, currentView: 'device' as const };

    it('replaces the baseline and clears the log, loading from storage first if idle', async () => {
      const res = await request(app)
        .post(`/internal/sessions/${id}/replace`)
        .set('x-internal-key', 'topsecret')
        .send(NEW_BASELINE)
        .expect(200);
      expect(res.body).toEqual({ ok: true });
      const rec = await loadSessionRecord(storage, id);
      expect(rec?.baseline).toEqual(NEW_BASELINE);
      expect(rec?.log).toEqual([]);
    });

    it('404s for an unknown id', async () => {
      await request(app)
        .post('/internal/sessions/does-not-exist/replace')
        .set('x-internal-key', 'topsecret')
        .send(NEW_BASELINE)
        .expect(404);
    });

    it('broadcasts a resync via ioRef when populated', async () => {
      const emit = vi.fn();
      const to = vi.fn(() => ({ emit }));
      ioRef.current = { to } as never;

      await request(app)
        .post(`/internal/sessions/${id}/replace`)
        .set('x-internal-key', 'topsecret')
        .send(NEW_BASELINE)
        .expect(200);

      expect(to).toHaveBeenCalledWith(id);
      expect(emit).toHaveBeenCalledWith('sync-init', expect.objectContaining({ ok: true, baseline: NEW_BASELINE, actions: [] }));
    });

    it('does not throw when ioRef has no live socket server (ACA1-role-only usage)', async () => {
      await request(app)
        .post(`/internal/sessions/${id}/replace`)
        .set('x-internal-key', 'topsecret')
        .send(NEW_BASELINE)
        .expect(200);
    });
  });
});
