import { Router } from 'express';
import type { SessionStore } from '../collab/sessionStore.js';
import type { AppConfig } from '../config.js';
import { broadcastResync, type IoRef } from '../collab/collabSocket.js';

// ACA1 -> ACA2 control plane. All routes assume internalAuth ran first.
// `ioRef` is populated with the socket server's `io` after startServer()
// creates it (see index.ts) — the router is built before that exists.
export function createInternalRouter(store: SessionStore, config: AppConfig, ioRef: IoRef = { current: null }): Router {
  const router = Router();

  // Load a session into memory on this replica and mark it live. Idempotent.
  router.post('/internal/sessions/:id/activate', async (req, res) => {
    const { id } = req.params;
    const ok = await store.activateFromStorage(id);
    if (!ok) {
      res.status(404).json({ ok: false, error: 'no such session' });
      return;
    }
    const status = await store.markActivated(id, {
      relayInstance: config.instanceId,
      relayUrl: config.publicRelayUrl,
    });
    res.json({ ok: true, relayUrl: config.publicRelayUrl, status });
  });

  // Liveness + participant count for ACA1's session list.
  router.get('/internal/sessions/:id/live', async (req, res) => {
    const live = await store.getLive(req.params.id);
    res.json({ ok: true, ...live });
  });

  // Flush + drop from memory + set the at-rest status to close. Used by ACA1
  // when the owner deletes a session.
  router.post('/internal/sessions/:id/evict', async (req, res) => {
    const { id } = req.params;
    await store.evict(id);
    await store.markClosed(id);
    res.json({ ok: true });
  });

  // Replace an existing session's whole baseline (clears its action log) and
  // broadcast a resync to anyone currently connected to it. Used for a
  // create-time overwrite of a session with a duplicate name. ACA1 already
  // validated the baseline shape before calling this.
  router.post('/internal/sessions/:id/replace', async (req, res) => {
    const { id } = req.params;
    const ok = await store.replaceBaseline(id, req.body);
    if (!ok) {
      res.status(404).json({ ok: false, error: 'no such session' });
      return;
    }
    if (ioRef.current) broadcastResync(ioRef.current, store, id);
    res.json({ ok: true });
  });

  return router;
}
