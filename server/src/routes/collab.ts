import { Router } from 'express';
import type { StorageClient } from '../collab/storage/storageClient.js';
import { createSessionRecord, hashOwnerToken, metaKey } from '../collab/persistence.js';

// The routes the local/Electron frontend calls to start and name a session.
// In the cloud split these move to ACA1 (server/src/aca1); here they stay so
// the packaged desktop app keeps working unchanged. Session state now goes
// through a StorageClient (in-memory for ROLE=local) instead of a module Map.
export function createLocalCollabRouter(deps: { storage: StorageClient }): Router {
  const router = Router();

  router.post('/collab/sessions', async (req, res) => {
    const { name, schedule, envConfig, currentView } = (req.body ?? {}) as {
      name?: string;
      schedule?: unknown;
      envConfig?: unknown;
      currentView?: 'worker' | 'device';
    };
    if (!name?.trim() || !schedule || !envConfig || (currentView !== 'worker' && currentView !== 'device')) {
      res.status(400).json({ ok: false, error: 'name, schedule, envConfig, currentView are required' });
      return;
    }
    const sessionId = await createSessionRecord(deps.storage, {
      name: name.trim(),
      baseline: { schedule, envConfig, currentView },
      // Local mode has no owner-lock flow; a fixed hash keeps the record shape valid.
      ownerTokenHash: hashOwnerToken('local'),
    });
    res.json({ ok: true, sessionId });
  });

  router.get('/collab/sessions/:id/name', async (req, res) => {
    const meta = await deps.storage.getJson<{ name: string }>(metaKey(req.params.id));
    if (!meta) {
      res.status(404).json({ ok: false });
      return;
    }
    res.json({ ok: true, name: meta.name });
  });

  return router;
}
