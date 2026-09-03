import { Router } from 'express';
import * as store from '../collab/sessionStore.js';

export const collabRouter = Router();

collabRouter.post('/collab/sessions', (req, res) => {
  const { name, schedule, envConfig, currentView } = req.body as {
    name?: string;
    schedule?: unknown;
    envConfig?: unknown;
    currentView?: 'worker' | 'device';
  };
  if (!name || !schedule || !envConfig || (currentView !== 'worker' && currentView !== 'device')) {
    res.status(400).json({ ok: false, error: 'name, schedule, envConfig, currentView are required' });
    return;
  }
  const sessionId = store.createSession(name, { schedule, envConfig, currentView });
  res.json({ ok: true, sessionId });
});

collabRouter.get('/collab/sessions/:id/name', (req, res) => {
  const name = store.getSessionName(req.params.id);
  if (!name) {
    res.status(404).json({ ok: false });
    return;
  }
  res.json({ ok: true, name });
});
