import { Router } from 'express';
import * as store from '../collab/sessionStore.js';

export const collabRouter = Router();

collabRouter.post('/collab/sessions', (req, res) => {
  const { schedule, envConfig, currentView } = req.body as {
    schedule?: unknown;
    envConfig?: unknown;
    currentView?: 'worker' | 'device';
  };
  if (!schedule || !envConfig || (currentView !== 'worker' && currentView !== 'device')) {
    res.status(400).json({ ok: false, error: 'schedule, envConfig, currentView are required' });
    return;
  }
  const sessionId = store.createSession({ schedule, envConfig, currentView });
  res.json({ ok: true, sessionId });
});
