import express from 'express';
import cors from 'cors';
import { rateLimit } from 'express-rate-limit';
import type { StorageClient } from '../collab/storage/storageClient.js';
import type { AppConfig } from '../config.js';
import type { Aca2Client } from './aca2Client.js';
import { createSessionApiRouter } from './sessionApi.js';

export interface Aca1AppDeps {
  storage: StorageClient;
  aca2: Aca2Client;
  config: AppConfig;
}

// The ACA1 HTTP app: the public session API, no sockets. Kept as a factory so
// tests can drive it with supertest and a fake Aca2Client.
export function createAca1App(deps: Aca1AppDeps): express.Express {
  const app = express();
  app.set('trust proxy', 1); // one hop (ACA ingress) — so the rate limiter keys on the real client IP
  app.use(cors({ origin: deps.config.webOrigin ?? true }));
  app.use(express.json({ limit: '10mb' }));

  app.get('/api/health', (_req, res) => {
    res.json({ ok: true, role: 'aca1', time: new Date().toISOString() });
  });

  // LAN auto-discovery (see server/src/lan/discoveryBeacon.ts) only exists
  // for ROLE=local — ACA1 has a well-known URL already, so there's nothing to
  // discover. Answering with an always-empty list (not 404) keeps the client
  // able to call this route unconditionally without a role check of its own.
  app.get('/api/lan-hosts', (_req, res) => {
    res.json({ ok: true, hosts: [] });
  });

  // Abuse guard on session creation only (reads/opens are cheap).
  const createLimiter = rateLimit({
    windowMs: 60 * 60 * 1000,
    limit: deps.config.limits.createPerHourPerIp,
    standardHeaders: 'draft-7',
    legacyHeaders: false,
    message: { ok: false, error: 'too many sessions created from this address; try again later' },
  });
  app.post('/api/sessions', createLimiter);

  app.use('/api', createSessionApiRouter(deps));

  return app;
}
