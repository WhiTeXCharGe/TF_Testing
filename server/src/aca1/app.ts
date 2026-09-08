import express from 'express';
import cors from 'cors';
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
  app.use(cors({ origin: deps.config.webOrigin ?? true }));
  app.use(express.json({ limit: '10mb' }));

  app.get('/api/health', (_req, res) => {
    res.json({ ok: true, role: 'aca1', time: new Date().toISOString() });
  });

  app.use('/api', createSessionApiRouter(deps));

  return app;
}
