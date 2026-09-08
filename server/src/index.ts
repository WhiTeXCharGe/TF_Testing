import express from 'express';
import cors from 'cors';
import path from 'node:path';
import { createServer, Server as HttpServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { constraintsRouter } from './routes/constraints.js';
import { handoffRouter } from './routes/handoff.js';
import { networkInfoRouter } from './routes/networkInfo.js';
import { createLocalCollabRouter } from './routes/collab.js';
import { createCollabSocketServer } from './collab/collabSocket.js';
import { createInternalRouter } from './routes/internal.js';
import { internalAuth } from './internalAuth.js';
import { isLocalOrLanOrigin } from './lanOrigin.js';
import { loadConfig, type AppConfig } from './config.js';
import { makeStorage } from './collab/storage/index.js';
import { createSessionStore } from './collab/sessionStore.js';
import { createAca1App } from './aca1/app.js';
import { createAca2Client } from './aca1/aca2Client.js';
import { startSweep } from './aca1/sweep.js';

export interface RunningServer {
  port: number;
  close: () => Promise<void>;
}

// ROLE=local — today's Electron/LAN relay: every route, sockets, static
// hosting. Session state is in-memory (config defaults STORAGE=memory here).
function buildLocalApp(storage: ReturnType<typeof makeStorage>): express.Express {
  const app = express();
  app.use(cors({
    // No Origin (packaged Electron / curl) or a LAN origin only. Reflecting
    // any origin would expose /api/save-files (writes an absolute path from
    // the body) to every site the user's browser visits. See lanOrigin.ts.
    origin: (origin, callback) => callback(null, !origin || isLocalOrLanOrigin(origin)),
  }));
  app.use(express.json({ limit: '10mb' }));

  app.use('/api', constraintsRouter);
  app.use('/api', handoffRouter);
  app.use('/api', networkInfoRouter);
  app.use('/api', createLocalCollabRouter({ storage }));

  app.get('/api/health', (_req, res) => {
    res.json({ ok: true, role: 'local', server: 'gantt-editor-api', time: new Date().toISOString() });
  });

  app.post('/api/save-files', async (req, res) => {
    const { envPath, schedulePath, envYaml, scheduleYaml } = req.body as {
      envPath?: string; schedulePath?: string; envYaml?: string; scheduleYaml?: string;
    };
    try {
      const writes: Promise<void>[] = [];
      if (envPath && envYaml) writes.push(writeFile(envPath, envYaml, 'utf-8'));
      if (schedulePath && scheduleYaml) writes.push(writeFile(schedulePath, scheduleYaml, 'utf-8'));
      await Promise.all(writes);
      res.json({ ok: true });
    } catch (err) {
      res.status(500).json({ ok: false, error: String(err) });
    }
  });

  const staticDir = process.env.SERVE_STATIC_DIR;
  if (staticDir) {
    app.use(express.static(staticDir));
    app.get('*', (_req, res) => res.sendFile(path.join(staticDir, 'index.html')));
  }

  return app;
}

// ROLE=aca2 — the public Socket.IO relay + the ACA1-only /internal/* control
// plane. No local-file routes, no static hosting.
function buildAca2App(store: ReturnType<typeof createSessionStore>, config: AppConfig): express.Express {
  const app = express();
  app.use(cors({ origin: config.webOrigin ?? true }));
  app.use(express.json({ limit: '10mb' }));
  app.get('/api/health', (_req, res) => {
    res.json({ ok: true, role: 'aca2', instance: config.instanceId, time: new Date().toISOString() });
  });
  // Auth only gates the /internal/* prefix; the router itself is mounted at
  // root because its route paths already start with /internal.
  app.use('/internal', internalAuth(config.internalKey));
  app.use(createInternalRouter(store, config));
  return app;
}

export async function startServer(config: AppConfig = loadConfig()): Promise<RunningServer> {
  const storage = makeStorage(config.storage);
  const store = createSessionStore({ storage });
  let stopSweep: (() => void) | undefined;

  let app: express.Express;
  let withSockets = false;

  if (config.role === 'aca1') {
    const aca2 = createAca2Client(config);
    app = createAca1App({ storage, aca2, config });
    stopSweep = startSweep({ storage, aca2, config });
  } else if (config.role === 'aca2') {
    app = buildAca2App(store, config);
    withSockets = true;
  } else {
    app = buildLocalApp(storage);
    withSockets = true;
  }

  const httpServer: HttpServer = createServer(app);
  if (withSockets) createCollabSocketServer(httpServer, store, config);

  await new Promise<void>((resolve) => httpServer.listen(config.port, resolve));
  const port = (httpServer.address() as AddressInfo).port;

  return {
    port,
    close: () =>
      new Promise<void>((resolve, reject) => {
        stopSweep?.();
        httpServer.close((err) => (err ? reject(err) : resolve()));
      }),
  };
}

// Direct execution (node dist/index.js / tsx src/index.ts) — not when imported by a test.
const invokedDirectly = process.argv[1]
  ? import.meta.url === pathToFileURL(process.argv[1]).href
  : false;
if (invokedDirectly) {
  const config = loadConfig();
  startServer(config)
    .then(({ port }) => console.log(`[server] role=${config.role} listening on http://localhost:${port}`))
    .catch((err) => {
      console.error('[server] failed to start:', err);
      process.exit(1);
    });
}
