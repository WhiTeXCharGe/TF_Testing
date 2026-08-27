import express from 'express';
import cors from 'cors';
import path from 'node:path';
import { createServer } from 'node:http';
import { writeFile } from 'node:fs/promises';
import { constraintsRouter } from './routes/constraints.js';
import { handoffRouter } from './routes/handoff.js';
import { networkInfoRouter } from './routes/networkInfo.js';
import { collabRouter } from './routes/collab.js';
import { createCollabSocketServer } from './collab/collabSocket.js';
import { isLocalOrLanOrigin } from './lanOrigin.js';

const app = express();
const PORT = Number(process.env.PORT ?? 3010);

// A collab joiner opens the shared link on their own PC as
// http://<lan-ip>:5173/?session=... , so their browser sends
// Origin: http://192.168.x.x:5173 — the old fixed localhost allowlist
// preflight-rejected every /api/collab/*, /api/network-info, … call they made,
// breaking the LAN case this feature exists for (invisible both in packaged
// Electron, which is same-origin, and in a localhost-only test).
//
// Permissive enough for that, and no more: reflecting *any* origin would also
// hand every website the user's browser visits a working cross-origin POST to
// /api/save-files, which writes an absolute path taken from the request body.
// See lanOrigin.ts.
app.use(cors({
  origin: (origin, callback) => {
    // No Origin header at all: same-origin (packaged Electron serves the
    // frontend from this very server), or a non-browser caller like curl or
    // the Electron main process. Nothing for CORS to gate either way.
    callback(null, !origin || isLocalOrLanOrigin(origin));
  },
}));
app.use(express.json({ limit: '10mb' }));

app.use('/api', constraintsRouter);
app.use('/api', handoffRouter);
app.use('/api', networkInfoRouter);
app.use('/api', collabRouter);

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, server: 'gantt-editor-api', time: new Date().toISOString() });
});

app.post('/api/save-files', async (req, res) => {
  const { envPath, schedulePath, envYaml, scheduleYaml } = req.body as {
    envPath?: string;
    schedulePath?: string;
    envYaml?: string;
    scheduleYaml?: string;
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

// Packaged Electron builds serve the built frontend from this same origin so
// relative fetch('/api/...') calls in the renderer keep working without CORS.
// SERVE_STATIC_DIR is set by electron/main.cts only in packaged mode — unset
// in dev, where Vite serves the frontend on its own port instead.
const staticDir = process.env.SERVE_STATIC_DIR;
if (staticDir) {
  app.use(express.static(staticDir));
  app.get('*', (_req, res) => {
    res.sendFile(path.join(staticDir, 'index.html'));
  });
}

// Plain app.listen() can't also host Socket.IO on the same port, so the
// collab relay wraps app in its own http.Server first.
const httpServer = createServer(app);
createCollabSocketServer(httpServer);

httpServer.listen(PORT, () => {
  console.log(`[server] running on http://localhost:${PORT}`);
});