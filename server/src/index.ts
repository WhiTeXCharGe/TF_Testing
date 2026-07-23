import express from 'express';
import cors from 'cors';
import path from 'node:path';
import { writeFile } from 'node:fs/promises';
import { constraintsRouter } from './routes/constraints.js';
import { handoffRouter } from './routes/handoff.js';

const app = express();
const PORT = Number(process.env.PORT ?? 3010);

app.use(cors({ origin: ['http://localhost:5173', 'http://localhost:5174'] }));
app.use(express.json({ limit: '10mb' }));

app.use('/api', constraintsRouter);
app.use('/api', handoffRouter);

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

app.listen(PORT, () => {
  console.log(`[server] running on http://localhost:${PORT}`);
});