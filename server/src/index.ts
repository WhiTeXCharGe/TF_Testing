import express from 'express';
import cors from 'cors';
import { writeFile } from 'node:fs/promises';
import { constraintsRouter } from './routes/constraints.js';

const app = express();
const PORT = process.env.PORT ? Number(process.env.PORT) : 3001;

app.use(cors({ origin: 'http://localhost:5173' }));
app.use(express.json({ limit: '10mb' }));

app.use('/api', constraintsRouter);

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

app.listen(PORT, () => {
  console.log(`[server] running on http://localhost:${PORT}`);
});
