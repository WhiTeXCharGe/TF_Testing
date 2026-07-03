import express from 'express';
import cors from 'cors';
import { constraintsRouter } from './routes/constraints.js';

const app = express();
const PORT = process.env.PORT ? Number(process.env.PORT) : 3001;

app.use(cors({ origin: 'http://localhost:5173' }));
app.use(express.json({ limit: '10mb' }));

app.use('/api', constraintsRouter);

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, server: 'gantt-editor-api', time: new Date().toISOString() });
});

app.listen(PORT, () => {
  console.log(`[server] running on http://localhost:${PORT}`);
});
