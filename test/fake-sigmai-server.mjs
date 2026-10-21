/**
 * Fake Sigma-I API for testing bridge.sh without Docker / the wheel / Azure.
 * Mimics the parts of the real API the bridge uses (verified against the delivered source):
 *   POST /jobs (multipart, X-API-Key) → 202 | 409 JOB_ALREADY_EXISTS
 *   GET  /jobs/{id}/status  → { job_status:{status}, progress:{estimated_progress_rate}, error:{type,message} }
 *   GET  /jobs/{id}/result  → { summary, stage1_result, stage2_result, kpi, error }
 *   GET  /jobs/{id}/download → result_Schedule.yaml
 *   DELETE /jobs/{id} → 202 (cancel)
 *   GET  /health
 *
 * FAKE_MODE: ok (default) | fail (job fails OPTIMIZATION_FAILED) | slow (never finishes — for cancel/timeout)
 * FAKE_POLLS_TO_FINISH: how many status polls before "finished" (default 3)
 *
 * Usage:  node test/fake-sigmai-server.mjs [port]     (default 8000)
 */
import http from 'node:http';
import express from 'express';
import multer from 'multer';

const KEY = process.env.FAKE_API_KEY ?? 'test-key';
const MODE = process.env.FAKE_MODE ?? 'ok';
const POLLS = Number(process.env.FAKE_POLLS_TO_FINISH ?? 3);

export function createFakeSigmai({ mode = MODE, polls: POLLS_TO_FINISH = POLLS } = {}) {
  const jobs = new Map();       // id → { fields, files, polls, canceled }
  const app = express();
  const upload = multer({ storage: multer.memoryStorage() });

  app.get('/health', (_q, r) => r.json({ status: 'ok', service: 'fake-sigmai' }));
  app.use('/jobs', (q, r, next) => (q.get('X-API-Key') === KEY ? next()
    : r.status(401).json({ error: { type: 'UNAUTHORIZED', message: 'bad key' } })));

  app.post('/jobs', upload.fields([{ name: 'environment' }, { name: 'schedule' }]), (q, r) => {
    const id = q.body.job_id;
    if (!id) return r.status(422).json({ error: { type: 'VALIDATION_ERROR', message: 'Invalid request.' } });
    if (jobs.has(id)) return r.status(409).json({ error: { type: 'JOB_ALREADY_EXISTS', message: 'exists' } });
    if (!q.files?.environment || !q.files?.schedule) {
      return r.status(422).json({ error: { type: 'VALIDATION_ERROR', message: 'Invalid request.' } });
    }
    jobs.set(id, { fields: q.body, schedule: q.files.schedule[0].buffer.toString(), polls: 0, canceled: false });
    r.status(202).json({ job_id: id, status: 'queued', created_at: new Date().toISOString() });
  });

  const state = (j) => {
    if (j.canceled) return 'canceled';
    if (mode === 'slow') return 'running';
    if (j.polls < 1) return 'queued';
    if (j.polls < POLLS_TO_FINISH) return 'running';
    return mode === 'fail' ? 'failed' : 'finished';
  };

  app.get('/jobs/:id/status', (q, r) => {
    const j = jobs.get(q.params.id);
    if (!j) return r.status(404).json({ error: { type: 'JOB_NOT_FOUND' } });
    const st = state(j);
    j.polls++;
    r.json({
      job_id: q.params.id, job_info: { scenario_name: j.fields.scenario_name ?? null },
      job_status: { status: st },
      progress: { elapsed_seconds: j.polls, time_limit_seconds: 100, estimated_progress_rate: Math.min(100, j.polls * 30) },
      error: st === 'failed'
        ? { type: 'OPTIMIZATION_FAILED', message: 'Optimization failed.', stage: 'system' }
        : { type: null, message: null, stage: null },
    });
  });

  app.get('/jobs/:id/result', (q, r) => {
    const j = jobs.get(q.params.id);
    if (!j) return r.status(404).json({ error: { type: 'JOB_NOT_FOUND' } });
    r.json({
      job_id: q.params.id, job_info: {}, job_status: { status: state(j) },
      summary: { final_status: 'ok', validation_passed: mode !== 'invalid', final_score: 123.4 },
      stage1_result: { status: 'optimal' }, stage2_result: { status: 'optimal' },
      kpi: { makespan: 40 }, error: {},
    });
  });

  app.get('/jobs/:id/download', (q, r) => {
    const j = jobs.get(q.params.id);
    if (!j || state(j) !== 'finished') return r.status(409).json({ error: { type: 'JOB_NOT_FINISHED' } });
    r.type('application/x-yaml').send(`# result for ${q.params.id}\n${j.schedule}`);
  });

  app.delete('/jobs/:id', (q, r) => {
    const j = jobs.get(q.params.id);
    if (!j) return r.status(404).json({ error: { type: 'JOB_NOT_FOUND' } });
    j.canceled = true;
    r.status(202).json({ job_id: q.params.id, status: 'cancel_requested', accepted: true });
  });

  app.jobs = jobs;
  return app;
}

if (import.meta.url === `file:///${process.argv[1].replace(/\\/g, '/')}`) {
  const port = Number(process.argv[2] ?? 8000);
  http.createServer(createFakeSigmai()).listen(port, '127.0.0.1', () =>
    console.log(`fake Sigma-I on http://127.0.0.1:${port}  key=${KEY} mode=${MODE}`));
}
