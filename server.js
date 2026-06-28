/**
 * Timefold Local API Service
 *
 * Replaces Azure Container Apps + Blob Storage + Azure Batch during local development.
 * Mirrors the exact HTTP contract the webapp (runService.ts) expects.
 *
 * Endpoints:
 *   POST /runSolver              — upload EnvConfig.yaml + Schedule.yaml, start solve
 *   GET  /status/:runId          — poll solve progress
 *   GET  /download/:runId        — download output YAML when completed
 *
 * Test-only helpers (Postman / manual testing):
 *   PUT  /status/:runId          — manually set status (simulates Docker writing status)
 *   POST /output/:runId          — manually upload output YAML (simulates Docker output)
 *
 * Data layout (all under ./data/):
 *   input/{runId}/EnvConfig.yaml
 *   input/{runId}/Schedule.yaml
 *   output/{runId}/result_Schedule.yaml
 *   status/{runId}.json
 */

'use strict';

const express        = require('express');
const multer         = require('multer');
const cors           = require('cors');
const fs             = require('fs');
const path           = require('path');
const { spawn }      = require('child_process');

const app  = express();
const PORT = 3001;

// ── Middleware ───────────────────────────────────────────────────────────────

app.use(cors());
app.use(express.json());

// multer: keep files in memory so we can write them ourselves
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 }, // 50 MB per file
  fileFilter: (_req, file, cb) => {
    if (/\.ya?ml$/i.test(file.originalname)) return cb(null, true);
    cb(new Error(`Only .yaml / .yml files are accepted (got: ${file.originalname})`));
  },
});

// ── Data directories ─────────────────────────────────────────────────────────

const DATA_DIR   = path.join(__dirname, 'data');
const INPUT_DIR  = path.join(DATA_DIR, 'input');
const OUTPUT_DIR = path.join(DATA_DIR, 'output');
const STATUS_DIR = path.join(DATA_DIR, 'status');

[INPUT_DIR, OUTPUT_DIR, STATUS_DIR].forEach(d =>
  fs.mkdirSync(d, { recursive: true })
);

// ── Concurrency queue ─────────────────────────────────────────────────────────
// MAX_CONCURRENT_RUNS env var sets the cap (default: 2).
// Extra runs wait in `pendingQueue` and auto-start when a slot opens.

const MAX_CONCURRENT = parseInt(process.env.MAX_CONCURRENT_RUNS || '2', 10);
let activeCount = 0;
const pendingQueue = []; // Array of { runId, inputDir }

/** Map of runId → child_process so we can kill containers on cancel/delete. */
const runningProcs = new Map();

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Generate a run ID like 20260610_143022500 */
function generateRunId() {
  const d   = new Date();
  const pad = (n, w = 2) => String(n).padStart(w, '0');
  return (
    `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_` +
    `${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}${pad(d.getMilliseconds(), 3)}`
  );
}

/** Block path traversal attacks. */
function isValidRunId(id) {
  return typeof id === 'string' && /^[A-Za-z0-9_-]+$/.test(id);
}

function readStatus(runId) {
  const file = path.join(STATUS_DIR, `${runId}.json`);
  if (!fs.existsSync(file)) return null;
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

function writeStatus(runId, payload) {
  fs.writeFileSync(
    path.join(STATUS_DIR, `${runId}.json`),
    JSON.stringify(payload, null, 2),
    'utf8'
  );
}

/** Free one concurrency slot and start the next queued run (if any). */
function releaseSlot() {
  activeCount--;
  console.log(`[queue] slot released — active: ${activeCount}/${MAX_CONCURRENT}, pending: ${pendingQueue.length}`);
  if (pendingQueue.length > 0) {
    const next = pendingQueue.shift();
    launchContainer(next.runId, next.inputDir);
  }
}

/**
 * Actually spawn the Docker container for a run.
 * Only called when a concurrency slot is available.
 */
function launchContainer(runId, inputDir) {
  activeCount++;
  console.log(`[queue] launching ${runId} — active: ${activeCount}/${MAX_CONCURRENT}, pending: ${pendingQueue.length}`);

  // path.join() on Windows produces backslashes; Docker needs forward slashes.
  const dp = p => p.replace(/\\/g, '/');

  const outputDir = path.join(OUTPUT_DIR, runId);
  fs.mkdirSync(outputDir, { recursive: true });

  const containerName = `tf-${runId}`;

  const args = [
    'run', '--rm',
    '--name', containerName,
    '-e', `RUN_ID=${runId}`,
    '-v', `${dp(inputDir)}:/work/input:ro`,
    '-v', `${dp(outputDir)}:/work/output`,
    '-v', `${dp(STATUS_DIR)}:/work/status`,
    'timefold-scheduler:local',
  ];

  console.log(`[docker:${runId}] starting — docker run ${args.join(' ')}`);

  const proc = spawn('docker', args, { stdio: 'pipe' });
  runningProcs.set(runId, proc);

  proc.stdout.on('data', d => process.stdout.write(`[docker:${runId}] ${d}`));
  proc.stderr.on('data', d => process.stderr.write(`[docker:${runId}] ${d}`));

  proc.on('error', err => {
    console.error(`[docker:${runId}] spawn failed:`, err.message);
    runningProcs.delete(runId);
    writeStatus(runId, {
      status: 'Failed',
      stage: null,
      progress: 0,
      error: { type: 'UnknownError', message: `Cannot start Docker: ${err.message}` },
    });
    releaseSlot();
  });

  proc.on('close', code => {
    console.log(`[docker:${runId}] container exited (code ${code})`);
    runningProcs.delete(runId);
    if (code !== 0) {
      const current = readStatus(runId);
      if (!current || current.status === 'Submitted' || current.status === 'Running') {
        writeStatus(runId, {
          status: 'Failed',
          stage: null,
          progress: 0,
          error: {
            type: 'UnknownError',
            message: `Container exited with code ${code} before writing status — check: docker logs ${containerName}`,
          },
        });
      }
    }
    releaseSlot();
  });
}

/**
 * Queue a run for execution. Starts immediately if a slot is free;
 * otherwise adds to the pending queue (status stays "Submitted" until launched).
 * Max concurrent containers is controlled by MAX_CONCURRENT_RUNS env var (default 2).
 */
function triggerDocker(runId, inputDir) {
  if (activeCount < MAX_CONCURRENT) {
    launchContainer(runId, inputDir);
  } else {
    pendingQueue.push({ runId, inputDir });
    console.log(`[queue:${runId}] queued at position ${pendingQueue.length} (active: ${activeCount}/${MAX_CONCURRENT})`);
  }
}

/** Find the first YAML file in the output folder for a run. */
function findOutputYaml(runId) {
  const dir = path.join(OUTPUT_DIR, runId);
  if (!fs.existsSync(dir)) return null;
  const found = fs.readdirSync(dir).find(f => /\.ya?ml$/i.test(f));
  return found ? path.join(dir, found) : null;
}

// ── POST /runSolver ───────────────────────────────────────────────────────────
//
// Body: multipart/form-data
//   env   (file) — EnvConfig.yaml
//   sched (file) — Schedule.yaml
//
// Response 202: { runId, status: "Submitted" }
// Response 400: { error: "..." }   — missing files or wrong format
// Response 500: { error: "..." }   — disk write failure

app.post(
  '/runSolver',
  upload.fields([
    { name: 'env',   maxCount: 1 },
    { name: 'sched', maxCount: 1 },
  ]),
  (req, res) => {
    try {
      const files = req.files ?? {};

      if (!files.env || files.env.length === 0) {
        return res.status(400).json({ error: 'Missing required field: env (EnvConfig.yaml)' });
      }
      if (!files.sched || files.sched.length === 0) {
        return res.status(400).json({ error: 'Missing required field: sched (Schedule.yaml)' });
      }

      // Prefer the runId the webapp already created (keeps IDs in sync).
      // Fall back to generating one if the field is absent or invalid.
      const proposed = (req.body && typeof req.body.runId === 'string') ? req.body.runId.trim() : '';
      const runId    = proposed && isValidRunId(proposed) ? proposed : generateRunId();
      const inputDir = path.join(INPUT_DIR, runId);
      fs.mkdirSync(inputDir, { recursive: true });

      const envFile   = files.env[0];
      const schedFile = files.sched[0];

      fs.writeFileSync(path.join(inputDir, envFile.originalname),   envFile.buffer);
      fs.writeFileSync(path.join(inputDir, schedFile.originalname), schedFile.buffer);

      writeStatus(runId, {
        status:   'Submitted',
        stage:    null,
        progress: 0,
        error:    null,
      });

      console.log(`[runSolver] created run ${runId}`);
      console.log(`  env  → ${envFile.originalname} (${envFile.size} bytes)`);
      console.log(`  sched→ ${schedFile.originalname} (${schedFile.size} bytes)`);

      triggerDocker(runId, inputDir);

      return res.status(202).json({ runId, status: 'Submitted' });
    } catch (err) {
      console.error('[runSolver] error:', err);
      return res.status(500).json({ error: `Upload failed: ${err.message}` });
    }
  }
);

// ── GET /status/:runId ────────────────────────────────────────────────────────
//
// Response 200: { status, stage, progress, error }
//   status: "Submitted" | "Running" | "Completed" | "Failed"
// Response 400: { error: "..." }   — invalid runId format
// Response 404: { error: "..." }   — run not found
// Response 500: { error: "..." }   — can't read status file

app.get('/status/:runId', (req, res) => {
  const { runId } = req.params;

  if (!isValidRunId(runId)) {
    return res.status(400).json({ error: 'Invalid runId format' });
  }

  try {
    const status = readStatus(runId);
    if (!status) {
      return res.status(404).json({ error: `Run "${runId}" not found` });
    }

    console.log(`[status] ${runId} → ${status.status}`);
    return res.status(200).json(status);
  } catch (err) {
    console.error(`[status] error reading ${runId}:`, err);
    return res.status(500).json({ error: `Failed to read status: ${err.message}` });
  }
});

// ── GET /download/:runId ──────────────────────────────────────────────────────
//
// Sends the output YAML file as a file attachment.
//
// Response 200: file download (application/octet-stream)
// Response 400: { error: "..." }   — invalid runId format
// Response 404: { error: "..." }   — run not found or output not ready
// Response 409: { error: "..." }   — run failed, no output to download
// Response 500: { error: "..." }   — unexpected error

app.get('/download/:runId', (req, res) => {
  const { runId } = req.params;

  if (!isValidRunId(runId)) {
    return res.status(400).json({ error: 'Invalid runId format' });
  }

  try {
    const status = readStatus(runId);

    if (!status) {
      return res.status(404).json({ error: `Run "${runId}" not found` });
    }

    if (status.status === 'Failed') {
      return res.status(409).json({
        error: `Run "${runId}" failed — no output available`,
        detail: status.error ?? 'No error detail recorded',
      });
    }

    if (status.status !== 'Completed') {
      return res.status(409).json({
        error: `Run "${runId}" is not completed yet (current status: ${status.status})`,
      });
    }

    const yamlPath = findOutputYaml(runId);
    if (!yamlPath) {
      return res.status(404).json({
        error: `Output YAML not found for run "${runId}" — folder may be missing`,
      });
    }

    console.log(`[download] ${runId} → ${path.basename(yamlPath)}`);
    return res.download(yamlPath, path.basename(yamlPath));
  } catch (err) {
    console.error(`[download] error for ${runId}:`, err);
    return res.status(500).json({ error: `Download failed: ${err.message}` });
  }
});

// ── GET /queue ────────────────────────────────────────────────────────────────
//
// Returns the current concurrency state: how many containers are running,
// the limit, and what's waiting in the queue.
//
// Response 200: { maxConcurrent, active, pending: string[] }

app.get('/queue', (_req, res) => {
  return res.status(200).json({
    maxConcurrent: MAX_CONCURRENT,
    active:  activeCount,
    pending: pendingQueue.map(r => r.runId),
  });
});

// ── DELETE /run/:runId ────────────────────────────────────────────────────────
//
// Cancel a running container (if active) and delete all service data for the run.
//
// Response 200: { ok: true, runId }
// Response 400: { error: "..." }   — invalid runId format
// Response 500: { error: "..." }   — unexpected error

app.delete('/run/:runId', (req, res) => {
  const { runId } = req.params;

  if (!isValidRunId(runId)) {
    return res.status(400).json({ error: 'Invalid runId format' });
  }

  try {
    // 1. Remove from pending queue (if waiting to start).
    const queueIdx = pendingQueue.findIndex(r => r.runId === runId);
    if (queueIdx !== -1) {
      pendingQueue.splice(queueIdx, 1);
      console.log(`[cancel:${runId}] removed from pending queue`);
    }

    // 2. Kill the running Docker container (via proc kill + docker kill).
    const proc = runningProcs.get(runId);
    if (proc) {
      proc.kill('SIGTERM');
      runningProcs.delete(runId);
      console.log(`[cancel:${runId}] sent SIGTERM to docker run process`);
    }
    // Also attempt via docker kill in case proc reference is stale.
    const containerName = `tf-${runId}`;
    spawn('docker', ['kill', containerName], { stdio: 'ignore' });

    // 3. Mark status as Cancelled so the webapp reflects the state.
    writeStatus(runId, {
      status: 'Cancelled',
      stage: null,
      progress: 0,
      error: null,
    });

    // 4. Clean up service-side data folders.
    const toRemove = [
      path.join(INPUT_DIR,  runId),
      path.join(OUTPUT_DIR, runId),
      path.join(STATUS_DIR, `${runId}.json`),
    ];
    for (const p of toRemove) {
      if (fs.existsSync(p)) {
        const stat = fs.statSync(p);
        if (stat.isDirectory()) fs.rmSync(p, { recursive: true, force: true });
        else fs.unlinkSync(p);
      }
    }

    console.log(`[cancel:${runId}] cleaned up service data`);
    return res.status(200).json({ ok: true, runId });
  } catch (err) {
    console.error(`[cancel:${runId}] error:`, err);
    return res.status(500).json({ error: `Cancel failed: ${err.message}` });
  }
});

// ── GET /docker ───────────────────────────────────────────────────────────────
//
// Shows real Docker container state for all tf-* containers plus the
// service's internal queue counters. Good for debugging in Postman or curl.
//
// Response 200: { maxConcurrent, active, pending, containers: [...] }

app.get('/docker', (_req, res) => {
  const proc = spawn('docker', [
    'ps', '-a',
    '--filter', 'name=tf-',
    '--format', '{{.Names}}\t{{.Status}}\t{{.RunningFor}}\t{{.ID}}',
  ], { stdio: 'pipe' });

  let stdout = '';
  let stderr = '';
  proc.stdout.on('data', d => { stdout += d; });
  proc.stderr.on('data', d => { stderr += d; });

  proc.on('error', err => {
    return res.status(500).json({ error: `docker ps failed: ${err.message}` });
  });

  proc.on('close', code => {
    const containers = stdout
      .trim()
      .split('\n')
      .filter(Boolean)
      .map(line => {
        const [name, status, runningFor, id] = line.split('\t');
        return { name, status, runningFor, id };
      });

    return res.status(200).json({
      maxConcurrent: MAX_CONCURRENT,
      active:  activeCount,
      pending: pendingQueue.map(r => r.runId),
      watching: [...runningProcs.keys()],
      containers,
      dockerExitCode: code,
      dockerStderr: stderr.trim() || null,
    });
  });
});

// ── PUT /status/:runId  (TEST HELPER — simulates Docker writing status) ───────
//
// Body: { "status": "Running", "stage": 1, "progress": 0.4, "error": null }
//
// Use from Postman to manually advance the run state without Docker.

app.put('/status/:runId', (req, res) => {
  const { runId } = req.params;

  if (!isValidRunId(runId)) {
    return res.status(400).json({ error: 'Invalid runId format' });
  }

  const allowed = ['Submitted', 'Running', 'Completed', 'Failed'];
  const { status, stage = null, progress = 0, error = null } = req.body ?? {};

  if (!allowed.includes(status)) {
    return res.status(400).json({
      error: `Invalid status "${status}". Must be one of: ${allowed.join(', ')}`,
    });
  }

  try {
    writeStatus(runId, { status, stage, progress, error });
    console.log(`[status:put] ${runId} → ${status}`);
    return res.status(200).json({ ok: true, runId, status });
  } catch (err) {
    console.error(`[status:put] error for ${runId}:`, err);
    return res.status(500).json({ error: `Failed to write status: ${err.message}` });
  }
});

// ── POST /output/:runId  (TEST HELPER — simulates Docker writing output) ──────
//
// Body: multipart/form-data
//   result (file) — the output YAML (e.g. result_Schedule.yaml)
//
// Use from Postman to upload a fake solver output without running Docker.

app.post(
  '/output/:runId',
  upload.single('result'),
  (req, res) => {
    const { runId } = req.params;

    if (!isValidRunId(runId)) {
      return res.status(400).json({ error: 'Invalid runId format' });
    }

    if (!req.file) {
      return res.status(400).json({ error: 'Missing required field: result (output YAML file)' });
    }

    try {
      const outDir = path.join(OUTPUT_DIR, runId);
      fs.mkdirSync(outDir, { recursive: true });

      const filename = req.file.originalname || 'result_Schedule.yaml';
      fs.writeFileSync(path.join(outDir, filename), req.file.buffer);

      console.log(`[output:post] ${runId} → saved ${filename} (${req.file.size} bytes)`);
      return res.status(200).json({ ok: true, runId, file: filename });
    } catch (err) {
      console.error(`[output:post] error for ${runId}:`, err);
      return res.status(500).json({ error: `Failed to save output: ${err.message}` });
    }
  }
);

// ── Multer error handler ──────────────────────────────────────────────────────

app.use((err, _req, res, _next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(413).json({ error: 'File too large (max 50 MB)' });
    }
    return res.status(400).json({ error: `Upload error: ${err.message}` });
  }
  if (err) {
    return res.status(400).json({ error: err.message });
  }
});

// ── Start ─────────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`\nTimefold API service running at http://localhost:${PORT}`);
  console.log(`Data directory: ${DATA_DIR}`);
  console.log(`Max concurrent containers: ${MAX_CONCURRENT} (set MAX_CONCURRENT_RUNS env var to change)\n`);
  console.log('Endpoints:');
  console.log(`  POST http://localhost:${PORT}/runSolver`);
  console.log(`  GET  http://localhost:${PORT}/status/:runId`);
  console.log(`  GET  http://localhost:${PORT}/download/:runId`);
  console.log(`  GET  http://localhost:${PORT}/queue`);
  console.log('\nTest helpers (Postman only):');
  console.log(`  PUT  http://localhost:${PORT}/status/:runId`);
  console.log(`  POST http://localhost:${PORT}/output/:runId\n`);
});
