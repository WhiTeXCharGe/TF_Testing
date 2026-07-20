/**
 * Timefold API Controller — TEST/PIPELINE-ONLY edition.
 *
 * Same HTTP contract as ../api-controller/src/server.js (and the webapp
 * expects the same thing from either), but the Batch task does NOT run the
 * Timefold solver, does NOT use a container, and does NOT touch ACR at all.
 * It just copies the uploaded Schedule.yaml to Schedule_result.yaml and
 * writes a Completed status — proving the webapp -> API -> Blob -> Batch ->
 * Blob -> API -> webapp pipeline works, independent of solver correctness
 * or any ACR/Docker complications.
 *
 * Endpoints (identical shapes to the real api-controller):
 *   POST   /runSolver      — upload EnvConfig.yaml + Schedule.yaml, create a Batch task
 *   GET    /status/:runId  — read status/{runId}.json from Blob
 *   GET    /download/:runId — stream output/{runId}/Schedule_result.yaml from Blob
 *   DELETE /run/:runId     — terminate the Batch task (if any) + delete all blobs for the run
 *   GET    /health
 */

import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { DefaultAzureCredential } from '@azure/identity';
import { BlobServiceClient } from '@azure/storage-blob';

// ── Config — same env vars as the real controller, minus anything ACR/solver-specific ──
const STORAGE_ACCOUNT     = process.env.STORAGE_ACCOUNT;
const BLOB_CONTAINER      = process.env.BLOB_CONTAINER;
const BATCH_ACCOUNT_URL   = process.env.BATCH_ACCOUNT_URL;
const BATCH_JOB_ID        = process.env.BATCH_JOB_ID;
const POOL_MI_RESOURCE_ID = process.env.POOL_MI_RESOURCE_ID;
const PORT                = Number(process.env.PORT ?? 8080);

for (const [k, v] of Object.entries({
  STORAGE_ACCOUNT, BLOB_CONTAINER, BATCH_ACCOUNT_URL, BATCH_JOB_ID, POOL_MI_RESOURCE_ID,
})) {
  if (!v) console.warn(`[config] WARNING: env var ${k} is not set — requests that need it will fail`);
}

const BATCH_API_VERSION = '2024-07-01.20.0';
const BATCH_SCOPE       = 'https://batch.core.windows.net/.default';

const credential  = new DefaultAzureCredential();
const blobService = new BlobServiceClient(`https://${STORAGE_ACCOUNT}.blob.core.windows.net`, credential);
const container    = blobService.getContainerClient(BLOB_CONTAINER);

// ── Helpers ─────────────────────────────────────────────────────────────

function isValidRunId(id) {
  return typeof id === 'string' && /^[A-Za-z0-9_-]+$/.test(id);
}

function makeRunId() {
  const d = new Date(), pad = (n, w = 2) => String(n).padStart(w, '0');
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_` +
         `${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}${pad(d.getMilliseconds(), 3)}`;
}

async function writeJsonBlob(name, obj) {
  const body = Buffer.from(JSON.stringify(obj, null, 2), 'utf8');
  await container.getBlockBlobClient(name).uploadData(body, {
    blobHTTPHeaders: { blobContentType: 'application/json' },
  });
}

async function readJsonBlob(name) {
  const blob = container.getBlockBlobClient(name);
  if (!(await blob.exists())) return null;
  const buf = await blob.downloadToBuffer();
  return JSON.parse(buf.toString('utf8'));
}

async function batchToken() {
  const t = await credential.getToken(BATCH_SCOPE);
  return t.token;
}

async function batchFetch(path, options = {}) {
  const token = await batchToken();
  const sep = path.includes('?') ? '&' : '?';
  const url = `${BATCH_ACCOUNT_URL}${path}${sep}api-version=${BATCH_API_VERSION}`;
  return fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json; odata=minimalmetadata',
      Authorization: `Bearer ${token}`,
      ...(options.headers ?? {}),
    },
  });
}

/**
 * Create a PLAIN (non-container) Batch task that just copies the uploaded
 * Schedule.yaml to Schedule_result.yaml and writes a Completed status.
 *
 * No containerSettings at all — runs directly on the pool node's own shell.
 * No ACR, no Docker image, no entrypoint.sh. The status JSON is built in JS
 * and passed through as base64 to sidestep nested-quoting problems entirely
 * (base64 has no quote characters, so it's safe inside the single-quoted
 * `bash -c '...'` wrapper with zero escaping headaches).
 *
 * resourceFiles with filePath "input" + blobPrefix "input/<runId>/" lands
 * files at "input/input/<runId>/..." relative to the task working
 * directory — Batch preserves the full blob path under filePath, it does
 * not strip the prefix. (Same behavior whether containerized or not.)
 */
async function createBatchTask(runId) {
  const containerUrl = `https://${STORAGE_ACCOUNT}.blob.core.windows.net/${BLOB_CONTAINER}`;
  const nowIso = new Date().toISOString();

  const statusPayload = JSON.stringify({
    runId, status: 'Completed', stage: 1, progress: 1,
    startedAt: nowIso, updatedAt: nowIso, finishedAt: nowIso,
    error: null, output: `output/Schedule_result.yaml`,
  });
  const statusB64 = Buffer.from(statusPayload, 'utf8').toString('base64');

  const commandLine =
    `/bin/bash -c 'set -e; mkdir -p output status; ` +
    `cp "input/input/${runId}/Schedule.yaml" "output/Schedule_result.yaml"; ` +
    `echo ${statusB64} | base64 -d > "status/${runId}.json"'`;

  const task = {
    id: runId,
    commandLine,
    userIdentity: { autoUser: { elevationLevel: 'admin', scope: 'task' } },
    resourceFiles: [
      {
        autoStorageContainerName: BLOB_CONTAINER,
        blobPrefix: `input/${runId}/`,
        filePath: 'input',
        identityReference: { resourceId: POOL_MI_RESOURCE_ID },
      },
    ],
    outputFiles: [
      {
        filePattern: 'output/Schedule_result.yaml',
        destination: {
          container: {
            containerUrl,
            path: `output/${runId}/Schedule_result.yaml`,
            identityReference: { resourceId: POOL_MI_RESOURCE_ID },
          },
        },
        uploadOptions: { uploadCondition: 'taskSuccess' },
      },
      {
        filePattern: `status/${runId}.json`,
        destination: {
          container: {
            containerUrl,
            path: `status/${runId}.json`,
            identityReference: { resourceId: POOL_MI_RESOURCE_ID },
          },
        },
        uploadOptions: { uploadCondition: 'taskCompletion' },
      },
    ],
  };

  const res = await batchFetch(`/jobs/${BATCH_JOB_ID}/tasks`, {
    method: 'POST',
    body: JSON.stringify(task),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Batch task create failed (${res.status}): ${text}`);
  }
}

async function terminateBatchTask(runId) {
  const res = await batchFetch(`/jobs/${BATCH_JOB_ID}/tasks/${runId}/terminate`, { method: 'POST' });
  if (!res.ok && res.status !== 404 && res.status !== 409) {
    console.warn(`[cancel] terminate returned ${res.status}: ${await res.text()}`);
  }
}

// ── App ─────────────────────────────────────────────────────────────────

const app = express();
app.use(cors());
app.use(express.json());

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 },
});

app.get('/health', (_req, res) => res.json({ ok: true, mode: 'test-pipeline-only' }));

// POST /runSolver — same multipart shape as the real controller (env + sched),
// even though this test never looks at env — keeps the webapp unchanged.
app.post(
  '/runSolver',
  upload.fields([{ name: 'env', maxCount: 1 }, { name: 'sched', maxCount: 1 }]),
  async (req, res) => {
    try {
      const envFile   = req.files?.env?.[0];
      const schedFile = req.files?.sched?.[0];
      if (!envFile)   return res.status(400).json({ error: 'Missing required field: env (EnvConfig.yaml)' });
      if (!schedFile) return res.status(400).json({ error: 'Missing required field: sched (Schedule.yaml)' });

      const proposed = typeof req.body?.runId === 'string' ? req.body.runId.trim() : '';
      const runId = proposed && isValidRunId(proposed) ? proposed : makeRunId();

      // Task's commandLine hardcodes the filename "Schedule.yaml" — upload
      // under that exact name regardless of what the browser sent, so the
      // cp in createBatchTask always finds it.
      await container.getBlockBlobClient(`input/${runId}/${envFile.originalname}`).uploadData(envFile.buffer);
      await container.getBlockBlobClient(`input/${runId}/Schedule.yaml`).uploadData(schedFile.buffer);

      await writeJsonBlob(`status/${runId}.json`, {
        runId, status: 'Submitted', stage: null, progress: 0,
        startedAt: new Date().toISOString(), updatedAt: new Date().toISOString(),
        finishedAt: null, error: null, output: null,
      });

      await createBatchTask(runId);

      console.log(`[runSolver] created TEST run ${runId} (pipeline-only, no solver)`);
      res.status(202).json({ runId, status: 'Submitted' });
    } catch (e) {
      console.error('[runSolver]', e);
      res.status(500).json({ error: String(e.message || e) });
    }
  },
);

app.get('/status/:runId', async (req, res) => {
  const { runId } = req.params;
  if (!isValidRunId(runId)) return res.status(400).json({ error: 'Invalid runId format' });
  try {
    const status = await readJsonBlob(`status/${runId}.json`);
    if (!status) return res.status(404).json({ error: `Run "${runId}" not found` });
    res.json(status);
  } catch (e) {
    console.error('[status]', e);
    res.status(500).json({ error: String(e.message || e) });
  }
});

// GET /download/:runId — note the filename is Schedule_result.yaml here, not result_Schedule.yaml
app.get('/download/:runId', async (req, res) => {
  const { runId } = req.params;
  if (!isValidRunId(runId)) return res.status(400).json({ error: 'Invalid runId format' });
  try {
    const status = await readJsonBlob(`status/${runId}.json`);
    if (!status) return res.status(404).json({ error: `Run "${runId}" not found` });
    if (status.status === 'Failed') {
      return res.status(409).json({ error: `Run "${runId}" failed — no output available`, detail: status.error });
    }
    if (status.status !== 'Completed') {
      return res.status(409).json({ error: `Run "${runId}" is not completed yet (current status: ${status.status})` });
    }

    const blob = container.getBlockBlobClient(`output/${runId}/Schedule_result.yaml`);
    if (!(await blob.exists())) {
      return res.status(404).json({ error: `Output YAML not found for run "${runId}"` });
    }

    res.setHeader('Content-Disposition', 'attachment; filename="Schedule_result.yaml"');
    res.setHeader('Content-Type', 'application/octet-stream');
    const downloadResponse = await blob.download();
    downloadResponse.readableStreamBody.pipe(res);
  } catch (e) {
    console.error('[download]', e);
    res.status(500).json({ error: String(e.message || e) });
  }
});

app.delete('/run/:runId', async (req, res) => {
  const { runId } = req.params;
  if (!isValidRunId(runId)) return res.status(400).json({ error: 'Invalid runId format' });
  try {
    await terminateBatchTask(runId);

    let deleted = 0;
    for (const prefix of [`input/${runId}/`, `output/${runId}/`]) {
      for await (const blob of container.listBlobsFlat({ prefix })) {
        await container.deleteBlob(blob.name);
        deleted++;
      }
    }
    const statusBlobName = `status/${runId}.json`;
    if (await container.getBlockBlobClient(statusBlobName).exists()) {
      await container.deleteBlob(statusBlobName);
      deleted++;
    }

    console.log(`[cancel] ${runId} — terminated task, deleted ${deleted} blob(s)`);
    res.json({ ok: true, runId, deleted });
  } catch (e) {
    console.error('[cancel]', e);
    res.status(500).json({ error: String(e.message || e) });
  }
});

app.use((err, _req, res, _next) => {
  if (err?.name === 'MulterError') return res.status(400).json({ error: `Upload error: ${err.message}` });
  if (err) return res.status(400).json({ error: err.message });
});

app.listen(PORT, () => {
  console.log(`TEST API Controller (pipeline-only, no solver/ACR) listening on :${PORT}`);
  console.log(`  storage=${STORAGE_ACCOUNT}  container=${BLOB_CONTAINER}`);
  console.log(`  batch=${BATCH_ACCOUNT_URL}  job=${BATCH_JOB_ID}`);
});
