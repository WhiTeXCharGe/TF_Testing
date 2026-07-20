/**
 * Timefold API Controller — Azure edition.
 *
 * Same HTTP contract as ../service/server.js (the local dev mock), but backed
 * by real Azure Blob Storage (input/output/status) and Azure Batch (runs the
 * Timefold container as a task) instead of local disk + `docker run`.
 *
 * Endpoints (must match webapp/src/services/runService.ts exactly):
 *   POST   /runSolver      — upload EnvConfig.yaml + Schedule.yaml, create a Batch task
 *   GET    /status/:runId  — read status/{runId}.json from Blob
 *   GET    /download/:runId — stream output/{runId}/result_Schedule.yaml from Blob
 *   DELETE /run/:runId     — terminate the Batch task (if running) + delete all blobs for the run
 *   GET    /health          — smoke test
 *
 * Auth: DefaultAzureCredential — picks up the Container App's system-assigned
 * Managed Identity automatically. No secrets/keys anywhere in this file.
 */

import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { DefaultAzureCredential } from '@azure/identity';
import { BlobServiceClient } from '@azure/storage-blob';

// ── Config (set as env vars on the Container App — see Azure-Company-06) ──
const STORAGE_ACCOUNT     = process.env.STORAGE_ACCOUNT;
const BLOB_CONTAINER      = process.env.BLOB_CONTAINER;
const BATCH_ACCOUNT_URL   = process.env.BATCH_ACCOUNT_URL;    // https://<batch-account-name>.<region>.batch.azure.com
const BATCH_JOB_ID        = process.env.BATCH_JOB_ID;
const SOLVER_IMAGE        = process.env.SOLVER_IMAGE;         // <acr-name>.azurecr.io/timefold:v1
const ACR_LOGIN_SERVER    = process.env.ACR_LOGIN_SERVER;     // <acr-name>.azurecr.io
const POOL_MI_RESOURCE_ID = process.env.POOL_MI_RESOURCE_ID;  // /subscriptions/.../userAssignedIdentities/<name>
const PORT                = Number(process.env.PORT ?? 8080);

for (const [k, v] of Object.entries({
  STORAGE_ACCOUNT, BLOB_CONTAINER, BATCH_ACCOUNT_URL, BATCH_JOB_ID,
  SOLVER_IMAGE, ACR_LOGIN_SERVER, POOL_MI_RESOURCE_ID,
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
 * Create one Batch task for this run. The pool/job already exist (created in
 * Azure-Company-05) — this only submits the task.
 *
 * Both outputFiles entries use an EXACT filePattern (no wildcard) so the
 * destination.path is used as the literal blob name — no ambiguity about
 * where the file lands, unlike a wildcard pattern with a prefix path.
 */
async function createBatchTask(runId) {
  const containerUrl = `https://${STORAGE_ACCOUNT}.blob.core.windows.net/${BLOB_CONTAINER}`;

  const task = {
    id: runId,
    // Run entrypoint, then chmod the outputs so Batch's outputFiles upload can
    // read them, then exit with the solver's real return code. `-c` (not
    // `/bin/bash -c`) because containerRunOptions overrides the entrypoint to
    // /bin/bash below.
    commandLine:
      "-c 'mkdir -p /work/output /work/status && /app/entrypoint.sh; " +
      "rc=$?; chmod -R a+rX /work/output /work/status; " +
      "echo ---PERMS---; ls -la /work/output; exit $rc'",
    userIdentity: { autoUser: { elevationLevel: 'admin', scope: 'task' } },
    containerSettings: {
      imageName: SOLVER_IMAGE,
      containerRunOptions:
        '--rm --entrypoint /bin/bash --user root --workdir /app ' +
        // resourceFiles (filePath "input" + blobPrefix "input/<runId>/") lands
        // the files at $AZ_BATCH_TASK_WORKING_DIR/input/input/<runId>/, because
        // Batch preserves the full blob path under filePath. Mount THAT dir.
        `-v $AZ_BATCH_TASK_WORKING_DIR/input/input/${runId}:/work/input:ro ` +
        '-v $AZ_BATCH_TASK_WORKING_DIR/output:/work/output ' +
        '-v $AZ_BATCH_TASK_WORKING_DIR/status:/work/status ' +
        `-e RUN_ID=${runId}`,
      registry: {
        registryServer: ACR_LOGIN_SERVER,
        identityReference: { resourceId: POOL_MI_RESOURCE_ID },
      },
    },
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
        filePattern: 'output/result_Schedule.yaml',
        destination: {
          container: {
            containerUrl,
            path: `output/${runId}/result_Schedule.yaml`,
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
        // taskCompletion = always upload, even on failure/cancel, so status.json
        // (written by entrypoint.sh's write_failed/write_cancelled) always reaches Blob.
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
  // 404/409 = task doesn't exist or already finished — both fine to ignore.
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

app.get('/health', (_req, res) => res.json({ ok: true }));

// POST /runSolver — multipart: env (file), sched (file), runId (optional)
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

      await container.getBlockBlobClient(`input/${runId}/${envFile.originalname}`).uploadData(envFile.buffer);
      await container.getBlockBlobClient(`input/${runId}/${schedFile.originalname}`).uploadData(schedFile.buffer);

      await writeJsonBlob(`status/${runId}.json`, {
        runId, status: 'Submitted', stage: null, progress: 0,
        startedAt: new Date().toISOString(), updatedAt: new Date().toISOString(),
        finishedAt: null, error: null, output: null,
      });

      await createBatchTask(runId);

      console.log(`[runSolver] created run ${runId}`);
      res.status(202).json({ runId, status: 'Submitted' });
    } catch (e) {
      console.error('[runSolver]', e);
      res.status(500).json({ error: String(e.message || e) });
    }
  },
);

// GET /status/:runId
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

// GET /download/:runId — streams the file directly (webapp expects a raw blob response, not JSON)
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

    const blob = container.getBlockBlobClient(`output/${runId}/result_Schedule.yaml`);
    if (!(await blob.exists())) {
      return res.status(404).json({ error: `Output YAML not found for run "${runId}"` });
    }

    res.setHeader('Content-Disposition', 'attachment; filename="result_Schedule.yaml"');
    res.setHeader('Content-Type', 'application/octet-stream');
    const downloadResponse = await blob.download();
    downloadResponse.readableStreamBody.pipe(res);
  } catch (e) {
    console.error('[download]', e);
    res.status(500).json({ error: String(e.message || e) });
  }
});

// DELETE /run/:runId — cancel the Batch task (if any) + purge all blobs for this run
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
  console.log(`API Controller listening on :${PORT}`);
  console.log(`  storage=${STORAGE_ACCOUNT}  container=${BLOB_CONTAINER}`);
  console.log(`  batch=${BATCH_ACCOUNT_URL}  job=${BATCH_JOB_ID}`);
  console.log(`  solverImage=${SOLVER_IMAGE}`);
});