/**
 * API Controller v2 — Timefold + Sigma-I.
 *
 * Same HTTP contract as ../api-controller (v1, Timefold only) plus an `engine` field on
 * POST /runSolver. engine=timefold behaves exactly like v1 (same Batch task, same job).
 * engine=sigmai creates a "bridge" task in the Sigma-I Batch job (pool-sigmai).
 *
 * Endpoints:
 *   POST   /runSolver       — env + sched (files) + options → Blob input/ + Batch task
 *   GET    /status/:runId   — Blob status (+ live status from the node, + queuePosition)
 *   GET    /download/:runId — stream output/{runId}/result_Schedule.yaml
 *   DELETE /run/:runId      — terminate the Batch task + delete input/output/status blobs
 *   GET    /health          — { ok, version, engines: [...] }  ← SchedulerWeb checks `engines`
 *
 * Also writes runs/{runId}.json (history manifest; survives DELETE — see design §6.4).
 *
 * Auth: DefaultAzureCredential (Managed Identity on ACA, `az login` locally).
 * Design: documents/SigmaI/SigmaI_Azure_DetailedDesign20261013.md
 */

import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { createHash } from 'node:crypto';
import { DefaultAzureCredential } from '@azure/identity';
import { BlobServiceClient } from '@azure/storage-blob';
import {
  isValidRunId, parseRunOptions, buildSigmaiOptionsFile, wallClockDuration, queuePositionFor,
  buildInitialStatus, buildManifest, finalizeManifest, bridgeExitedWithoutStatus, TERMINAL,
} from './options.js';

// ── Config ──────────────────────────────────────────────────────────────
const STORAGE_ACCOUNT     = process.env.STORAGE_ACCOUNT;
const BLOB_CONTAINER      = process.env.BLOB_CONTAINER;
const BATCH_ACCOUNT_URL   = process.env.BATCH_ACCOUNT_URL;
const BATCH_JOB_ID        = process.env.BATCH_JOB_ID;               // Timefold job
const BATCH_JOB_ID_SIGMAI = process.env.BATCH_JOB_ID_SIGMAI || '';  // Sigma-I job; empty = Sigma-I off
const SOLVER_IMAGE        = process.env.SOLVER_IMAGE;
const ACR_LOGIN_SERVER    = process.env.ACR_LOGIN_SERVER;
const POOL_MI_RESOURCE_ID = process.env.POOL_MI_RESOURCE_ID;
const PORT                = Number(process.env.PORT ?? 8080);

const APP_VERSION       = process.env.APP_VERSION ?? '2.0.0';
const SIGMAI_SETUP_VERSION = process.env.SIGMAI_SETUP_VERSION ?? 'unset';
const SIGMAI_DWAVE_ENABLED = (process.env.SIGMAI_DWAVE_ENABLED ?? 'false').toLowerCase() === 'true';
const SIGMAI_MAX_STAGE_TIMEOUT_SEC = Number(process.env.SIGMAI_MAX_STAGE_TIMEOUT_SEC ?? 7200);
const SIGMAI_WALLCLOCK_MARGIN_SEC  = Number(process.env.SIGMAI_WALLCLOCK_MARGIN_SEC ?? 900);
const STATUS_GRACE_SEC             = Number(process.env.STATUS_GRACE_SEC ?? 120);
const SIGMAI_IMAGE_LABEL           = process.env.SIGMAI_IMAGE_LABEL ?? 'sigmai/screenas-workerassign-api';

for (const [k, v] of Object.entries({
  STORAGE_ACCOUNT, BLOB_CONTAINER, BATCH_ACCOUNT_URL, BATCH_JOB_ID,
  SOLVER_IMAGE, ACR_LOGIN_SERVER, POOL_MI_RESOURCE_ID,
})) {
  if (!v) console.warn(`[config] WARNING: env var ${k} is not set — requests that need it will fail`);
}

const OPTION_CFG = {
  sigmaiEnabled: Boolean(BATCH_JOB_ID_SIGMAI),
  dwaveEnabled: SIGMAI_DWAVE_ENABLED,
  maxStageTimeoutSec: SIGMAI_MAX_STAGE_TIMEOUT_SEC,
};

const BATCH_API_VERSION = '2024-07-01.20.0';
const BATCH_SCOPE       = 'https://batch.core.windows.net/.default';

const credential  = new DefaultAzureCredential();
const blobService = new BlobServiceClient(`https://${STORAGE_ACCOUNT}.blob.core.windows.net`, credential);
const container   = blobService.getContainerClient(BLOB_CONTAINER);

// ── Helpers ─────────────────────────────────────────────────────────────

const jobIdFor = (engine) => (engine === 'sigmai' ? BATCH_JOB_ID_SIGMAI : BATCH_JOB_ID);
const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

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

async function batchFetch(path, options = {}) {
  const token = (await credential.getToken(BATCH_SCOPE)).token;
  const sep = path.includes('?') ? '&' : '?';
  return fetch(`${BATCH_ACCOUNT_URL}${path}${sep}api-version=${BATCH_API_VERSION}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json; odata=minimalmetadata',
      Authorization: `Bearer ${token}`,
      ...(options.headers ?? {}),
    },
  });
}

// ── Batch tasks ─────────────────────────────────────────────────────────

const containerUrl = () => `https://${STORAGE_ACCOUNT}.blob.core.windows.net/${BLOB_CONTAINER}`;
const miRef = () => ({ resourceId: POOL_MI_RESOURCE_ID });
const inputResourceFiles = (runId) => [{
  autoStorageContainerName: BLOB_CONTAINER,
  blobPrefix: `input/${runId}/`,
  filePath: 'input',
  identityReference: miRef(),
}];
const outputFile = (pattern, path, condition) => ({
  filePattern: pattern,
  destination: { container: { containerUrl: containerUrl(), path, identityReference: miRef() } },
  uploadOptions: { uploadCondition: condition },
});

/** Timefold task — IDENTICAL to api-controller v1 (do not change without testing Timefold). */
function timefoldTaskSpec(runId) {
  return {
    id: runId,
    commandLine:
      "-c 'mkdir -p /work/output /work/status && /app/entrypoint.sh; " +
      "rc=$?; chmod -R a+rX /work/output /work/status; " +
      "echo ---PERMS---; ls -la /work/output /work/status; exit $rc'",
    userIdentity: { autoUser: { elevationLevel: 'admin', scope: 'task' } },
    containerSettings: {
      imageName: SOLVER_IMAGE,
      containerRunOptions:
        '--rm --entrypoint /bin/bash --user root --workdir /app ' +
        `-v $AZ_BATCH_TASK_WORKING_DIR/input/input/${runId}:/work/input:ro ` +
        '-v $AZ_BATCH_TASK_WORKING_DIR/output:/work/output ' +
        '-v $AZ_BATCH_TASK_WORKING_DIR/status:/work/status ' +
        `-e RUN_ID=${runId}`,
      registry: { registryServer: ACR_LOGIN_SERVER, identityReference: miRef() },
    },
    resourceFiles: inputResourceFiles(runId),
    outputFiles: [
      outputFile('output/result_Schedule.yaml', `output/${runId}/result_Schedule.yaml`, 'taskSuccess'),
      outputFile(`status/${runId}.json`, `status/${runId}.json`, 'taskCompletion'),
    ],
  };
}

/**
 * Sigma-I "bridge" task — a host task (no container) on a pool-sigmai node. The node's
 * StartTask already brought the Sigma-I stack up; bridge.sh just talks to 127.0.0.1:8000.
 * See sigmai-setup/bridge.sh and design §8.3.
 */
function sigmaiTaskSpec(runId, parsed) {
  return {
    id: runId,
    displayName: `sigmai ${parsed.option} ${runId}`,
    commandLine: "/bin/bash -c 'exec bash $AZ_BATCH_NODE_SHARED_DIR/sigmai/bridge.sh'",
    userIdentity: { autoUser: { elevationLevel: 'admin', scope: 'pool' } },
    environmentSettings: [{ name: 'RUN_ID', value: runId }],
    constraints: {
      maxWallClockTime: wallClockDuration(parsed.solver, SIGMAI_WALLCLOCK_MARGIN_SEC),
      maxTaskRetryCount: 1,
      retentionTime: 'P1D',
    },
    resourceFiles: inputResourceFiles(runId),
    outputFiles: [
      outputFile('output/result_Schedule.yaml', `output/${runId}/result_Schedule.yaml`, 'taskSuccess'),
      outputFile('output/sigmai/*', `output/${runId}/sigmai`, 'taskCompletion'),
      outputFile(`status/${runId}.json`, `status/${runId}.json`, 'taskCompletion'),
    ],
  };
}

async function createBatchTask(runId, parsed) {
  const spec = parsed.engine === 'sigmai' ? sigmaiTaskSpec(runId, parsed) : timefoldTaskSpec(runId);
  const res = await batchFetch(`/jobs/${jobIdFor(parsed.engine)}/tasks`, {
    method: 'POST',
    body: JSON.stringify(spec),
  });
  if (!res.ok) throw new Error(`Batch task create failed (${res.status}): ${await res.text()}`);
}

/** Full Batch task JSON, or null if it does not exist. */
async function getBatchTask(engine, runId) {
  const res = await batchFetch(`/jobs/${jobIdFor(engine)}/tasks/${runId}`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`Batch task lookup failed (${res.status}): ${await res.text()}`);
  return res.json();
}

async function readLiveStatusFile(engine, runId) {
  // The task dir is addressed as "wd/..." — Timefold bind-mounts /work/status there; the
  // Sigma-I bridge writes wd/status/{runId}.json directly.
  const res = await batchFetch(`/jobs/${jobIdFor(engine)}/tasks/${runId}/files/wd/status/${runId}.json`);
  if (!res.ok) return null;
  try { return JSON.parse(await res.text()); } catch { return null; }
}

async function getQueuePosition(engine, runId, creationTime) {
  const filter = encodeURIComponent("state eq 'active'");
  const res = await batchFetch(`/jobs/${jobIdFor(engine)}/tasks?$filter=${filter}&$select=id,creationTime`);
  if (!res.ok) return null;
  const data = await res.json();
  return queuePositionFor(runId, creationTime, data.value ?? []);
}

async function terminateBatchTask(engine, runId) {
  const res = await batchFetch(`/jobs/${jobIdFor(engine)}/tasks/${runId}/terminate`, { method: 'POST' });
  if (!res.ok && res.status !== 404 && res.status !== 409) {
    console.warn(`[cancel] terminate returned ${res.status}: ${await res.text()}`);
  }
}

// ── Manifest (runs/{runId}.json) ────────────────────────────────────────

/** Best-effort: history must never break a run. */
async function finalizeRunManifest(runId, status, task) {
  try {
    const manifest = await readJsonBlob(`runs/${runId}.json`);
    if (!manifest || manifest.result) return;
    const sigmaiResult = manifest.engine === 'sigmai'
      ? await readJsonBlob(`output/${runId}/sigmai/result.json`).catch(() => null)
      : null;
    await writeJsonBlob(`runs/${runId}.json`, finalizeManifest(manifest, status, task, sigmaiResult));
  } catch (e) {
    console.warn(`[manifest] finalize ${runId} failed: ${e.message}`);
  }
}

async function findEngineOfRun(runId) {
  const status = await readJsonBlob(`status/${runId}.json`).catch(() => null);
  if (status?.engine) return status.engine;
  const manifest = await readJsonBlob(`runs/${runId}.json`).catch(() => null);
  return manifest?.engine ?? 'timefold';
}

// ── App ─────────────────────────────────────────────────────────────────

const app = express();
app.use(cors());
app.use(express.json());

const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 50 * 1024 * 1024 } });

app.get('/health', (_req, res) => {
  const engines = ['timefold'];
  if (OPTION_CFG.sigmaiEnabled) engines.push('sigmai');
  res.json({ ok: true, version: APP_VERSION, engines, sigmaiDwave: OPTION_CFG.sigmaiEnabled && SIGMAI_DWAVE_ENABLED });
});

app.post(
  '/runSolver',
  upload.fields([{ name: 'env', maxCount: 1 }, { name: 'sched', maxCount: 1 }]),
  async (req, res) => {
    let runId = null;
    let parsed = null;
    try {
      const envFile   = req.files?.env?.[0];
      const schedFile = req.files?.sched?.[0];
      if (!envFile)   return res.status(400).json({ error: 'Missing required field: env (EnvConfig.yaml)' });
      if (!schedFile) return res.status(400).json({ error: 'Missing required field: sched (Schedule.yaml)' });

      const opt = parseRunOptions(req.body, OPTION_CFG);
      if (!opt.ok) return res.status(400).json({ error: opt.error });
      parsed = opt.value;

      const proposed = typeof req.body?.runId === 'string' ? req.body.runId.trim() : '';
      runId = proposed && isValidRunId(proposed) ? proposed : makeRunId();
      const nowIso = new Date().toISOString();

      // Timefold keeps v1 behaviour (original file names); Sigma-I uses fixed names.
      const envName   = parsed.engine === 'sigmai' ? 'EnvConfig.yaml' : envFile.originalname;
      const schedName = parsed.engine === 'sigmai' ? 'Schedule.yaml'  : schedFile.originalname;
      await container.getBlockBlobClient(`input/${runId}/${envName}`).uploadData(envFile.buffer);
      await container.getBlockBlobClient(`input/${runId}/${schedName}`).uploadData(schedFile.buffer);
      if (parsed.engine === 'sigmai') {
        await writeJsonBlob(`input/${runId}/sigmai_options.json`, buildSigmaiOptionsFile(parsed));
      }

      await writeJsonBlob(`runs/${runId}.json`, buildManifest({
        runId, parsed, nowIso,
        batchJobId: jobIdFor(parsed.engine),
        inputInfo: {
          envFile: `input/${runId}/${envName}`, schedFile: `input/${runId}/${schedName}`,
          envSha256: sha256(envFile.buffer), schedSha256: sha256(schedFile.buffer),
          originalEnvPath: parsed.originalEnvPath, originalSchedPath: parsed.originalSchedPath,
        },
        versions: {
          apiController: APP_VERSION,
          solverImage: parsed.engine === 'sigmai' ? SIGMAI_IMAGE_LABEL : SOLVER_IMAGE,
          sigmaiSetup: parsed.engine === 'sigmai' ? SIGMAI_SETUP_VERSION : null,
        },
      }));
      await writeJsonBlob(`status/${runId}.json`, buildInitialStatus({ runId, parsed, nowIso }));

      try {
        await createBatchTask(runId, parsed);
      } catch (e) {
        // v1 left the status at "Submitted" forever here — record the failure instead.
        const failed = {
          ...buildInitialStatus({ runId, parsed, nowIso }),
          status: 'Failed', finishedAt: new Date().toISOString(),
          error: { type: 'SubmitFailed', message: String(e.message || e) },
        };
        await writeJsonBlob(`status/${runId}.json`, failed);
        await finalizeRunManifest(runId, failed, null);
        throw e;
      }

      console.log(`[runSolver] created run ${runId} engine=${parsed.engine} option=${parsed.option}`);
      res.status(202).json({ runId, status: 'Submitted', engine: parsed.engine });
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
    const engine = status.engine ?? 'timefold';

    if (TERMINAL.includes(status.status)) {
      void finalizeRunManifest(runId, status, null); // no-op when already finalized
      return res.json(status);
    }

    // Blob only has "Submitted" (at /runSolver) or the final status (uploaded at task end).
    // While the task is queued/running, prefer the live file on the node.
    const task = await getBatchTask(engine, runId).catch(() => null);
    const state = task?.state;
    let out = status;

    if (state === 'active' || state === 'preparing' || state === 'running') {
      const live = await readLiveStatusFile(engine, runId).catch(() => null);
      if (live) out = { ...live, engine, option: status.option, solver: status.solver };
      if (state === 'active') {
        const pos = await getQueuePosition(engine, runId, task.creationTime).catch(() => null);
        if (pos != null) out = { ...out, queuePosition: pos };
      }
    } else if (state === 'completed' && engine === 'sigmai') {
      const failed = bridgeExitedWithoutStatus(status, task, STATUS_GRACE_SEC);
      if (failed) {
        await writeJsonBlob(`status/${runId}.json`, failed).catch(() => {});
        void finalizeRunManifest(runId, failed, task);
        return res.json(failed);
      }
    }

    if (TERMINAL.includes(out.status)) void finalizeRunManifest(runId, out, task);
    res.json(out);
  } catch (e) {
    console.error('[status]', e);
    res.status(500).json({ error: String(e.message || e) });
  }
});

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
    (await blob.download()).readableStreamBody.pipe(res);
  } catch (e) {
    console.error('[download]', e);
    res.status(500).json({ error: String(e.message || e) });
  }
});

app.delete('/run/:runId', async (req, res) => {
  const { runId } = req.params;
  if (!isValidRunId(runId)) return res.status(400).json({ error: 'Invalid runId format' });
  try {
    const engine = await findEngineOfRun(runId);
    await terminateBatchTask(engine, runId);

    // Record the outcome in the history before the status blob is deleted.
    const status = await readJsonBlob(`status/${runId}.json`).catch(() => null);
    if (status) {
      const final = TERMINAL.includes(status.status)
        ? status
        : { ...status, status: 'Cancelled', finishedAt: new Date().toISOString() };
      const task = await getBatchTask(engine, runId).catch(() => null);
      await finalizeRunManifest(runId, final, task);
    }
    const manifest = await readJsonBlob(`runs/${runId}.json`).catch(() => null);
    if (manifest) await writeJsonBlob(`runs/${runId}.json`, { ...manifest, deleted: true }).catch(() => {});

    let deleted = 0;
    for (const prefix of [`input/${runId}/`, `output/${runId}/`]) {
      for await (const blob of container.listBlobsFlat({ prefix })) {
        await container.deleteBlob(blob.name);
        deleted++;
      }
    }
    const statusBlob = container.getBlockBlobClient(`status/${runId}.json`);
    if (await statusBlob.exists()) { await statusBlob.delete(); deleted++; }

    console.log(`[cancel] ${runId} (${engine}) — terminated task, deleted ${deleted} blob(s)`);
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
  console.log(`API Controller v${APP_VERSION} listening on :${PORT}`);
  console.log(`  storage=${STORAGE_ACCOUNT}  container=${BLOB_CONTAINER}`);
  console.log(`  batch=${BATCH_ACCOUNT_URL}  timefoldJob=${BATCH_JOB_ID}  sigmaiJob=${BATCH_JOB_ID_SIGMAI || '(disabled)'}`);
  console.log(`  sigmai dwave=${SIGMAI_DWAVE_ENABLED}  maxStageTimeout=${SIGMAI_MAX_STAGE_TIMEOUT_SEC}s`);
});
