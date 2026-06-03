# Phase 6 — API Controller (Blob only, no Batch yet)

**Goal of this phase:** a real Node/Express API running on your ACA app
(`ca-tf-api`) that the webapp will talk to. It exposes the same endpoint
shapes locked in [Azure.md](../Azure.md), but with **Batch orchestration
stubbed out for now** — perfect for demoing the API + Blob plumbing without
needing the compute layer ready.

By the end:
- `https://ca-tf-api.<region>.azurecontainerapps.io/runSolver` (and friends)
  is live
- Webapp uploads → land in Blob input/{runId}/
- Status flips Submitted → Completed via a `mock-complete` shortcut
- Download returns a real SAS URL the browser can follow

**Time:** ~60 min (most of it is writing the code + first build/push).
**Cost:** $0 extra (ACA stays in free tier; ACR already $5/mo).
**Prereqs:** Phases 1–4 done. Docker Desktop running. Node 18+ on your laptop.

> When Batch is ready in the company environment, you swap the mock-complete
> path for "create Batch task" — about 30 lines of code. The rest is unchanged.

---

## What the API does (and doesn't, in v0)

| Endpoint                      | v0 (this phase)                                                          | v1 (with Batch later)                          |
| ----------------------------- | ------------------------------------------------------------------------ | ---------------------------------------------- |
| `POST /runSolver`             | Upload to Blob, write status=Submitted, return runId                     | Same + create Batch task                       |
| `GET /status/{runId}`         | Read status.json from Blob                                               | Same                                           |
| `GET /download/{runId}`       | Check output yaml exists, return SAS URL                                 | Same                                           |
| `POST /cancel/{runId}`        | Set status=Cancelled                                                     | Same + Batch task terminate                    |
| `DELETE /run/{runId}`         | Delete blobs                                                             | Same + cancel Batch task first if active       |
| `POST /mock-complete/{runId}` | Copy input/schedule.yaml → output/result_Schedule.yaml, status=Completed | **REMOVE** (real Batch fills the role)         |

---

## Step 0 — Set up the API project locally

```bash
mkdir -p /c/Users/Seiya/Desktop/work/Timefold/web/api-controller
cd /c/Users/Seiya/Desktop/work/Timefold/web/api-controller

# Tiny Node project
npm init -y

# Runtime deps
npm install express multer cors @azure/identity @azure/storage-blob

# Dev-only
npm install --save-dev nodemon
```

Set `"type": "module"` in `package.json` so we can use `import`:

```bash
node -e "const p=require('./package.json'); p.type='module'; p.scripts={start:'node src/server.js', dev:'nodemon src/server.js'}; require('fs').writeFileSync('package.json', JSON.stringify(p, null, 2))"
```

---

## Step 1 — Write the API (one file, ~180 lines)

Create `src/server.js`:

```bash
mkdir -p src
```

Save the following as `src/server.js`:

```javascript
// API Controller — Blob-only v0. Batch wiring stubbed (see /mock-complete).
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import { DefaultAzureCredential } from '@azure/identity';
import {
  BlobServiceClient,
  generateBlobSASQueryParameters,
  BlobSASPermissions,
  UserDelegationKey,
} from '@azure/storage-blob';

// ── Config (from env vars set on the ACA app) ─────────────────────────────
const STORAGE_ACCOUNT   = process.env.STORAGE_ACCOUNT  ?? 'sttimefolddevseiya';
const BLOB_CONTAINER    = process.env.BLOB_CONTAINER   ?? 'timefold';
const PORT              = Number(process.env.PORT ?? 8080);

// ── Azure Blob client using the ACA app's Managed Identity ────────────────
const credential = new DefaultAzureCredential();
const blobService = new BlobServiceClient(
  `https://${STORAGE_ACCOUNT}.blob.core.windows.net`,
  credential
);
const container = blobService.getContainerClient(BLOB_CONTAINER);

// ── Helpers ───────────────────────────────────────────────────────────────
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 25 * 1024 * 1024 } });

function makeRunId() {
  const d = new Date(), pad = (n, w = 2) => String(n).padStart(w, '0');
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_` +
         `${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}${pad(d.getMilliseconds(), 3)}`;
}

async function writeJsonBlob(name, obj) {
  const body = Buffer.from(JSON.stringify(obj, null, 2), 'utf8');
  const blob = container.getBlockBlobClient(name);
  await blob.uploadData(body, { blobHTTPHeaders: { blobContentType: 'application/json' } });
}

async function readJsonBlob(name) {
  const blob = container.getBlockBlobClient(name);
  if (!await blob.exists()) return null;
  const buf = await blob.downloadToBuffer();
  return JSON.parse(buf.toString('utf8'));
}

async function makeSasUrl(blobName, minutesValid = 15) {
  // User-delegation SAS — signed by the MI's AAD token, no account key.
  const now = new Date();
  const startsOn  = new Date(now.getTime() - 5 * 60 * 1000);
  const expiresOn = new Date(now.getTime() + minutesValid * 60 * 1000);
  const udk = await blobService.getUserDelegationKey(startsOn, expiresOn);
  const sas = generateBlobSASQueryParameters({
    containerName: BLOB_CONTAINER,
    blobName,
    permissions: BlobSASPermissions.parse('r'),
    startsOn, expiresOn,
    protocol: 'https',
  }, udk, STORAGE_ACCOUNT).toString();
  return `https://${STORAGE_ACCOUNT}.blob.core.windows.net/${BLOB_CONTAINER}/${blobName}?${sas}`;
}

// ── App ───────────────────────────────────────────────────────────────────
const app = express();
app.use(cors());                          // open CORS for demo; lock down later
app.use(express.json());

app.get('/health', (_req, res) => res.json({ ok: true }));

// POST /runSolver  multipart: env (file), sched (file), originalEnvPath, originalSchedPath
app.post('/runSolver',
  upload.fields([{ name: 'env', maxCount: 1 }, { name: 'sched', maxCount: 1 }]),
  async (req, res) => {
    try {
      const envFile   = req.files?.env?.[0];
      const schedFile = req.files?.sched?.[0];
      if (!envFile || !schedFile) return res.status(400).json({ error: 'env and sched files required' });

      const runId = req.body.runId?.match(/^[\w-]+$/) ? req.body.runId : makeRunId();
      const envBlobName   = `input/${runId}/${envFile.originalname}`;
      const schedBlobName = `input/${runId}/${schedFile.originalname}`;

      await container.getBlockBlobClient(envBlobName).uploadData(envFile.buffer);
      await container.getBlockBlobClient(schedBlobName).uploadData(schedFile.buffer);

      const status = {
        runId, status: 'Submitted',
        startedAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
        originalEnvPath:   req.body.originalEnvPath   ?? null,
        originalSchedPath: req.body.originalSchedPath ?? null,
        savedEnvPath:   `/${BLOB_CONTAINER}/${envBlobName}`,
        savedSchedPath: `/${BLOB_CONTAINER}/${schedBlobName}`,
        output: null,
        error: null,
      };
      await writeJsonBlob(`status/${runId}.json`, status);
      res.status(202).json({ runId });
    } catch (e) {
      console.error(e);
      res.status(500).json({ error: String(e.message || e) });
    }
  });

app.get('/status/:runId', async (req, res) => {
  const s = await readJsonBlob(`status/${req.params.runId}.json`);
  if (!s) return res.status(404).json({ error: 'not found' });
  res.json(s);
});

app.get('/download/:runId', async (req, res) => {
  const runId = req.params.runId;
  const status = await readJsonBlob(`status/${runId}.json`);
  if (!status) return res.status(404).json({ error: 'unknown runId' });
  if (status.status !== 'Completed') return res.status(404).json({ ready: false, status: status.status });
  const outBlob = `output/${runId}/result_Schedule.yaml`;
  if (!await container.getBlockBlobClient(outBlob).exists()) {
    return res.status(404).json({ ready: false, status: status.status, error: 'output missing' });
  }
  const url = await makeSasUrl(outBlob, 15);
  res.json({ url, expiresInMinutes: 15 });
});

app.post('/cancel/:runId', async (req, res) => {
  const runId = req.params.runId;
  const status = await readJsonBlob(`status/${runId}.json`);
  if (!status) return res.status(404).json({ error: 'unknown runId' });
  if (['Completed', 'Failed', 'Cancelled'].includes(status.status)) {
    return res.status(409).json({ ok: false, status: status.status });
  }
  status.status = 'Cancelled';
  status.finishedAt = new Date().toISOString();
  status.updatedAt  = status.finishedAt;
  await writeJsonBlob(`status/${runId}.json`, status);
  res.json({ ok: true, status: 'Cancelled' });
});

app.delete('/run/:runId', async (req, res) => {
  const runId = req.params.runId;
  let deleted = 0;
  for await (const blob of container.listBlobsFlat({ prefix: `input/${runId}/` }))   { await container.deleteBlob(blob.name); deleted++; }
  for await (const blob of container.listBlobsFlat({ prefix: `output/${runId}/` }))  { await container.deleteBlob(blob.name); deleted++; }
  for await (const blob of container.listBlobsFlat({ prefix: `status/${runId}` }))   { await container.deleteBlob(blob.name); deleted++; }
  res.json({ ok: true, deleted });
});

// DEV ONLY — simulate solver completion without Batch by copying input → output.
// Remove this endpoint once Batch is wired in.
app.post('/mock-complete/:runId', async (req, res) => {
  const runId = req.params.runId;
  const status = await readJsonBlob(`status/${runId}.json`);
  if (!status) return res.status(404).json({ error: 'unknown runId' });

  // Find the schedule blob from status
  const schedBlobName = status.savedSchedPath?.replace(`/${BLOB_CONTAINER}/`, '');
  if (!schedBlobName) return res.status(400).json({ error: 'no savedSchedPath in status' });

  const src = container.getBlockBlobClient(schedBlobName);
  const dstName = `output/${runId}/result_Schedule.yaml`;
  const dst = container.getBlockBlobClient(dstName);
  // simple server-side copy
  await dst.beginCopyFromURL(src.url + '?' + (await makeSasUrl(schedBlobName, 5)).split('?')[1]);

  status.status = 'Completed';
  status.finishedAt = new Date().toISOString();
  status.updatedAt  = status.finishedAt;
  status.output = `/${BLOB_CONTAINER}/${dstName}`;
  await writeJsonBlob(`status/${runId}.json`, status);

  res.json({ ok: true, status: 'Completed', output: status.output });
});

app.listen(PORT, () => console.log(`API listening on :${PORT}  storage=${STORAGE_ACCOUNT}  container=${BLOB_CONTAINER}`));
```

---

## Step 2 — Dockerfile for the API

Create `Dockerfile` in `web/api-controller/`:

```dockerfile
FROM node:20-alpine AS deps
WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev

FROM node:20-alpine
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY package*.json ./
COPY src ./src
ENV NODE_ENV=production
EXPOSE 8080
CMD ["node", "src/server.js"]
```

And a `.dockerignore`:

```
node_modules
.git
*.log
.env
```

---

## Step 3 — Test locally first

```bash
cd /c/Users/Seiya/Desktop/work/Timefold/web/api-controller

# Run against Azure storage using your own AAD identity (DefaultAzureCredential picks it up)
STORAGE_ACCOUNT=$ST BLOB_CONTAINER=$CONTAINER npm run dev
```

In another terminal, test:
```bash
curl http://localhost:8080/health
# {"ok":true}

curl -X POST http://localhost:8080/runSolver \
  -F "env=@/c/Users/Seiya/Desktop/work/Timefold/web/Timefold/src/main/resource/EnvConfig.yaml" \
  -F "sched=@/c/Users/Seiya/Desktop/work/Timefold/web/Timefold/src/main/resource/Schedule.yaml" \
  -F "originalEnvPath=C:/work/EnvConfig.yaml" \
  -F "originalSchedPath=C:/work/Schedule.yaml"
# {"runId":"20260605_120000123"}

# Grab the runId from above and substitute it:
RUNID=20260605_120000123
curl http://localhost:8080/status/$RUNID
# {"runId":"...","status":"Submitted",...}

curl -X POST http://localhost:8080/mock-complete/$RUNID
# {"ok":true,"status":"Completed","output":"/timefold/output/.../result_Schedule.yaml"}

curl http://localhost:8080/download/$RUNID
# {"url":"https://...","expiresInMinutes":15}

curl -X DELETE http://localhost:8080/run/$RUNID
# {"ok":true,"deleted":3}
```

If all of those work locally → you're ready to deploy.

---

## Step 4 — Build + push the API image to ACR

```bash
cd /c/Users/Seiya/Desktop/work/Timefold/web/api-controller
TAG=v1
docker build -t $ACR_LOGIN/api-controller:$TAG .

az acr login --name $ACR
docker push $ACR_LOGIN/api-controller:$TAG
```

Verify:
```bash
az acr repository show-tags --name $ACR --repository api-controller -o table
```

---

## Step 5 — Grant the ACA app's MI permission to pull from ACR

(Same portal-flow as Phase 3.2 — using portal because CLI is broken for role assignments.)

1. Portal → Container registries → `acrtimefolddevseiya` → **Access control (IAM)**
2. **+ Add** → **Add role assignment**
3. Role: `AcrPull` → Next
4. Members: **Managed identity** → **+ Select members** → Container app → `ca-tf-api` → Select
5. Review + assign

---

## Step 6 — Deploy a new revision of `ca-tf-api` pointing at the API image

```bash
source ~/azure-timefold-env.sh

# Update the ACA app to use our image, pass storage env vars, expose port 8080
az containerapp update \
  --name $ACA_APP \
  --resource-group $RG \
  --image $ACR_LOGIN/api-controller:v1 \
  --set-env-vars STORAGE_ACCOUNT=$ST BLOB_CONTAINER=$CONTAINER

# Update the ingress target port (was 80 for the hello-world; ours is 8080)
az containerapp ingress update \
  --name $ACA_APP \
  --resource-group $RG \
  --target-port 8080
```

Wait ~30 seconds for the new revision to come up, then test:

```bash
echo "API base: https://$APP_URL"

curl https://$APP_URL/health
# {"ok":true}

curl -X POST https://$APP_URL/runSolver \
  -F "env=@/c/Users/Seiya/Desktop/work/Timefold/web/Timefold/src/main/resource/EnvConfig.yaml" \
  -F "sched=@/c/Users/Seiya/Desktop/work/Timefold/web/Timefold/src/main/resource/Schedule.yaml"
# {"runId":"..."}
```

If you get a `runId` back, your API is **live on Azure** talking to **real Blob Storage**. Confirm in the portal: Storage account → Containers → `timefold` → `input/<runId>/` should have both YAMLs.

---

## What you should have at the end of Phase 6

- [ ] `web/api-controller/` exists with `src/server.js`, `Dockerfile`, `package.json`
- [ ] Local `npm run dev` works against Azure Blob
- [ ] Image pushed to ACR as `$ACR_LOGIN/api-controller:v1`
- [ ] ACA app `ca-tf-api`'s MI has `AcrPull` on ACR (portal-verified)
- [ ] ACA app updated to pull the API image, port 8080, env vars set
- [ ] `curl https://$APP_URL/health` returns `{"ok":true}`
- [ ] `curl -X POST .../runSolver` returns a `runId` and blobs appear in storage
- [ ] `curl -X POST .../mock-complete/<runId>` flips status to Completed
- [ ] `curl .../download/<runId>` returns a SAS URL the browser can open

Tell me **"Phase 6 done"** and we'll do Phase 7 — wiring the React webapp to call this Azure API.

---

## Troubleshooting

| Symptom                                            | Cause                                              | Fix                                                                          |
| -------------------------------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------- |
| Local `npm run dev` → `AuthorizationPermissionMismatch` | Your user doesn't have Blob role yet            | Phase 2 Step 3 / portal grant                                                |
| `docker build` errors on `npm ci`                  | Lock file out of sync                              | Delete `package-lock.json`, run `npm install`, rebuild                       |
| ACA revision deploy stuck or app returns 503       | Image pull failing (no AcrPull on MI)              | Step 5 above; check portal IAM                                               |
| API returns `AuthorizationPermissionMismatch`      | ACA MI doesn't have Blob role yet                  | Phase 3 Step 6 portal grant (you may have skipped this)                      |
| `/runSolver` returns 400 "env and sched files required" | Wrong field name in multipart                  | Use `env` and `sched` exactly (not `envFile` etc.)                           |
| `/download` returns 404 with `status: "Submitted"` | No output yet — that's correct                     | Call `/mock-complete/{runId}` first to simulate completion                   |

---

## When Batch is ready (later, ~30 LOC change)

Replace `/mock-complete` and add ~30 lines to `/runSolver` to also create
a Batch task. Pattern:

```javascript
// Inside POST /runSolver, after the blob uploads + status write:
const batch = new BatchServiceClient(credential, `https://${BATCH}.batch.azure.com`);
await batch.task.add('job-timefold-runs', {
  id: runId,
  commandLine: '/bin/bash -c "/app/entrypoint.sh"',
  containerSettings: { /* same as Phase 5 Step 7 JSON */ },
  resourceFiles: [ /* same */ ],
  outputFiles: [ /* same */ ],
});
```

The status updates and SAS URL flow stay identical — Batch writes status.json
from the compute node, the API just reads it.
