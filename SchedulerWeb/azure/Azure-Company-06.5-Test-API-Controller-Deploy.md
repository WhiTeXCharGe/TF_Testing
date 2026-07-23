# Company Phase 6.5 — Deploy the pipeline-only test controller

**Why this phase exists:** the real Timefold solver has open questions
(input/output YAML format, Java/Maven behavior) that are separate from
whether the *pipeline itself* (webapp → API → Blob → Batch → Blob → API →
webapp) works. This phase deploys
[web/api-controller-test/](../../api-controller-test/) — a stripped-down
controller that skips the solver entirely: its Batch task just copies the
uploaded `Schedule.yaml` to `Schedule_result.yaml` and reports Completed.
**No container, no Docker image for the task, no ACR dependency for the
compute step at all** — only plain shell commands on the pool node. This
also cleanly isolates today's `FileUploadMiscError` problem: if this
simpler task hits the same error, that proves it's a Storage-side
network/RBAC issue, not anything about containers or entrypoint.sh.

Same HTTP contract as the real controller
([Azure-Company-06](./Azure-Company-06-API-Controller-Deploy.md)) — the
webapp needs **zero code changes** to point at this instead; only the file
it gets back is renamed (`Schedule_result.yaml` instead of
`result_Schedule.yaml`) and unmodified from whatever was uploaded.

**Time:** ~20 minutes. **Prereqs:** Phase 1–5 done — `$ST`, `$CONTAINER`,
`$BATCH`, `$BATCH_URL`, `$JOB_ID`, `$MI_ID` all set. The pool does **not**
need to exist/be healthy for you to build and push this image — it's only
needed at actual run time.

---

## Step 1 — Test locally first (fastest way to catch mistakes)

```bash
cd /c/Users/Seiya/Desktop/work/Timefold/web/api-controller-test
npm install

source ~/azure-timefold-company-env.sh
export STORAGE_ACCOUNT="$ST"
export BLOB_CONTAINER="$CONTAINER"
export BATCH_ACCOUNT_URL="$BATCH_URL"
export BATCH_JOB_ID="$JOB_ID"
export POOL_MI_RESOURCE_ID="$MI_ID"
export PORT=8080

npm run dev
```

In a second terminal:
```bash
curl http://localhost:8080/health
# {"ok":true,"mode":"test-pipeline-only"}

curl -X POST http://localhost:8080/runSolver \
  -F "env=@/c/Users/Seiya/Desktop/work/Timefold/web/Timefold/src/main/resource/EnvConfig.yaml" \
  -F "sched=@/c/Users/Seiya/Desktop/work/Timefold/web/Timefold/src/main/resource/Schedule.yaml"
# {"runId":"...","status":"Submitted"}

RUNID=<paste the runId>
curl http://localhost:8080/status/$RUNID
# poll this a few times — Submitted -> (no Running, see note below) -> Completed

curl http://localhost:8080/download/$RUNID -o /tmp/Schedule_result.yaml
diff /c/Users/Seiya/Desktop/work/Timefold/web/Timefold/src/main/resource/Schedule.yaml /tmp/Schedule_result.yaml
# should be identical — it's a pure passthrough, no solving happened

curl -X DELETE http://localhost:8080/run/$RUNID
```

If `diff` shows no differences and the whole cycle completes, the pipeline
is proven end-to-end **and** this also proves `FileUploadMiscError` isn't
container-related — output upload works fine for a plain task. If you
*still* hit `FileUploadMiscError` here, that's now conclusively a
Storage-side network/RBAC issue (see the diagnostic in
[Azure-Company-05 troubleshooting](./Azure-Company-05-Batch-Setup-And-Run.md#troubleshooting))
— fix that first before deploying anything.

Stop the local server (Ctrl+C) once confirmed.

## Step 2 — Build the image

```bash
cd /c/Users/Seiya/Desktop/work/Timefold/web/api-controller-test
TAG=test1
docker build -t "$ACR_LOGIN/api-controller-test:$TAG" .
```

## Step 3 — Push it

Same network caveat as every ACR step today — confirm you're on the
company network first (see the warning at the top of
[Azure-Company-01](./Azure-Company-01-Access-And-Resources.md)).

```bash
az acr login --name "$ACR"
docker push "$ACR_LOGIN/api-controller-test:$TAG"
az acr repository show-tags --name "$ACR" --repository api-controller-test -o table
```

## Step 4 — Deploy to the existing ACA app

This reuses `$ACA_APP` rather than creating a new Container App — no new
resource, no new admin ask. It **replaces** whatever revision is currently
running there (the real api-controller, if you'd already deployed it) —
that's fine, switching back later is a one-line command (Step 6 below).

```bash
source ~/azure-timefold-company-env.sh

az containerapp update \
  --name "$ACA_APP" \
  --resource-group "$RG" \
  --image "$ACR_LOGIN/api-controller-test:$TAG" \
  --set-env-vars \
    STORAGE_ACCOUNT="$ST" \
    BLOB_CONTAINER="$CONTAINER" \
    BATCH_ACCOUNT_URL="$BATCH_URL" \
    BATCH_JOB_ID="$JOB_ID" \
    POOL_MI_RESOURCE_ID="$MI_ID"

az containerapp ingress update \
  --name "$ACA_APP" \
  --resource-group "$RG" \
  --target-port 8080
```

Note: no `SOLVER_IMAGE` / `ACR_LOGIN_SERVER` env vars needed — this
controller's Batch task never references a container image at all.

## Step 5 — Smoke test the deployed app

```bash
echo "https://$APP_URL"
curl https://$APP_URL/health
# {"ok":true,"mode":"test-pipeline-only"}
```

Then repeat the same `curl` sequence from Step 1, swapping `localhost:8080`
for `https://$APP_URL`. If it completes, you have a **fully working, fully
deployed, real Azure pipeline demo** — independent of whether the Timefold
solver itself is solved yet.

## Step 6 — Point the webapp at it for a live demo

Same as [Azure-Company-07](./Azure-Company-07-Webapp-Connect.md) — no
webapp code changes needed:

```bash
cd /c/Users/Seiya/Desktop/work/Timefold/web/SchedulerWeb
cat > .env.local <<EOF
VITE_API_BASE_URL=https://$APP_URL
EOF
npm run dev
```

Click through New Run in the browser — the file you get back on Show
Result / download will be an exact copy of the Schedule.yaml you uploaded,
renamed. That's the whole point: it proves the UI, upload, Blob storage,
Batch task execution, output retrieval, and download all work — the only
thing not being tested is the solver's actual scheduling logic.

## Switching back to the real (solver) controller later

```bash
az containerapp update \
  --name "$ACA_APP" \
  --resource-group "$RG" \
  --image "$ACR_LOGIN/api-controller:v1" \
  --set-env-vars \
    STORAGE_ACCOUNT="$ST" \
    BLOB_CONTAINER="$CONTAINER" \
    BATCH_ACCOUNT_URL="$BATCH_URL" \
    BATCH_JOB_ID="$JOB_ID" \
    SOLVER_IMAGE="$SOLVER_IMAGE" \
    ACR_LOGIN_SERVER="$ACR_LOGIN" \
    POOL_MI_RESOURCE_ID="$MI_ID"
```
(Full env var set restored — the test deploy only set a subset.)

---

## What you should have at the end of this phase

- [ ] Local test: full Submitted → Completed cycle, downloaded file identical to upload
- [ ] Image pushed to `$ACR` as `api-controller-test:test1`
- [ ] `$ACA_APP` updated, `/health` returns `{"ok":true,"mode":"test-pipeline-only"}`
- [ ] A real run through the deployed app completes and downloads correctly
- [ ] Webapp demo works end-to-end against this controller

---

## Troubleshooting

| Symptom | Cause | Fix |
| ------- | ----- | --- |
| `FileUploadMiscError` even on this non-container task | Confirms it's not container/entrypoint-related — Storage-side network or RBAC | Check `az storage account show --name "$ST" --query "{publicNetworkAccess:publicNetworkAccess, defaultAction:networkRuleSet.defaultAction}"` — same pattern as the ACR fix earlier |
| `cp: cannot stat 'input/input/<runId>/Schedule.yaml'` in task logs | The `filePath`/`blobPrefix` double-nesting behavior didn't apply the way expected — confirm the actual path via `az batch task file list` | Download `stdout.txt`/`stderr.txt` and check what's actually under `input/` in the task working directory |
| Task fails immediately, no useful log | Base64 status payload malformed (rare — only if `runId` somehow contains characters breaking JSON, which `isValidRunId` should already prevent) | Check `az batch task file download --file-path stderr.txt` |
