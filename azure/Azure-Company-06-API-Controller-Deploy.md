# Company Phase 6 — Deploy the real API Controller

**Goal of this phase:** the code already written for you at
[web/api-controller/](../../api-controller/) — a small Node/Express app that
does exactly what you did by hand in Phase 5, but as an HTTP API — gets
built, pushed to `$ACR`, and deployed onto the real `$ACA_APP`. By the end,
`https://$APP_URL/runSolver` runs a full Timefold solve on Azure Batch.

**What this code does (already written, nothing to author):**

| Endpoint                | What it does |
| ------------------------ | -------------- |
| `POST /runSolver`        | Upload EnvConfig.yaml + Schedule.yaml to Blob, write `status/{runId}.json` = Submitted, create the Batch task (same JSON as Phase 5 Step 8) |
| `GET /status/:runId`     | Read `status/{runId}.json` from Blob and return it as-is |
| `GET /download/:runId`   | Stream `output/{runId}/result_Schedule.yaml` straight from Blob as a file download |
| `DELETE /run/:runId`     | Terminate the Batch task if still running, delete all blobs for that run |
| `GET /health`            | Returns `{ok:true}` — smoke test |

This is the same HTTP contract the webapp already calls in
[webapp/src/services/runService.ts](../src/services/runService.ts) — no
webapp code changes needed, just point it at the right URL (Phase 7).

**Time:** ~30–40 minutes.
**Prereqs:** Phases 1–5 done. `$SOLVER_IMAGE`, `$MI_ID`, `$JOB_ID`,
`$POOL_ID` all set. RBAC from
[Azure-Company-02-RBAC.md §2.2](./Azure-Company-02-RBAC.md#22--aca-apps-system-assigned-managed-identity-3-roles)
confirmed on `$ACA_APP`'s identity.

---

## Step 1 — Look at the code once (optional but recommended)

Open [web/api-controller/src/server.js](../../api-controller/src/server.js).
You don't need to change anything in it — it reads all the Azure-specific
values (storage account, container, Batch URL, image name, etc.) from
environment variables, which you'll set on the Container App in Step 6.
Skim it once so the env var names in Step 6 make sense.

## Step 2 — Test it locally against the real Azure resources first

Testing locally (using your own `az login` credentials instead of a Managed
Identity) catches config mistakes before you've spent time building and
pushing a Docker image.

```bash
cd /c/Users/Seiya/Desktop/work/Timefold/web/api-controller
npm install
```

Source your env script and export the variables the app expects (it uses
plain `process.env`, same names as your script but without needing the
`$MI_ID` dollar sign inside the JSON this time — plain values):

```bash
source ~/azure-timefold-company-env.sh

export STORAGE_ACCOUNT="$ST"
export BLOB_CONTAINER="$CONTAINER"
export BATCH_ACCOUNT_URL="$BATCH_URL"
export BATCH_JOB_ID="$JOB_ID"
export SOLVER_IMAGE="$SOLVER_IMAGE"
export ACR_LOGIN_SERVER="$ACR_LOGIN"
export POOL_MI_RESOURCE_ID="$MI_ID"
export PORT=8080

npm run dev
```

> `DefaultAzureCredential` (used inside `server.js`) automatically tries
> your local `az login` session when there's no Managed Identity available
> — so this works on your laptop with zero extra config, and automatically
> switches to the ACA app's Managed Identity once deployed. Same code, two
> environments.

In a second terminal, run through the whole contract:

```bash
curl http://localhost:8080/health
# {"ok":true}

curl -X POST http://localhost:8080/runSolver \
  -F "env=@/c/Users/Seiya/Desktop/work/Timefold/web/Timefold/src/main/resource/EnvConfig.yaml" \
  -F "sched=@/c/Users/Seiya/Desktop/work/Timefold/web/Timefold/src/main/resource/Schedule.yaml"
# {"runId":"...","status":"Submitted"}

RUNID=<paste the runId from above>

curl http://localhost:8080/status/$RUNID
# watch this a few times — status should move Submitted -> Running -> Completed as Batch solves it

curl http://localhost:8080/download/$RUNID -o /tmp/result_from_api.yaml
cat /tmp/result_from_api.yaml   # only works once status is Completed

curl -X DELETE http://localhost:8080/run/$RUNID
# {"ok":true,"runId":"...","deleted":N}
```

If all of these work locally, the logic is sound — anything that goes wrong
after deploying to ACA is almost certainly an environment variable or RBAC
issue, not a code issue.

Stop the local server (Ctrl+C) once confirmed.

## Step 3 — Build the image

```bash
cd /c/Users/Seiya/Desktop/work/Timefold/web/api-controller
TAG=v1
docker build -t "$ACR_LOGIN/api-controller:$TAG" .
```

## Step 4 — Push it

```bash
az acr login --name "$ACR"
docker push "$ACR_LOGIN/api-controller:$TAG"

az acr repository show-tags --name "$ACR" --repository api-controller -o table
```

## Step 5 — Confirm the ACA app's identity can pull it

This should already be true from
[Azure-Company-02-RBAC.md §2.2](./Azure-Company-02-RBAC.md#22--aca-apps-system-assigned-managed-identity-3-roles)
(`AcrPull` on `$ACR`). Double check in the portal:
Container registry `$ACR` → Access control (IAM) → Role assignments →
confirm `$ACA_APP` appears under `AcrPull`.

## Step 6 — Deploy the image with the right environment variables

```bash
source ~/azure-timefold-company-env.sh

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

# Make sure ingress points at port 8080 (the hello-world/previous image may have used a different port)
az containerapp ingress update \
  --name "$ACA_APP" \
  --resource-group "$RG" \
  --target-port 8080
```

Wait ~30 seconds for the new revision to become active.

## Step 7 — Smoke test the deployed app

```bash
echo "https://$APP_URL"

curl https://$APP_URL/health
# {"ok":true}

curl -X POST https://$APP_URL/runSolver \
  -F "env=@/c/Users/Seiya/Desktop/work/Timefold/web/Timefold/src/main/resource/EnvConfig.yaml" \
  -F "sched=@/c/Users/Seiya/Desktop/work/Timefold/web/Timefold/src/main/resource/Schedule.yaml"
```

If you get back `{"runId":"...","status":"Submitted"}`, the API is live on
Azure, has correctly talked to Blob, and has successfully submitted a real
Batch task using its own Managed Identity (not your personal login this
time — this is the real proof RBAC on the ACA app's identity is correct).

Poll status and download exactly like Step 2, but against `https://$APP_URL`
instead of `localhost:8080`. Confirm the full loop (Submitted → Running →
Completed → download succeeds) works against the deployed app.

## Step 8 — Watch the logs if anything fails

```bash
az containerapp logs show --name "$ACA_APP" --resource-group "$RG" --follow
```

Ctrl+C to stop following. Cross-reference any error against the
Troubleshooting table below.

---

## What you should have at the end of this phase

- [ ] `npm run dev` locally completed a full run → status → download → delete cycle against real Azure resources
- [ ] `web/api-controller` image pushed to `$ACR` as `api-controller:v1`
- [ ] `$ACA_APP` updated to that image, ingress on port 8080, all 7 env vars set
- [ ] `curl https://$APP_URL/health` returns `{"ok":true}`
- [ ] A real `/runSolver` call against the deployed app completes a solve and downloads successfully

Next: [Azure-Company-07-Webapp-Connect.md](./Azure-Company-07-Webapp-Connect.md)
— point the webapp itself at `https://$APP_URL` and test the actual UI.

---

## Troubleshooting

| Symptom                                                            | Cause                                                        | Fix |
| --------------------------------------------------------------------- | --------------------------------------------------------------- | ----- |
| Local `npm run dev` → `AuthorizationPermissionMismatch` on Blob calls  | Your own account missing Storage Blob Data Contributor          | Re-check Phase 2.1 |
| Local `npm run dev` → Batch task create fails with 403                | Your own account missing Azure Batch Account Contributor         | Re-check Phase 2.1 |
| Deployed app returns 503 / won't start                                 | Image pull failed — ACA identity lacks AcrPull                   | Re-check Phase 2.2, Step 5 above |
| Deployed `/runSolver` returns 500 with a Blob auth error                | ACA identity lacks Storage Blob Data Contributor                  | Re-check Phase 2.2 |
| Deployed `/runSolver` returns 500 with a Batch auth/403 error           | ACA identity lacks Azure Batch Account Contributor, or `BATCH_JOB_ID`/`BATCH_ACCOUNT_URL` env var is wrong | Re-check Phase 2.2 and the env vars in Step 6 |
| `/runSolver` succeeds but task never appears in Batch                   | `POOL_MI_RESOURCE_ID` or `SOLVER_IMAGE` env var doesn't match the pool's actual config | Re-verify against Phase 5 Step 3's output |
| `/download/:runId` returns 404 forever                                  | Status never reached Completed — check `/status/:runId` first     | If stuck on Running/Submitted, check the Batch task directly (`az batch task show`) for the real state |
| curl works but the exact same call from the webapp fails with a CORS error | Shouldn't happen — `cors()` is enabled for all origins in this code | Hard-refresh the browser (Ctrl+Shift+R); if it persists, check `az containerapp logs show` for the actual incoming request |
