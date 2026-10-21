# Guide 03 — Create the Azure resources for Sigma-I (TEST environment first)

Same style and variable names as `documents/SchedulerWeb/azure/Azure-Company-0x`. Run the commands
in **Git Bash** after `source ~/azure-timefold-company-env.sh`.

**Goal:** a working Sigma-I run through Azure on a **test** setup that real users never touch.

**Time:** ~2–3 hours the first time, mostly waiting for node boot / image pushes.
**Prereqs:** your Timefold setup from Azure-Company-01…05 already exists (`$RG $ST $ACR $ACR_LOGIN $BATCH $BATCH_URL $POOL_ID $JOB_ID $MI_ID $ACA_ENV $ACA_APP`).
Code from guides 01/02/05 is ready. **Company network** is needed for the ACR/Storage steps (same warning as
Azure-Company-01).

---

## 0. What is new, and who creates it

| New resource | Name used here | Reuses | Who |
|---|---|---|---|
| Blob container for the **test** run data + setup files | `tf-test` (`$CONTAINER_TEST`) | storage account `$ST` | you |
| **Private** Blob container for the D-Wave key | `sigmai-secrets` (`$SECRETS_CONTAINER`) | `$ST` | you |
| ACR repos `sigmai/screenas-workerassign-api`, `mirror/redis`, `api-controller` (tag `v2-test1`) | — | `$ACR` | you (AcrPush) |
| Batch **pool** `pool-sigmai` | `$POOL_ID_SIGMAI` | Batch account `$BATCH`, pool identity `$MI_ID` | you |
| Batch **job** `job-sigmai-test` | `$JOB_ID_SIGMAI` | — | you |
| **Test** Container App | `$ACA_APP_TEST` | ACA environment `$ACA_ENV` | you |
| Role assignments for the test app's identity | — | — | **admin** (§2) |

Not created: no Key Vault (key lives in Blob as you asked), no new Batch account, no new storage account, no new ACR.

## 1. Add variables to your env script

```bash
cat >> ~/azure-timefold-company-env.sh <<'EOF'

# ── Sigma-I (test) ──
export CONTAINER_TEST="tf-test"
export SECRETS_CONTAINER="sigmai-secrets"
export POOL_ID_SIGMAI="pool-sigmai"
export JOB_ID_SIGMAI="job-sigmai-test"
export ACA_APP_TEST="${ACA_APP}-test"          # check the name rules / ask for a convention
export SIGMAI_VER="0.1.0-t1"                   # version folder under sigmai-setup/
export SIGMAI_IMAGE="${ACR_LOGIN}/sigmai/screenas-workerassign-api:0.1.0"
export REDIS_IMAGE="${ACR_LOGIN}/mirror/redis:7-alpine"
EOF
source ~/azure-timefold-company-env.sh
echo "$CONTAINER_TEST $POOL_ID_SIGMAI $JOB_ID_SIGMAI $ACA_APP_TEST $SIGMAI_IMAGE"
```

> Ask the admin if a naming convention exists before inventing names (as in Azure-Company-05).

## 2. Permissions — send this to the admin **today**

Everything in Azure-Company-02 already covers the pool identity and your own account on the storage
account and ACR. What is new is only the **test ACA app's identity** (it exists after §9 step 2, so this
request can be a standing "when I create it, please grant"):

| Identity | Role | Scope | Why |
|---|---|---|---|
| `$ACA_APP_TEST` (system-assigned) | Storage Blob Data Contributor | storage account `$ST` | read/write input, output, status, runs |
| `$ACA_APP_TEST` | AcrPull | `$ACR` | pull its own image |
| `$ACA_APP_TEST` | Azure Batch Account Contributor | `$BATCH` | create / terminate Sigma-I and Timefold tasks |
| you | Storage Blob Data Contributor | `$ST` | upload setup files + D-Wave key (you have it per 02 §2.1) |

Already-existing roles that this design relies on (no new request): `$MI_NAME` has *Storage Blob Data Contributor*
on `$ST` and *AcrPull* on `$ACR`.

> **Note on "private":** because the pool identity's blob role is at **storage-account** level, it can read
> `sigmai-secrets` (needed) — but so can any other identity with that account-level role, including the API
> controller's. For stricter isolation, ask the admin to assign *Storage Blob Data Reader* on the
> **container** `sigmai-secrets` only to `$MI_NAME`, and keep the broader role off it. Fine to do later.

CLI form for the admin (or for you if you may):
```bash
az role assignment create --assignee-object-id "<principalId of $ACA_APP_TEST>" --assignee-principal-type ServicePrincipal \
  --role "Storage Blob Data Contributor" --scope "$(az storage account show -n $ST -g $RG --query id -o tsv)"
az role assignment create --assignee-object-id "<principalId>" --assignee-principal-type ServicePrincipal \
  --role "AcrPull" --scope "$(az acr show -n $ACR --query id -o tsv)"
az role assignment create --assignee-object-id "<principalId>" --assignee-principal-type ServicePrincipal \
  --role "Azure Batch Account Contributor" --scope "$(az batch account show -n $BATCH -g $RG --query id -o tsv)"
```

## 3. Blob containers

```bash
az login   # if needed
for C in "$CONTAINER_TEST" "$SECRETS_CONTAINER"; do
  az storage container create --account-name "$ST" --name "$C" --auth-mode login --public-access off
done
az storage container list --account-name "$ST" --auth-mode login --query "[].{name:name, access:properties.publicAccess}" -o table
```
`access` must be empty/`None` for both (no anonymous access).

The test controller uses `$CONTAINER_TEST` as its `BLOB_CONTAINER`, so test runs never mix with real ones
(Batch `autoStorageContainerName` works for any container in the linked storage account — confirm with
Azure-Company-05 Step 0 that `$ST` is the Batch auto-storage account).

## 4. D-Wave key in Blob (instead of Key Vault)

You need the D-Wave Leap API token (company decision, design Q5). **Option ② needs it; option ③ does not**,
so you can skip this section until the token exists and run with `DWAVE_ENABLED=false`.

```bash
# 1) put the token in a local file — ONE line, nothing else (no quotes)
printf '%s' 'PASTE-TOKEN-HERE' > /tmp/dwave-token.txt

# 2) upload to the private container
az storage blob upload --account-name "$ST" --container-name "$SECRETS_CONTAINER" --auth-mode login \
  --name dwave-token.txt --file /tmp/dwave-token.txt --overwrite

# 3) delete the local copy
shred -u /tmp/dwave-token.txt 2>/dev/null || rm -f /tmp/dwave-token.txt

# 4) verify (shows size only — do NOT print the content into a shared log)
az storage blob show --account-name "$ST" --container-name "$SECRETS_CONTAINER" --auth-mode login \
  --name dwave-token.txt --query "{size:properties.contentLength, modified:properties.lastModified}" -o table
```

How it reaches the solver: at node boot `start-sigmai.sh` asks the node's Managed Identity for a storage
token (IMDS), downloads `dwave-token.txt`, and writes it to `DWAVE_API_TOKEN=` in a `chmod 600` file on the node.
It is **never** put in Batch `environmentSettings` or logs.

Trade-offs vs Key Vault (so you can tell the team): Key Vault gives per-secret access control, audit logs and
rotation features; a blob gives you none of those, and anyone with account-level blob read can read it.
If the team later wants Key Vault, only the "fetch the token" block in `start-sigmai.sh` §4 changes.

**Rotate the key:** overwrite the blob (step 2), then re-run the StartTask on each node:
`az batch node reboot --pool-id "$POOL_ID_SIGMAI" --node-id <id>` (StartTask runs again after a reboot).

## 5. Push the Sigma-I image and a Redis copy to ACR

Sigma-i delivered the image as a tar (`screenas-workerassign-api_0.1.0.image.tar` in the `sigmaI` folder).
Docker Desktop must be running.

```bash
SIGMA_DIR="/c/Users/PC_USER/OneDrive/Desktop/work/Timefold/web/sigmaI/2.Web APIソースコード一式/APIコンテナ（2026-07-15提供物と同一）/screenas-workerassign-api"
docker load -i "$SIGMA_DIR/screenas-workerassign-api_0.1.0.image.tar"
docker images | grep screenas-workerassign-api        # expect: sigmai/screenas-workerassign-api  0.1.0

az acr login --name "$ACR"
docker tag sigmai/screenas-workerassign-api:0.1.0 "$SIGMAI_IMAGE"
docker push "$SIGMAI_IMAGE"

# Redis copy so nodes never depend on Docker Hub (rate limits / egress)
docker pull redis:7-alpine
docker tag redis:7-alpine "$REDIS_IMAGE"
docker push "$REDIS_IMAGE"

az acr repository list --name "$ACR" -o table
```

> Platform: the image is Linux/amd64 (README). On an ARM laptop add `--platform linux/amd64` to `docker pull`.
> **Licence (design Q2):** confirm with Sigma-i that hosting the image/wheel in your ACR is allowed.

## 6. Upload the setup files (compose, scripts, wheel)

The **core wheel** `screenas_workerassign_opt-*.whl` is delivered separately by Sigma-i. Put exactly one into
`api-controller-v2/sigmai-setup/wheels/` (git-ignored — do not commit it), then:

```bash
cd /c/Users/PC_USER/OneDrive/Desktop/work/Timefold/web/api-controller-v2/sigmai-setup
CONTAINER="$CONTAINER_TEST" bash upload-setup.sh "$SIGMAI_VER"
```
Expected last lines:
```
sigmai-setup/0.1.0-t1/bridge.sh
sigmai-setup/0.1.0-t1/docker-compose.azure.yml
sigmai-setup/0.1.0-t1/start-sigmai.sh
sigmai-setup/0.1.0-t1/wheels/screenas_workerassign_opt-0.1.x-py3-none-any.whl
```
Changing a script later? Upload under a **new version** (`0.1.0-t2`), create a new pool/resize with the new
`blobPrefix`, so a node's behaviour always matches a version you can name.

## 7. Batch: quota, pool, job

### 7.1 Quota
```bash
az batch account login --name "$BATCH" --resource-group "$RG"
az batch account show --name "$BATCH" --resource-group "$RG" \
  --query "{dedicated:dedicatedCoreQuota, perFamilyEnforced:dedicatedCoreQuotaPerVmFamilyEnforced}" -o table
az batch account show --name "$BATCH" --resource-group "$RG" --query "dedicatedCoreQuotaPerVmFamily" -o table
```
`Standard_F8s_v2` is in the **same family (Fsv2) as your Timefold `F16s_v2`**, so you need **8 free vCPU** of
Fsv2 quota per Sigma-I node (design Q3: the right size is still to be confirmed with Sigma-i).

### 7.2 Copy the image settings from the working Timefold pool
```bash
az batch pool show --pool-id "$POOL_ID" --query "virtualMachineConfiguration.{image:imageReference, agent:nodeAgentSkuId}" -o json
```
Use the same values below (the example is the `ubuntu-hpc` one from Azure-Company-05).

### 7.3 Create the pool (1 fixed node for testing)

```bash
mkdir -p ~/azure-timefold-company
cat > ~/azure-timefold-company/pool-sigmai.json <<EOF
{
  "id": "${POOL_ID_SIGMAI}",
  "vmSize": "STANDARD_F8S_V2",
  "virtualMachineConfiguration": {
    "imageReference": { "publisher": "microsoft-dsvm", "offer": "ubuntu-hpc", "sku": "2204", "version": "latest" },
    "nodeAgentSKUId": "batch.node.ubuntu 22.04",
    "containerConfiguration": {
      "type": "dockerCompatible",
      "containerRegistries": [
        { "registryServer": "${ACR_LOGIN}", "identityReference": { "resourceId": "${MI_ID}" } }
      ],
      "containerImageNames": [ "${SIGMAI_IMAGE}", "${REDIS_IMAGE}" ]
    }
  },
  "targetDedicatedNodes": 1,
  "targetLowPriorityNodes": 0,
  "enableAutoScale": false,
  "taskSlotsPerNode": 1,
  "taskSchedulingPolicy": { "nodeFillType": "Spread" },
  "identity": { "type": "UserAssigned", "userAssignedIdentities": [ { "resourceId": "${MI_ID}" } ] },
  "startTask": {
    "commandLine": "/bin/bash -c 'bash \"\$(find \"\$AZ_BATCH_TASK_WORKING_DIR\" -name start-sigmai.sh | head -1)\"'",
    "userIdentity": { "autoUser": { "scope": "pool", "elevationLevel": "admin" } },
    "waitForSuccess": true,
    "maxTaskRetryCount": 2,
    "resourceFiles": [
      {
        "autoStorageContainerName": "${CONTAINER_TEST}",
        "blobPrefix": "sigmai-setup/${SIGMAI_VER}/",
        "filePath": "setup",
        "identityReference": { "resourceId": "${MI_ID}" }
      }
    ],
    "environmentSettings": [
      { "name": "STORAGE_ACCOUNT",     "value": "${ST}" },
      { "name": "SECRETS_CONTAINER",   "value": "${SECRETS_CONTAINER}" },
      { "name": "DWAVE_SECRET_BLOB",   "value": "dwave-token.txt" },
      { "name": "DWAVE_ENABLED",       "value": "false" },
      { "name": "POOL_MI_RESOURCE_ID", "value": "${MI_ID}" },
      { "name": "ACR_LOGIN_SERVER",    "value": "${ACR_LOGIN}" },
      { "name": "SIGMAI_IMAGE",        "value": "${SIGMAI_IMAGE}" },
      { "name": "REDIS_IMAGE",         "value": "${REDIS_IMAGE}" }
    ]
  }
}
EOF
az batch pool create --json-file ~/azure-timefold-company/pool-sigmai.json
```

Notes:
- `userAssignedIdentities` is an **array** of `{resourceId}` (see Azure-Company-05 note).
- `DWAVE_ENABLED` is `false` for the first test. To enable option ②, delete the pool (when idle) and create it again
  with `"true"` — StartTask settings are fixed when the pool is created.
- **Dedicated** nodes only — a low-priority eviction would kill the warm stack and a running solve.

### 7.4 Create the job and wait for the node
```bash
az batch job create --id "$JOB_ID_SIGMAI" --pool-id "$POOL_ID_SIGMAI"

until [ "$(az batch node list --pool-id "$POOL_ID_SIGMAI" --query '[0].state' -o tsv 2>/dev/null)" = "idle" ]; do
  echo "$(date +%T) node state: $(az batch node list --pool-id "$POOL_ID_SIGMAI" --query '[0].state' -o tsv 2>/dev/null)"
  sleep 20
done
echo "Sigma-I node is idle = StartTask succeeded"
```
First boot takes **~6–15 min** (VM allocation + image pull + the wheel `pip install`). Write down the real time — it
decides the autoscale design (design §8.4).

### 7.5 If the node ends as `starttaskfailed`
```bash
NODE=$(az batch node list --pool-id "$POOL_ID_SIGMAI" --query '[0].id' -o tsv)
az batch node show --pool-id "$POOL_ID_SIGMAI" --node-id "$NODE" --query "startTaskInfo.{state:state,exit:exitCode,fail:failureInfo}" -o json
az batch node file download --pool-id "$POOL_ID_SIGMAI" --node-id "$NODE" --file-path startup/stdout.txt --destination /tmp/sigmai-start-stdout.txt
az batch node file download --pool-id "$POOL_ID_SIGMAI" --node-id "$NODE" --file-path startup/stderr.txt --destination /tmp/sigmai-start-stderr.txt
tail -60 /tmp/sigmai-start-stdout.txt
```
The script prints compose/api/worker logs on failure. Common causes are in §11 and guide 05.

## 8. Prove the node works — one manual bridge task (no API controller yet)

Mirrors Azure-Company-05 Step 7–9. Uses the same test data you used for Timefold.

```bash
RUN_ID="sigmai-smoke-001"
IN="/c/Users/PC_USER/Desktop/file/SchedulerWeb/GanttChartEditor/Test_data"     # your EnvConfig.yaml / Schedule.yaml folder

for F in EnvConfig.yaml Schedule.yaml; do
  az storage blob upload --account-name "$ST" --container-name "$CONTAINER_TEST" --auth-mode login \
    --name "input/${RUN_ID}/${F}" --file "$IN/$F" --overwrite
done
cat > /tmp/sigmai_options.json <<'EOF'
{ "engine":"sigmai","stage1Solver":"scip","stage2Solver":"scip","stage1TimeoutSec":120,"stage2TimeoutSec":120,
  "stage1NumSolutionsLimit":null,"stage2NumSolutionsLimit":null,"weightPreset":null,"disabledConstraints":[],"scenarioName":"smoke" }
EOF
az storage blob upload --account-name "$ST" --container-name "$CONTAINER_TEST" --auth-mode login \
  --name "input/${RUN_ID}/sigmai_options.json" --file /tmp/sigmai_options.json --overwrite

CONT_URL="https://${ST}.blob.core.windows.net/${CONTAINER_TEST}"
cat > ~/azure-timefold-company/task-${RUN_ID}.json <<EOF
{
  "id": "${RUN_ID}",
  "displayName": "sigmai smoke",
  "commandLine": "/bin/bash -c 'exec bash \$AZ_BATCH_NODE_SHARED_DIR/sigmai/bridge.sh'",
  "userIdentity": { "autoUser": { "scope": "pool", "elevationLevel": "admin" } },
  "environmentSettings": [ { "name": "RUN_ID", "value": "${RUN_ID}" } ],
  "constraints": { "maxWallClockTime": "PT1H", "maxTaskRetryCount": 1, "retentionTime": "P1D" },
  "resourceFiles": [ { "autoStorageContainerName": "${CONTAINER_TEST}", "blobPrefix": "input/${RUN_ID}/", "filePath": "input",
                       "identityReference": { "resourceId": "${MI_ID}" } } ],
  "outputFiles": [
    { "filePattern": "output/result_Schedule.yaml",
      "destination": { "container": { "containerUrl": "${CONT_URL}", "path": "output/${RUN_ID}/result_Schedule.yaml", "identityReference": { "resourceId": "${MI_ID}" } } },
      "uploadOptions": { "uploadCondition": "taskSuccess" } },
    { "filePattern": "output/sigmai/*",
      "destination": { "container": { "containerUrl": "${CONT_URL}", "path": "output/${RUN_ID}/sigmai", "identityReference": { "resourceId": "${MI_ID}" } } },
      "uploadOptions": { "uploadCondition": "taskCompletion" } },
    { "filePattern": "status/${RUN_ID}.json",
      "destination": { "container": { "containerUrl": "${CONT_URL}", "path": "status/${RUN_ID}.json", "identityReference": { "resourceId": "${MI_ID}" } } },
      "uploadOptions": { "uploadCondition": "taskCompletion" } }
  ]
}
EOF
az batch task create --job-id "$JOB_ID_SIGMAI" --json-file ~/azure-timefold-company/task-${RUN_ID}.json

# watch
until [ "$(az batch task show --job-id "$JOB_ID_SIGMAI" --task-id "$RUN_ID" --query state -o tsv)" = "completed" ]; do
  echo "$(date +%T) $(az batch task show --job-id "$JOB_ID_SIGMAI" --task-id "$RUN_ID" --query state -o tsv)"; sleep 15
done
az batch task show --job-id "$JOB_ID_SIGMAI" --task-id "$RUN_ID" --query "{exit:executionInfo.exitCode, result:executionInfo.result, node:nodeInfo.nodeId}" -o table
az storage blob list --account-name "$ST" --container-name "$CONTAINER_TEST" --auth-mode login --prefix "output/${RUN_ID}/" --query "[].name" -o tsv
az storage blob download --account-name "$ST" --container-name "$CONTAINER_TEST" --auth-mode login --name "status/${RUN_ID}.json" --file /tmp/s.json && cat /tmp/s.json
```
Expected: `exit 0`, `output/sigmai-smoke-001/result_Schedule.yaml`, `…/sigmai/{result.json,bridge.log,…}`, status `Completed`.
**Also check the blob names under `output/<id>/sigmai/`** — wildcard `outputFiles` treat `path` as a prefix
(design §8.3); if names look wrong, adjust the pattern/path pair in `sigmaiTaskSpec()` and in this task JSON.

If it fails with `OPTIMIZATION_FAILED`, that is the Sigma-I input problem (design P0 / Q11), not Azure — the logs are in
`output/<id>/sigmai/` (`error.log`, `optimization.log`, `bridge.log`).

Second task on the same node while the first one runs: it must sit in `active` and then run **without** a reinstall — that proves the queue behaviour.

## 9. Deploy API controller v2 as a **test** app

### 9.1 Build and push
```bash
cd /c/Users/PC_USER/OneDrive/Desktop/work/Timefold/web/api-controller-v2
npm install && npm test                      # 24 tests green first (run from Git Bash)
TAG=v2-test1
docker build -t "$ACR_LOGIN/api-controller:$TAG" .
az acr login --name "$ACR"
docker push "$ACR_LOGIN/api-controller:$TAG"
```

### 9.2 Create the app (placeholder image first — it has no AcrPull yet)
```bash
az containerapp create --name "$ACA_APP_TEST" --resource-group "$RG" --environment "$ACA_ENV" \
  --image mcr.microsoft.com/k8se/quickstart:latest --target-port 80 --ingress external \
  --system-assigned --min-replicas 1 --max-replicas 1

PRINCIPAL=$(az containerapp show --name "$ACA_APP_TEST" --resource-group "$RG" --query identity.principalId -o tsv)
echo "$PRINCIPAL"        # ← give this to the admin for the 3 roles in §2 (or assign them yourself if allowed)
```
**Wait until the 3 roles exist** (a few minutes to propagate), then:

### 9.3 Point it at the real image and set env vars
```bash
az containerapp registry set --name "$ACA_APP_TEST" --resource-group "$RG" --server "$ACR_LOGIN" --identity system

az containerapp update --name "$ACA_APP_TEST" --resource-group "$RG" \
  --image "$ACR_LOGIN/api-controller:$TAG" \
  --set-env-vars \
    STORAGE_ACCOUNT="$ST" \
    BLOB_CONTAINER="$CONTAINER_TEST" \
    BATCH_ACCOUNT_URL="$BATCH_URL" \
    BATCH_JOB_ID="$JOB_ID" \
    BATCH_JOB_ID_SIGMAI="$JOB_ID_SIGMAI" \
    SOLVER_IMAGE="$SOLVER_IMAGE" \
    ACR_LOGIN_SERVER="$ACR_LOGIN" \
    POOL_MI_RESOURCE_ID="$MI_ID" \
    SIGMAI_DWAVE_ENABLED=false \
    SIGMAI_SETUP_VERSION="$SIGMAI_VER" \
    APP_VERSION="2.0.0-test1"

az containerapp ingress update --name "$ACA_APP_TEST" --resource-group "$RG" --target-port 8080

APP_URL_TEST=$(az containerapp show --name "$ACA_APP_TEST" --resource-group "$RG" --query properties.configuration.ingress.fqdn -o tsv)
echo "export APP_URL_TEST=$APP_URL_TEST" >> ~/azure-timefold-company-env.sh
curl "https://$APP_URL_TEST/health"
# {"ok":true,"version":"2.0.0-test1","engines":["timefold","sigmai"],"sigmaiDwave":false}
```
Note the **Timefold** part of the test app uses `$JOB_ID` (the real Timefold job/pool) — a Timefold regression run on the
test app does consume the real Timefold pool, but writes its blobs into `$CONTAINER_TEST`.

Logs: `az containerapp logs show --name "$ACA_APP_TEST" --resource-group "$RG" --follow`

### 9.4 Point SchedulerWeb at it
Edit `SchedulerWeb/config.txt` (next to the exe in the packaged app):
```
mode=online
azure_url=https://<APP_URL_TEST>
```
Restart the app. Now run the checks in guide 02 §6 / guide 04 part D.

## 10. Cost control and rollback

- A fixed **1-node** pool bills while it exists. When you are not testing:
  `az batch pool resize --pool-id "$POOL_ID_SIGMAI" --target-dedicated-nodes 0` (back up with `…=1`, StartTask re-runs).
- Remove everything test-only later: delete the app (`az containerapp delete`), the pool/job, the `tf-test` container.
  The real Timefold setup is not touched by any of this.
- Roll the **real** controller back (after promotion) by redeploying the old tag:
  `az containerapp update --name "$ACA_APP" --resource-group "$RG" --image "$ACR_LOGIN/api-controller:v1"`.

## 11. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Node `starttaskfailed`, log says *docker-compose.azure.yml not found* | StartTask `resourceFiles` prefix/version wrong, or files not uploaded | `az storage blob list … --prefix sigmai-setup/$SIGMAI_VER/`; names must match §6 |
| …*expected exactly 1 wheel* | no wheel or two wheels uploaded | upload exactly one; use a new version folder |
| …*could not get a storage/ARM token from IMDS* | pool has no user-assigned identity attached | check `identity` in the pool JSON (§7.3) |
| …*ACR token exchange failed* / *docker login failed* | pool MI lacks **AcrPull**, or ACR not reachable (firewall) | check roles; run from the company network; Azure-Company-04 note on ACR network access |
| …*could not read the D-Wave token blob* (only when `DWAVE_ENABLED=true`) | blob missing/empty, or MI has no blob read | §4; role at account level |
| StartTask waits 20 min then *did not become ready* | worker `pip install` of the wheel can't reach PyPI (no outbound internet) | design Q4 — ask IT to allow `pypi.org`, `files.pythonhosted.org`; or ask Sigma-i for a pre-baked image (Q1) |
| Task `FileUploadMiscError` | Storage network/RBAC (same as Timefold) | Azure-Company-06.5 troubleshooting |
| Bridge task `exit 1`, status `InvalidInput` | missing `sigmai_options.json` or Sigma-I 4xx | see `bridge.log` in `output/<id>/sigmai/` |
| Bridge `SigmaIUnavailable` | stack not running on that node | `docker ps` on the node; node reboot re-runs StartTask |
| `/health` shows only `["timefold"]` | `BATCH_JOB_ID_SIGMAI` not set on the app | §9.3 |
| SchedulerWeb: *backend does not support "sigmai"* | `config.txt` points at the old controller | set `azure_url` to the **test** app |
| 400 *D-Wave is not enabled* | `SIGMAI_DWAVE_ENABLED=false` | set true **and** pool `DWAVE_ENABLED=true` + token in Blob |
