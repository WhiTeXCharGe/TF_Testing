# Phase 5 — Batch account + pool + first end-to-end run

**Goal of this phase:** the milestone. By the end:
- Azure Batch account exists, linked to your storage account
- A compute pool exists, scaled to 0 idle, can scale up to 1 on demand
- The pool's Managed Identity can pull from ACR and read/write Blob
- You submit ONE Batch task → it pulls the Timefold image → reads input YAMLs
  from Blob → solves → writes `result_Schedule.yaml` + `status.json` to Blob
- You download the result via SAS URL

This is your **architecture-validates** moment. No API code yet — Phase 6
automates this manual flow.

**Time:** ~45 min setup + however long Timefold takes to solve (could be 5
minutes, could be 60 — depends on your input size).
**Cost:** ~$0.05–$0.15 for the test (one VM running ~30 min on low-priority).
**Prereqs:** Phases 1–4 done. Env script up to date with `$ACR_LOGIN`, etc.

> **Heads-up:** Personal Azure accounts often start with a **Batch vCPU quota
> of 0**. We address this in Step 0. If you hit it later, that's also where
> to come back to.

---

## Concepts (3 min read)

| Concept                  | What it is                                                                                                |
| ------------------------ | --------------------------------------------------------------------------------------------------------- |
| **Batch account**        | The top-level Azure resource that owns pools, jobs, and tasks. One per project is enough.                 |
| **Pool**                 | A fleet of VMs (compute nodes) that run tasks. Has autoscale rules — can drop to 0 nodes when no work.    |
| **Compute node**         | One VM in a pool. Each node can run N tasks in parallel; we'll keep it at 1.                              |
| **Job**                  | A logical container for tasks. We use one long-lived job; each solve = one task in it.                    |
| **Task**                 | The unit of work. References a container image (from ACR), env vars, and a command line.                  |
| **Container task**       | Batch can run any Docker image. The image is pulled from ACR using the pool's MI.                         |
| **Auto-storage**         | If you link a storage account to your Batch account, Batch auto-generates SAS URLs for blob references — no manual SAS needed for resourceFiles/outputFiles. |
| **resourceFiles**        | Blobs Batch downloads into the task's working dir BEFORE the task runs.                                   |
| **outputFiles**          | Files Batch uploads from the task's working dir to Blob AFTER the task runs (always, on success, or on failure). |

The whole pattern: **Batch downloads inputs from Blob → starts container →
container reads inputs from a mounted dir → writes output to mounted dir →
container exits → Batch uploads outputs to Blob.** Clean separation, no Blob
SDK needed inside the container.

---

## Step 0 — Prereqs and quota check

### 0a. Source env, register Batch provider

```bash
source ~/azure-timefold-env.sh
echo "RG=$RG  LOC=$LOC  ACR=$ACR  ST=$ST"

# Microsoft.Batch should already be Registering from Phase 2.2
az provider show --namespace Microsoft.Batch --query registrationState -o tsv

# If not Registered yet, wait
until [ "$(az provider show --namespace Microsoft.Batch --query registrationState -o tsv)" = "Registered" ]; do
  echo "waiting for Microsoft.Batch..."; sleep 5
done
```

### 0b. Check your Batch vCPU quota

Personal accounts often have 0 quota for Batch. Check:

```bash
az batch location quotas show --location $LOC -o table
```

Look at the columns. You want to see at least:
- `DedicatedCoreQuotaPerVMFamily` ≥ 2 for the F-series, OR
- `LowPriorityCoreQuota` ≥ 2

If both say 0, you need to request an increase first.

### 0c. Request a quota increase (if needed — takes 1–48 hours)

1. Portal → search **Quotas** → **Compute** → filter by your subscription + region
2. Find a row like `Standard FSv2 Family vCPUs` (or `Total Regional vCPUs`)
3. Click the pencil icon → New limit: `4` (enough for our F2s_v2 test)
4. Justification: `Personal learning project — running Azure Batch tutorials`
5. Submit

Usually approved automatically in minutes for small requests. While you wait
you can still do Steps 1–3 below (Batch account, MI, role assignments).

---

## Step 1 — Create a user-assigned Managed Identity for the pool

The pool's nodes will use this MI to (a) pull the Timefold image from ACR
and (b) read/write blobs. We use a USER-assigned MI (not system-assigned)
because pools attach existing MIs at create time.

```bash
MI_NAME=mi-tf-pool

az identity create \
  --name $MI_NAME \
  --resource-group $RG \
  --location $LOC

# Grab the two ids we need later
MI_ID=$(az identity show --name $MI_NAME --resource-group $RG --query id -o tsv)
MI_OID=$(az identity show --name $MI_NAME --resource-group $RG --query principalId -o tsv)
MI_CLIENT_ID=$(az identity show --name $MI_NAME --resource-group $RG --query clientId -o tsv)

echo "MI_ID=$MI_ID"
echo "MI_OID=$MI_OID"
echo "MI_CLIENT_ID=$MI_CLIENT_ID"

# Save to env script
cat >> ~/azure-timefold-env.sh <<EOF

# Phase 5 additions — pool MI
export MI_NAME=$MI_NAME
export MI_ID=$MI_ID
export MI_OID=$MI_OID
export MI_CLIENT_ID=$MI_CLIENT_ID
EOF
```

---

## Step 2 — Grant the MI two roles (do this via PORTAL per Phase 3.2)

The CLI's role assignment is still broken on your machine — use the portal.

### 2a. Grant `AcrPull` on the ACR

1. Portal → **Container registries** → `acrtimefolddevseiya`
2. **Access control (IAM)** → **+ Add** → **Add role assignment**
3. **Role:** search `AcrPull` → select → **Next**
4. **Members:**
   - Assign access to: **Managed identity**
   - **+ Select members** → Subscription = yours → Managed identity = **User-assigned managed identity** → pick `mi-tf-pool` → **Select**
5. **Review + assign** → **Review + assign**

### 2b. Grant `Storage Blob Data Contributor` on the storage account

1. Portal → **Storage accounts** → `sttimefolddevseiya`
2. **Access control (IAM)** → **+ Add** → **Add role assignment**
3. **Role:** search `Storage Blob Data Contributor` → select → **Next**
4. **Members:** Managed identity → User-assigned → `mi-tf-pool`
5. **Review + assign**

### 2c. Verify in portal

On each resource's **Access control (IAM) → Role assignments** tab, expand
the role row and confirm `mi-tf-pool` is listed.

---

## Step 3 — Create the Batch account, linked to your storage account

```bash
BATCH=batchtimefolddevseiya       # 3-24 lowercase alphanumeric, GLOBALLY unique

az batch account create \
  --name $BATCH \
  --resource-group $RG \
  --location $LOC \
  --storage-account $ST
```

The `--storage-account $ST` is the magic that **links auto-storage** — Batch
can now reference blobs in that account by container/prefix without you
generating SAS URLs.

Save to env script:
```bash
BATCH_URL=$(az batch account show --name $BATCH --resource-group $RG --query accountEndpoint -o tsv)
echo "BATCH_URL=https://$BATCH_URL"

cat >> ~/azure-timefold-env.sh <<EOF
export BATCH=$BATCH
export BATCH_URL=https://$BATCH_URL
EOF

source ~/azure-timefold-env.sh
```

### Sign the CLI into Batch with AAD

```bash
az batch account login --name $BATCH --resource-group $RG
```

This caches your AAD token for `az batch ...` commands.

---

## Step 4 — Create the pool (with container support + MI)

Pool config is the most complex part. We'll put it in a JSON file so it's
readable and re-usable.

### 4a. Create the pool JSON

```bash
mkdir -p ~/azure-timefold
cat > ~/azure-timefold/pool-config.json <<EOF
{
  "id": "pool-timefold-dev",
  "vmSize": "STANDARD_F2S_V2",
  "virtualMachineConfiguration": {
    "imageReference": {
      "publisher": "microsoft-azure-batch",
      "offer": "ubuntu-server-container",
      "sku": "20-04-lts",
      "version": "latest"
    },
    "nodeAgentSKUId": "batch.node.ubuntu 20.04",
    "containerConfiguration": {
      "type": "dockerCompatible",
      "containerRegistries": [
        {
          "registryServer": "${ACR_LOGIN}",
          "identityReference": { "resourceId": "${MI_ID}" }
        }
      ],
      "containerImageNames": [
        "${ACR_LOGIN}/timefold:v1"
      ]
    }
  },
  "targetDedicatedNodes": 0,
  "targetLowPriorityNodes": 1,
  "enableAutoScale": false,
  "taskSlotsPerNode": 1,
  "identity": {
    "type": "UserAssigned",
    "userAssignedIdentities": {
      "${MI_ID}": {}
    }
  }
}
EOF
```

Key parts:
- `vmSize STANDARD_F2S_V2` — 2 vCPU / 4 GB, low-priority eligible
- `imageReference` — pre-built Ubuntu-with-Docker image from Microsoft. Docker is already installed.
- `containerRegistries` — tells the pool how to authenticate to ACR (using `mi-tf-pool`)
- `containerImageNames` — pre-pulls the image when nodes start (fewer cold-start surprises)
- `targetDedicatedNodes: 0, targetLowPriorityNodes: 1` — one cheap node, no expensive dedicated ones
- `identity.userAssignedIdentities` — attaches `mi-tf-pool` to every node

### 4b. Create the pool

```bash
az batch pool create --json-file ~/azure-timefold/pool-config.json
```

Right after creation the pool exists but the node is provisioning. Check:
```bash
az batch pool show --pool-id pool-timefold-dev --query "{state:state, allocationState:allocationState, currentDedicatedNodes:currentDedicatedNodes, currentLowPriorityNodes:currentLowPriorityNodes}" -o table
```

Loop until the node is `idle`:
```bash
until [ "$(az batch node list --pool-id pool-timefold-dev --query '[0].state' -o tsv 2>/dev/null)" = "idle" ]; do
  echo "waiting for node..."; sleep 15
  az batch node list --pool-id pool-timefold-dev --query "[].{id:id, state:state}" -o table 2>/dev/null
done
echo "Node is idle and ready."
```

**Initial provisioning takes 5–10 minutes** (Azure allocates a VM, installs
the node agent, pulls the Timefold image from ACR). Subsequent task starts
on an already-up node are seconds, not minutes.

---

## Step 5 — Create a long-lived job

The job is just a logical container — cheap, no compute attached:

```bash
az batch job create \
  --id "job-timefold-runs" \
  --pool-id "pool-timefold-dev"
```

---

## Step 6 — Verify inputs are still in Blob from Phase 2

```bash
az storage blob list \
  --account-name $ST \
  --container-name $CONTAINER \
  --prefix "input/test-run-001/" \
  --auth-mode login \
  --query "[].name" -o tsv
```

You should see at least `input/test-run-001/EnvConfig.yaml`. If you don't
also have `Schedule.yaml`, upload it now:
```bash
az storage blob upload \
  --account-name $ST \
  --container-name $CONTAINER \
  --auth-mode login \
  --name "input/test-run-001/Schedule.yaml" \
  --file "/c/Users/Seiya/Desktop/work/Timefold/web/Timefold/src/main/resource/Schedule.yaml" \
  --overwrite
```

---

## Step 7 — Define and submit the task

This is where everything comes together. Save the task JSON:

```bash
RUN_ID=test-run-001
cat > ~/azure-timefold/task-${RUN_ID}.json <<EOF
{
  "id": "${RUN_ID}",
  "commandLine": "/bin/bash -c 'mkdir -p /work/output /work/status && /app/entrypoint.sh'",
  "containerSettings": {
    "imageName": "${ACR_LOGIN}/timefold:v1",
    "containerRunOptions": "--rm --workdir /app -v \$AZ_BATCH_TASK_WORKING_DIR/input:/work/input:ro -v \$AZ_BATCH_TASK_WORKING_DIR/output:/work/output -v \$AZ_BATCH_TASK_WORKING_DIR/status:/work/status -e RUN_ID=${RUN_ID}",
    "registry": {
      "registryServer": "${ACR_LOGIN}",
      "identityReference": { "resourceId": "${MI_ID}" }
    }
  },
  "resourceFiles": [
    {
      "autoStorageContainerName": "${CONTAINER}",
      "blobPrefix": "input/${RUN_ID}/",
      "filePath": "input",
      "identityReference": { "resourceId": "${MI_ID}" }
    }
  ],
  "outputFiles": [
    {
      "filePattern": "output/**/*",
      "destination": {
        "container": {
          "containerUrl": "https://${ST}.blob.core.windows.net/${CONTAINER}",
          "path": "output/${RUN_ID}",
          "identityReference": { "resourceId": "${MI_ID}" }
        }
      },
      "uploadOptions": { "uploadCondition": "taskCompletion" }
    },
    {
      "filePattern": "status/**/*",
      "destination": {
        "container": {
          "containerUrl": "https://${ST}.blob.core.windows.net/${CONTAINER}",
          "path": "status/${RUN_ID}",
          "identityReference": { "resourceId": "${MI_ID}" }
        }
      },
      "uploadOptions": { "uploadCondition": "taskCompletion" }
    }
  ]
}
EOF
```

What's happening in this JSON:

| Block               | What it does                                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------------- |
| `commandLine`       | Inside the container: create output/status dirs, then run our entrypoint script.                              |
| `containerRunOptions` | Mount three task-working-dir subfolders into the container at `/work/{input,output,status}` so our entrypoint sees them as expected. Pass `RUN_ID` env var. |
| `resourceFiles`     | BEFORE the task runs, download every blob under `input/test-run-001/` from Blob into the `input/` subfolder of the task working dir. |
| `outputFiles`       | AFTER the task completes (regardless of success), upload everything from `output/**` and `status/**` back to Blob under matching prefixes. |

Submit:
```bash
az batch task create \
  --job-id "job-timefold-runs" \
  --json-file ~/azure-timefold/task-${RUN_ID}.json
```

---

## Step 8 — Watch the task run

### Check task state
```bash
az batch task show \
  --job-id job-timefold-runs \
  --task-id $RUN_ID \
  --query "{state:state, exitCode:executionInfo.exitCode, startTime:executionInfo.startTime, endTime:executionInfo.endTime}" \
  -o table
```

States it goes through: `active` → `running` → `completed`.

Polling loop (auto-updates every 15 s):
```bash
until [ "$(az batch task show --job-id job-timefold-runs --task-id $RUN_ID --query state -o tsv)" = "completed" ]; do
  state=$(az batch task show --job-id job-timefold-runs --task-id $RUN_ID --query state -o tsv)
  echo "$(date +%T) state=$state"
  sleep 15
done
echo "Task completed."
```

### Tail the container logs (while running)

```bash
az batch task file list \
  --job-id job-timefold-runs \
  --task-id $RUN_ID \
  -o table
```

You'll see `stdout.txt`, `stderr.txt`, and after completion files under `wd/` for the working directory.

Download stdout to see Timefold's solver progress:
```bash
az batch task file download \
  --job-id job-timefold-runs \
  --task-id $RUN_ID \
  --file-path stdout.txt \
  --destination /tmp/task-stdout.txt
tail -30 /tmp/task-stdout.txt
```

---

## Step 9 — Verify output in Blob, then download via SAS

After task `state` is `completed`, check what landed:

```bash
az storage blob list \
  --account-name $ST \
  --container-name $CONTAINER \
  --prefix "output/${RUN_ID}/" \
  --auth-mode login \
  --query "[].{name:name, size:properties.contentLength}" -o table

az storage blob list \
  --account-name $ST \
  --container-name $CONTAINER \
  --prefix "status/${RUN_ID}/" \
  --auth-mode login \
  --query "[].{name:name, size:properties.contentLength}" -o table
```

You should see:
- `output/test-run-001/result_Schedule.yaml`
- `status/test-run-001/test-run-001.json`

Generate a SAS URL and open it in your browser (same pattern as Phase 2 Step 7):
```bash
EXPIRY=$(date -u -d '+1 hour' '+%Y-%m-%dT%H:%MZ')

az storage blob generate-sas \
  --account-name $ST \
  --container-name $CONTAINER \
  --name "output/${RUN_ID}/result_Schedule.yaml" \
  --permissions r \
  --expiry "$EXPIRY" \
  --auth-mode login \
  --as-user \
  --https-only \
  --full-uri \
  --output tsv
```

Paste the printed URL into your browser → save the file → open in VS Code →
that's the solver's output. **End-to-end run complete.** 🎉

---

## What you should have at the end of Phase 5

- [ ] `Microsoft.Batch` Registered
- [ ] User-assigned MI `mi-tf-pool` exists
- [ ] MI has `AcrPull` on ACR (portal-verified)
- [ ] MI has `Storage Blob Data Contributor` on Storage (portal-verified)
- [ ] Batch account `batchtimefolddevseiya` linked to storage account
- [ ] Pool `pool-timefold-dev` exists, one low-priority node idle
- [ ] Job `job-timefold-runs` exists
- [ ] Task `test-run-001` completed successfully (exit code 0)
- [ ] `output/test-run-001/result_Schedule.yaml` exists in Blob
- [ ] `status/test-run-001/test-run-001.json` exists in Blob with `"status": "Completed"`
- [ ] You downloaded the result via SAS URL in your browser

Tell me **"Phase 5 done"** and we'll do Phase 6 — wire all of this into the
**real API Controller** deployed to your ACA app, so the webapp can drive it.

---

## Cost reality check

| Item                                       | Cost                                                                |
| ------------------------------------------ | ------------------------------------------------------------------- |
| Batch account                              | $0 (free)                                                           |
| Pool idle (0 dedicated, 1 low-pri target)  | ~$0.01/hour ($7/mo if left running) — **scale to 0 when not using** |
| Per task run (F2s_v2 low-pri, 30 min)      | ~$0.01                                                              |
| ACR Basic                                  | $5/mo (from Phase 4)                                                |
| Storage / ACA                              | ~$0 (unchanged)                                                     |

**To stop the pool from billing between sessions:**
```bash
az batch pool resize \
  --pool-id pool-timefold-dev \
  --target-dedicated-nodes 0 \
  --target-low-priority-nodes 0
```
Costs $0 while at zero nodes. Resize back to `--target-low-priority-nodes 1`
when you want to run a task.

**To delete everything at the end of the project:**
```bash
az group delete --name $RG --yes --no-wait
```

---

## Troubleshooting

| Symptom                                                              | Cause                                                            | Fix                                                                          |
| -------------------------------------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `az batch pool create` → "vCPU quota exceeded"                       | No quota on this subscription                                    | Step 0c — request a quota increase                                           |
| Node stuck in `starting` for >15 min                                 | Image pull failing                                               | Check `az batch node list` for `errors` field; usually MI doesn't have AcrPull yet — re-verify in portal |
| Task `state: completed, exitCode: 125 or 126`                        | Container failed to start (usually mount or image issue)         | Download `stderr.txt` and `wd/stdout.txt`; check `containerRunOptions` flags |
| Task `state: completed, exitCode: 2`                                 | Entrypoint script said "input not found"                         | `resourceFiles` didn't download — check the MI has Blob Data role and that input blobs exist |
| Task `state: completed, exitCode: 0` but no output blobs             | `outputFiles` pattern doesn't match what was written             | Download stdout to confirm solver wrote to `/work/output/`; check filePattern in JSON |
| `az batch task file download` → "task not running"                   | Task finished — file is in working dir not stdout anymore        | Use `--file-path wd/output/result_Schedule.yaml` etc.                        |
| Solver runs forever                                                  | Default termination is hours                                      | Cancel: `az batch task terminate --job-id job-timefold-runs --task-id test-run-001`; or reduce solve time in the Java code |

---

## What's next (Phase 6 preview)

Phase 6 is where we **automate everything you just did manually** into a
real API Controller:

1. Write a small Node/Express app exposing `POST /runSolver`,
   `GET /status/{runId}`, `GET /download/{runId}`, `POST /cancel/{runId}`,
   `DELETE /run/{runId}`
2. The app uses the Azure Blob SDK (via the ACA app's MI from Phase 3)
   to upload input YAMLs
3. The app uses the Azure Batch SDK to create tasks (using the same JSON
   shape from Phase 5 Step 7)
4. Deploy the app to your existing ACA `ca-tf-api` (replacing the
   hello-world image)

Then Phase 7 flips `VITE_API_BASE_URL` in the webapp and you have the full
end-to-end demo with a real UI.
