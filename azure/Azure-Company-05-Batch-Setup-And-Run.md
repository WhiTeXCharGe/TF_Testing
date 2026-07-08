# Company Phase 5 — Verify the Batch pool and run one real task

**Goal of this phase:** confirm the Batch pool is correctly wired to pull
`$SOLVER_IMAGE` and read/write `$ST`/`$CONTAINER`, then submit **one manual
task** end-to-end — proving the whole compute path works before any API
code exists. If this phase works, Phase 6 (the API Controller) only has to
wrap this exact task-submission call in an HTTP endpoint.

**Time:** ~45 minutes, most of it waiting for the node to provision and the
solve to run.
**Prereqs:** Phase 1–4 done. `$SOLVER_IMAGE`, `$MI_ID`, `$ST`, `$CONTAINER`,
`$BATCH`, `$POOL_ID` all set in your env script. RBAC roles from
[Azure-Company-02-RBAC.md §2.3](./Azure-Company-02-RBAC.md#23--batch-pools-user-assigned-managed-identity-2-roles)
confirmed on `$MI_NAME`.

---

## Step 1 — Source env, sign in to Batch

```bash
source ~/azure-timefold-company-env.sh
echo "BATCH=$BATCH  BATCH_URL=$BATCH_URL  POOL_ID=$POOL_ID  SOLVER_IMAGE=$SOLVER_IMAGE"

az batch account login --name "$BATCH" --resource-group "$RG"
```

## Step 2 — Check the vCPU quota

New subscriptions (including some company ones) sometimes start with 0
Batch quota, which silently caps the pool at 0 running nodes forever.

```bash
az batch location quotas show --location "$LOC" -o table
```

You want `DedicatedCoreQuotaPerVMFamily` or `LowPriorityCoreQuota` above 0
for whatever VM family the pool uses. If both are 0:
- Portal → **Quotas** → **Compute** → filter by subscription + `$LOC` →
  find the Batch row → request increase (or ask the admin, since this may
  require elevated permission on a company subscription).

## Step 3 — Inspect the existing pool

```bash
az batch pool show --pool-id "$POOL_ID" \
  --query "{id:id, state:state, vmSize:vmSize, targetLowPriority:targetLowPriorityNodes, targetDedicated:targetDedicatedNodes, identity:identity.type}" \
  -o table
```

Confirm:
- [ ] `state` is `active`
- [ ] `identity.type` is `UserAssigned`

Then check the container configuration specifically (this is the part most
likely to be wrong if the pool predates you knowing `$SOLVER_IMAGE`):

```bash
az batch pool show --pool-id "$POOL_ID" \
  --query "virtualMachineConfiguration.containerConfiguration" -o json
```

Look at `containerImageNames` — does it list your `$SOLVER_IMAGE` (or at
least the same repository, `timefold`)? And does `containerRegistries[0].registryServer`
match `$ACR_LOGIN`?

**If everything matches** — skip to Step 5.

**If the pool references a different image/tag, or doesn't exist at all** —
continue to Step 4.

## Step 4 — Only if needed: create or update the pool

### 4a. If the pool doesn't exist yet

```bash
mkdir -p ~/azure-timefold-company
cat > ~/azure-timefold-company/pool-config.json <<EOF
{
  "id": "${POOL_ID}",
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
        { "registryServer": "${ACR_LOGIN}", "identityReference": { "resourceId": "${MI_ID}" } }
      ],
      "containerImageNames": [ "${SOLVER_IMAGE}" ]
    }
  },
  "targetDedicatedNodes": 0,
  "targetLowPriorityNodes": 1,
  "enableAutoScale": false,
  "taskSlotsPerNode": 1,
  "identity": {
    "type": "UserAssigned",
    "userAssignedIdentities": { "${MI_ID}": {} }
  }
}
EOF

az batch pool create --json-file ~/azure-timefold-company/pool-config.json
```

Ask the admin first if a naming convention exists for `$POOL_ID` — don't
invent one silently if this pool is meant to be company-standard.

### 4b. If the pool exists but the image reference is stale

Pools don't support editing `containerConfiguration` in place. Two options,
in order of preference:
1. **Ask the admin** to update it, if they own pool configuration.
2. If you're expected to manage it yourself: delete and recreate using the
   JSON above (this drops any running nodes/tasks — only do this if the
   pool is genuinely idle):
   ```bash
   az batch pool delete --pool-id "$POOL_ID" --yes
   # wait for deletion to finish, then re-run the `az batch pool create` command above
   ```

## Step 5 — Wait for a node to be ready

```bash
until [ "$(az batch node list --pool-id "$POOL_ID" --query '[0].state' -o tsv 2>/dev/null)" = "idle" ]; do
  echo "waiting for node..."; sleep 15
done
echo "Node is idle and ready."
```

If the pool's `targetLowPriorityNodes` is 0, resize it up first:
```bash
az batch pool resize --pool-id "$POOL_ID" --target-low-priority-nodes 1
```

## Step 6 — Confirm or create the standing job

```bash
az batch job show --job-id "$JOB_ID" -o table 2>/dev/null || echo "Job not found — will create it"
```

If it doesn't exist:
```bash
JOB_ID="job-timefold-runs"
az batch job create --id "$JOB_ID" --pool-id "$POOL_ID"

cat >> ~/azure-timefold-company-env.sh <<EOF

export JOB_ID=$JOB_ID
EOF
source ~/azure-timefold-company-env.sh
```

## Step 7 — Upload test input files

```bash
RUN_ID=company-smoke-test-001

az storage blob upload \
  --account-name "$ST" --container-name "$CONTAINER" --auth-mode login \
  --name "input/${RUN_ID}/EnvConfig.yaml" \
  --file "/c/Users/Seiya/Desktop/work/Timefold/web/Timefold/src/main/resource/EnvConfig.yaml" --overwrite

az storage blob upload \
  --account-name "$ST" --container-name "$CONTAINER" --auth-mode login \
  --name "input/${RUN_ID}/Schedule.yaml" \
  --file "/c/Users/Seiya/Desktop/work/Timefold/web/Timefold/src/main/resource/Schedule.yaml" --overwrite
```

## Step 8 — Submit the task

This JSON is exactly what
[web/api-controller/src/server.js](../../api-controller/src/server.js)
sends automatically once it's deployed in Phase 6 — running it manually
here first isolates any Batch/permissions problem from any API code problem.

Note the two `outputFiles` entries use an **exact filename**, not a wildcard
pattern — that makes `destination.path` the literal final blob name, with
no ambiguity about nested folders.

```bash
cat > ~/azure-timefold-company/task-${RUN_ID}.json <<EOF
{
  "id": "${RUN_ID}",
  "commandLine": "/bin/bash -c 'mkdir -p /work/output /work/status && /app/entrypoint.sh'",
  "containerSettings": {
    "imageName": "${SOLVER_IMAGE}",
    "containerRunOptions": "--rm --workdir /app -v \$AZ_BATCH_TASK_WORKING_DIR/input:/work/input:ro -v \$AZ_BATCH_TASK_WORKING_DIR/output:/work/output -v \$AZ_BATCH_TASK_WORKING_DIR/status:/work/status -e RUN_ID=${RUN_ID}",
    "registry": { "registryServer": "${ACR_LOGIN}", "identityReference": { "resourceId": "${MI_ID}" } }
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
      "filePattern": "output/result_Schedule.yaml",
      "destination": {
        "container": {
          "containerUrl": "https://${ST}.blob.core.windows.net/${CONTAINER}",
          "path": "output/${RUN_ID}/result_Schedule.yaml",
          "identityReference": { "resourceId": "${MI_ID}" }
        }
      },
      "uploadOptions": { "uploadCondition": "taskSuccess" }
    },
    {
      "filePattern": "status/${RUN_ID}.json",
      "destination": {
        "container": {
          "containerUrl": "https://${ST}.blob.core.windows.net/${CONTAINER}",
          "path": "status/${RUN_ID}.json",
          "identityReference": { "resourceId": "${MI_ID}" }
        }
      },
      "uploadOptions": { "uploadCondition": "taskCompletion" }
    }
  ]
}
EOF

az batch task create --job-id "$JOB_ID" --json-file ~/azure-timefold-company/task-${RUN_ID}.json
```

## Step 9 — Watch it run

```bash
until [ "$(az batch task show --job-id "$JOB_ID" --task-id "$RUN_ID" --query state -o tsv)" = "completed" ]; do
  state=$(az batch task show --job-id "$JOB_ID" --task-id "$RUN_ID" --query state -o tsv)
  echo "$(date +%T) state=$state"
  sleep 15
done

az batch task show --job-id "$JOB_ID" --task-id "$RUN_ID" \
  --query "{state:state, exitCode:executionInfo.exitCode}" -o table
```

`exitCode` should be `0`.

If something goes wrong, pull the logs:
```bash
az batch task file download --job-id "$JOB_ID" --task-id "$RUN_ID" \
  --file-path stdout.txt --destination /tmp/task-stdout.txt
tail -40 /tmp/task-stdout.txt

az batch task file download --job-id "$JOB_ID" --task-id "$RUN_ID" \
  --file-path stderr.txt --destination /tmp/task-stderr.txt
tail -40 /tmp/task-stderr.txt
```

## Step 10 — Verify the output landed in Blob at the right path

```bash
az storage blob list \
  --account-name "$ST" --container-name "$CONTAINER" --auth-mode login \
  --prefix "output/${RUN_ID}/" \
  --query "[].{name:name, size:properties.contentLength}" -o table

az storage blob show \
  --account-name "$ST" --container-name "$CONTAINER" --auth-mode login \
  --name "status/${RUN_ID}.json" \
  --query "{name:name, size:properties.contentLength}" -o table
```

You should see exactly:
- `output/${RUN_ID}/result_Schedule.yaml`
- `status/${RUN_ID}.json`

Download and check the status content:
```bash
az storage blob download \
  --account-name "$ST" --container-name "$CONTAINER" --auth-mode login \
  --name "status/${RUN_ID}.json" --file "/tmp/status-check.json"
cat /tmp/status-check.json
```
Should show `"status": "Completed"`.

## Step 11 — Clean up the smoke-test data

```bash
for prefix in "input/${RUN_ID}/" "output/${RUN_ID}/"; do
  for blob in $(az storage blob list --account-name "$ST" --container-name "$CONTAINER" --auth-mode login --prefix "$prefix" --query "[].name" -o tsv); do
    az storage blob delete --account-name "$ST" --container-name "$CONTAINER" --auth-mode login --name "$blob"
  done
done
az storage blob delete --account-name "$ST" --container-name "$CONTAINER" --auth-mode login --name "status/${RUN_ID}.json"
```

## Step 12 — Scale the pool back down if you're stopping for now

```bash
az batch pool resize --pool-id "$POOL_ID" --target-low-priority-nodes 0 --target-dedicated-nodes 0
```
Costs ~$0 while at zero nodes.

---

## What you should have at the end of this phase

- [ ] Pool `$POOL_ID` confirmed `active`, `UserAssigned` identity, references `$SOLVER_IMAGE`
- [ ] Job `$JOB_ID` exists (created or confirmed)
- [ ] One task ran to `state: completed`, `exitCode: 0`
- [ ] `output/${RUN_ID}/result_Schedule.yaml` and `status/${RUN_ID}.json` both appeared in Blob at the exact paths shown above
- [ ] Smoke-test blobs cleaned up

Next: [Azure-Company-06-API-Controller-Deploy.md](./Azure-Company-06-API-Controller-Deploy.md)
— deploy the real API Controller that automates everything you just did by
hand.

---

## Troubleshooting

| Symptom                                                              | Cause                                                            | Fix |
| ------------------------------------------------------------------- | ----------------------------------------------------------------- | ----- |
| `az batch pool create` → vCPU quota exceeded                        | No quota on this subscription/region                              | Step 2 — request an increase |
| Node stuck in `starting` for 15+ minutes                             | Image pull failing                                                | `az batch node list` for an `errors` field; usually the pool's MI doesn't actually have `AcrPull` — recheck Phase 2.3 |
| Task `completed`, `exitCode: 125` or `126`                            | Container failed to start — mount or image problem                 | Download stderr.txt / stdout.txt (Step 9) |
| Task `completed`, `exitCode: 2`                                      | Entrypoint said input not found                                    | `resourceFiles` didn't download — confirm the MI has `Storage Blob Data Contributor` and the input blobs actually exist at `input/${RUN_ID}/` |
| Task `completed`, `exitCode: 0`, but no output blob                  | `outputFiles` filePattern doesn't match what was actually written  | Check stdout.txt for where the solver wrote its output; confirm it matches `/work/output/result_Schedule.yaml` exactly |
| `status/${RUN_ID}.json` missing even though the task ran              | `uploadCondition` was wrong (e.g. `taskSuccess` on a task that failed) | The JSON in Step 8 already uses `taskCompletion` for status — double check you copied it exactly |
