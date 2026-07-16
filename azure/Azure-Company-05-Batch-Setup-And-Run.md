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

## Step 0 — If the resource group has more than one Storage account, confirm the link

Skip this if there's only one Storage account in the resource group. If
there are two or more, this is the single most important check in this
phase — the pool's `resourceFiles` step (Step 8) uses
`autoStorageContainerName`, which resolves to whichever storage account is
linked to the Batch account's own auto-storage setting, **not** whichever
one you assumed was "for this project."

```bash
az batch account show --name "$BATCH" --resource-group "$RG" --query "autoStorage.storageAccountId" -o tsv
```

Compare the storage account name at the end of that resource ID against
`$ST` in your env script. If they don't match, `$ST` is wrong — go back to
[Azure-Company-01-Access-And-Resources.md Step 7](./Azure-Company-01-Access-And-Resources.md#step-7--record-the-storage-account),
fix `$ST` and `$CONTAINER` to point at the correct account, and redo the
RBAC checks in [Azure-Company-02-RBAC.md](./Azure-Company-02-RBAC.md) and
the verification in [Azure-Company-03-Storage-Verify.md](./Azure-Company-03-Storage-Verify.md)
against the corrected account before continuing here. A task submitted
against the wrong `$ST` will create input blobs Batch can never find.

## Step 1 — Source env, sign in to Batch

```bash
source ~/azure-timefold-company-env.sh
echo "BATCH=$BATCH  BATCH_URL=$BATCH_URL  POOL_ID=$POOL_ID  SOLVER_IMAGE=$SOLVER_IMAGE"

az batch account login --name "$BATCH" --resource-group "$RG"
```

## Step 2 — Check the vCPU quota

New subscriptions (including some company ones) sometimes start with 0
Batch quota, which silently caps the pool at 0 running nodes forever.

**Use the Batch account itself, not `az batch location quotas show`.**
That older command queries a subscription+region quota model most Batch
accounts don't use anymore — for a default account (`Pool Allocation Mode:
BatchService`), quota is enforced **per Batch account**, and the
subscription/location command just returns empty/null instead of erroring,
which is confusing rather than helpful.

```bash
# Aggregate quota on the account
az batch account show --name "$BATCH" --resource-group "$RG" \
  --query "{dedicatedCoreQuota:dedicatedCoreQuota, lowPriorityCoreQuota:lowPriorityCoreQuota, poolQuota:poolQuota, perFamilyEnforced:dedicatedCoreQuotaPerVmFamilyEnforced}" \
  -o table

# Per-VM-family breakdown — this is the one that actually gates a specific VM size
az batch account show --name "$BATCH" --resource-group "$RG" \
  --query "dedicatedCoreQuotaPerVmFamily" -o table
```

You need `dedicatedCoreQuota` (or `lowPriorityCoreQuota`, matching whichever
tier your pool uses) high enough to cover `node count × vCPUs per node` for
the VM size you're about to use. If `perFamilyEnforced` is `true`, the
specific VM family (e.g. Fsv2 for F-series) in the per-family list also
needs to individually cover that number — the aggregate alone isn't enough
in that case. If either is short:
- Portal → your **Batch account** → left sidebar **Quotas** (a blade on the
  account itself — not the generic subscription-wide Quotas page) → request
  an increase (or ask the admin, since this may require elevated permission
  on a company subscription).

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

**Don't assume the image reference below still exists.** Azure Batch's list
of supported (verified, container-capable) images changes over time and
varies by subscription — the classic `microsoft-azure-batch` /
`ubuntu-server-container` offer that used to be the standard pick may not
be available anymore. Confirm the real current options first:

```bash
az batch pool supported-images list --query "length(@)"   # sanity check you're authenticated (should be dozens, not 0/error)

# List every image that actually supports containers, any distro/publisher
az batch pool supported-images list \
  --query "[?capabilities && contains(capabilities, 'DockerCompatible')].{publisher:imageReference.publisher, offer:imageReference.offer, sku:imageReference.sku, nodeAgentSku:nodeAgentSkuId, osType:osType}" \
  -o table
```

(The `capabilities &&` guard matters — some entries have `capabilities: null`,
and `contains()` errors on `null` instead of just skipping it.)

Pick a `linux` row matching your pool's VM architecture (skip `arm64` rows
if you're on a standard x86_64 size like `Standard_F16s_v2`). As of writing,
many subscriptions no longer offer `microsoft-azure-batch` at all and
instead show `microsoft-dsvm` / `ubuntu-hpc` as the container-capable Ubuntu
option — heavier (it's the Data Science VM image, with extra tooling
preinstalled) but functionally fine, since your container runs isolated
regardless of what's on the host. Use whatever your own query actually
returns, not the values below verbatim.

**VM size and node count are a per-workload decision, not a fixed rule.**
For this project we settled on `Standard_F16s_v2` (16 vCPU / 32 GiB) ×
**2 dedicated nodes** — Dedicated rather than Low-priority because a
16-vCPU node getting evicted mid-solve (Low-priority can be reclaimed with
~30s notice) wastes a genuinely expensive VM's time; worth the extra cost
for a size this large. Adjust both values below if your own quota/cost
constraints differ — see [Azure-Products-Required.md](./Azure-Products-Required.md)
for the general VM-size tradeoff table.

```bash
mkdir -p ~/azure-timefold-company
cat > ~/azure-timefold-company/pool-config.json <<EOF
{
  "id": "${POOL_ID}",
  "vmSize": "STANDARD_F16S_V2",
  "virtualMachineConfiguration": {
    "imageReference": {
      "publisher": "microsoft-dsvm",
      "offer": "ubuntu-hpc",
      "sku": "2204",
      "version": "latest"
    },
    "nodeAgentSKUId": "batch.node.ubuntu 22.04",
    "containerConfiguration": {
      "type": "dockerCompatible",
      "containerRegistries": [
        { "registryServer": "${ACR_LOGIN}", "identityReference": { "resourceId": "${MI_ID}" } }
      ],
      "containerImageNames": [ "${SOLVER_IMAGE}" ]
    }
  },
  "targetDedicatedNodes": 2,
  "targetLowPriorityNodes": 0,
  "enableAutoScale": false,
  "taskSlotsPerNode": 1,
  "identity": {
    "type": "UserAssigned",
    "userAssignedIdentities": [
      { "resourceId": "${MI_ID}" }
    ]
  }
}
EOF

az batch pool create --json-file ~/azure-timefold-company/pool-config.json
```

**Status as of this writing: not yet confirmed reaching `idle`.** Every
fix above (image reference, `userAssignedIdentities` array shape, ACR
network access) resolved the specific error hit at that step, but the
final end-to-end confirmation (both nodes actually reaching `idle`, then
Step 5 onward) is still pending — blocked by the network-access issue
described in the box below, not by anything wrong with this JSON. Re-run
Step 5 (node wait loop) once back on the company network before assuming
this configuration is fully proven.

> **Note on `userAssignedIdentities` shape:** this is an **array** of
> `{resourceId: ...}` objects, not the `{"<resourceId>": {}}` dict form
> used elsewhere in Azure (e.g. on an ACA app or VM). If you copy an
> identity block from ARM/Bicep examples elsewhere, you'll hit
> `(InvalidRequestBody) ... A 'StartArray' node was expected` from
> `az batch pool create` — the dict form is valid ARM JSON but not what
> this data-plane API expects here.

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
| `az batch pool create` → `(InvalidRequestBody) ... A 'StartArray' node was expected` | `identity.userAssignedIdentities` was written as an ARM-style dict instead of the array-of-objects form this API expects | Use `"userAssignedIdentities": [{ "resourceId": "..." }]`, not `{"<resourceId>": {}}` — see the note under Step 4a |
| Same error, but the JSON validates fine locally (`python -m json.tool`)     | Confirms it's a schema mismatch, not malformed JSON — check every array/object field against the exact shapes in Step 4a rather than assuming a typo | Re-diff your file against Step 4a's JSON field-by-field |
| Portal's pool wizard won't let you set Container configuration to Custom (it's greyed out) | The selected VM image (Publisher/Offer/Sku) isn't container-capable — plain distro images like `canonical`/`Ubuntu Server` don't support it, only specific verified images do | Run the `az batch pool supported-images list` query at the top of Step 4a to find a real `DockerCompatible` image, and select that exact Publisher/Offer/Sku in the portal |
| `az batch pool supported-images list` with a `contains(capabilities, ...)` filter errors with `invalid type for value: None` | Some images have `capabilities: null` instead of an array, and `contains()` can't run against `null` | Guard it: `[?capabilities && contains(capabilities, 'DockerCompatible')]` — the `capabilities &&` short-circuits past the null entries |
| Node stuck in `starting` for 15+ minutes                             | Image pull failing                                                | `az batch node list` for an `errors` field; usually the pool's MI doesn't actually have `AcrPull` — recheck Phase 2.3 |
| Node reaches `unusable` (not just slow — this is a real error state, waiting longer never fixes it) | `az batch node show --pool-id "$POOL_ID" --node-id <id> --query "errors"` for the real cause — a common one: `NodePreparationError` / `"ACR token exchange failed ... client with IP '...' is not allowed access"` | This is an ACR **network firewall** issue, not RBAC — `AcrPull` being correctly assigned doesn't matter if the node's IP is blocked before auth is even evaluated. Check `az acr show --name "$ACR" --query "{publicNetworkAccess:publicNetworkAccess, defaultAction:networkRuleSet.defaultAction, ipRules:networkRuleSet.ipRules}"` — if restricted, ask the admin to allow the pool's outbound IP (or all networks temporarily for a dev/test pool), since changing ACR firewall rules needs management-plane permission beyond `AcrPush`/`AcrPull` |
| `az batch pool resize` / `az batch node delete` / `az batch job create` → `(AuthorizationFailure) This request is not authorized to perform this operation`, even though your RBAC roles look correct | Two different possible causes — don't assume it's the same one twice | **First** re-check you're on the company network (see the warning at the top of [Azure-Company-01](./Azure-Company-01-Access-And-Resources.md)) — a network-routing failure can sometimes surface as an authorization-style error rather than a clean connection-timeout, depending on where in the request path it fails. **If genuinely on the right network and still failing**, this may be a custom/scoped role or Deny Assignment that permits pool/task create + read but excludes specific "action"-verb operations like resize/node-delete/job-create — `az role assignment list --assignee <you> --scope <batch-account-id>` to see the exact role name; if it's not literally the built-in `Azure Batch Account Contributor`, that's the admin's decision to revisit, not something to work around |
| Task `completed`, `exitCode: 125` or `126`                            | Container failed to start — mount or image problem                 | Download stderr.txt / stdout.txt (Step 9) |
| Task `completed`, `exitCode: 2`                                      | Entrypoint said input not found                                    | `resourceFiles` didn't download — confirm the MI has `Storage Blob Data Contributor` and the input blobs actually exist at `input/${RUN_ID}/` |
| Task `completed`, `exitCode: 0`, but no output blob                  | `outputFiles` filePattern doesn't match what was actually written  | Check stdout.txt for where the solver wrote its output; confirm it matches `/work/output/result_Schedule.yaml` exactly |
| `status/${RUN_ID}.json` missing even though the task ran              | `uploadCondition` was wrong (e.g. `taskSuccess` on a task that failed) | The JSON in Step 8 already uses `taskCompletion` for status — double check you copied it exactly |
