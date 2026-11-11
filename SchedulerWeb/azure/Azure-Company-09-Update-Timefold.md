# Company Phase 9 — Update Timefold in Azure (new solver image, new API controller)

**Goal of this phase:** roll out a newer Timefold solver image and the API controller that can pass a third solver input
(`SolverParams.yaml`) to it, without breaking the setup that already works. Phases 4 to 8 built the setup once; this phase is the
repeatable "update" routine.

**Status:** written from the Phase 4–6 steps and the current code. Not yet run against the company Azure (it is not reachable
from the development PC), so treat the first run as the test of this document and note any difference at the end.

**Time:** 30–60 minutes (mostly build and upload). **Prereqs:** Phases 1–6 done, company network (ACR is private, see the
warning in [Azure-Company-01](./Azure-Company-01-Access-And-Resources.md)), Docker Desktop running, role `AcrPush`.

---

## What changed since Phase 4 and Phase 6

| Part | Change | Why it matters for Azure |
| --- | --- | --- |
| Solver (`web/Timefold`) | the solver reads up to **three** inputs from `/work/input`: `EnvConfig.yaml`, `Schedule.yaml` (required) and `SolverParams.yaml` (optional). Without the third one every parameter has a default. Any `TF_*` environment variable of the container also works | a new image is needed; the task has to mount the third file when it exists |
| Solver | trip list input / output, block weights, stay rules, exit code 5 (`InvalidInputData`) | see [Timefold_ChangesAndHowToRun20261022](../../Timefold/Timefold_ChangesAndHowToRun20261022.md) and the `README.md` "Parameters" section of `web/Timefold` |
| Solver defaults | stage 1 up to **300 minutes**, stage 2 on for up to **60 minutes**, stage 1 also stops after 600 s without improvement | a run without `SolverParams.yaml` can take hours. The weekly tools send **stage 2 on, 45 + 15 minutes = 1 hour** unless told otherwise |
| API controller (`web/api-controller` = v1, Timefold only; v2 also routes Sigma-I and is not needed here) | `POST /runSolver` accepts an optional file field **`params`** and stores the uploads under the fixed names `EnvConfig.yaml`, `Schedule.yaml`, `SolverParams.yaml` (before, the original file names were kept, which only worked if the client used exactly these names) | the controller image has to be rebuilt and redeployed. Batch already copies everything under `input/<runId>/` to the node, so the task definition needs no change |

The HTTP contract is otherwise unchanged (`/runSolver`, `/status/:runId`, `/download/:runId`, `/run/:runId`, `/health`), so the webapp
and the tools 2-6 / 3-1 use the same URLs.

## Which update do I need?

| You changed | Do |
| --- | --- |
| only the solver code in `web/Timefold` | Steps 1, 2, 3, 5 (new solver image, point `SOLVER_IMAGE` at it) |
| only `web/api-controller` | Steps 4, 5 |
| both (this time) | all steps; do the solver first, then the controller in one `az containerapp update` |

Rule: **never overwrite an existing tag.** Give every build a new tag (`v2`, `v3`, or a date such as `v20261111`) so that the previous
one stays available for a rollback.

---

## Step 1 — Test the new solver locally first

From `web/Timefold`, build and run the sample once. This is the fastest way to catch a broken image before Azure is involved.

```bash
cd web/Timefold
docker compose up --build solver
```

Check in the output: the `[params]` line shows the values you expect, the last line is `done — result at /work/output/result_Schedule.yaml`,
and `work/output/result_Schedule.yaml` exists. To test the third input, put a small `SolverParams.yaml` (for example `stage1_minutes: 2`,
`stage2_enabled: false`) next to the inputs in `work/input` and see that the `[params]` line changes.

## Step 2 — Build and push the new solver image

```bash
source ~/azure-timefold-company-env.sh
az acr login --name "$ACR"

cd web/Timefold
TAG=v2                                   # a NEW tag, never one that already exists
docker build -t "$ACR_LOGIN/timefold:$TAG" .
docker push "$ACR_LOGIN/timefold:$TAG"

az acr repository show-tags --name "$ACR" --repository timefold -o table      # old and new tag must both be listed
```

The Dockerfile builds from `pom.xml` in `web/Timefold` (the `docker/pom-standalone.xml` mentioned in Phase 4 is no longer used). Run
the command from that folder.

## Step 3 — The Batch pool needs no change

The pool lists the image in `containerConfiguration.containerImageNames` (Phase 5), but that list only **pre-loads** images when a node
starts. A task whose `SOLVER_IMAGE` is a different tag is pulled from `$ACR` when the task starts (the pool already has the registry and the
managed identity for it), so the pool does not have to be recreated. The first task on each node pays the pull time (about 0.5 GB).
If you want the new tag pre-loaded, recreate the pool as in Phase 5 Step 4b with the new tag in `containerImageNames`; only do this when
the pool is idle.

## Step 4 — Build, push and deploy the API controller

Only when `web/api-controller` changed (this time: the `params` input).

```bash
cd web/api-controller
npm install
```

Optional local test against the real Azure resources, as in Phase 6 Step 2, now with the third input:

```bash
curl -X POST http://localhost:8080/runSolver \
  -F "env=@<EnvConfig.yaml>" -F "sched=@<Schedule.yaml>" -F "params=@<SolverParams.yaml>"
```

Build and push (new tag again), then deploy together with the new solver tag:

```bash
CTAG=v2
docker build -t "$ACR_LOGIN/api-controller:$CTAG" .
docker push "$ACR_LOGIN/api-controller:$CTAG"

az containerapp update \
  --name "$ACA_APP" --resource-group "$RG" \
  --image "$ACR_LOGIN/api-controller:$CTAG" \
  --set-env-vars SOLVER_IMAGE="$ACR_LOGIN/timefold:$TAG"
```

`--set-env-vars` only changes the variables you name; the other six of Phase 6 stay as they are. `SOLVER_IMAGE` is the single place that decides
which solver image the Batch tasks run. For a solver-only update, run the same command without `--image`. Wait about 30 seconds for the new
revision, and save the new tags in your env script (`export SOLVER_IMAGE=...`).

## Step 5 — Smoke test the deployed version

```bash
curl https://$APP_URL/health
# {"ok":true}

curl -X POST https://$APP_URL/runSolver \
  -F "env=@<EnvConfig.yaml>" -F "sched=@<Schedule.yaml>" -F "params=@<SolverParams.yaml>"
# {"runId":"...","status":"Submitted"}
```

Use a short `SolverParams.yaml` for the test (for example `stage1_minutes: 2`, `stage1_unimproved_seconds: 20`, `stage2_enabled: false`).
Poll `/status/<runId>` until `Completed`, then `/download/<runId>`. Prove that the parameters arrived and which image ran:

```bash
az batch task show --job-id "$JOB_ID" --task-id "<runId>" --query "containerSettings.imageName" -o tsv
az batch task file download --job-id "$JOB_ID" --task-id "<runId>" --file-path stdout.txt --destination /tmp/task-stdout.txt
grep "\[params\]" /tmp/task-stdout.txt
```

The image name must be the new tag, and the `[params]` line must show your values (`stage1=2min/20s ... stage2Enabled=false`). Without the third input it shows the defaults.

## Step 6 — Roll back if it goes wrong

The previous images are still in `$ACR`, so a rollback is another update with the previous values:

```bash
az containerapp update --name "$ACA_APP" --resource-group "$RG" \
  --image "$ACR_LOGIN/api-controller:<previous tag>" \
  --set-env-vars SOLVER_IMAGE="$ACR_LOGIN/timefold:<previous tag>"
```

Do not delete old tags from `$ACR` until the new version has run a real weekly input.

## Step 7 — Write down what is deployed

| Date | Solver image tag | API controller tag | Note |
| --- | --- | --- | --- |
| | | | |

---

## Known points to decide or check

- **Memory.** The solver container uses `JVM_MAX_HEAP` (image default 6g). A full-size weekly input ran out of memory at 6g locally and passed at 12g, so the controller now starts every solver container with `-e JVM_MAX_HEAP=12g`. To change it, set the optional controller variable `SOLVER_JVM_MAX_HEAP` (for example `16g`; the node has 32 GiB) with `az containerapp update ... --set-env-vars SOLVER_JVM_MAX_HEAP=16g`. A run that still dies with `SolverError` (exit 137 or out-of-memory in `stdout.txt`) needs a larger value.
- **Time limit.** The Batch task has no maximum run time in this controller. Without `SolverParams.yaml` the solver defaults allow hours, so always send the third input (the tools 2-6 do) or set `TF_STAGE1_MINUTES` / `TF_STAGE2_MINUTES` in `containerRunOptions` as a safety net.
- **Output.** Only `result_Schedule.yaml` and the status file are uploaded. `trip_warnings.txt` stays on the node; the status says `InvalidInputData` and points to it when `trip_validation` is `strict`.
- **Tools 2-6 and 3-1** (`tools/weekly_planning`) talk to this API with the same contract. Each has a `config.txt` in its own folder (keep `mode` and the address the same in both): `mode = local` uses the MockServer on the development PC, `mode = online` uses `azure_url`, which is `https://$APP_URL` of this phase (no sign-in is needed, like SchedulerWeb). After this phase: set `mode = online` and `azure_url`, run 2-6 with a short test (`stage1_minutes = 2`, `stage2_enabled = false`), then 3-1. The PC running the tools needs a network path to `$APP_URL`. Run ids are used once on Azure, so leave `run_id` empty in `config.txt`.

## What you should have at the end of this phase

- [ ] the new solver image is in `$ACR` under a new tag, the old tag is still there
- [ ] the new API controller image is in `$ACR` (if it changed) and `$ACA_APP` runs it with the new `SOLVER_IMAGE`
- [ ] `curl https://$APP_URL/health` returns `{"ok":true}`
- [ ] a run with `params` completed, the Batch task used the new image, and the `[params]` line shows the values that were sent
- [ ] the "what is deployed" table in Step 7 is filled in

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `/runSolver` answers `400 Upload error: Unexpected field` | the running controller is the old one: it does not know the `params` field | deploy the new controller image (Step 4) |
| the `[params]` line shows the defaults although `params` was sent | the controller stored the file under another name, or the old controller ignored it | check that the blob `input/<runId>/SolverParams.yaml` exists; redeploy the controller |
| the Batch task still runs the old solver | `SOLVER_IMAGE` was not updated, or the new revision is not the active one | `az containerapp show -n "$ACA_APP" -g "$RG" --query "properties.template.containers[0].env"` and `az containerapp revision list` |
| task `completed` with exit code 125 / 126 | the image could not be pulled or started on the node | download `stderr.txt` and `stdout.txt` (Phase 5 Step 9); check `AcrPull` on the pool identity and that the tag exists |
| `docker push` is denied or cannot connect | not on the company network, or the `az acr login` token expired | Phase 4 troubleshooting; check the network first |
| the run stays `Running` for hours | no `SolverParams.yaml`, so the solver defaults (300 + 60 minutes) apply | send the third input or cancel with `DELETE /run/<runId>` |
