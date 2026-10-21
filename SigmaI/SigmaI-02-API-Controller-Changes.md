# Guide 02 — API Controller changes (v1 → v2)

## 1. Backup first (done)

| Folder | What it is |
|---|---|
| `api-controller/` | the current Timefold-only controller — **left untouched** |
| `api-controller-backup-20261020/` | exact copy (no `node_modules`, no `.git`). It includes the **uncommitted** `src/server.js` edits that existed in `api-controller/` today; `UNCOMMITTED_server.js.diff` shows that diff against the last commit |
| `api-controller-v2/` | the new controller (this guide) |

If you ever need the old behaviour back: redeploy the old image tag (guide 03 §10) — or build from the backup folder.

## 2. What v2 does differently

Same endpoints; Timefold behaviour is unchanged when `engine` is missing or `timefold`.

| Endpoint | v1 | v2 |
|---|---|---|
| `GET /health` | `{ok:true}` | `{ok, version, engines:[…], sigmaiDwave}` — SchedulerWeb uses `engines` |
| `POST /runSolver` | Timefold only | reads `engine` + Sigma-I fields, validates them (400 on bad input), routes to the right Batch job, writes `runs/{runId}.json` |
| `GET /status/:id` | blob status + live file | same, plus `engine`, `option`, `queuePosition` (while waiting), and a `BridgeExitedWithoutStatus` failure if a Sigma-I task ends without a final status |
| `GET /download/:id` | stream result | unchanged |
| `DELETE /run/:id` | terminate + delete blobs | terminates in the **right** job, records the outcome in `runs/{runId}.json` (`deleted:true`) **before** deleting input/output/status |

Other behaviour changes (all deliberate):
- A failed Batch submit now ends as **`Failed (SubmitFailed)`** instead of staying `Submitted` forever.
- Sigma-I runs store inputs under fixed names `EnvConfig.yaml` / `Schedule.yaml`; Timefold keeps `originalname` (as v1).
- Sigma-I fields sent with `engine=timefold` → **400** (catches a wrong UI state).
- The Timefold Batch task spec is copied from v1 verbatim — do not edit it without re-testing Timefold.
- Dockerfile base is **`node:22-alpine`** (v1 used 20; the installed `@azure/*` packages declare `node >=22`).

## 3. Code map (`api-controller-v2/`)

| File | Role |
|---|---|
| `src/server.js` | Express app, Blob + Batch calls, task specs (Timefold + Sigma-I) |
| `src/options.js` | **pure** logic: option validation, `sigmai_options.json`, queue position, manifest, status fallbacks |
| `test/options.test.js` | 17 unit tests for `options.js` |
| `test/bridge.test.js` | 7 tests that run the real `bridge.sh` against `test/fake-sigmai-server.mjs` |
| `sigmai-setup/` | scripts that run **on the Sigma-I node** (guide 05) |

## 4. Environment variables

| Var | Required | Example | Note |
|---|---|---|---|
| `STORAGE_ACCOUNT` `BLOB_CONTAINER` `BATCH_ACCOUNT_URL` `BATCH_JOB_ID` `SOLVER_IMAGE` `ACR_LOGIN_SERVER` `POOL_MI_RESOURCE_ID` | yes | as v1 | same meaning as v1 |
| `BATCH_JOB_ID_SIGMAI` | for Sigma-I | `job-sigmai-test` | empty ⇒ Sigma-I disabled, `/health` lists only `timefold` |
| `SIGMAI_DWAVE_ENABLED` | no (`false`) | `true` | `false` ⇒ option ② is rejected with a clear message |
| `SIGMAI_MAX_STAGE_TIMEOUT_SEC` | no (`7200`) | | upper limit per stage |
| `SIGMAI_WALLCLOCK_MARGIN_SEC` | no (`900`) | | added to stage1+stage2 for Batch `maxWallClockTime` |
| `SIGMAI_SETUP_VERSION` | no | `0.1.0-t1` | written into the run history |
| `SIGMAI_IMAGE_LABEL` | no | | label recorded in history |
| `STATUS_GRACE_SEC` | no (`120`) | | wait before declaring `BridgeExitedWithoutStatus` |
| `APP_VERSION` | no (`2.0.0`) | `2.0.0-test1` | in `/health` and the history |
| `PORT` | no (`8080`) | | |

## 5. Run history — `runs/{runId}.json`

Written for **every** run (both engines) at submit; "finalized" (adds execution info + result) the first
time the controller sees a terminal status. It survives `DELETE /run` (marked `deleted:true`). Schema:
design doc §6.4. It is best-effort — a history write error never fails a run.

## 6. Test first, then promote (what your team asked for)

1. **Test deployment** = a *separate* ACA app (`$ACA_APP_TEST`) with its own Blob container, its own
   Batch Sigma-I job, same Batch account. Real users never touch it. Steps: guide 03 §9.
2. In the test app verify, in this order (guide 04 part D):
   - [ ] `/health` shows `engines:["timefold","sigmai"]`
   - [ ] **Timefold regression**: a normal Timefold run completes and downloads (proves v2 didn't break v1)
   - [ ] Sigma-I option ③ (SCIP+SCIP) completes; result opens in the Gantt editor
   - [ ] Cancel (DELETE) while queued and while running
   - [ ] A failing input gives `Failed` with a readable `error.type`
   - [ ] Option ② only after the D-Wave key/contract exist
3. **Promote** (only after the team agrees):
   ```bash
   source ~/azure-timefold-company-env.sh
   # keep the old image tag handy for rollback, e.g. $ACR_LOGIN/api-controller:v1
   az containerapp update --name "$ACA_APP" --resource-group "$RG" \
     --image "$ACR_LOGIN/api-controller:v2" \
     --set-env-vars BATCH_JOB_ID_SIGMAI="$JOB_ID_SIGMAI" SIGMAI_DWAVE_ENABLED=false \
                    SIGMAI_SETUP_VERSION="0.1.0" APP_VERSION="2.0.0"
   ```
   The real app already has the v1 env vars; this only adds the Sigma-I ones. **Rollback:**
   `az containerapp update --name "$ACA_APP" --resource-group "$RG" --image "$ACR_LOGIN/api-controller:v1"`.
   The real app then needs its own Sigma-I job/pool decision (reuse `pool-sigmai`, new job
   `job-sigmai`) and the same RBAC as the test app.
4. After promotion: when the team is happy, replace the contents of `api-controller/` with v2 (git commit
   in that repo) so source and deployment match; keep the backup folder for one release.

## 7. Known limits of v2

- No `GET /runs` list, no Blob index tags, no sweeper (design §7.5/7.6) — later phases.
- Queue position counts **active** tasks in the Sigma-I job older than yours; it's approximate while nodes boot.
- The Batch/Blob code paths can only be exercised against real Azure (guide 04 part D). Locally you can
  test validation, `/health`, the pure logic and the bridge.
