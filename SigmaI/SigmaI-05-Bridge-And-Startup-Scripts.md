# Guide 05 — The two scripts that run on a Sigma-I node

Both live in `api-controller-v2/sigmai-setup/` and are uploaded to Blob by `upload-setup.sh`
(guide 03 §6). They are plain bash (+ `curl`, `python3`), already present on the Batch Ubuntu image.

```
 node boots ──► StartTask: start-sigmai.sh ──► Sigma-I stack stays running (api :8000, worker, redis)
                                                        ▲
 each run ──► Batch task: bridge.sh ──── HTTP + X-API-Key (127.0.0.1:8000) ──┘
```

## 1. `start-sigmai.sh` — the StartTask (once per node boot)

Runs as pool-scope admin. Batch gives the node **no tasks until it exits 0** (`waitForSuccess:true`).
Idempotent: Batch re-runs it after a reboot; an existing API key is kept.

| Step | What it does |
|---|---|
| 1 | finds the setup files Batch downloaded (`docker-compose.azure.yml`, `bridge.sh`, `wheels/*.whl`) and copies them to `$SIGMAI_HOME` (`$AZ_BATCH_NODE_SHARED_DIR/sigmai`). **Exactly one wheel** is required (Sigma-I refuses otherwise) |
| 2 | finds `docker compose` (installs the plugin if missing — needs internet) |
| 3 | creates `API_KEY` once (32 random bytes). It only ever lives in `chmod 600` files on the node |
| 4 | `DWAVE_ENABLED=true` → asks IMDS for a storage token with the pool's Managed Identity and downloads `$SECRETS_CONTAINER/$DWAVE_SECRET_BLOB` into `DWAVE_API_TOKEN` |
| 5 | writes `.env.server` (for compose) and `.env.client` (for the bridge: `API_BASE_URL`, `API_KEY`) |
| 6 | logs in to ACR via the MI (IMDS ARM token → `/oauth2/exchange` → `docker login`) and `compose pull` |
| 7 | `docker compose up -d` |
| 8 | waits (max 20 min) for **both** `GET /health` OK **and** the worker log line `celery@… ready` — `/health` alone answers before the wheel is installed |
| fail | prints `compose ps` + api/worker logs to `startup/stdout.txt` and exits 1 → node `starttaskfailed` |

Settings (pool `startTask.environmentSettings`; full JSON in guide 03 §7.3):

| Var | Meaning |
|---|---|
| `STORAGE_ACCOUNT`, `SECRETS_CONTAINER` (`sigmai-secrets`), `DWAVE_SECRET_BLOB` (`dwave-token.txt`) | where the D-Wave key is |
| `DWAVE_ENABLED` | `true` to fetch the key; `false` = SCIP-only node |
| `POOL_MI_RESOURCE_ID` | pool identity used for IMDS tokens |
| `ACR_LOGIN_SERVER`, `SIGMAI_IMAGE`, `REDIS_IMAGE` | images to pull |
| optional: `SIGMAI_HOME`, `SIGMAI_PORT` (8000), `READY_TIMEOUT_SEC` (1200), `SKIP_ACR_LOGIN`, `SETUP_DIR` | local rehearsal / tuning |

Security: the port is bound to `127.0.0.1` only; secrets are never in `environmentSettings` or logs.

## 2. `bridge.sh` — one per run

Does not solve anything; it is an HTTP client for the node's Sigma-I.

| Phase | What it does |
|---|---|
| inputs | reads `EnvConfig.yaml`, `Schedule.yaml`, `sigmai_options.json` from `wd/input/input/<RUN_ID>/` (Batch keeps the full blob path under `filePath`) |
| submit | `POST /jobs` with `job_id=<RUN_ID>` + options (`--form-string`, so a label starting with `@` is never read as a file). `202` ok · `JOB_ALREADY_EXISTS` → resume polling (task retry) · other `4xx` → `Failed InvalidInput` · `5xx`/no connection → 3 retries then `Failed SigmaIUnavailable` |
| poll | every `POLL_INTERVAL_SEC` (10): maps `queued/running/cancel_requested → Running (+progress)`, writes `status/<RUN_ID>.json` atomically; the API controller reads this file live |
| deadline | stage1 + stage2 + `DEADLINE_MARGIN_SEC` (600); exceeded → `DELETE /jobs/<id>` and `Failed TIMEOUT` |
| finish | `finished` → `GET /download` → `output/result_Schedule.yaml`; `GET /result` + job logs → `output/sigmai/`; status `Completed` (progress 100) with `sigmai.summary.validationPassed` |
| failure | `failed` → logs copied, status `Failed` with Sigma-I's `error.type` (e.g. `OPTIMIZATION_FAILED`, `STAGE1_FAILED`, `TIMEOUT`) |
| cancel | SIGTERM (Batch terminate) → `DELETE /jobs/<id>`, waits ≤ `CANCEL_WAIT_SEC` (45) for `canceled`, status `Cancelled`, exit 143 |
| lost contact | 6 failed polls in a row → `Failed SigmaIUnavailable` |

Exit codes: `0` Completed · `1` Failed · `143` Cancelled. Everything it prints is also in `output/sigmai/bridge.log`
(uploaded even on failure via `uploadCondition: taskCompletion`).

Outputs → Blob (via the task's `outputFiles`):
```
output/<runId>/result_Schedule.yaml            (only on success)
output/<runId>/sigmai/{result.json, result_summary.json, optimization.log, worker.log, error.log, bridge.log}
status/<runId>.json
```

Settings: `RUN_ID` (required, set by the task) and optional overrides — `BRIDGE_INPUT_DIR`, `BRIDGE_WORK_DIR`,
`SIGMAI_HOME`, `SIGMAI_API_URL`, `SIGMAI_API_KEY`, `POLL_INTERVAL_SEC`, `CANCEL_WAIT_SEC`, `DEADLINE_MARGIN_SEC`,
`MAX_STATUS_FAILURES` — used by the local tests (guide 04).

## 3. How to change a script safely

1. Edit in `api-controller-v2/sigmai-setup/`; run `npm test` (the bridge tests run the real file).
2. Upload under a **new version**: `CONTAINER=$CONTAINER_TEST bash upload-setup.sh 0.1.0-t2`.
3. Recreate/resize the pool with the new `blobPrefix` (or resize 0 → 1 after updating the pool JSON) and set
   `SIGMAI_SETUP_VERSION` on the controller so the history records which version ran.
4. A node keeps the scripts it booted with; a new version only applies to freshly booted nodes.

## 4. Troubleshooting (on the node side)

```bash
NODE=$(az batch node list --pool-id "$POOL_ID_SIGMAI" --query '[0].id' -o tsv)
# StartTask output
az batch node file download --pool-id "$POOL_ID_SIGMAI" --node-id "$NODE" --file-path startup/stdout.txt --destination /tmp/st.txt
# a run's task output
az batch task file download --job-id "$JOB_ID_SIGMAI" --task-id <runId> --file-path stdout.txt --destination /tmp/t.txt
```
Causes and fixes for the common failures are in guide 03 §11. The most likely first-run problems:
**no outbound internet for the wheel install** (design Q4), **Sigma-I input rejected** (`OPTIMIZATION_FAILED`, design Q11),
and **`docker compose` missing on the image** (the script tries to install it).

## 5. What is verified and what is not

| Verified here | Not verified yet (first real run will tell) |
|---|---|
| `bridge.sh` against a fake Sigma-I built from the delivered API source: success, failure, wrong key, unreachable, missing input, deadline, task retry, cancel | `start-sigmai.sh` end-to-end (needs Docker daemon + wheel) |
| `docker-compose.azure.yml` parses (`docker compose config`): ACR images, 127.0.0.1 binding, restart policy | `outputFiles` wildcard naming for `output/sigmai/*` in Batch |
| `bash -n` syntax of all scripts | IMDS/ACR login steps on a real node; wheel install time; real cold-start time |
