# Guide 04 — Test locally (before and while Azure is being set up)

Run everything from **Git Bash** (the bridge is a bash script; Windows' `bash.exe` from WSL on the PATH can
break the tests — `which bash` should point into `Git\usr\bin`).

| Part | Needs | Proves |
|---|---|---|
| A | Node only | validation / status logic; `/health`; request checks |
| B | Node + bash + curl + python | `bridge.sh` against a **fake** Sigma-I (success, failure, wrong key, unreachable, deadline, retry, cancel) |
| C | Docker Desktop + Sigma-I image + wheel | the **real** Sigma-I stack, `start-sigmai.sh`, `bridge.sh` end-to-end on your PC |
| D | Azure resources from guide 03 | controller → Blob/Batch → node → result, with SchedulerWeb |
| E | D | Timefold regression |

---

## Part A — controller logic (no Azure)

```bash
cd api-controller-v2
npm install
npm test        # 24 tests: 17 logic + 7 bridge (part B). All green expected.
```

Start it with dummy settings and poke the validation paths (no Azure call is reached for these):
```bash
STORAGE_ACCOUNT=x BLOB_CONTAINER=c BATCH_ACCOUNT_URL=https://x BATCH_JOB_ID=j BATCH_JOB_ID_SIGMAI=js \
SOLVER_IMAGE=i ACR_LOGIN_SERVER=a POOL_MI_RESOURCE_ID=m PORT=8099 node src/server.js &

curl localhost:8099/health
# {"ok":true,"version":"2.0.0","engines":["timefold","sigmai"],"sigmaiDwave":false}

printf 'a: 1' > /tmp/e.yaml
curl -F env=@/tmp/e.yaml -F sched=@/tmp/e.yaml -F engine=sigmai -F stage1Solver=dwave localhost:8099/runSolver
# {"error":"D-Wave is not enabled on this API controller (SIGMAI_DWAVE_ENABLED=false)"}
curl -F env=@/tmp/e.yaml -F sched=@/tmp/e.yaml -F engine=timefold -F stage2Solver=scip localhost:8099/runSolver
# {"error":"Sigma-I field(s) not allowed with engine=timefold: stage2Solver"}
```
(Anything that passes validation will try to reach Azure and fail — that's expected without resources.)

## Part B — the bridge against a fake Sigma-I (no Docker, no Azure)

`npm test` already does this. To watch it by hand:

```bash
cd api-controller-v2
FAKE_MODE=ok node test/fake-sigmai-server.mjs 8765 &        # modes: ok | fail | slow

W=$PWD/local-work/b1; mkdir -p $W/input/input/r1
printf 'environment: {}\n' > $W/input/input/r1/EnvConfig.yaml
printf 'schedule: {}\n'    > $W/input/input/r1/Schedule.yaml
echo '{"engine":"sigmai","stage1Solver":"scip","stage2Solver":"scip","stage1TimeoutSec":5,"stage2TimeoutSec":5,"scenarioName":"x"}' \
  > $W/input/input/r1/sigmai_options.json

RUN_ID=r1 BRIDGE_WORK_DIR=$W SIGMAI_API_URL=http://127.0.0.1:8765 SIGMAI_API_KEY=test-key \
SIGMAI_HOME=$W/home POLL_INTERVAL_SEC=1 bash sigmai-setup/bridge.sh; echo "exit=$?"
cat $W/status/r1.json ; ls $W/output $W/output/sigmai
```
> Use paths under your project (or `C:/…`) — not `/tmp` — for `BRIDGE_WORK_DIR` if you call Windows Python/curl yourself;
> the script itself converts the paths it passes to curl.

**Cancel test** (a `kill -TERM` must stop the Sigma-I job and write `Cancelled`):
```bash
FAKE_MODE=slow node test/fake-sigmai-server.mjs 8767 &
RUN_ID=r3 BRIDGE_WORK_DIR=$W3 SIGMAI_API_URL=http://127.0.0.1:8767 SIGMAI_API_KEY=test-key SIGMAI_HOME=$W3/home \
  POLL_INTERVAL_SEC=1 CANCEL_WAIT_SEC=10 bash sigmai-setup/bridge.sh &
BP=$!; sleep 5; kill -TERM $BP; wait $BP; echo "exit=$?"     # 143 expected; status.json → "Cancelled"
```
(Batch sends the same SIGTERM on `DELETE /run/:id`.) Stop background servers when done: `kill %1 %2` or close the terminal.

## Part C — the real Sigma-I on your PC (Docker Desktop must be running)

This is the **P0 gate** from the design: do it before spending time on Azure. Needs the delivered image tar and
the core wheel `screenas_workerassign_opt-*.whl` (from Sigma-i; not in this repo).

```bash
cd api-controller-v2
SIGMA_DIR="/c/Users/PC_USER/OneDrive/Desktop/work/Timefold/web/sigmaI/2.Web APIソースコード一式/APIコンテナ（2026-07-15提供物と同一）/screenas-workerassign-api"
docker load -i "$SIGMA_DIR/screenas-workerassign-api_0.1.0.image.tar"
cp /path/to/screenas_workerassign_opt-0.1.x-py3-none-any.whl sigmai-setup/wheels/      # exactly one

# Dress rehearsal of the real StartTask on your PC (no ACR login; image is local):
export SIGMAI_HOME="$PWD/local-work/sigmai-home"            # use a C:/…-style path if Docker complains
SKIP_ACR_LOGIN=true SETUP_DIR="$PWD/sigmai-setup" \
SIGMAI_IMAGE=sigmai/screenas-workerassign-api:0.1.0 REDIS_IMAGE=redis:7-alpine \
  bash sigmai-setup/start-sigmai.sh
# → "READY (api + worker)"   (first start installs the wheel: needs internet, a few minutes)
```
Then run one real job through the bridge with your data:
```bash
IN="$PWD/local-work/real1/input"; mkdir -p "$IN"
cp /path/to/EnvConfig.yaml /path/to/Schedule.yaml "$IN/"
echo '{"engine":"sigmai","stage1Solver":"scip","stage2Solver":"scip","stage1TimeoutSec":120,"stage2TimeoutSec":120,
       "stage1NumSolutionsLimit":null,"stage2NumSolutionsLimit":null,"weightPreset":null,"disabledConstraints":[],
       "scenarioName":"local real"}' > "$IN/sigmai_options.json"

RUN_ID=real-001 BRIDGE_INPUT_DIR="$IN" BRIDGE_WORK_DIR="$PWD/local-work/real1" bash sigmai-setup/bridge.sh
cat local-work/real1/status/real-001.json
ls local-work/real1/output local-work/real1/output/sigmai
```
Check `result_Schedule.yaml` opens in the Gantt editor. **If the status is `Failed` with `OPTIMIZATION_FAILED`
(`validation_passed:false`)** that is the known open issue with `test-001` — take `output/sigmai/{error.log,optimization.log}`
to Sigma-i. Option ② locally: put the token in `DWAVE_API_TOKEN=` in `$SIGMAI_HOME/.env.server`, `docker compose … up -d`
again, and use `"stage1Solver":"dwave"`.

Stop it: `cd local-work/sigmai-home && docker compose -p sigmai --env-file .env.server down`.

> I could not run Part C myself (no Docker daemon / wheel on the build machine), so `start-sigmai.sh` and the compose
> file are validated for syntax (`docker compose config`) and logic review only — **Part C is the first real run**.
> Tell me what its output says and I'll fix anything.

## Part D — controller on your PC → real Azure → SchedulerWeb (needs guide 03 §3–§8)

Your own account needs *Storage Blob Data Contributor* and *Azure Batch Account Contributor* (Azure-Company-06 troubleshooting).

```bash
az login
source ~/azure-timefold-company-env.sh
cd api-controller-v2
STORAGE_ACCOUNT="$ST" BLOB_CONTAINER="$CONTAINER_TEST" BATCH_ACCOUNT_URL="$BATCH_URL" \
BATCH_JOB_ID="$JOB_ID" BATCH_JOB_ID_SIGMAI="$JOB_ID_SIGMAI" SOLVER_IMAGE="$SOLVER_IMAGE" \
ACR_LOGIN_SERVER="$ACR_LOGIN" POOL_MI_RESOURCE_ID="$MI_ID" SIGMAI_DWAVE_ENABLED=false \
SIGMAI_SETUP_VERSION="$SIGMAI_VER" PORT=8080 npm run dev
```
Second terminal — one Sigma-I run with curl:
```bash
curl localhost:8080/health
curl -F env=@EnvConfig.yaml -F sched=@Schedule.yaml -F engine=sigmai -F stage1Solver=scip -F stage2Solver=scip \
     -F stage1TimeoutSec=120 -F stage2TimeoutSec=120 -F label="local→azure" localhost:8080/runSolver
RUN=<runId>
curl localhost:8080/status/$RUN                    # Submitted (queuePosition?) → Running → Completed
curl localhost:8080/download/$RUN -o result.yaml
```
Look at `runs/<runId>.json` in the `$CONTAINER_TEST` container: after the first terminal status poll it contains
`execution` (node, queued/run seconds) and `result`.

Then SchedulerWeb → controller on your PC:
```
# SchedulerWeb/config.txt
mode=local
local_url=http://localhost:8080
```
`npm run dev` in `SchedulerWeb`, New Run → Submit Run → choose **Sigma-I: SCIP + SCIP** → watch status in the run list.

## Part E — Timefold regression on the same controller

Choose **Timefold** in the dialog (or `engine=timefold` / no engine in curl). Expect the same behaviour as v1:
`Submitted → Running → Completed`, `result_Schedule.yaml` downloads, plus a new `runs/<id>.json`. Compare one result with a run
from the old controller to be sure nothing changed. Also: cancel a Timefold run and a Sigma-I run (`DELETE /run/:id`).

## Checklist for your report

- [ ] A: `npm test` 24/24
- [ ] B: bridge manual run + cancel exit 143
- [ ] C: real Sigma-I run produced `result_Schedule.yaml` (or the failure log sent to Sigma-i)
- [ ] D: Sigma-I run through Azure completed; `runs/<id>.json` finalized
- [ ] E: Timefold regression same as v1; cancel works for both
