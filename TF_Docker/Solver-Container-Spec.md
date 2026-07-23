# Timefold Solver Container — Behaviour Spec

> **Who this is for:** the team/AI building or modifying the Timefold Docker container
> (`web/Timefold/`). This covers what the container **must do** from the webapp's
> perspective — status updates, error handling, output files, and how to test using
> Postman + the local Express service instead of real Azure.
>
> For Docker setup, build steps, and run commands → see `Docker.md` in this folder.
> For Azure infrastructure details → see `Azure-Company-Permission-Request.md`.

---

## 1. The container's job (summary)

```
START
  │
  ├─ write status.json → { status: "Running", stage: 1 }
  ├─ read EnvConfig.yaml + Schedule.yaml from /work/input/
  ├─ run Stage 1 solve
  │     ├─ update status.json → { stage: 1, progress: 0.0 … 1.0 }
  │     └─ on error → write status.json → { status: "Failed", error: { type, message } }
  │
  ├─ write status.json → { status: "Running", stage: 2 }
  ├─ run Stage 2 solve
  │     ├─ update status.json → { stage: 2, progress: 0.0 … 1.0 }
  │     └─ on error → write status.json → { status: "Failed", error: { type, message } }
  │
  ├─ write result_Schedule.yaml → /work/output/
  └─ write status.json → { status: "Completed", output: "/work/output/result_Schedule.yaml" }

CANCEL (SIGTERM received at any time)
  └─ write status.json → { status: "Cancelled" } → exit
```

---

## 2. status.json — exact schema

The webapp reads this file via `GET /status/:runId`. The schema **must** match exactly.

### Full schema

```json
{
  "runId":      "20260611_143022500",
  "status":     "Running",
  "stage":      1,
  "progress":   0.45,
  "startedAt":  "2026-06-11T14:30:22Z",
  "updatedAt":  "2026-06-11T14:35:10Z",
  "finishedAt": null,
  "error":      null,
  "output":     null
}
```

### Field definitions

| Field | Type | Description |
|---|---|---|
| `runId` | string | The run ID passed in as env var `RUN_ID` |
| `status` | string | One of: `"Submitted"` `"Running"` `"Completed"` `"Failed"` `"Cancelled"` |
| `stage` | number \| null | `1` or `2` while running. `null` before starting or after finished |
| `progress` | number | `0.0` to `1.0` within the current stage. Optional but useful |
| `startedAt` | ISO string \| null | When the container started processing |
| `updatedAt` | ISO string \| null | Last time this file was written |
| `finishedAt` | ISO string \| null | When the container finished (Completed, Failed, or Cancelled) |
| `error` | object \| string \| null | Only set on `"Failed"`. See error schema below |
| `output` | string \| null | Path to the output YAML. Only set on `"Completed"` |

### Error field schema (when status = "Failed")

```json
"error": {
  "type":    "InvalidInputData",
  "message": "worker_list is empty in EnvConfig.yaml"
}
```

| `type` value | When to use |
|---|---|
| `"InvalidInputData"` | YAML is malformed, missing required fields, wrong format |
| `"SolverError"` | Solver ran but hit an internal error (exception, OOM, timeout) |
| `"OutputError"` | Solver finished but writing the result YAML failed |
| `"UnknownError"` | Anything else |

---

## 3. Status update timeline — what to write and when

### On container start

```json
{
  "runId": "<RUN_ID>",
  "status": "Running",
  "stage": 1,
  "progress": 0,
  "startedAt": "<now ISO>",
  "updatedAt": "<now ISO>",
  "finishedAt": null,
  "error": null,
  "output": null
}
```

### During Stage 1 (periodic updates, e.g. every 30 seconds)

```json
{
  "status": "Running",
  "stage": 1,
  "progress": 0.4,
  "updatedAt": "<now ISO>"
}
```

### When Stage 1 finishes, Stage 2 begins

```json
{
  "status": "Running",
  "stage": 2,
  "progress": 0,
  "updatedAt": "<now ISO>"
}
```

### On successful completion

```json
{
  "status": "Completed",
  "stage": 2,
  "progress": 1,
  "updatedAt": "<now ISO>",
  "finishedAt": "<now ISO>",
  "error": null,
  "output": "/work/output/result_Schedule.yaml"
}
```

### On SIGTERM (cancel)

Write this before exiting:
```json
{
  "status": "Cancelled",
  "stage": <whatever stage was running>,
  "updatedAt": "<now ISO>",
  "finishedAt": "<now ISO>"
}
```

---

## 4. Error handling — what the container must catch

### Case A — Invalid input YAML

Triggered when: `EnvConfig.yaml` or `Schedule.yaml` is missing, malformed,
or missing required fields (e.g. `worker_list` is empty, `plan_range` is absent).

**Detect before starting the solve.** Do not write "Running" first — write "Failed" immediately.

```json
{
  "status": "Failed",
  "stage": null,
  "updatedAt": "<now ISO>",
  "finishedAt": "<now ISO>",
  "error": {
    "type": "InvalidInputData",
    "message": "plan_range.start_date is missing in Schedule.yaml"
  },
  "output": null
}
```

### Case B — Solver exception (OOM, internal error)

Triggered when: the Java solver throws an uncaught exception or the JVM runs
out of memory.

Write this before the process exits:
```json
{
  "status": "Failed",
  "stage": 1,
  "updatedAt": "<now ISO>",
  "finishedAt": "<now ISO>",
  "error": {
    "type": "SolverError",
    "message": "Java heap space exceeded — increase container memory limit"
  },
  "output": null
}
```

### Case C — Output write failure

Triggered when: the solver finishes but writing `result_Schedule.yaml` fails
(e.g. disk full, permission denied on the output volume).

```json
{
  "status": "Failed",
  "stage": 2,
  "updatedAt": "<now ISO>",
  "finishedAt": "<now ISO>",
  "error": {
    "type": "OutputError",
    "message": "Failed to write result_Schedule.yaml: No space left on device"
  },
  "output": null
}
```

---

## 5. Volume mount structure (local testing)

When running locally (without Azure), the container uses bind mounts:

```
web/Timefold/work/
  input/
    EnvConfig.yaml        ← container reads from /work/input/
    Schedule.yaml
  output/
    result_Schedule.yaml  ← container writes to /work/output/
  status/
    <RUN_ID>.json         ← container writes to /work/status/
```

Docker run command (matches entrypoint expectations):
```bash
docker run --rm \
  -e RUN_ID=20260611_143022 \
  -v "$(pwd)/work/input:/work/input:ro" \
  -v "$(pwd)/work/output:/work/output" \
  -v "$(pwd)/work/status:/work/status" \
  timefold-scheduler:local
```

The entrypoint script reads `RUN_ID` from environment and writes to
`/work/status/${RUN_ID}.json`.

---

## 6. Testing with Postman + local Express service (no Azure)

> **Context:** Azure is not provisioned yet. To test the full flow locally,
> the local Express service (`web/service/server.js`) acts as the API Controller.
> Postman is used to control the flow manually.

### Full local test flow

```
Postman                   Express service           Docker container
   │                      (web/service/)            (timefold-scheduler:local)
   │                      port 3001                 
   │                            │                          │
   ├─ POST /runSolver ─────────>│                          │
   │  (env + sched files)       │                          │
   │<─ 202 { runId } ───────────│                          │
   │                            │ saves files to           │
   │                            │ data/input/{runId}/      │
   │                            │                          │
   │                            │ [manually trigger]       │
   │                            │ docker run ... \         │
   │                            │   -e RUN_ID={runId} ────>│
   │                            │                          │ reads input/
   │                            │                          │ writes status Running
   │                            │                          │ solves...
   │                            │                          │ writes output/
   │                            │                          │ writes status Completed
   │                            │                          │
   ├─ GET /status/{runId} ─────>│                          │
   │<─ { status: "Completed" } ─│ reads data/status/       │
   │                            │ {runId}.json              │
   │                            │                          │
   ├─ GET /download/{runId} ───>│                          │
   │<─ result_Schedule.yaml ────│ reads data/output/       │
                                │ {runId}/                  │
```

### Step-by-step

**Step 1 — Start the Express service**
```bash
cd web/service
npm install
npm run dev    # runs on http://localhost:3001
```

**Step 2 — Upload input YAMLs via Postman**

`POST http://localhost:3001/runSolver`  
Body → form-data:
- `env` (File) → your `EnvConfig.yaml`
- `sched` (File) → your `Schedule.yaml`

Response: `{ "runId": "20260611_143022500", "status": "Submitted" }`

Files are now at: `web/service/data/input/20260611_143022500/`

**Step 3 — Run the Docker container with the uploaded files**

```bash
cd web/Timefold

docker run --rm \
  -e RUN_ID=20260611_143022500 \
  -v "C:/Users/Seiya/Desktop/work/Timefold/web/service/data/input/20260611_143022500:/work/input:ro" \
  -v "C:/Users/Seiya/Desktop/work/Timefold/web/service/data/output/20260611_143022500:/work/output" \
  -v "C:/Users/Seiya/Desktop/work/Timefold/web/service/data/status:/work/status" \
  timefold-scheduler:local
```

The container reads from `web/service/data/input/` and writes status + output
back into `web/service/data/` — the same folders the Express service uses.

**Step 4 — Poll status via Postman**

`GET http://localhost:3001/status/20260611_143022500`

Watch the status change: `Submitted` → `Running` (Stage 1) → `Running` (Stage 2) → `Completed`

**Step 5 — Download output via Postman**

`GET http://localhost:3001/download/20260611_143022500`

Click **Save Response → Save to a file** → downloads `result_Schedule.yaml`

---

## 7. Manual status override for testing (no Docker needed)

To test the webapp's error dialogs without actually running Docker:

```
PUT http://localhost:3001/status/20260611_143022500
Body (JSON):
{
  "status": "Failed",
  "stage": 1,
  "progress": 0.2,
  "error": {
    "type": "InvalidInputData",
    "message": "worker_list is empty in EnvConfig.yaml"
  }
}
```

This writes directly to `data/status/{runId}.json`. The webapp will show the
error dialog when Show Result is clicked.

---

## 8. What the entrypoint script must implement

The shell script at `docker/entrypoint.sh` must:

1. Read `RUN_ID` from environment (`$RUN_ID`)
2. Set a `SIGTERM` trap → write `status=Cancelled` → exit cleanly
3. Validate input files exist and are readable → if not, write `status=Failed` with `type=InvalidInputData`
4. Write initial `status=Running, stage=1` to `/work/status/${RUN_ID}.json`
5. Run Stage 1 solver → catch non-zero exit → write `status=Failed, type=SolverError`
6. Update `status=Running, stage=2`
7. Run Stage 2 solver → catch non-zero exit → write `status=Failed, type=SolverError`
8. Check output file was written → if missing, write `status=Failed, type=OutputError`
9. Write `status=Completed, output=/work/output/result_Schedule.yaml`

All status writes should be atomic (write to temp file, then rename) to avoid
the webapp reading a half-written JSON.

---

## 9. What is NOT the container's job

| Thing | Who does it |
|---|---|
| Upload input YAMLs to Blob Storage | API Controller (ACA) |
| Poll the container for status | Webapp + API Controller |
| Generate the SAS download URL | API Controller |
| Serve the output YAML to the browser | API Controller |
| Store run history / run list | Webapp (runs.json locally, or DB later) |

The container only reads from its input volume and writes to its output and
status volumes. It does not make HTTP calls.

---

## 10. Azure equivalent (for when permissions are granted)

| Local (now) | Azure (later) |
|---|---|
| `/work/input/` bind mount | Blob Storage: `input/{runId}/` read via Managed Identity |
| `/work/output/` bind mount | Blob Storage: `output/{runId}/` write via Managed Identity |
| `/work/status/` bind mount | Blob Storage: `status/{runId}.json` write via Managed Identity |
| `docker run` manually | Azure Batch task created by API Controller |
| Express service reads status file | API Controller reads Blob, returns to webapp |

The container code does not need to change between local and Azure — only the
volume mounts change (local folders → Azure Blob via FUSE or AzCopy in the
entrypoint).
