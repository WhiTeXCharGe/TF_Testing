# Sigma-I on Azure — Implementation Plan (Overview)

> Visual summary. Deep version: [SigmaI_Azure_DetailedDesign20261013.md](./SigmaI_Azure_DetailedDesign20261013.md)
> 日本語版: [SigmaI_Azure_ImplementationPlan20261013(JP).md](./SigmaI_Azure_ImplementationPlan20261013(JP).md)

**Goal:** SchedulerWeb can choose **Timefold** or **Sigma-I (SCIP / D-Wave)**, using the same Azure stack we already run for Timefold (ACA + ACR + Blob + Batch).

---

## 1. Today vs. Target

### Today — Timefold only

```mermaid
flowchart LR
    W["SchedulerWeb"] -->|"POST /runSolver<br/>EnvConfig + Schedule"| API["API Controller<br/>(ACA)"]
    API -->|"input/{runId}/*.yaml<br/>status/{runId}.json"| BLOB[("Blob Storage")]
    API -->|"create task"| TFPOOL["Batch pool: Timefold"]
    TFPOOL -->|"1 task = 1 container<br/>start → solve → exit"| BLOB
    ACR[("ACR<br/>timefold image")] -.->|pull| TFPOOL
```

### Target — Timefold + Sigma-I

```mermaid
flowchart LR
    W["SchedulerWeb<br/>engine = timefold | sigmai"] -->|"POST /runSolver"| API["API Controller<br/>(ACA)"]
    API --> BLOB[("Blob Storage")]
    API -->|"engine = timefold"| TFPOOL["Batch pool: Timefold<br/>(unchanged)"]
    API -->|"engine = sigmai"| SPOOL["Batch pool: Sigma-I<br/>(NEW, warm node)"]
    TFPOOL --> BLOB
    SPOOL --> BLOB
    ACR[("ACR")] -.-> TFPOOL
    ACR -.-> SPOOL
    KV[("Key Vault<br/>DWAVE_API_TOKEN")] -.-> SPOOL
    SPOOL -.->|"Stage 1 = dwave only"| DWAVE["D-Wave Leap<br/>(cloud)"]
```

---

## 2. Why Sigma-I needs a different Batch design

| | Timefold | Sigma-I |
|---|---|---|
| What runs | 1 container | 3 containers: **API + worker + Redis** |
| How you call it | Command line (`entrypoint.sh`) | **HTTP API** (`POST /jobs` → poll → download) |
| Startup cost | Small (JVM start) | **Large** — installs the core wheel (`pip`) every start |
| Best lifecycle | New container per run | **Keep running (warm)**, reuse for many runs |
| Solvers | Timefold | Stage 1: `scip` or `dwave` / Stage 2: `scip` only |
| Secrets | none | `API_KEY` (local), `DWAVE_API_TOKEN` (only for D-Wave) |

➡ **Key idea:** start the Sigma-I stack **once per node** (Batch *StartTask*), then each run is a small **Bridge task** that talks to it.

---

## 3. Inside the Sigma-I Batch node

```mermaid
flowchart TB
    subgraph NODE["Batch node (pool-sigmai)"]
        direction TB
        subgraph START["StartTask — runs ONCE when node boots"]
            S1["pull images from ACR"] --> S2["download wheel + compose<br/>from Blob"] --> S3["get DWAVE token<br/>from Key Vault"] --> S4["docker compose up -d<br/>(pip install wheel)"] --> S5["wait /health = OK"]
        end
        subgraph STACK["Sigma-I stack (stays running)"]
            A["api :8000"] --- WK["worker (SCIP / D-Wave)"] --- R["redis"]
        end
        subgraph TASKS["Bridge task — 1 per run"]
            B["bridge script"]
        end
        START --> STACK
        B -->|"localhost:8000"| A
    end
    BLOB[("Blob")] -->|"input yaml"| B
    B -->|"status / result / logs"| BLOB
```

---

## 4. The Bridge task — 3 jobs

```mermaid
flowchart LR
    subgraph ONE["① Send"]
        a1["read EnvConfig + Schedule<br/>+ sigmai_options.json<br/>from Blob"] --> a2["POST /jobs<br/>job_id = runId"]
    end
    subgraph TWO["② Watch"]
        b1["GET /jobs/{id}/status<br/>every 10 s"] --> b2["write status/{runId}.json<br/>Running + progress"]
        b2 --> b1
    end
    subgraph THREE["③ Save"]
        c1["GET /download<br/>GET /result"] --> c2["upload result_Schedule.yaml<br/>+ summary + logs to Blob"]
    end
    ONE --> TWO --> THREE
```

---

## 5. End-to-end sequence

```mermaid
sequenceDiagram
    autonumber
    participant W as SchedulerWeb
    participant API as API Controller (ACA)
    participant BLOB as Blob
    participant BQ as Batch job queue (job-sigmai)
    participant BR as Bridge task (node)
    participant SG as Sigma-I stack (node)

    W->>API: POST /runSolver (engine=sigmai, yaml x2, options)
    API->>BLOB: input/{runId}/ EnvConfig, Schedule, sigmai_options.json
    API->>BLOB: status = Submitted
    API->>BQ: create task {runId}
    API-->>W: 202 {runId}

    Note over BQ: waits here if the node is busy (queue)
    BQ->>BR: start task on the warm node
    BR->>SG: POST /jobs (job_id=runId, stage1/2 solver, timeouts)
    SG-->>BR: 202 queued

    loop every 10 s
        BR->>SG: GET /jobs/{runId}/status
        BR->>BR: write live status.json (Running, progress)
        W->>API: GET /status/{runId}
        API-->>W: Running 45%
    end

    alt finished
        BR->>SG: GET /download + /result
        BR->>BLOB: output/{runId}/result_Schedule.yaml + sigmai/*
        BR->>BLOB: status = Completed
    else failed
        BR->>BLOB: output/{runId}/sigmai/ error.log, optimization.log
        BR->>BLOB: status = Failed (errorType)
    end

    W->>API: GET /download/{runId}
    API-->>W: result_Schedule.yaml
```

---

## 6. Queue — same node, runs wait their turn

```mermaid
gantt
    title pool-sigmai — 1 node, 1 task slot
    dateFormat HH:mm
    axisFormat %H:%M
    section Node
    StartTask (boot + pip install)  :crit, s0, 09:00, 8m
    section Runs
    run A (SCIP)                    :a, after s0, 30m
    run B (waiting in Batch queue)  :done, bq, 09:10, 28m
    run B (SCIP)                    :b, after a, 30m
    run C (D-Wave stage 1)          :c, after b, 20m
    section Node
    idle keep-alive, then scale to 0 :idle, after c, 30m
```

- Batch **job queue** = our queue. `taskSlotsPerNode = 1` → one run at a time.
- Waiting runs show as **Submitted (queue #2)** in SchedulerWeb.
- Stack is **started once** → runs B and C pay **no** install cost.

---

## 7. Run status (what SchedulerWeb sees)

```mermaid
stateDiagram-v2
    [*] --> Submitted: API created task
    Submitted --> Submitted: waiting in Batch queue (queuePosition)
    Submitted --> Running: bridge started
    Running --> Completed: Sigma-I finished
    Running --> Failed: Sigma-I failed / timeout
    Submitted --> Cancelled: user cancel
    Running --> Cancelled: user cancel
    Completed --> [*]
    Failed --> [*]
    Cancelled --> [*]
```

| Sigma-I status | → SchedulerWeb status |
|---|---|
| *(Batch task waiting)* | `Submitted` + `queuePosition` |
| `queued`, `running`, `cancel_requested` | `Running` |
| `finished` | `Completed` |
| `failed` | `Failed` (+ `errorType`) |
| `canceled` | `Cancelled` |

Same 5 states as Timefold → **minimal UI change**.

---

## 8. Cancel

```mermaid
sequenceDiagram
    participant W as SchedulerWeb
    participant API as API Controller
    participant BQ as Batch
    participant BR as Bridge
    participant SG as Sigma-I

    W->>API: DELETE /run/{runId}
    API->>BQ: terminate task
    alt still waiting in queue
        BQ-->>API: removed (never started)
    else running
        BQ->>BR: SIGTERM
        BR->>SG: DELETE /jobs/{runId}
        BR->>BR: status = Cancelled
    end
    API-->>W: 200
```

---

## 9. Node lifecycle (keep warm)

```mermaid
stateDiagram-v2
    [*] --> Zero: no runs
    Zero --> Booting: run submitted
    Booting --> Ready: StartTask OK (~5-15 min, measure)
    Ready --> Busy: bridge task
    Busy --> Ready: task done
    Ready --> Zero: idle > 60 min (autoscale)
    Booting --> Failed: StartTask failed
```

---

## 10. Changes per component

| Component | Change | Size |
|---|---|---|
| **SchedulerWeb** | engine selector + Sigma-I options + queue display | S |
| **API Controller** | `engine` field, route to `job-sigmai`, queue position, write `sigmai_options.json` | M |
| **Blob** | add `sigmai-setup/` + `output/{runId}/sigmai/` | XS |
| **ACR** | add Sigma-I API image + `redis:7-alpine` mirror | XS |
| **Key Vault** | `DWAVE_API_TOKEN` secret | XS |
| **Batch** | new `pool-sigmai` + `job-sigmai` + StartTask + bridge script | **L** |
| **Timefold path** | **no change** | — |

---

## 11. Roadmap

```mermaid
flowchart LR
    P0["P0<br/>Local run OK<br/>+ ask Sigma-i"] --> P1["P1<br/>ACR + Blob<br/>+ Key Vault"] --> P2["P2<br/>pool-sigmai<br/>StartTask"] --> P3["P3<br/>manual bridge<br/>task test"] --> P4["P4<br/>API Controller<br/>engine routing"] --> P5["P5<br/>SchedulerWeb UI"] --> P6["P6<br/>autoscale<br/>+ monitoring"]
```

⚠ **P0 blocker:** local test `test-001` ended `OPTIMIZATION_FAILED` (`validation_passed: false`). Fix with Sigma-i first — Azure won't fix input/model problems.

---

## 12. Open questions (for Sigma-i / company IT)

| # | Question | Who |
|---|---|---|
| 1 | Can the wheel be **pre-installed in the image** (skip pip at every start)? | Sigma-i |
| 2 | Are we allowed to run the image + wheel on our Azure? | Sigma-i |
| 3 | Recommended VM size (CPU / RAM) for SCIP? | Sigma-i |
| 4 | Outbound internet from Batch nodes (PyPI, D-Wave)? | Company IT |
| 5 | D-Wave Leap plan / token owner / cost? | Company |
| 6 | Keep node warm all business hours, or idle timeout only? | Us + Company |
