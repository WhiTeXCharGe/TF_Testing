# Sigma-I on Azure — Implementation Plan (Overview)

> Visual summary. Deep version: [SigmaI_Azure_DetailedDesign20261013.md](./SigmaI_Azure_DetailedDesign20261013.md)
> 日本語版: [SigmaI_Azure_ImplementationPlan20261013(JP).md](./SigmaI_Azure_ImplementationPlan20261013(JP).md)

**Goal:** In SchedulerWeb the user picks a solver option. The run uses the same Azure stack we already have for Timefold (ACA + ACR + Blob + **the same Batch account**). Sigma-I gets **its own pool of nodes** inside that Batch account.

---

## 1. Solver options

| # | Option in SchedulerWeb | Stage 1 | Stage 2 | Where it runs |
|---|---|---|---|---|
| ① | **Timefold** | Timefold | Timefold | Batch → **Timefold pool** (existing) |
| ② | **Sigma-I: D-Wave + SCIP** | D-Wave (quantum annealing, cloud) | SCIP | Batch → **Sigma-I pool** (new) |
| ③ | **Sigma-I: SCIP + SCIP** | SCIP | SCIP | Batch → **Sigma-I pool** (new) |

> Sigma-I Stage 2 is always SCIP (fixed by the Sigma-I API). D-Wave can only be used in Stage 1.

```mermaid
flowchart LR
    U(["User picks option"]) --> O1["① Timefold"]
    U --> O2["② Sigma-I<br/>D-Wave + SCIP"]
    U --> O3["③ Sigma-I<br/>SCIP + SCIP"]

    O1 --> API["API Controller (ACA)<br/>POST /runSolver"]
    O2 --> API
    O3 --> API

    subgraph BA["Batch account (same as today)"]
        direction TB
        subgraph TP["Timefold pool (existing)"]
            TN1["node"]
            TN2["node"]
        end
        subgraph SP["Sigma-I pool (NEW)"]
            SN1["node 1"]
            SN2["node 2"]
            SN3["node … N"]
        end
    end

    API -->|"engine = timefold"| TP
    API -->|"engine = sigmai<br/>stage1 = dwave"| SP
    API -->|"engine = sigmai<br/>stage1 = scip"| SP
    SP -.->|"② only: Stage 1 calls"| DW["D-Wave Leap (cloud)"]
```

- **Same Batch account**, **different pools**: each pool has its own nodes, VM size and startup.
- The Sigma-I pool can have **more than one node**. More nodes = more runs at the same time. Adding nodes is fine.
- Options ② and ③ use the **same Sigma-I node type**. The solver choice is only a parameter of the request.

---

## 2. Big picture — blocks

```mermaid
flowchart LR
    W["SchedulerWeb"] -->|"POST /runSolver<br/>yaml x2 + option"| ACA

    subgraph ACA["ACA — API Controller"]
        R["route by engine"]
    end

    subgraph BATCH["Azure Batch account (shared)"]
        direction TB
        subgraph TFP["Timefold pool (existing)"]
            TFN["Node<br/>┌ Timefold container ┐<br/>1 task = start → solve → exit"]
        end
        subgraph SGP["Sigma-I pool (new) — node 1..N"]
            subgraph SGN["Sigma-I node"]
                direction TB
                ST["① StartTask<br/>(once per node boot)"]
                subgraph STACK["Sigma-I stack (stays running)"]
                    SA["api :8000"]
                    SW["worker"]
                    SR["redis"]
                end
                BR["② Bridge task<br/>(1 per run)"]
                ST -->|"docker compose up"| STACK
                BR -->|"localhost:8000"| SA
            end
        end
    end

    R -->|"timefold"| TFP
    R -->|"sigmai"| SGP

    BLOB[("Blob Storage<br/>input / output / status<br/>runs (history)")]
    ACR[("ACR<br/>images")]
    KV[("Key Vault<br/>D-Wave token")]

    ACA <--> BLOB
    TFN <--> BLOB
    BR <--> BLOB
    ACR -.-> TFN
    ACR -.-> ST
    KV -.-> ST
```

---

## 3. What SchedulerWeb sends (per option)

All three use the **same endpoint**: `POST /runSolver` (multipart/form-data).

### ① Timefold — same as today, plus `engine`

```text
POST /runSolver
  env              = EnvConfig.yaml        (file)
  sched            = Schedule.yaml         (file)
  label            = "Plan Sep-Nov"
  engine           = timefold               ← new (default if missing)
```

### ② Sigma-I — D-Wave (Stage 1) + SCIP (Stage 2)

```text
POST /runSolver
  env              = EnvConfig.yaml        (file)
  sched            = Schedule.yaml         (file)
  label            = "Plan Sep-Nov (D-Wave)"
  engine           = sigmai
  stage1Solver     = dwave
  stage2Solver     = scip
  stage1TimeoutSec = 600
  stage2TimeoutSec = 1800
  weightPreset     = standard              (optional)
```

### ③ Sigma-I — SCIP (Stage 1) + SCIP (Stage 2)

```text
POST /runSolver
  env              = EnvConfig.yaml        (file)
  sched            = Schedule.yaml         (file)
  label            = "Plan Sep-Nov (SCIP)"
  engine           = sigmai
  stage1Solver     = scip
  stage2Solver     = scip
  stage1TimeoutSec = 1800
  stage2TimeoutSec = 1800
  weightPreset     = standard              (optional)
```

### What happens to the fields

```mermaid
flowchart LR
    subgraph SW["SchedulerWeb sends"]
        f1["engine"]
        f2["stage1Solver / stage2Solver"]
        f3["stage1TimeoutSec / stage2TimeoutSec"]
        f4["weightPreset, disabledConstraints"]
        f5["env, sched (files)"]
    end
    subgraph API["API Controller"]
        a1["choose pool / Batch job"]
        a2["runs/{runId}.json<br/>(manifest + history)"]
        a3["input/{runId}/sigmai_options.json"]
        a4["input/{runId}/*.yaml"]
    end
    subgraph SG["Bridge → Sigma-I POST /jobs"]
        s1["stage1_solver / stage2_solver"]
        s2["stage1_timeout / stage2_timeout"]
        s3["weight_preset, disabled_constraints"]
        s4["environment, schedule (files)<br/>job_id = runId"]
    end
    f1 --> a1
    f1 --> a2
    f2 --> a2 & a3
    f3 --> a2 & a3
    f4 --> a3
    f5 --> a4
    a3 --> s1 & s2 & s3
    a4 --> s4
```

---

## 4. Why Sigma-I needs its own pool

| | Timefold pool | Sigma-I pool |
|---|---|---|
| What runs | 1 container | 3 containers: **API + worker + Redis** |
| How you call it | Command line (`entrypoint.sh`) | **HTTP API** (`POST /jobs` → poll → download) |
| Startup cost | Small (JVM start) | **Large**: installs the core wheel (`pip`) at every start |
| Best lifecycle | New container per run | **Keep running (warm)** and reuse for many runs |
| Node startup | — | **StartTask** brings the Sigma-I stack up once |
| Per run | Timefold container task | **Bridge task** (small script) |
| Secrets | none | `API_KEY` (made on the node), `DWAVE_API_TOKEN` (option ② only) |

---

## 5. Inside one Sigma-I node

```mermaid
flowchart TB
    subgraph NODE["Sigma-I node (each node has its own copy)"]
        direction TB
        subgraph START["StartTask — runs ONCE when the node boots"]
            S1["pull images from ACR"] --> S2["get wheel + compose<br/>from Blob"] --> S3["get D-Wave token<br/>from Key Vault"] --> S4["docker compose up -d<br/>(pip install wheel)"] --> S5["wait /health = OK"]
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
    BLOB[("Blob")] -->|"input yaml + options"| B
    B -->|"status / result / logs"| BLOB
```

---

## 6. The Bridge task — 3 jobs

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

## 7. End-to-end sequence (option ② or ③)

```mermaid
sequenceDiagram
    autonumber
    participant W as SchedulerWeb
    participant API as API Controller (ACA)
    participant BLOB as Blob
    participant BQ as Batch (Sigma-I job queue)
    participant BR as Bridge task (node)
    participant SG as Sigma-I stack (node)

    W->>API: POST /runSolver (engine=sigmai, stage1Solver, yaml x2)
    API->>BLOB: input/{runId}/ EnvConfig, Schedule, sigmai_options.json
    API->>BLOB: runs/{runId}.json (manifest), status = Submitted
    API->>BQ: create task {runId}
    API-->>W: 202 {runId}

    Note over BQ: waits here if all Sigma-I nodes are busy
    BQ->>BR: start task on a free warm node
    BR->>SG: POST /jobs (job_id=runId, solvers, timeouts)
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

    API->>BLOB: runs/{runId}.json += result, times, node
    W->>API: GET /download/{runId}
    API-->>W: result_Schedule.yaml
```

---

## 8. Several nodes and the queue

```mermaid
gantt
    title Sigma-I pool — 2 nodes, 1 run per node
    dateFormat HH:mm
    axisFormat %H:%M
    section Node 1
    StartTask (boot + pip install) :crit, n1s, 09:00, 8m
    run A (③ SCIP+SCIP)            :a, after n1s, 30m
    run C (② D-Wave+SCIP)          :c, after a, 20m
    section Node 2
    StartTask (boot + pip install) :crit, n2s, 09:05, 8m
    run B (③ SCIP+SCIP)            :b, after n2s, 30m
    section Batch queue
    run C waiting                  :done, cw, 09:15, 23m
```

- 1 node = 1 run at a time (`taskSlotsPerNode = 1`).
- More runs than nodes → the extra runs **wait in the Batch queue**. SchedulerWeb shows **Submitted (queue #n)**.
- Each node starts its stack **once**, so later runs on the same node skip the install.

---

## 9. Run status (what SchedulerWeb sees)

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

These are the same 5 states as Timefold, so the UI changes very little.

---

## 10. Cancel

```mermaid
sequenceDiagram
    participant W as SchedulerWeb
    participant API as API Controller
    participant BQ as Batch
    participant BR as Bridge
    participant SG as Sigma-I
    participant BLOB as Blob

    W->>API: DELETE /run/{runId}
    API->>BQ: terminate task
    alt still waiting in queue
        BQ-->>API: removed (never started)
    else running
        BQ->>BR: SIGTERM
        BR->>SG: DELETE /jobs/{runId}
        BR->>BR: status = Cancelled
    end
    API->>BLOB: runs/{runId}.json status = Cancelled
    API-->>W: 200
```

---

## 11. Blob Storage — what we store

```mermaid
flowchart TB
    C[("Blob container")]
    C --> I["input/{runId}/<br/>EnvConfig.yaml · Schedule.yaml<br/><b>sigmai_options.json</b> (new)"]
    C --> O["output/{runId}/<br/>result_Schedule.yaml<br/><b>sigmai/</b> result.json · result_summary.json · logs (new)"]
    C --> S["status/{runId}.json<br/>live state (+ engine, solver)"]
    C --> RN["<b>runs/{runId}.json</b> (new)<br/>run manifest = history record"]
    C --> SU["<b>sigmai-setup/{version}/</b> (new)<br/>wheel · compose · scripts"]
```

| Prefix | What | Written by | Kept |
|---|---|---|---|
| `input/{runId}/` | input YAML + Sigma-I options | API | 30 days |
| `output/{runId}/` | result YAML + Sigma-I summary/logs | Batch task | 30 days |
| `status/{runId}.json` | live state while running | API → task | 30 days |
| **`runs/{runId}.json`** | **who / when / which solver / settings / versions / node / durations / final result** | API (create + finalize) | **long term (history)** |
| **`sigmai-setup/{ver}/`** | Sigma-I wheel, compose, StartTask + bridge scripts | us (deploy) | per version |

`runs/{runId}.json` (short example):

```json
{
  "runId": "20261014_093000123",
  "label": "Plan Sep-Nov (D-Wave)",
  "engine": "sigmai",
  "solver": { "stage1": "dwave", "stage2": "scip", "stage1TimeoutSec": 600, "stage2TimeoutSec": 1800, "weightPreset": "standard" },
  "versions": { "sigmaiImage": "0.1.0", "sigmaiWheel": "0.1.x", "apiController": "v2" },
  "submittedAt": "2026-10-14T00:30:00Z",
  "execution": { "pool": "pool-sigmai", "node": "tvm-…-1", "queuedSec": 95, "runSec": 1260 },
  "result": { "status": "Completed", "validationPassed": true, "finalScore": 123.4 }
}
```

The history uses **Blob index tags** (`engine`, `status`, `month`), so we can filter "all D-Wave runs in October" without a database.

---

## 12. Sigma-I node lifecycle

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

Autoscale: number of nodes = number of waiting + running Sigma-I runs (max N). Each node stays warm for 60 min after its last run.

---

## 13. What changes

| Component | Change | Size |
|---|---|---|
| **SchedulerWeb** | option picker (①②③), time limits, queue display, engine in run list/history | S |
| **API Controller** | `engine` routing, option validation, `sigmai_options.json`, `runs/` manifest (all engines), queue position, optional `GET /runs` | M |
| **Blob** | new prefixes `runs/`, `sigmai-setup/`, `output/{runId}/sigmai/`; index tags; lifecycle rules | S |
| **ACR** | add the Sigma-I API image + a `redis:7-alpine` copy | XS |
| **Key Vault** | `DWAVE_API_TOKEN` (option ②) | XS |
| **Batch** | new **Sigma-I pool** + job + StartTask + bridge script (same Batch account) | **L** |
| **Timefold pool / container** | **no change** | — |

---

## 14. Roadmap

```mermaid
flowchart LR
    P0["P0<br/>Local run OK<br/>+ ask Sigma-i"] --> P1["P1<br/>ACR + Blob<br/>+ Key Vault"] --> P2["P2<br/>Sigma-I pool<br/>StartTask"] --> P3["P3<br/>manual bridge<br/>task test"] --> P4["P4<br/>API Controller<br/>routing + runs/"] --> P5["P5<br/>SchedulerWeb UI"] --> P6["P6<br/>multi-node,<br/>autoscale, D-Wave"]
```

⚠ **P0 blocker:** the local test `test-001` ended with `OPTIMIZATION_FAILED` (`validation_passed: false`). Fix this with Sigma-i first. Azure will not fix input or model problems.

Focus is **running on Azure first**. Batch cost tuning (sharing nodes with Timefold, smarter queueing) comes later.

---

## 15. Open questions (for Sigma-i / company IT)

| # | Question | Who |
|---|---|---|
| 1 | Can the wheel be **pre-installed in the image** (skip pip at every start)? | Sigma-i |
| 2 | Are we allowed to run the image + wheel on our Azure? | Sigma-i |
| 3 | Recommended VM size (CPU / RAM) for SCIP? | Sigma-i |
| 4 | Outbound internet from Batch nodes (PyPI, D-Wave)? | Company IT |
| 5 | D-Wave Leap plan / token owner / cost? | Company |
| 6 | How many Sigma-I nodes at most (cost cap)? Keep warm in business hours? | Us + Company |
| 7 | How long to keep run history (`runs/`)? | Us + Company |
