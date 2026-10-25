# Sigma-I — Implementation Guide (start here)

Design lives in [SigmaI_Azure_DetailedDesign20261013.md](./SigmaI_Azure_DetailedDesign20261013.md)
(*what* we build and *why*). This set of guides is the **how-to**, in the order you do it.

| # | Guide | What it covers |
|---|-------|----------------|
| 00 | this file | order of work, what was built, decisions, checklist |
| 01 | [SchedulerWeb changes](./SigmaI-01-SchedulerWeb-Changes.md) | solver-choice dialog, request fields, safety guard |
| 02 | [API Controller changes](./SigmaI-02-API-Controller-Changes.md) | backup, v2 behaviour, env vars, test-first deploy, going live |
| 03 | [Azure resources](./SigmaI-03-Azure-Resources.md) | step-by-step: containers, D-Wave key, ACR, pool + StartTask, test app |
| 04 | [Local testing](./SigmaI-04-Local-Testing.md) | everything you can verify on your PC before/while Azure is being set up |
| 05 | [Bridge & start-up scripts](./SigmaI-05-Bridge-And-Startup-Scripts.md) | what `start-sigmai.sh` / `bridge.sh` do, settings, troubleshooting |
| 06 | [Diagnose `S1_INVALID`](./SigmaI-06-Diagnose-S1_INVALID.md) | analysis of job `20261013_01` and the step-by-step experiment to find the input to fix |

---

## 1. The three solver options

| Dialog choice | Request sent | Runs on |
|---|---|---|
| **Timefold** | `engine=timefold` | existing Timefold pool (unchanged) |
| **Sigma-I: D-Wave + SCIP** | `engine=sigmai stage1Solver=dwave stage2Solver=scip` | new **pool-sigmai** |
| **Sigma-I: SCIP + SCIP** | `engine=sigmai stage1Solver=scip stage2Solver=scip` | new **pool-sigmai** |

Stage 2 is always SCIP (the Sigma-I API only allows that).

## 2. Order of work (what you asked for)

```
 1. SchedulerWeb          ← DONE in code (guide 01)         no Azure needed
 2. API Controller v2     ← DONE in code (guide 02)         no Azure needed to unit-test
 3. Bridge + start-up scripts ← DONE in code (guide 05)     tested against a fake Sigma-I
 4. Local tests           ← guide 04, part A–C              no Azure needed
 5. Create Azure resources (TEST) ← guide 03                needs admin for role assignments
 6. Local controller → real Azure  ← guide 04, part D       uses the resources from step 5
 7. Deploy v2 as a separate TEST app ← guide 03 §9 / guide 02
 8. Team review → promote to the real app ← guide 02 §6
```

You can do 1–4 today without any Azure resource.

## 3. What exists now

| Item | Location | State |
|---|---|---|
| Backup of the old controller | `api-controller-backup-20261020/` | exact copy of `api-controller/` **including its uncommitted `server.js` change** (also saved as `UNCOMMITTED_server.js.diff`) |
| Original controller | `api-controller/` | **untouched** — this is what users run today |
| New controller (v2) | `api-controller-v2/` | Timefold + Sigma-I routing, run history, queue position |
| Bridge script | `api-controller-v2/sigmai-setup/bridge.sh` | per-run task |
| Node start-up script | `api-controller-v2/sigmai-setup/start-sigmai.sh` | pool StartTask |
| Compose file for a node | `api-controller-v2/sigmai-setup/docker-compose.azure.yml` | |
| Upload helper | `api-controller-v2/sigmai-setup/upload-setup.sh` | |
| SchedulerWeb dialog | `SchedulerWeb/src/components/common/SolverPickerDialog.tsx` + `services/solverOptions.ts` | |
| Fake Sigma-I (for tests) | `api-controller-v2/test/fake-sigmai-server.mjs` | |

## 4. Decisions made for you (change if you disagree)

1. **v2 lives in its own folder** (`api-controller-v2/`) instead of overwriting `api-controller/`. The
   running version stays as it is; "promotion" is a deliberate step (guide 02 §6). The backup is
   extra insurance because `api-controller/` has an uncommitted change.
2. **D-Wave key in a private Blob container** (`sigmai-secrets`), not Key Vault — as you asked.
   The node reads it at start-up through its Managed Identity. Trade-offs are in guide 03 §4.
3. **Selecting Sigma-I against a backend that can't run it is refused** (clear error, no run created).
   Old controllers ignore unknown fields, so without this guard a Sigma-I choice would silently run Timefold.
4. **Test environment is separate from the real one:** its own Blob container, its own ACA app, its
   own Batch job. Only the Batch account, the pool Managed Identity and ACR are shared.
5. **D-Wave is off by default** (`SIGMAI_DWAVE_ENABLED=false`) until the D-Wave contract/token exist
   (design Q5). Option ② is then rejected by the API with a clear message.
6. `finished` + `validation_passed=false` → reported as **Completed** with a warning in the status
   (design Q10, proposal) — change later if the team wants it to be Failed.
7. Not built yet (later phases in the design): `GET /runs` history list, autoscale, Blob index tags,
   SchedulerWeb history view / queue-position display. The history file `runs/{runId}.json` **is** written.

## 5. Blockers that are not code

- **P0:** the local Sigma-I test `test-001` ended with `OPTIMIZATION_FAILED` (`validation_passed:false`).
  Azure will not fix that — resolve it with Sigma-i first (design Q11).
- Ask company IT: can Batch nodes reach PyPI / D-Wave (design Q4)? Without outbound internet the
  Sigma-I wheel install at node start fails.
- Role assignments (RBAC) need an admin — see guide 03 §2.

## 6. Master checklist (copy into your report)

**Code (done)**
- [x] Backup of the current API controller
- [x] SchedulerWeb: solver-choice dialog after "Submit Run"
- [x] SchedulerWeb: guard so Sigma-I is never silently run as Timefold
- [x] API controller v2 (routing, validation, run history, queue position, `/health` engines)
- [x] Bridge script, node start-up script, compose file, upload helper
- [x] Unit tests (17) + bridge tests against a fake Sigma-I (7)

**You do next**
- [ ] Read guide 01 → run SchedulerWeb locally and look at the dialog
- [ ] Guide 04 parts A–C (no Azure)
- [ ] Ask admin for the roles in guide 03 §2 (can be sent today)
- [ ] Guide 03: containers → D-Wave key → ACR → pool → test controller app
- [ ] Guide 04 part D: one real Sigma-I run through Azure
- [ ] Timefold regression run on the **test** app
- [ ] Team review → promote (guide 02 §6)
