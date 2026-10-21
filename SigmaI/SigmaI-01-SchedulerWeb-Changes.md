# Guide 01 — SchedulerWeb changes

**Goal:** after the user clicks **Submit Run**, a second dialog asks which solver to use:
Timefold, Sigma-I (D-Wave + SCIP), or Sigma-I (SCIP + SCIP).

Needs no Azure. Time to try it: ~5 minutes.

## 1. What the user sees

```
New Run modal  ──[Submit Run]──►  "Choose a solver" dialog  ──[Start run]──►  upload + submit
                                   ○ Timefold                    (default)
                                   ○ Sigma-I: D-Wave + SCIP      Stage 1 [600] s   Stage 2 [1800] s
                                   ○ Sigma-I: SCIP + SCIP        Stage 1 [1800] s  Stage 2 [1800] s
                                   [Back]                [Start run]
```

- **Timefold** behaves exactly as before.
- **Sigma-I** options show the Stage 1 / Stage 2 time limits (whole seconds, 1–7200). Defaults
  are the design-doc proposal; final values are an open question (design Q7).
- **Back** returns to the file form. Any error (backend unreachable, Sigma-I not supported…) is
  shown in the New Run modal and **no run row is created**.

## 2. Files changed / added

| File | Change |
|---|---|
| `src/services/solverOptions.ts` | **new** — the 3 options, defaults, and `toRequestFields()` (what is sent) |
| `src/components/common/SolverPickerDialog.tsx` | **new** — the dialog |
| `src/services/runService.ts` | `submitRun(..., selection)` adds the fields; `ensureEngineSupported()` checks `GET /health`; `RunStatus` gets optional `engine`, `queuePosition` |
| `src/hooks/useRuns.ts` | `submitToSolver(..., selection)` passes it through |
| `src/pages/RunLogPage.tsx` | "Submit Run" opens the dialog; `handleSubmit(payload, selection)` |
| `src/config/uiConfig.ja.ts` / `uiConfig.en.ts` | new `solverPicker` strings + `errors.engineNotSupported` (both languages, same shape) |
| `.claude/launch.json` | added `schedulerweb-dev` (port 5174) for the preview tool |

## 3. What is sent to the API

`POST /runSolver` (multipart) — same as today plus:

| Choice | Extra fields |
|---|---|
| Timefold | `engine=timefold` |
| Sigma-I D-Wave+SCIP | `engine=sigmai`, `stage1Solver=dwave`, `stage2Solver=scip`, `stage1TimeoutSec`, `stage2TimeoutSec` |
| Sigma-I SCIP+SCIP | `engine=sigmai`, `stage1Solver=scip`, `stage2Solver=scip`, `stage1TimeoutSec`, `stage2TimeoutSec` |

## 4. The safety guard (important)

The **current** API controller ignores unknown fields. If SchedulerWeb just sent `engine=sigmai` to it,
the run would be accepted and **silently executed by Timefold**. To prevent that, before a Sigma-I
submit SchedulerWeb calls `GET {solver URL}/health`:

- v2 controller answers `{"ok":true,"engines":["timefold","sigmai"],...}` → allowed.
- v1 controller answers `{"ok":true}` (no `engines`) → treated as Timefold-only → Sigma-I refused with:
  *"This solver backend does not support "sigmai" yet…"*
- no solver URL configured, or backend unreachable → refused / network error.

The check happens **before** the local row/files are created.

## 5. Try it

```bash
cd SchedulerWeb
npm install          # first time only
npm run dev          # http://localhost:5174
```

1. Open the app → **New Run** → attach any two YAML files → **Submit Run**.
2. The solver dialog appears. Click through the three options; the time-limit fields appear for Sigma-I.
3. With no Sigma-I-capable backend running, **Start run** on a Sigma-I option shows an error in the modal
   (this is the guard working).
4. Type-check: `npx tsc --noEmit -p .` (clean).

To see the full flow, run API controller v2 (guide 04) and point SchedulerWeb at it in `config.txt`:

```
mode=local
local_url=http://localhost:8080
```
Restart the app (or reload the page) after editing `config.txt`.

## 6. Not done yet (later)

- Show the chosen solver as a badge in the run list, and "Waiting (#n)" from `queuePosition`.
- A history view from the future `GET /runs`.
- Remember the last chosen solver as a setting.
- The local mock backend (`localhost:3001`) has no Sigma-I support — Sigma-I choices are refused there by design.
