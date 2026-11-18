# Weekly Planning with Timefold — How the Whole Flow Works

> **Date:** 2026-11-18
> **Code:** `tools/weekly_planning/` (the tools, `common/` shared code, `01_StartSchedule/` the program for the weekly user)
> **Audience:** the planning team and the developers who maintain the tools.
> No real names, customers or file names are used in this document; the tools' own READMEs give the exact options.

---

## 1. What this is, in one minute

Every week the planner keeps a workbook (**SU_Others**) that shows, per worker and per day, which seiban (module) the worker is on. Timefold can plan
part of that automatically. The weekly planning tools do the work in between:

1. read the workbook the planner just edited,
2. decide what the planner **fixed by hand** (Timefold must not touch it) and what is **still open** (Timefold may move it),
3. add the new seiban of the week,
4. send everything to Timefold,
5. take the result back and write it into a **copy** of the workbook for the planner to review.

The planner stays in control: whatever the planner moves or renames in the workbook becomes fixed the next week.
The tools never overwrite the planner's workbook.

---

## 2. The flow at a glance

```
 PART 1  01_StartSchedule  (before the solver)                         PART 2  02_UpdateSchedule  (after the solver)

 SU_Others ──┐                                                          run record (run_info.json)
 Seiban list ─┤                                                                │
 last week's  │                                                           ┌─────▼─────┐
 FlexibleSchedule + EnvConfig                                             │   3-1     │ ask Timefold, download result
              ▼                                                           └─────┬─────┘
        ┌──────────┐  01_*.yaml   ┌─────────┐  02_*.yaml  ┌─────────┐          04_Schedule / 04_EnvConfig
        │  2-2-3   ├─────────────►│   2-4   ├────────────►│   2-5   │                │
        │ prepare  │              │ new     │             │ plan    │          ┌─────▼─────┐
        └──────────┘              │ seiban  │             │ period  │          │   3-2     │ keep the flexible part
                                  └─────────┘             └────┬────┘          └─────┬─────┘
                                                      03_*.yaml│                FlexibleSchedule.yaml
                                                          ┌────▼────┐                │
                                                          │   2-6   │ send     ┌─────▼─────┐
                                                          │ submit  │          │   3-3     │ write into a COPY of SU_Others
                                                          └────┬────┘          └─────┬─────┘
                                                               │                     │
                                                         ┌─────▼──────┐        rewritten SU_Others
                                                         │  Timefold  │──────► (the planner reviews it,
                                                         │  solver    │         edits it, and it is next week's input)
                                                         └────────────┘
```

| Step | Tool (folder in `weekly_planning`) | Reads | Writes |
|---|---|---|---|
| 2-2 + 2-3 | `2-2-3prepare_schedule` | last week's FlexibleSchedule and EnvConfig, SU_Others, seiban list (+ skill workbook, first week only) | `<date>_FlexibleWork.yaml`, `01_EnvConfig.yaml`, `01_Schedule.yaml` |
| 2-4 | `2-4apply_seiban_list` | `01_*`, seiban list | `02_*` |
| 2-5 | `2-5set_plan_period` | `02_*` | `03_*` |
| 2-6 | `2-6submit_to_timefold` | `03_*` | `run_info.json`, `SolverParams.yaml` |
| (solve) | Timefold | EnvConfig, Schedule, SolverParams | the planned Schedule |
| 3-1 | `3-1receive_from_timefold` | `run_info.json` | `04_Schedule.yaml`, `04_EnvConfig.yaml` |
| 3-2 | `3-2extract_flexible_schedule` | `04_*` | `FlexibleSchedule.yaml` |
| 3-3 | `3-3write_su_others` | `FlexibleSchedule.yaml`, `04_EnvConfig.yaml`, a copy of SU_Others | `<name>_rewritten.xlsm` |

The number in a file name (`01_`, `02_`, `03_`, `04_`) says which step wrote it; each step reads the previous one. The numbers may change if the flow changes.

---

## 3. Words used in the tools

| Word | Meaning |
|---|---|
| **Module / seiban** | one piece of work with a code such as `530N00001A`. In Timefold it is a *workflow task* with phases and operations. |
| **Flexible** | Timefold may move it. A module is flexible while nobody has touched it in the workbook. |
| **Fixed** | Timefold must keep it exactly as it is: bars the planner changed, finished work, personal plans, FI/FO the planner moved. |
| **Misc task** | how a Fixed bar is stored in the Schedule: a task with Fixed assignment rows (10 hours per worker-day). Misc tasks are not calculated by Timefold; they only block the worker. |
| **FI / FO** | fly-in / fly-out marks in the worker rows. A pair makes a **trip**: the worker is away from the first to the last day. |
| **EnvConfig** | the master data of a run: regions, fabs, customers, companies, transit days, workflows, workers and their skills. Ids in it never change from week to week. |
| **FlexibleSchedule** | what the solver planned and the planner has not touched yet: the modules, their assignment rows and their trips. It is the memory between two weeks. |
| **`cell_name`** | the text a bar has in the workbook. It is stored on every assignment row so the next week can tell a renamed bar from an untouched one. |
| **Skill** | a level (1–3) per kind of work (`Mech`, `Elec`, `QC`) and a type (regular / spot). Level above 0 means "can do". |

---

## 4. The steps one by one

### 4.1 `2-2-3 prepare_schedule` — what did the planner change?

It compares last week's plan (FlexibleSchedule) with the workbook as the planner left it.

**Deciding what stays flexible**
- A module **stays flexible** if the set of (worker, day) cells in the workbook still equals last week's assignment.
- It becomes **Fixed** (a misc task) if the planner moved its work, only renamed its bar, or all its work is before today. Hours are not compared.
- A module that was not in last week's FlexibleSchedule is always a misc task here; new seiban come in at step 2-4.
- FI/FO **never** decide a module. A flexible trip whose FI or FO day moved becomes a Fixed trip; an unchanged flexible trip stays flexible.
- FI/FO are read tolerantly (typos, notes in brackets). Marks without a partner are repaired from the worker's busy days; every repaired trip is listed in the report as a guess. A Fixed trip without a known region gets `r_other` (away, place unknown).

**Building the solver input**
- Flexible modules and trips are copied from last week; every other bar becomes a misc task with Fixed rows; grey personal cells become personal misc tasks; red cells become unavailable days.
- **The worker list follows the workbook.** A new worker is added, a worker who is no longer in it is removed (the report lists both). The modules of a removed worker turn into misc tasks, so nothing points to a worker who is gone.
- **Workflows first.** Every usable process template of the seiban list gets a workflow in the EnvConfig (an existing one with the same operations is reused; otherwise a new `wf<n>`). This also makes the very first week work.
- **Skills follow the work name.** Every operation named `Mech`, `Elec` or `QC`, in every workflow, gets the worker's level for that name. The level comes from, in this order: the skill workbook (first week only; level 0 there means "no skill, on purpose"); last week's EnvConfig (kept, and operations of a new workflow get the worker's best level for that name); for a worker who is new, the 担当職種 (role) column of the workbook at level 1, as spot.
- Customer, fab and region of every module come from the seiban list; a module missing there gets region `r_other` without a warning.

### 4.2 `2-4 apply_seiban_list` — new and deleted seiban

The seiban list marks each row by colour of the seiban cell:

| Colour | Meaning | Action |
|---|---|---|
| white | new seiban | add a workflow task: one phase per process of the template, one operation per work, hours from the template |
| grey | already planned | nothing (2-2-3 looks after changes) |
| pink | to be deleted | remove the workflow task and all its assignment rows |

For a white row it finds or creates the customer, fab and region, and the workflow that matches the template (same processes, works and minimum / maximum workers). Rows with a missing date, an unknown template or a seiban that already exists are skipped and listed in the report. Existing workflows are never edited; a template that changed creates a new workflow.

### 4.3 `2-5 set_plan_period` — the planning window

- Start = the later of the earliest phase start and today + 7 days. End = the latest phase end + 30 days (at least 30 days after the start).
- Every assignment row that has work before the start date becomes Fixed (the whole row).

### 4.4 `2-6 submit_to_timefold` — send

Sends `EnvConfig.yaml`, `Schedule.yaml` and `SolverParams.yaml` to the solver and writes a run record (`run_info.json`) that step 3-1 reads.
Defaults: two-stage solving, 45 + 15 minutes at most, Fixed trips are not cut (`allow_cut_fixed_trip: false`). Two modes: **online** (the API controller on Azure, same as SchedulerWeb; it checks `/health` first, which also wakes the app) and **local** (a solver on the PC, started in Docker). On Azure a run id can be used only once; an empty run id means date and time.

### 4.5 Timefold — the solving

Timefold assigns workers to the Flexible operations within the plan period, respecting skills, minimum / maximum workers, unavailable days, trips and the Fixed rows. It does not change the EnvConfig.

### 4.6 `3-1 receive_from_timefold` — result

Asks for the state of the run (waiting, running with stage and progress, finished, failed, cancelled) and downloads the result when it has finished. Exit codes: 0 done, 1 not finished, 2 failed, 3 cancelled, 4 server not reachable, 5 bad arguments or settings. `04_EnvConfig.yaml` is a copy of the EnvConfig that was sent.

### 4.7 `3-2 extract_flexible_schedule` — what to remember

Keeps the part that is still flexible: the modules, their assignment rows and the Flexible trips that have module work inside. Misc rows and Fixed trips are dropped (they come from the workbook again next week). Every row gets its `cell_name`; new modules get colours that other modules used longest ago. The result is `FlexibleSchedule.yaml`.

### 4.8 `3-3 write_su_others` — back into the workbook

Writes the plan into a **copy** of the workbook: bars of modules are made to match the schedule, FI/FO of flexible trips are written, cells that are no longer part of a module are emptied. It never overwrites other tasks, red cells or empty grey cells (they are listed as conflicts). Cells before today are untouched. The file is edited inside the zip, so macros and buttons stay byte for byte, and the tool verifies this after writing. A VBA macro (`vba/CompareSuOthers.bas`) shows old and new side by side so the planner can review before replacing the real workbook.

**The loop closes here:** the planner edits the rewritten workbook; next week step 2-2-3 sees which bars were moved or renamed and turns exactly those modules into Fixed.

---

## 5. The programs for the weekly user

Two programs, split at the solver, so the user gives only two files:

| Program | Runs | Status |
|---|---|---|
| `01_StartSchedule` | 2-2-3 → 2-4 → 2-5 → 2-6 | written; runs as a Python script and as an `.exe` (`01_StartSchedule/build_exe.py`) |
| `02_UpdateSchedule` | 3-1 → 3-2 → 3-3 | to be written by someone else, in `02_UpdateSchedule/` next to `01_StartSchedule/`, using the same `common` code |

**What the user does:** open `01_StartSchedule`, pick the SU_Others workbook and the seiban list in a small window, press Start. Later, when the solver is done, run `02_UpdateSchedule`.

### Folder structure on the user's PC

```
plan_assist/
├─ 01_StartSchedule.exe
├─ 02_UpdateSchedule.exe
├─ plan_assist.config.txt        folders, online/local mode and address, solver minutes, year-specific sheet names
├─ Weekly_planning/              a copy of the tools (so they can be updated without rebuilding the exe)
├─ Master/                       first week only: skill workbook and EnvConfig_start.yaml
└─ Work/
   └─ 20261117/                  one folder per week, named by the run date
      ├─ FlexibleSchedule.yaml   ┐ written by part 2: what the next week reads
      ├─ EnvConfig.yaml          ┘
      ├─ SU_Others_rewritten.xlsm   the result for the planner (part 2)
      ├─ week_status.json        started / ready / submitted / failed, the step, the run id
      ├─ input/                  copies of the two files the user picked (the originals are never changed)
      ├─ steps/                  every file of the tools: 01_*, 02_*, 03_*, reports, the settings written for each tool, run_info.json
      └─ logs/                   start_log.txt
```

### How the programs behave
- **Previous week** = the newest earlier week folder that has both `FlexibleSchedule.yaml` and `EnvConfig.yaml`. No hidden state. If none exists it is the first week: the EnvConfig starts from `Master/EnvConfig_start.yaml` and the skill workbook of `Master/` is used. Later weeks never read the skill workbook.
- **Settings:** the programs write a small settings file for each tool into `steps/` and pass it with `--config`; the tools' own `config.txt` files are not used, and the folder keeps exactly what was used.
- **Stops at the first failure** and says which tool and why; the week is marked `failed` and can simply be run again.
- **Running a week again:** the old `steps/` and `logs/` are renamed (`steps_old_<time>`), never deleted. A week already sent to the solver asks first.
- **No-send mode:** `--no-submit` (or `submit = false`) stops after the 03 files, for checking by hand.
- **Command line** (no window): `python start_schedule.py --su-others <file> --seiban-list <file> [--no-submit] [--yes] [--date YYYYMMDD]`.

### Testing without the solver
2-6 and 3-1 only need a server that speaks the solver API, not Timefold itself. `common/testing/stub_solver.py` is such a server (no solving), and a `run_info.json` can be prepared by hand for part 2. How, and what to write in the `config.txt` of 2-6 and 3-1: `WeeklyPlanning_TestWithoutTimefold20261119.md`.

### The tools run on their own too
Every tool is still a command line program with its own README, options and tests (`python -m unittest discover -s tests` in its folder). Tools never import from each other; shared code is in `common`.

---

## 6. First week and normal weeks

| | First week | Later weeks |
|---|---|---|
| Previous EnvConfig | `Master/EnvConfig_start.yaml` (empty if missing, with a warning) | `EnvConfig.yaml` of the previous week |
| Previous FlexibleSchedule | none: every bar is a misc task | `FlexibleSchedule.yaml` of the previous week |
| Skill workbook | read, from `Master/` | not needed: skills are carried over; new workers get skills from their role |
| New workflows | made from the templates of the seiban list | made only when a template is new |

---

## 7. Things to know and open points

- **The exe** (`01_StartSchedule.exe`, built with PyInstaller, one folder of about 80 MB) was tested on real input files without sending to the solver: same result as the Python program. It holds Python and the libraries; the tools are loaded from the `Weekly_planning` folder next to it, so a tool can be updated without a new build (a new library import needs one). It has no console: errors appear in a message box and in `plan_assist_error.txt`. `02_UpdateSchedule` has no exe yet.
- **`02_UpdateSchedule`** is not written; part 2 must write `FlexibleSchedule.yaml` and `EnvConfig.yaml` into the week folder, because that is how the next week finds its previous week.
- **2-5 fixes a whole row** if any of its work is before the plan start, including the days after it. The earlier design split the row at the start date. Check with the author whether that is intended.
- **3-2 drops a flexible trip** that has only Fixed module work inside; 3-3 then does not write its FI/FO.
- **A changed template** creates a new workflow and leaves the old one (to be decided with the team). Workflow matching includes minimum / maximum workers; confirm this is wanted.
- **Two tools could be merged** (2-3 and 2-4) if pink rows must be deleted before their modules turn into misc tasks; to be discussed with the team.
- **Role-only skills:** a new worker with no skill level gets level 1 as spot from the role column. Planners can adjust it in the EnvConfig later; the level is a setting (`default_skill_level`).
- **Sheet names carry the year** (for example the task sheet) and are settings, not code.
- **Shared `Work` folder:** if several people use the same `Work` folder, a lock is needed (not built).
- **Online mode** was developed against a test server; confirm it with the first real run on Azure.
- **Repaired FI/FO are guesses** and are listed in the report; `--no-fifo-repair` switches the repair off.

---

## 8. Where to look when something is wrong

| Symptom | Look at |
|---|---|
| The program stopped | the last lines of `logs/start_log.txt`: the line `STOPPED at <step>` names the tool and the reason |
| A module became Fixed that should be flexible | `steps/<date>_prepare_report.txt`, the `FIXED module …` lines say why (moved cells, renamed bar, finished) |
| A seiban was not added | `steps/02_apply_report.txt`: skipped rows are listed with the reason and the Excel row number |
| A worker has no skills | `steps/<date>_prepare_report.txt`: the lines about workers without a skill level, role or skill in last week's EnvConfig |
| A trip looks wrong | the `INFERRED` lines of the 2-2-3 report (repaired FI/FO) |
| The solver result is odd | `steps/SolverParams.yaml` and `steps/03_*` (what was sent); the state in 3-1 |
| Cells were not written back | the `CONFLICT` lines of the 3-3 report |
