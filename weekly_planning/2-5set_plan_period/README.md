# set_plan_period  (weekly planning, step 2-5)

Decides the plan period of the solver input. `N/02_Schedule.yaml` + `N/02_EnvConfig.yaml` in, `N/03_Schedule.yaml` + `N/03_EnvConfig.yaml` out.

## Rules (from the team plan, section 2-5)

| | Rule |
|---|---|
| `start_date` | the later of (a) the earliest phase `start_date` of the modules to be optimised and (b) today + **K** days |
| `end_date` | the latest phase `end_date` (due date) of the modules to be optimised + **L** days |
| assignments before `start_date` | only these become `Fixed`; a row that crosses `start_date` is split in two rows (past days Fixed, the rest unchanged) |
| `03_EnvConfig.yaml` | byte-for-byte copy of `02_EnvConfig.yaml` |

"Modules to be optimised" = the workflow tasks of the schedule. In the weekly flow the 2-3 schedule only holds the flexible modules
(everything else is a misc task), plus the modules the planner added in 2-4.

K and L are constants "to suit the operation"; the defaults are **K = 7** (the run is on the weekend and the planner reviews the result
in the week, so the next 7 days are frozen) and **L = 30** (a month of slack after the latest due date). Change them with `--k` / `--l`.

## Run

```
python set_plan_period.py --schedule N/02_Schedule.yaml --env N/02_EnvConfig.yaml --out-dir N --today YYYY/MM/DD [--k 7] [--l 30] [--report]
```

**Example** (PowerShell, in this folder; `20261106` is this week's folder `N`, with the files saved by the planner in step 2-4):

```powershell
python set_plan_period.py --schedule 20261106\02_Schedule.yaml --env 20261106\02_EnvConfig.yaml --out-dir 20261106 --today 2026/11/06 --report
```

With the defaults K = 7 and L = 30 this freezes the plan until 2026/11/13 (if the modules may start earlier) and ends it 30 days after the latest due date.
Another K / L: add `--k 10 --l 45`. Output in `20261106\`: `03_Schedule.yaml`, `03_EnvConfig.yaml`, `03_plan_period_report.txt`.

Output: `03_Schedule.yaml`, `03_EnvConfig.yaml` (and `03_plan_period_report.txt` with `--report`). The report / console lists the start and end
date with their reason, the number of worker-days made Fixed, and warnings (assignments or trips after `end_date`; end before start).

## Tried (2026-11-04)

On the 2-3 output made from `20260927 SU_Others` (26 modules), today 2026/09/27: plan_range `2026/01/01..2027/03/31` became
`2026/10/04..2027/08/03` (earliest start 2026/08/31, so today + 7 wins; latest due date 2027/07/04 + 30). Nothing was before the start date in
that file, so no row was split; the split logic is covered by unit tests (6 tests: `python -m unittest discover -s tests`).

## Open points

* The new `end_date` is later than the last date column of SU_Others (2027/03/31) and than the default due dates decoder7 writes
  (2027/06/05, 2027/03/31, ...). Check that the EnvConfig calendars (holidays, unavailable dates) are meant to cover that period.
* The tool has not been run through the Timefold solver; the solver ignores assignment days before `start_date` anyway (it only keeps days inside the plan range).
* Which phase dates count as "may start" / "due date" is an assumption: phase `start_date` / `end_date` of every phase of every module.

## Example with the `20261105Weeklyplanning` test folders (run from the `tools` folder, PowerShell)

Reads `20261105Weeklyplanning/20261030Initialplan/2-5/input`, writes `20261105Weeklyplanning/20261030Initialplan/2-5/output` (`03_Schedule.yaml`, `03_EnvConfig.yaml`, `03_plan_period_report.txt`).

```powershell
python -X utf8 weekly_planning/2-5set_plan_period/set_plan_period.py --schedule 20261105Weeklyplanning/20261030Initialplan/2-5/input/20261030_02_Schedule.yaml --env 20261105Weeklyplanning/20261030Initialplan/2-5/input/20261030_02_EnvConfig.yaml --out-dir 20261105Weeklyplanning/20261030Initialplan/2-5/output --today 2026/10/30 --report
```
