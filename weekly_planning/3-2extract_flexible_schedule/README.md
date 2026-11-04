# extract_flexible_schedule  (weekly planning, step 3-2)

Makes `N/FlexibleSchedule.yaml` from the two yaml files of the solver (`N/04_Schedule.yaml`, `N/04_EnvConfig.yaml`).

## What it keeps (team plan, section 3-2)

| Part of 04_Schedule.yaml | In FlexibleSchedule.yaml |
|---|---|
| `workflow_task_list` | unchanged (these are the modules that were optimised) |
| `assignment_list` | only the rows of those modules (normal tasks), unchanged, Fixed and Flexible rows both; misc rows are left out |
| `trip_list` | only trips that are **Flexible** and have work of one of those modules of that worker between start and end date, unchanged. Dropped: Flexible trips without module work (the solver writes those too) and Fixed trips (they come from SU_Others and are rebuilt every week by `prepare_schedule`) |
| `plan_range` | unchanged |
| `misc_task_list` | written as an empty list so the file has the same shape as a Schedule.yaml |

`04_EnvConfig.yaml` is only used to check that the workers of the kept assignments exist. Next week's tools still need it as the "previous
EnvConfig" (`prepare_schedule --prev-env`, `write_su_others --env`), because the schedule only has worker ids.

## Run

```
python extract_flexible_schedule.py --schedule N/04_Schedule.yaml --env N/04_EnvConfig.yaml --out-dir N [--report]
```

**Example** (PowerShell, in this folder; `20261106` is this week's folder `N`, with the two files downloaded from the planning management tool in step 3-1):

```powershell
python extract_flexible_schedule.py --schedule 20261106\04_Schedule.yaml --env 20261106\04_EnvConfig.yaml --out-dir 20261106 --report
```

Output in `20261106\`: `FlexibleSchedule.yaml` and `FlexibleSchedule_report.txt`. Next week's `prepare_schedule` reads this file as `--prev-flexible`.

Output: `FlexibleSchedule.yaml` (and `FlexibleSchedule_report.txt` with `--report`). The console shows the counts (modules, rows, trips kept
and dropped by reason) and warnings (a worker not in the EnvConfig; a module without any assignment).

## Tried (2026-11-04)

* A real Timefold result (the 5-module sample): 5 modules, 101 rows, 95 of 95 trips kept (each one wraps module work).
* The 2-5 output of the real data (26 modules, 291 module rows, 2,271 Fixed trips from the sheet, 5,541 misc rows): 291 rows kept, the Fixed trips and misc rows dropped.
* Round trip with `prepare_schedule` on the same sheet: modules that decoder7 had left without assignments (7 modules whose cells it converted to misc tasks)
  show up as "changed" again, because the misc rows are not in FlexibleSchedule. That only happens when decoder7 output is used as a stand-in; in the real flow
  3-3 writes the flexible modules into the sheet, so sheet and FlexibleSchedule agree.
* 6 unit tests: `python -m unittest discover -s tests`.

## Open points

* Fixed trips that contain flexible work are not kept (they are rebuilt from the sheet). If the plan wants them in FlexibleSchedule, say so.
* A trip counts as "with work inside" if any day of the worker's module assignment lies between its start and end date, inclusive.
