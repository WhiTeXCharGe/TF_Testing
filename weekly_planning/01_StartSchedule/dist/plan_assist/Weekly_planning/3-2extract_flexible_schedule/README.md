# 3-2 extract_flexible_schedule

Makes `FlexibleSchedule.yaml` from the solver result.

**In** `04_Schedule.yaml`, `04_EnvConfig.yaml`. **Out** `FlexibleSchedule.yaml` (and a report with `--report`).

## What it keeps
- `workflow_task_list` and the assignment rows of those modules, unchanged; misc rows are dropped.
- Each assignment row gets one more key, **`cell_name`**: the text the bar has in SU_Others. It is the row's `cell_name`, else its `description` (the solver keeps the text written by tool 2-2-3 there), else, for a new module, a name made from its fab and customer: `{customer}{fab}_{code}_新規`.
- `trip_list`: only Flexible trips that have work of those modules inside. Flexible trips without such work and Fixed trips (they come from the workbook) are dropped.
- `plan_range` unchanged.

Tool 3-3 writes `cell_name` into the cells; the next run of 2-2-3 compares it with the workbook, so a bar that was only renamed becomes a Fixed / misc module.

## Run
```
python extract_flexible_schedule.py --schedule <04_Schedule.yaml> --env <04_EnvConfig.yaml> --out-dir <folder> [--report]
```
