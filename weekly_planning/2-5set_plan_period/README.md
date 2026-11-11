# 2-5 set_plan_period

Decides the plan period of the solver input.

**In** `02_Schedule.yaml`, `02_EnvConfig.yaml`. **Out** `03_Schedule.yaml`, `03_EnvConfig.yaml` (a plain copy of the 02 file).

## What it does
- `start_date` = the later of the earliest start date of the modules and today + K days.
- `end_date` = the latest due date of the modules + L days.
- Assignments before `start_date` become Fixed (only those days; a row that crosses the date is split).
- Defaults: K = 7, L = 30.

## Run
```
python set_plan_period.py --schedule <02_Schedule.yaml> --env <02_EnvConfig.yaml> --out-dir <folder> --today YYYY/MM/DD [--k 7] [--l 30] [--report]
```

## Notes
- The new `end_date` can be later than the last date the SU_Others workbook shows; check that the EnvConfig calendars cover the period.
