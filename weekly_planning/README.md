# weekly_planning tools

One folder per tool; they run in this order every week (step numbers are the ones of the team plan).

| Step | Tool | In | Out |
|---|---|---|---|
| 2-2 + 2-3 | [prepare_schedule](prepare_schedule/README.md) | `(N-1)/FlexibleSchedule.yaml`, `(N-1)/04_EnvConfig.yaml`, SU_Others (+ skill file), 初期データ追加情報 | `N/FlexibleWork.yaml`, `N/01_EnvConfig.yaml`, `N/01_Schedule.yaml` |
| 2-4 | GanttChartEditor (by the planner) | `01_*` | `02_Schedule.yaml`, `02_EnvConfig.yaml` |
| 2-5 | [set_plan_period](set_plan_period/README.md) | `02_*` | `03_Schedule.yaml`, `03_EnvConfig.yaml` |
| (solver) | planning management tool | `03_*` | `04_Schedule.yaml`, `04_EnvConfig.yaml` |
| 3-2 | [extract_flexible_schedule](extract_flexible_schedule/README.md) | `04_*` | `N/FlexibleSchedule.yaml` |
| 3-3 | [write_su_others](write_su_others/README.md) | `N/FlexibleSchedule.yaml`, `04_EnvConfig.yaml`, SU_Others (a copy) | `<name>_rewritten.xlsm` |

`01_StartSchedule` keeps the combined run log at `logs/start_log.txt` in the week folder and does not write separate reports for steps 2-2-3 and 2-4.

`01_StartSchedule` and `02_UpdateSchedule` are the programs for the weekly user; `python build_exe.py` (in this folder) builds both as `.exe` files into `dist/plan_assist/` (see `01_StartSchedule/README.md`).

All tools run with plain `python` (3.12; `pyyaml`, `openpyxl`, `pandas`). Each has its own tests: `python -m unittest discover -s tests` in the tool folder.
The tools import `prepare_schedule` for small shared helpers, so keep the folders side by side.
