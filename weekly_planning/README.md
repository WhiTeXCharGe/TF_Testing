# weekly_planning tools

One folder per tool; they run in this order every week (step numbers are the ones of the team plan; the folder names start with the step).

| Step | Tool | In | Out |
|---|---|---|---|
| 2-2 + 2-3 | [2-2-3prepare_schedule](2-2-3prepare_schedule/README.md) ([日本語](2-2-3prepare_schedule/README_ja.md)) | `(N-1)/FlexibleSchedule.yaml`, `(N-1)/04_EnvConfig.yaml`, SU_Others (+ skill file), 初期データ追加情報 | `N/<date>_FlexibleWork.yaml`, `N/01_EnvConfig.yaml`, `N/01_Schedule.yaml` |
| 2-4 | GanttChartEditor (by the planner) | `01_*` | `02_Schedule.yaml`, `02_EnvConfig.yaml` |
| 2-5 | [2-5set_plan_period](2-5set_plan_period/README.md) | `02_*` | `03_Schedule.yaml`, `03_EnvConfig.yaml` |
| (solver) | planning management tool | `03_*` | `04_Schedule.yaml`, `04_EnvConfig.yaml` |
| 3-2 | [3-2extract_flexible_schedule](3-2extract_flexible_schedule/README.md) | `04_*` | `N/FlexibleSchedule.yaml` |
| 3-3 | [3-3write_su_others](3-3write_su_others/README.md) | `N/FlexibleSchedule.yaml`, `N/<date>_FlexibleWork.yaml`, `04_EnvConfig.yaml`, SU_Others (a copy) | `<name>_rewritten.xlsm` |

All tools run with plain `python` (3.12; `pyyaml`, `openpyxl`, `pandas`). Each has its own tests: `python -m unittest discover -s tests` in the tool folder.
The tools import `prepare_schedule.py` from `2-2-3prepare_schedule` for small shared helpers, and `2-2-3prepare_schedule` and `3-3write_su_others`
read `decoder7.py` from `tools/decorder`; keep these folders where they are.

## One week, command by command

`N` = this week's folder `20261106`, last week's folder = `20261030`. Run each command inside the folder of its tool (PowerShell).

```powershell
# 2-2 + 2-3   (in 2-2-3prepare_schedule)
python prepare_schedule.py --prev-flexible 20261030\FlexibleSchedule.yaml --prev-env 20261030\04_EnvConfig.yaml --su-others "..\..\decorder\20260726 SU_Others.xlsm" --skill-file "..\..\decorder\20260915 SU_Others_skill level有配布禁止_.xlsm" --seiban-info "..\..\decorder\初期データ追加情報 _20260930.rev1.xlsx" --out-dir 20261106 --today 2026/11/06 --report

# 2-4  the planner adds the new modules in GanttChartEditor: 01_* -> 02_*

# 2-5   (in 2-5set_plan_period)
python set_plan_period.py --schedule 20261106\02_Schedule.yaml --env 20261106\02_EnvConfig.yaml --out-dir 20261106 --today 2026/11/06 --report

# solver: run 03_* in the planning management tool, download 04_Schedule.yaml / 04_EnvConfig.yaml

# 3-2   (in 3-2extract_flexible_schedule)
python extract_flexible_schedule.py --schedule 20261106\04_Schedule.yaml --env 20261106\04_EnvConfig.yaml --out-dir 20261106 --report

# 3-3   (in 3-3write_su_others)  -- on a COPY of SU_Others
python write_su_others.py --schedule 20261106\FlexibleSchedule.yaml --env 20261106\04_EnvConfig.yaml --su-others 20261106\copy_SU_Others.xlsm --flexible-work 20261106\20261106_FlexibleWork.yaml --out-dir 20261106 --from-date 2026/11/06 --report
```

The paths of the Excel files and the `20261106\...` folders are examples: use your own.
