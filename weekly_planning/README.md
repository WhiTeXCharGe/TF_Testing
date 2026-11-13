# weekly_planning tools

One folder per tool, named after its step in the weekly flow. Run them in this order. The `common` folder is shared code, not a tool.

| Step | Folder | In | Out |
|---|---|---|---|
| 2-2 + 2-3 | `2-2-3prepare_schedule` ([日本語](2-2-3prepare_schedule/README_ja.md)) | previous `FlexibleSchedule.yaml` + previous EnvConfig, SU_Others workbook (+ skill workbook), seiban info workbook | `<date>_FlexibleWork.yaml`, `01_EnvConfig.yaml`, `01_Schedule.yaml` |
| 2-4 | GanttChartEditor (planner) | `01_*` | `02_*` |
| 2-5 | `2-5set_plan_period` | `02_*` | `03_*` |
| 2-6 | `2-6submit_to_timefold` | `03_*` | run record (`run_info.json`) |
| (solve) | Timefold | | |
| 3-1 | `3-1receive_from_timefold` | run record | `04_Schedule.yaml`, `04_EnvConfig.yaml` |
| 3-2 | `3-2extract_flexible_schedule` | `04_*` | `FlexibleSchedule.yaml` |
| 3-3 | `3-3write_su_others` | `FlexibleSchedule.yaml`, `04_EnvConfig.yaml`, SU_Others workbook (a copy) | rewritten SU_Others workbook |

- Python 3.12 with `pyyaml`, `openpyxl`, `pandas`. Tools 2-6 and 3-1 need only `pyyaml`.
- Tools 2-6 and 3-1 each have a `config.txt` in their own folder (local or online mode, the Azure address, input and output, run id, solver parameters), so they run with no options. A command line option replaces a setting for one run.
- Run a tool from the `tools` folder as `python -X utf8 weekly_planning/<folder>/<script>.py <options>`; `--help` lists the options.
- Every tool has its own tests: `python -m unittest discover -s tests` inside its folder.
- Code that several tools need (API client, config reader, yaml and date helpers, FI / FO, Excel readers) is in the `common` folder next to the tools ([README](common/README.md)). A tool never imports from another tool.
