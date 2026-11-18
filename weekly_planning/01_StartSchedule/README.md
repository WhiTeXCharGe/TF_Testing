# 01_StartSchedule

Part 1 of `plan_assist`, the programs the weekly user runs on top of the tools of `weekly_planning`: it runs 2-2-3, 2-4, 2-5 and 2-6 in
one go. The user gives two files; everything else is automatic. Design and reasons: [Design.md](Design.md).

| Program | Folder | Runs | Status |
|---|---|---|---|
| `01_StartSchedule` | `01_StartSchedule` (this folder) | 2-2-3, 2-4, 2-5, 2-6 | done (runs as a Python script; the `.exe` is not built yet) |
| `02_UpdateSchedule` | `02_UpdateSchedule` (by someone else) | 3-1, 3-2, 3-3 | not here yet |

Code shared with part 2 is in `common`: `plan_settings.py` (plan_assist.config.txt), `week_folders.py` (the week folders and their status),
`tool_runner.py` (runs a tool from its file). Here: `start_schedule.py` (entry), `start_run.py` (the steps), `start_window.py` (the window).

## 01_StartSchedule

```
python start_schedule.py                       # a window opens: choose the SU_Others workbook and the seiban list, press Start
python start_schedule.py --su-others <file> --seiban-list <file> [--no-submit] [--yes] [--date YYYYMMDD] [--config <file>]
```
- Both files on the command line: no window. `--no-submit` stops after the 03 files are made (nothing is sent). `--yes` runs a week again
  that was already sent to the solver. `--date` is the run date (default: today); it names the week folder and is `--today` of 2-2-3.
- It stops at the first tool that fails and says which one and why.

### Folders
On the user's PC (the exe next to a copy of `weekly_planning`); in the repository the program runs from this folder and the tools are the
folders above it.
```
plan_assist/
├─ 01_StartSchedule.exe        (in the repository: this folder, with start_schedule.py)
├─ plan_assist.config.txt      settings (see the comments in the file)
├─ Weekly_planning/            a copy of weekly_planning (common/, 2-2-3..., 2-4..., 2-5..., 2-6...)
├─ Master/                     first week only: EnvConfig_start.yaml and the skill workbook ("*skill*")
└─ Work/<YYYYMMDD>/            one folder per week
   ├─ week_status.json         started / ready / submitted / failed, the step, the run id
   ├─ input/                   copies of the two files (01_SU_Others, 02_SeibanList); the originals are never changed
   ├─ steps/                   every file of the tools (01_*, 02_*, 03_*, FlexibleWork, reports, config_2-4.txt, config_2-6.txt, run_info.json)
   └─ logs/start_log.txt
```
- **Previous week** = the newest earlier folder of `Work` that has `FlexibleSchedule.yaml` and `EnvConfig.yaml` (part 2 writes them). None:
  first week, which uses `Master/EnvConfig_start.yaml` and the skill workbook of `Master/`. Later weeks never read the skill workbook.
- **Running a week again:** the old `steps/` and `logs/` are renamed `steps_old_<time>` / `logs_old_<time>`, never deleted. A week that was
  already sent to the solver asks first (window) or needs `--yes`.
- The tools never read their own `config.txt` here: the program writes `steps/config_2-4.txt` and `steps/config_2-6.txt` for them.

## Tests
`python -m unittest discover -s tests` inside this folder (and in `common` for the shared code). They use small fake tools, so no Excel
file and no solver are needed (`test_ui.py` opens the window and is skipped when there is no screen).
