# plan_assist

The programs the weekly user runs, on top of the tools of `weekly_planning`. The user gives two files; everything else is automatic.
Design and reasons: [Design.md](Design.md).

| Program | Runs | Status |
|---|---|---|
| `01_StartSchedule` (`start_schedule.py`) | 2-2-3, 2-4, 2-5, 2-6 | done (runs as a Python script; the `.exe` is not built yet) |
| `02_UpdateSchedule` | 3-1, 3-2, 3-3 | part 2, not written yet |

## 01_StartSchedule

```
python start_schedule.py                       # a window opens: choose the SU_Others workbook and the seiban list, press Start
python start_schedule.py --su-others <file> --seiban-list <file> [--no-submit] [--yes] [--date YYYYMMDD] [--config <file>]
```
- Both files on the command line: no window. `--no-submit` stops after the 03 files are made (nothing is sent). `--yes` runs a week again
  that was already sent to the solver. `--date` is the run date (default: today); it names the week folder and is `--today` of 2-2-3.
- It stops at the first tool that fails and says which one and why.

### Folders
```
plan_assist/
├─ plan_assist.config.txt      settings (see the comments in the file)
├─ Weekly_planning/            the scripts of weekly_planning (next to the program; in the repository: ../weekly_planning)
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
`python -m unittest discover -s tests` inside this folder. They use small fake tools, so no Excel file and no solver are needed
(`test_ui.py` opens the window and is skipped when there is no screen).
