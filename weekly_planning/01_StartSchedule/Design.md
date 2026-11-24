# plan_assist — Design (part 1: 01_StartSchedule)

`plan_assist` is the layer the weekly user works with. It lives inside `weekly_planning`: one folder per program (`01_StartSchedule`,
later `02_UpdateSchedule`, which another person writes), and the code both need is in `common` (settings, week folders, tool runner). It runs the tools of `weekly_planning` in order, so the user gives only two files.
It contains **no planning logic**: it finds the files, writes a small settings file for each tool, calls the tool, checks the result and
keeps one folder per week. Every tool of `weekly_planning` still runs on its own, unchanged.

Two programs, split at the solver:

| Program | Tools | Status |
|---|---|---|
| `01_StartSchedule` | 2-2-3, 2-4, 2-5, 2-6 | this part |
| `02_UpdateSchedule` | 3-1, 3-2, 3-3 | part 2 |

## Folder structure (on the user's PC)

```
plan_assist/
├─ 01_StartSchedule.exe
├─ 02_UpdateSchedule.exe
├─ plan_assist.config.txt     settings of both programs
├─ Weekly_planning/           the scripts of weekly_planning (unchanged)
├─ Master/                    skill workbook and EnvConfig_start.yaml (first week only)
└─ Work/
   └─ 20261117/               one folder per week, named by the run date
      ├─ FlexibleSchedule.yaml, EnvConfig.yaml, SU_Others_rewritten.xlsm   (written by part 2; what the next week reads)
      ├─ week_status.json     where the week stands (started / ready / submitted / failed ...)
      ├─ input/               copies of the two files the user picked
      ├─ steps/               every file of the tools: 01_*, 02_*, 03_*, FlexibleWork, run_info.json, configs, reports
      └─ logs/                start_log.txt
```

## Decisions

- **Inputs:** the user picks two files in a small window (or gives them on the command line): the SU_Others workbook and the seiban list
  workbook. They are **copied** into `input/` (`01_SU_Others.xlsm`, `02_SeibanList.xlsm`, the original extension is kept); the originals are never changed.
- **Week folder:** `Work/<run date as YYYYMMDD>`, created by the program. The run date is also `--today` of 2-2-3.
- **Previous week:** the newest week folder *before* this one that has both `FlexibleSchedule.yaml` and `EnvConfig.yaml`. A week that was
  started but never finished is skipped. No state is kept outside the week folders.
- **First week** (no previous week): the previous EnvConfig is `Master/EnvConfig_start.yaml` (empty when missing, with a warning), the
  previous FlexibleSchedule is empty, and the skill workbook of `Master/` is given to 2-2-3. Later weeks never use the skill workbook.
- **Tools** are loaded from `Weekly_planning/` by file path and called as `main(argv)` in the same process. Their own `config.txt` files are
  never used: the program writes `steps/config_2-4.txt` and `steps/config_2-6.txt` and passes them with `--config`, so the folder also keeps
  exactly what was used.
- **Steps:** 2-2-3 → 2-4 → 2-5 → 2-6. It stops at the first tool that fails, says which one and why, and sets the status `failed`. 2-6 is
  skipped with `--no-submit` (or `submit = false`); the status is then `ready` and the 03 files can be checked by hand.
- **Re-running a week:** if the run was already sent to the solver (`run_info.json`), the program asks first. Old `steps/` and `logs/` are
  renamed (`steps_old_<time>`), never deleted. A week that was only prepared is simply run again (the old folders are renamed the same way).
- **Settings:** `plan_assist.config.txt` next to the program: folders, online/local mode and address, solver minutes, the year-specific sheet
  names. Relative paths are relative to that file. A missing file gives the defaults (folders next to the program).
- **Errors** are short sentences naming the step; details are in `logs/start_log.txt` and the reports in `steps/`.
- **Input from SharePoint (2026-11-20):** each of the two files is a path or a SharePoint link (`common/file_source.py`). A link is downloaded read only into `input/`; 01 never renames or locks (the planner just does not edit after the start; 02 warns if they did). One sign-in per run. A file named `_SU_Others...` is refused, because 02 uses that name as its lock. The SharePoint class itself is `common/file_operation_sharepoint.py` (not ours); we only subclass it to show the sign-in code in our window.

## Not in this part

- 02_UpdateSchedule (3-1, 3-2, 3-3), the top-level result files of the week.
- (Done afterwards) the `.exe`: `build_exe.py` builds a one-folder PyInstaller exe; the tools stay outside it, in `Weekly_planning/`.
- Locking for several people using the same `Work` folder.
