# 01_StartSchedule

Part 1 of `plan_assist`, the programs the weekly user runs on top of the tools of `weekly_planning`: it runs 2-2-3, 2-4, 2-5 and 2-6 in
one go. The user gives two files; everything else is automatic. Design and reasons: [Design.md](Design.md).

| Program | Folder | Runs | Status |
|---|---|---|---|
| `01_StartSchedule` | `01_StartSchedule` (this folder) | 2-2-3, 2-4, 2-5, 2-6 | done: runs as a Python script and as an `.exe` (`../build_exe.py`) |
| `02_UpdateSchedule` | `02_UpdateSchedule` (by someone else) | 3-1, 3-2, 3-3 | not here yet |

Code shared with part 2 is in `common`: `plan_settings.py` (plan_assist.config.txt), `week_folders.py` (the week folders and their status),
`tool_runner.py` (runs a tool from its file), `file_source.py` (a path or a SharePoint link -> a local copy). Here: `start_schedule.py` (entry), `start_run.py` (the steps), `start_window.py` (the window).

## 01_StartSchedule

```
python start_schedule.py                       # a window opens: choose the SU_Others workbook and the seiban list, press Start
python start_schedule.py --su-others <file> --seiban-list <file> [--no-submit] [--yes] [--date YYYYMMDD] [--config <file>]
```
- Both files on the command line: no window. `--no-submit` stops after the 03 files are made (nothing is sent). `--yes` runs a week again
  that was already sent to the solver. `--date` is the run date (default: today); it names the week folder and is `--today` of 2-2-3.
- It stops at the first tool that fails and says which one and why.

### The two files: a path or a SharePoint link
`--su-others` / `--seiban-list` (and the two boxes of the window) take **a path on this PC or a SharePoint link**. A link is downloaded into
`Work/<week>/input/` (read only: nothing on SharePoint is renamed, locked or changed; 01 is only a reader). Both files use one sign-in:
the user's own Microsoft account, asked in a small window when the first file is fetched.
- Settings: `tenant_id` and `client_id` in `plan_assist.config.txt` (company values; no secret). The libraries `msal` and `requests`
  are needed for links only (`python -m pip install msal requests`; also before building the exe).
- A file on SharePoint named `_SU_Others...` is refused: 02_UpdateSchedule renames the workbook that way while it writes the result.
- Code: `common/file_source.py` (path or link -> local copy) on top of `common/file_operation_sharepoint.py`. Tests use a fake client; a
  real SharePoint has not been tried yet.

### Folders
On the user's PC (both exes in one folder, with the scripts next to them); in the repository the program runs from this folder and the
scripts are the folders above it.
```
plan_assist/
├─ 01_StartSchedule.exe  + _internal_start/      (in the repository: this folder, with start_schedule.py)
├─ 02_UpdateSchedule.exe + _internal_update/
├─ config/plan_assist.config.txt   settings of both programs (see the comments in the file)
├─ common/, 2-2-3..., 2-4..., 2-5..., 2-6..., 3-1..., 3-2..., 3-3..., update_schedule/    the scripts (a copy of weekly_planning)
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

## Build the exe
One command builds **both** programs (01_StartSchedule and 02_UpdateSchedule); the script is `weekly_planning/build_exe.py`, one folder up:
```
python -m pip install pyinstaller msal requests          # once
python ../build_exe.py [--out <folder>] [--console] [--only 01|02] [--allow-missing]
```
Result: `dist/plan_assist/` (about 180 MB: each exe has its own Python and libraries, in `_internal_start/` and `_internal_update/`), the layout
shown above. Copy that whole folder to the user's PC.
- The scripts are **not** inside the exes: the programs load them from the folders next to the exe, so a script can be updated by replacing its
  folder. What is inside an exe is Python, tkinter and the libraries the scripts import (the build script finds them by reading the scripts).
  A script that starts to import a new library needs a new build. `common` and `update_schedule` are left out of the exes on purpose, so both
  programs and all tools use the same code.
- The build stops when a library the scripts import (msal, requests for SharePoint) is not installed here; `--allow-missing` builds anyway.
- The exes have no console. A start-up error is shown in a message box and written to `plan_assist_error.txt` next to the exe; the log of a run
  is `Work/<week>/logs/start_log.txt`. `--console` builds versions with a console window for finding problems. `--only 01` / `--only 02` builds one.
- Everything the user sees in 01_StartSchedule (window, messages, errors) is Japanese. The lines that the tools themselves print into the
  log area of the window are English (they are for finding problems, not for the user).
- `dist/` is build output: do not commit it.

## Tests
`python -m unittest discover -s tests` inside this folder (and in `common` for the shared code). They use small fake tools, so no Excel
file and no solver are needed (`test_ui.py` opens the window and is skipped when there is no screen).
