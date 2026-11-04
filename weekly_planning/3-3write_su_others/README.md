# write_su_others  (weekly planning, step 3-3)

Folder: `tools/weekly_planning/3-3write_su_others/`. Writes the schedule of the flexible modules back into a **copy** of SU_Others
(e.g. `20260726 SU_Others.xlsm`). The input workbook is only read; the result is a new file.

## What it does

* A module is a workflow task of the schedule whose name holds the module code (`530N03370A`). For every module that has
  assignments, the sheet cells of that module are made to match the schedule (worker row x date column).
* A cell that has to show a module gets the **text and style of a real cell of the same module** (the same worker's nearest cell
  first, otherwise the nearest cell of any worker), so `TSMCF21_SU3200_530N03370A_新規` is never invented or changed, and
  colour, border and font are the ones of the real bar.
* A cell that is no longer part of the module is emptied (the most common empty-cell style of the sheet, style 4 in the real file).
* Never overwritten: cells with another task or any other text or number, red cells, empty grey cells, and a cell that another
  module of the same run already took. These are listed as `CONFLICT` and left as they are.
* Cells before `--from-date` (default: today) are never touched, so the history stays.
* Everything else in the workbook is untouched on purpose: the file is edited **inside the zip** (only the sheet XML and the
  counter in `sharedStrings.xml` change), not through openpyxl, which would drop the macros, buttons, drawings and comments.
  The tool checks this itself after writing (`verify`): all other zip members are byte-identical, and only the planned cells differ.

## Run

```
python write_su_others.py ^
  --schedule        <FlexibleSchedule or decoder7 Schedule.yaml or Timefold result Schedule.yaml> ^
  --env             <EnvConfig.yaml of that schedule> ^
  --su-others       "<copy of the dated SU_Others>.xlsm" ^
  --flexible-work   <date>_FlexibleWork.yaml ^
  --out-dir         N ^
  --from-date       YYYY/MM/DD ^
  --report
```

**Example** (PowerShell, in this folder; `20261106` is this week's folder `N`; first copy the real SU_Others, the tool never writes to its input):

```powershell
Copy-Item "<folder of the real file>\20261106 SU_Others.xlsm" 20261106\copy_SU_Others.xlsm
python write_su_others.py --schedule 20261106\FlexibleSchedule.yaml --env 20261106\04_EnvConfig.yaml --su-others 20261106\copy_SU_Others.xlsm --flexible-work 20261106\20261106_FlexibleWork.yaml --out-dir 20261106 --from-date 2026/11/06 --report
```

Output in `20261106\`: `copy_SU_Others_rewritten.xlsm` and `copy_SU_Others_rewritten.txt`. Check it in Excel, then use it to overwrite the real SU_Others (step 3-4).
`--flexible-work` is optional but recommended in the weekly flow: only the modules listed in `<date>_FlexibleWork.yaml` are rewritten.

Output in `--out-dir`: `<old name>_rewritten.xlsm`, and with `--report` `<old name>_rewritten.txt` (per module: cells emptied / filled,
every conflict and every skipped item). Options: `--flexible-work <date>_FlexibleWork.yaml` (rewrite only the modules listed there),
`--sheet` (default `予定表_2026`), `--dry-run` (print the plan, write nothing), `--decoder-dir`.
Reading the 6 MB workbook takes about 25 seconds. The EnvConfig is needed because the schedule only has worker ids; names are
matched to the sheet rows like the other tools do (`name_key`).

## Tried on real data (copy of `20260927 SU_Others.xlsm`, 2026-11-02)

* decoder7 output of the same file as the schedule, `--from-date 2026/09/27`: 1 cell emptied, 0 filled, 0 conflicts. (The decoder had
  cut that one cell of 530N03519A, so the schedule does not contain it.)
* The same schedule with module 530N03498A moved by +7 days: 15 cells emptied, 6 filled, 8 conflicts (the new days were
  taken by other tasks). A cell-by-cell comparison of all 6 sheets (1.9 million cells: value, fill, font, border, alignment, number
  format) found exactly these 21 cells different. New cells showed `NST533_530N03498A` with the fill of the real bar (FFFCA2A2),
  emptied cells were blank again. The macro project, drawings, comments and all other sheets are byte-identical.
* Not tested: opening the result in Excel itself (only openpyxl and an XML parser), and a Timefold result as input.

## Limits

* A module with **no assignment** in the schedule is not rewritten (nothing says where it is). A module with assignments but
  **no cell at all** in the old sheet cannot be written, because its name and colour are unknown; it is reported.
* Only whole days are written; hours of the schedule are not shown in the sheet.
* An emptied cell gets the common empty style; if that cell was a coloured empty cell before the module was written, that colour is not restored.
* A worker that is not a row of the sheet, or a date outside the sheet's date columns, is skipped and reported.
