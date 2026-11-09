# write_su_others  (weekly planning, step 3-3)

Folder: `tools/weekly_planning/3-3write_su_others/`. Writes the schedule of the flexible modules (and their FI/FO) back into a **copy** of SU_Others
(e.g. `20260726 SU_Others.xlsm`). The input workbook is only read; the result is a new file.

## What it does

* A module is a workflow task of the schedule whose name holds the module code (`530N03370A`). **Every module that has assignments is written**,
  whether it was flexible last week or is new this week (added in step 2-4). The sheet cells of the module are made to match the schedule
  (worker row x date column).
* **The text of a cell is the `cell_name` of the assignment row** in `FlexibleSchedule.yaml` (3-2 writes it: the text the bar had in SU_Others, or, for a
  new module, a name made from its fab and customer). The **style** (colour, border, font) is the one of the nearest real cell of the module (the same
  worker's first, otherwise any worker's). If that real cell already has the text, the cell is copied as it is; otherwise the text is written as an inline string
  with the real cell's style.
* **Only when a row has no `cell_name`** (an older FlexibleSchedule, or a Timefold result used directly) the text of a real cell of the module is copied (the same
  worker's nearest cell first, otherwise the nearest cell of any worker), so `TSMCF21_SU3200_530N03370A_新規` is never invented or changed.
* A **new module** (no cell in the sheet yet) gets
  * its `cell_name`, or, if the row has none, the name `{customer}{fab}_{code}_新規` made here (customer and fab names from `04_EnvConfig.yaml`, e.g. `TSMCF21_530N03370A_新規`; change it with `--new-module-text`), and
  * the style (fill, font, border) of the **most common module bar of the sheet**, i.e. the same colour theme as the other bars
    (`--new-module-style-from <module code>` copies the style of one module instead).
  Every new module is listed as `NEW module ...` in the report.
* A cell that is no longer part of a module is emptied (the most common empty-cell style of the sheet, style 4 in the real file).
* **FI / FO** of the flexible trips of the schedule are written too: `FI` on the first day of the trip, `FO` on the last day, with the style of the
  plain FI / FO cells of the sheet. Where one trip ends and the next starts on the **same date**, only one mark, `FO`, is written (the FI is the same
  date; that is how the sheet is read back by `2-2-3prepare_schedule`). The flexible trips of last week (`<date>_FlexibleWork.yaml`, `--flexible-work`)
  are removed first, except a mark that is still needed by another trip of the sheet. One-day trips cannot be written with one mark and are reported.
  `--no-trips` skips all FI/FO.
* Never overwritten: cells with another task or any other text or number, red cells, empty grey cells, and a cell that another
  module of the same run already took. These are listed as `CONFLICT` and left as they are.
* Cells before `--from-date` (default: today) are never touched, so the history stays.
* Everything else in the workbook is untouched on purpose: the file is edited **inside the zip** (only the sheet XML and the
  counter in `sharedStrings.xml` change), not through openpyxl, which would drop the macros, buttons, drawings and comments.
  New texts are written as inline strings, so `sharedStrings.xml` does not grow.
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

* `--flexible-work` is optional. It is only used to take the **old flexible trips** away before the new FI/FO are written, and to add `INFO` / `WARN`
  lines (a module that is in the schedule but not in FlexibleWork is new this week). It does **not** limit the modules: an empty list (first week) is fine.
* Output in `--out-dir`: `<old name>_rewritten.xlsm`, and with `--report` `<old name>_rewritten.txt` (per module: cells emptied / filled,
  every conflict and every skipped item).
* Other options: `--sheet` (default `予定表_2026`), `--new-module-text`, `--new-module-style-from`, `--no-trips`, `--dry-run` (print the plan, write nothing), `--decoder-dir`.
* Reading the 6 MB workbook takes about 25 seconds. The EnvConfig is needed because the schedule only has worker ids; names are
  matched to the sheet rows like the other tools do (`name_key`).

## Tried on real data

* 2026-11-06, the `20261105Weeklyplanning` flow (first week, empty FlexibleWork, 5 new modules `530T90001A`..`530T90005A`, 37 flexible trips), on a copy of
  `20260927 SU_Others.xlsm`, `--from-date 2026/10/30`: 237 module cells and 67 FI/FO cells written, 6 conflicts (cells that already hold another task, a red cell, an existing FI).
  A cell-by-cell comparison of the sheet (value, fill, font, border, alignment, number format) found exactly these 304 cells different; the new cells carry
  `TSMCF21_530T90001A_新規` etc. with the most common bar style (style 115, yellow), the FI/FO cells look like the real ones (no fill, left aligned).
  Reading the result back with `2-2-3prepare_schedule`: 4 of 5 modules stay flexible, 34 of 37 trips are unchanged. The one module and the 3 trips that differ are
  exactly the cells that were conflicts (they could not be written), so next week they are seen as changed.
* 2026-11-02, decoder7 output of `20260927 SU_Others` as the schedule: 1 cell emptied; with one module moved by +7 days: 15 emptied, 6 filled, 8 conflicts;
  all 6 sheets compared cell by cell, the macro project, drawings, comments and the other sheets are byte-identical.
* Not tested: opening the result in Excel itself (only openpyxl and an XML parser).

## Limits

* A module with **no assignment** in the schedule is not rewritten (nothing says where it is).
* The name of a new module is made from the fab and customer, not the full planner-style name (model name etc.); the planner can rename the cells in Excel.
* Only whole days are written; hours of the schedule are not shown in the sheet.
* An emptied cell gets the common empty style; if that cell was a coloured empty cell before the module was written, that colour is not restored.
* A worker that is not a row of the sheet, or a date outside the sheet's date columns, is skipped and reported.
* FI/FO: only the marks of flexible trips are removed / written. The `移動` cells and the Fixed trips of the sheet are never touched.

## Example with the `20261105Weeklyplanning` test folders (run from the `tools` folder, PowerShell)

First copy the new FlexibleSchedule of step 3-2 into the input folder of 3-3 (the file there may be an older one without `cell_name`):

```powershell
Copy-Item 20261105Weeklyplanning/20261030Initialplan/3-2/output/FlexibleSchedule.yaml 20261105Weeklyplanning/20261030Initialplan/3-3/input/20261030_FlexibleSchedule.yaml
```

Then run 3-3. It reads `20261105Weeklyplanning/20261030Initialplan/3-3/input` (the SU_Others there is only read) and writes `20261105Weeklyplanning/20261030Initialplan/3-3/output` (`20260927 SU_Others_rewritten.xlsm` and `.txt`):

```powershell
python -X utf8 weekly_planning/3-3write_su_others/write_su_others.py --schedule 20261105Weeklyplanning/20261030Initialplan/3-3/input/20261030_FlexibleSchedule.yaml --env 20261105Weeklyplanning/20261030Initialplan/3-3/input/20261030_04_EnvConfig.yaml --su-others "20261105Weeklyplanning/20261030Initialplan/3-3/input/20260927 SU_Others.xlsm" --flexible-work 20261105Weeklyplanning/20261030Initialplan/3-3/input/20261030_FlexibleWork.yaml --out-dir 20261105Weeklyplanning/20261030Initialplan/3-3/output --from-date 2026/10/30 --report
```

In Excel, press Ctrl+G and go to the first written cell listed under `CELLS` in the report (for example `LN38`): the workbook opens far away from the changes.

## Compare the rewritten file with the old one (VBA, `vba/CompareSuOthers.bas`)

For reviewing a 3-3 result before it replaces the real SU_Others (step 3-4). It is a macro inside the workbook, not part of the Python tool: Python cannot add a macro
to the existing macro project (`vbaProject.bin`) without damaging it, and three static copies of the 6 MB task sheet would stay in the file that overwrites the real SU_Others.

**Install (once)**
1. Open the workbook that should hold the macro: the real SU_Others (then every `…_rewritten.xlsm` made from it later has the macro already, because 3-3 keeps the macro project byte for byte) or one rewritten file.
2. `Alt+F11`. If an older `CompareSuOthers` module is there, right-click it → Remove (answer No to "export"). Then File → Import File… → `vba/CompareSuOthers.bas`.
   (The file is ASCII with CRLF line ends so it imports on any Windows code page; the Japanese sheet and header names are built with `ChrW`.)
3. Run `CreateCompareButton` once (Alt+F8). It makes a sheet `Compare` with a button "Compare with old version". Save as `.xlsm`.

**Use**: open the rewritten file, click the button, choose the OLD SU_Others (the input of 3-3). The task sheet of the open workbook is the NEW version. Three sheets are added to the open workbook
(nothing else is changed; clicking again rebuilds them). **One click is enough**: the old sheet is no longer copied across workbooks (that copy merged the ~54,000 named styles of two files, was heavy and failed the first time).
Both views are built from a copy of the NEW sheet inside the workbook, and only the changed cells are taken over from the old file. If a step fails the message names the step and the half-built `Compare_*` sheets are deleted.

| Sheet | Content |
|---|---|
| `Compare_Old` | the old task sheet; a **red frame around every whole bar** that is going to change (one frame per run of neighbouring cells with the same old and new text; a place where something will be added is framed too) |
| `Compare_New` | the new task sheet; a red frame around every added or changed bar |
| both | a new first column **Changed** with a circle on every worker row that has a change, on the same rows in both sheets, and the filter arrow on that header: filter the circle to see only the rows that changed |
| `Compare_Summary` | a **list**, one row per bar: **Category** (`Add` green / `Change` yellow / `Delete` red), **Seiban**, **Worker**, **When**, **Before**, **After**, Text before, Text after. `Add` = a new bar (When = After), `Change` = a bar that moved, got longer or shorter, or was renamed (Before and After), `Delete` = a bar that is gone (When = Before). Sorted by category, seiban, date, worker; FI/FO marks (seiban `FI/FO`) come last; filter arrows on |

Rows are matched by company + name, columns by date, so the two files need not have the same layout. Preview (the same logic run in Python on the 20261105 test files, input vs output of 3-3):
304 changed cells, 104 red frames per sheet, 104 list rows (all `Add`, it was the first week). Moved, longer, renamed, new and deleted bars were checked on synthetic cases.

Not tested in Excel (no Excel here). Notes: Excel asks to enable macros for the file; the old file is opened read-only with its macros disabled.
If it still fails, send the message text: it names the step and the error number.
