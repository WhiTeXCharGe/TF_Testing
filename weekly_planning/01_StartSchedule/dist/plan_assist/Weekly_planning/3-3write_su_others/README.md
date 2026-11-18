# 3-3 write_su_others

Writes the flexible schedule back into a **copy** of the SU_Others workbook. The input workbook is only read; the result is a new file.

**In** `FlexibleSchedule.yaml`, `04_EnvConfig.yaml`, the workbook (a copy), optional `<date>_FlexibleWork.yaml`. **Out** `<name>_rewritten.xlsm` (and a report with `--report`, which lists every changed cell).

## What it does
- For every module with assignments (flexible last week or new this week) the cells are made to match the schedule. Cells that are no longer part of a module are emptied.
- The text of a cell is the row's `cell_name` (from 3-2). The style is copied from the nearest real cell of the module; a new module gets the most common module style of the sheet. Without a `cell_name` the text of a real cell is copied, or a name `{customer}{fab}_{code}_新規` is made.
- FI/FO of the flexible trips are written too (FI on the first day, FO on the last; trips that touch share one FO). Last week's flexible trips (`--flexible-work`) are removed first.
- Never overwritten: cells with other tasks or text, red cells, empty grey cells. They are listed as `CONFLICT`. Cells before `--from-date` (default today) are never touched.
- The workbook is edited inside the zip, so macros, buttons, drawings and comments stay byte for byte; the tool checks this after writing.

## Run
```
python write_su_others.py --schedule <FlexibleSchedule.yaml> --env <04_EnvConfig.yaml> --su-others <workbook copy> --out-dir <folder> --from-date YYYY/MM/DD [--flexible-work <FlexibleWork.yaml>] [--report]
```
Other options: `--sheet`, `--new-module-text`, `--new-module-style-from`, `--no-trips`, `--dry-run`. Reading the workbook takes about 25 seconds. In Excel the changes are far from where the file opens: use the cell list of the report.

## Compare old and new (VBA, `vba/CompareSuOthers.bas`)
A macro for reviewing the result before it replaces the real workbook.
1. Alt+F11 → File → Import File → `CompareSuOthers.bas` (remove an older copy of the module first), then run `CreateCompareButton` once; save as `.xlsm`.
2. Open the rewritten file, click the button, choose the old workbook.
3. Three sheets are added: `Compare_Old` and `Compare_New` (a red frame around every changed bar, and a first column `Changed` with a circle and a filter for the rows that changed) and `Compare_Summary` (a list of Add / Change / Delete with seiban, worker, dates before and after, and the text).

Not tested in Excel itself. If it fails, the message names the step and the half-built sheets are removed.
