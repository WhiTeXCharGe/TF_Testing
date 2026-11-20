# 3-3 write_su_others

Writes the flexible schedule back into a **copy** of the SU_Others workbook. The input workbook is only read; the result is a new file.

**In** new `FlexibleSchedule.yaml`, its EnvConfig, the workbook (a copy), optional previous `FlexibleSchedule.yaml` and its EnvConfig, optional `FlexibleWork.yaml`. **Out** `<name>_rewritten.xlsm` (and a report with `--report`, which lists every changed cell). The full solver Schedule is not required.

## What it does
- Only exact worker/date cells belonging to the previous flexible assignments may be cleared or reused. A matching module code elsewhere in the workbook does not grant permission to change that cell. If previous `cell_name` is present, its text must also match; changed or ambiguous old cells cause an error.
- An omitted, empty-file, or empty-mapping previous flexible schedule means no old assignments or FI/FO may be removed. New flexible work can be placed in any empty, unblocked target cell.
- `FlexibleWork.yaml`, when provided, limits removal to its listed modules and trips. Without it, the previous flexible assignments/trips define the removal scope. New modules do not need to be listed.
- The text of a cell is the row's `cell_name` (from 3-2). The style is copied from the nearest real cell of the module; a new module gets the most common module style of the sheet. Without a `cell_name` the text of a real cell is copied, or a name `{customer}{fab}_{code}_新規` is made.
- FI/FO of the flexible trips are written too (FI on the first day, FO on the last). When a trip ends and another starts on the same date, including an opposite existing FI/FO boundary, the cell becomes `FO/FI`. The reader recognizes both boundaries on the next cycle. Only previous flexible trip endpoints within the removal scope may be removed or reused; removing one half of a shared boundary preserves the other.
- Never overwritten: other occupied cells (even with the same module code), red cells, empty grey cells. Occupied or invalid targets and overlapping new assignments are errors: no rewritten workbook is generated. CLI exits with code 2; UpdateSchedule restores the selected workbook without publishing. Cells before `--from-date` (default today) are never touched.
- The workbook is edited inside the zip, so macros, buttons, drawings and comments stay byte for byte; the tool checks this after writing.

## Run
```
python write_su_others.py --schedule <new FlexibleSchedule.yaml> --env <new EnvConfig.yaml> --su-others <workbook copy> --out-dir <folder> [--prev-flexible <old FlexibleSchedule.yaml>] [--prev-env <old EnvConfig.yaml>] [--flexible-work <FlexibleWork.yaml>] [--from-date YYYY/MM/DD] [--report]
```
Other options: `--sheet`, `--new-module-text`, `--new-module-style-from`, `--no-trips`, `--dry-run`. Reading the workbook takes about 25 seconds. In Excel the changes are far from where the file opens: use the cell list of the report.

## Python API
`write_su_others(schedule_path, env_path, su_others_path, output_dir, flexible_work_path, *, sheet="予定表_2026", from_date=None, write_report=False, previous_flexible_path=None, previous_env_path=None)` performs the same workbook update without CLI output. It returns a `WriteResult` with the output path, optional report path, warning count, and conflict count. Conflicts raise `WriteConflictError` before writing. `flexible_work_path` may be `None`. If `previous_env_path` is omitted, the current EnvConfig resolves old worker IDs; pass the old EnvConfig if those IDs changed. UpdateSchedule automatically uses StartSchedule's recorded previous run and its EnvConfig (or no previous inputs for the first run). Input, output, or verification failures are raised as exceptions.

## Compare old and new (VBA, `vba/CompareSuOthers.bas`)
A macro for reviewing the result before it replaces the real workbook.
1. Alt+F11 → File → Import File → `CompareSuOthers.bas` (remove an older copy of the module first), then run `CreateCompareButton` once; save as `.xlsm`.
2. Open the rewritten file, click the button, choose the old workbook.
3. Three sheets are added: `Compare_Old` and `Compare_New` (a red frame around every changed bar, and a first column `Changed` with a circle and a filter for the rows that changed) and `Compare_Summary` (a list of Add / Change / Delete with seiban, worker, dates before and after, and the text).

Not tested in Excel itself. If it fails, the message names the step and the half-built sheets are removed.