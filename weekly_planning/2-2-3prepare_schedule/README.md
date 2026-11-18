# 2-2-3 prepare_schedule

Steps 2-2 and 2-3 in one tool: find what the planner changed in SU_Others, then build the solver input.
Japanese explanation (how FI/FO is read, repaired and compared): [README_ja.md](README_ja.md).

**In** previous `FlexibleSchedule.yaml`, previous EnvConfig, the SU_Others workbook (optional separate skill-level workbook), the seiban list workbook (the file of tool 2-4; its sheet `製番リスト` gives the customer, fab and region of every module).
**Out** `<date>_FlexibleWork.yaml` (flexible modules and flexible trips), `01_EnvConfig.yaml`, `01_Schedule.yaml` (and a report with `--report`).

## What it does
- A **module** stays flexible unless the planner moved its work (worker or date), only renamed its bar, or all its work is before today. Then it becomes Fixed and turns into misc tasks.
- **FI/FO never decide a module.** A flexible trip whose FI or FO day moved becomes a Fixed trip; unchanged flexible trips stay flexible and are listed in `FlexibleWork`.
- Flexible modules and trips are copied from last week; every other bar is a misc task, every other FI/FO a Fixed trip.
- FI/FO is read tolerantly (typos, notes), marks without a partner are repaired from the worker's busy days, and one trip may end on the date the next starts. Trips without a known region get `r_other`.
- Every flexible assignment row gets a `cell_name` (the text of its bar), also written to `description` when that is empty. 3-2 / 3-3 / the next run of this tool use it.
- EnvConfig: last week's lists are kept (ids never change), new names are added, `worker_list` is rebuilt from the workbook.
- **Workflows come first.** Every usable process template (sheets `工程テンプレート<N>` of the seiban list) gets a workflow in the EnvConfig before the skills are given: an existing workflow with the same operations (process, work, name, min / max workers) is used, otherwise a new `wf<n>` is made, exactly as tool 2-4 does. So the very first run, with an EnvConfig that has no workflow, already produces workflows and skills; tool 2-4 then finds the same workflows. A template that cannot be read is reported as a warning and gets no workflow.
- **Skills follow the work name.** The skill workbook gives a level per kind of work (`M` = Mech, `E` = Elec, `QC` = QC) and a status (`Primary` = regular, anything else = spot). Every operation with that name, in every workflow of the EnvConfig (also the `wf<n>` workflows made by tool 2-4), gets the level. A name that no operation has is reported as a warning; a worker who is not in the skill workbook keeps last week's skills.

## Run
```
python prepare_schedule.py --prev-flexible <FlexibleSchedule.yaml> --prev-env <EnvConfig.yaml> --su-others <workbook> --seiban-list <workbook> --out-dir <folder> --today YYYY/MM/DD [--skill-file <workbook>] [--report]
```
Other options: `--su-sheets`, `--skill-sheet`, `--seiban-sheet` (default `製番リスト`), `--default-trip-region` (default `r_other`, `none` skips such trips), `--no-fifo-repair`, `--max-gap-days` (14), `--chain-days` (3).
Reading the workbook takes about 25 seconds.

## Notes
- Set `allow_cut_fixed_trip: false` for the solver (tool 2-6 does by default), otherwise Fixed trips without same-region work inside are removed.
- The meaning of red FI/FO cells and of `移動` cells is not decided: red cells are read like others, `移動` stays a misc task.
- Repaired FI/FO are guesses; every one is listed in the report.
