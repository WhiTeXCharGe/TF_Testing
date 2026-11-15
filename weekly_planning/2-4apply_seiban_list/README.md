# 2-4 apply_seiban_list

Adds the new seiban of the week to the plan. It reads the seiban list workbook and changes both the EnvConfig and the Schedule. The full design is in [Design.md](Design.md) ([設計書.md](設計書.md) in Japanese).

**In** `01_Schedule.yaml`, `01_EnvConfig.yaml` and the seiban list workbook (sheet `計画対象`). **Out** `02_Schedule.yaml`, `02_EnvConfig.yaml` and, with `report = true` / `--report`, `02_apply_report.txt`. The input files are never changed.

## Run
Fill in the `config.txt` in this folder (see below), then:
```
python apply_seiban_list.py
```
`--config <file>` uses another settings file. Any option replaces the same setting of `config.txt` for that run; `--help` lists them, `--dry-run` reads and checks everything and writes nothing.

## config.txt
`key = value` lines, `#` starts a comment, an empty value means "not set", relative paths are relative to the folder of the file. Priority: command line, then `config.txt`, then the built-in default.

| Setting | Option | Meaning |
|---|---|---|
| `schedule`, `env` | `--schedule`, `--env` | the two input files of tool 2-2-3 |
| `seiban_list` | `--seiban-list` | the seiban list workbook |
| `sheet` | `--sheet` | sheet with the rows to handle (default `計画対象`) |
| `out_dir` | `--out-dir` | folder for the output (default: the folder of the schedule file) |
| `report` | `--report` / `--no-report` | write `02_apply_report.txt` (default: no) |

The tool stops with a message when an output file would overwrite an input file (for example when the inputs are already called `02_*` and `out_dir` is the same folder).

## What it does
The colour of the seiban cell tells what to do with a row:

| Colour | Meaning | What 2-4 does |
|---|---|---|
| white | new seiban | adds a workflow task to the Schedule (see below) |
| grey | already planned | nothing (2-2-3 handles changes of these) |
| pink | to be deleted | removes the workflow task of that seiban and every assignment row of its operations |
| other | unknown | skipped, listed in the report |

For every valid white row:
1. The customer, fab and region are found or added in the EnvConfig (matched by name; the same fab name under another customer is another fab). An empty Region is taken from `搬入先テーブル`.
2. The process template (`工程選択`) is matched to a workflow with the same operations and minimum / maximum workers. If there is none, a new workflow `wf<n>` (`wf<n>p<P>`, `wf<n>p<P>o<K>`) is created and the workers get skills for its operations (the highest level of the operations with the same work name).
3. One workflow task is added with the next free id `e<n>`: a phase task per process of the template (start and end date from the row) and an operation task per work (man-hours from the template).

A white row is skipped, and nothing is created for it, when its seiban is not a module code, the template is missing or unusable, a process of the template has no start or end date (or the end is before the start), or the seiban already is a workflow task of the Schedule. A seiban on both a white and a pink row is skipped on both.

Nothing is deleted from the EnvConfig, and existing workflows are never changed (a changed template creates a new workflow).

## Output
- The terminal shows counts only (modules added and deleted, rows skipped, grey rows ignored, entries added to the EnvConfig, number of warnings).
- The report has every change (`INFO`) and every doubt (`WARN`): skipped rows with the reason and the Excel row number, new customers, fabs, regions and workflows, skills, and the template of a new workflow that has the same works as an old one but other minimum / maximum workers.
- A file that did not change is copied unchanged.

## Notes
- The workbook is only read; recolouring the rows afterwards is done by the workbook's own macro.
- The output numbers (`02_*`) may change when the flow changes.
- Not handled yet: changing or swapping the template of a seiban that is already planned (to be decided with the team), the device name column, the available-from and desired due date columns.
- Tests: `python -m unittest discover -s tests` in this folder (they use small generated workbooks only).
