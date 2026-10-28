# prepare_schedule  (weekly planning, steps 2-2 + 2-3 in one tool)

Folder: `tools/weekly_planning/prepare_schedule/`. The other tools of the same weekly phase (2-5 window setter, 3-2 extractor,
3-3 Excel writer) can go next to it in `tools/weekly_planning/`.

## What it does

* **2-2** looks at last week's flexible plan and at `SU_Others.xlsm` and decides what the planner has changed:
  * a module whose worker-days in the sheet differ from last week's assignments becomes **Fixed** (it turns into misc tasks);
  * a module whose work all ended before `--today` becomes **Fixed**;
  * a flexible trip whose FI and FO are no longer on the same days in the sheet is no longer flexible: the sheet's FI/FO become a **Fixed trip**;
  * every other module and trip stays **flexible**.
* **2-3** builds the Timefold input:
  * flexible modules and their assignments are copied from last week unchanged (only the worker ids are re-mapped);
  * every other bar of the sheet becomes a misc task with Fixed rows (region from the 製番 sheet when the bar carries a module code, else `r_other`);
  * every FI/FO pair of the sheet that is not an unchanged flexible trip becomes a Fixed `trip_list` entry;
  * `EnvConfig`: workflows, fabs, regions, customers and companies of last week are kept (ids never change), new names from
    `初期データ追加情報` are added; `worker_list` is rebuilt from the sheet (ids of known workers are kept by name, skills and
    regular/spot are computed like decoder7).

## Run

There are two kinds of SU_Others workbook, and the tool takes both:

| Option | File | Used for |
|---|---|---|
| `--su-others` | the dated, planner-maintained file, e.g. `20260726 SU_Others.xlsm` | ongoing tasks (bars, FI/FO, personal business) and the worker list |
| `--skill-file` | `SU_Others_skill level有配布禁止_.xlsm` | skills, skill level and Primary/Secondary status only |

```
python prepare_schedule.py \
  --prev-flexible  <last week>/FlexibleSchedule.yaml \
  --prev-env       <last week>/EnvConfig.yaml \
  --su-others      "20260726 SU_Others.xlsm" \
  --skill-file     "20260915 SU_Others_skill level有配布禁止_.xlsm" \
  --seiban-info    "初期データ追加情報 _20260930.rev1.xlsx" \
  --out-dir        N \
  --today          2026/10/01
```

* Workers are matched to the skill file by ID, else by name. A worker that is **not** in the skill file keeps the skills of last week's
  EnvConfig (listed in the report); a brand-new worker not in the skill file gets no skills.
* `--skill-file` is optional: without it the skill sheet is looked for inside `--su-others`, like decoder7 does.
* Sheet names: `--su-sheets` is the task sheet in the dated file (default `予定表_2026`), `--skill-sheet` is the skill sheet in the skill file
  (default `予定表_2026 (2)`). If the name is wrong the tool stops and lists the sheets that exist.

Output in `--out-dir`: three yaml files, `<today>_FlexibleWork.yaml` (e.g. `20261028_FlexibleWork.yaml`, date = `--today`), `01_EnvConfig.yaml`
and `01_Schedule.yaml`. With `--report` there is a fourth file, `<today>_prepare_report.txt` (every decision and every
problem it could not resolve).

Options: `--su-sheets` (default `予定表_2026`), `--skill-sheet` (default `予定表_2026 (2)`), `--seiban-start-row` (default 3 = whole list),
`--default-trip-region <id>` (region for Fixed trips whose region cannot be found; default: they are skipped and listed),
`--decoder-dir` (folder of `decoder7.py`, default `../../../decorder`). Needs `pyyaml`, `openpyxl`, `pandas` (the same as decoder7).
Reading the 7 MB workbook takes about 20 seconds.

`--prev-flexible` can be a real `FlexibleSchedule.yaml` or simply a Timefold result `Schedule.yaml`: everything with a module is
treated as flexible, misc tasks in it are ignored. So until 3-2 exists, the Timefold output of last week works as it is.

## Rules in detail

| Topic | Rule |
|---|---|
| Module identity | the workflow task `name` (the module code such as `530N03359A`) is matched with the code in the sheet's bar text |
| "Changed" module | the set of (worker, date) in the sheet differs from last week's rows (hours are not compared). A worker that is renamed or gone counts as a change |
| Complete module | last assignment date `< --today` |
| FI/FO pairing | per worker, every `FI` is paired with the next `FO`. Orphans (FI without FO, FO without FI, FI FI, FO FO) are listed in the report and not guessed |
| Region of a Fixed trip | 1. the same trip of last week; 2. module codes between FI and FO (through 製番); 3. a fab name (3+ characters, one region) in a bar label between FI and FO; otherwise skipped and listed |
| Misc region | module code known in 製番: its region; everything else `r_other` (Timefold treats it as "no region") |
| Worker ids | known workers keep their id (by name); new workers get the next free `w###` |
| Skills | only operations that exist in the workflow of last week's EnvConfig are kept (others are dropped and reported) |

## Tests

```
python -m unittest discover -s tests -v
```

22 tests on plain data (decisions, FI/FO pairing, region rules, id stability, master lists, misc tasks, YAML round trip). No Excel needed.

## What was tried on real data (2026-10-28)

(The skill-level file was used as both `--su-others` and `--skill-file`, because no dated SU_Others was at hand; the dated file has not been tried.)
`decorder/decoder7_out` as "last week", the `20260915` SU_Others and `初期データ追加情報 _20260930.rev1`, `--today 2026/10/01`:
19 modules stay flexible, 7 become Fixed (the decoder had dropped cells of those modules, so the sheet has more than the plan);
1,632 misc tasks; 5,690 rows; 1,926 FI/FO pairs in the sheet, 1,106 became Fixed trips, 820 were skipped (no region could be
found), 579 FI/FO orphans reported. The generated files load in Timefold (strict trip parser lenient mode, 285 workers).
That input is large: Timefold needed `JVM_MAX_HEAP=12g` (it ran out of memory with 6g).

## Open points (not decided by the tool)

* **`allow_cut_fixed_trip`**: Timefold removes a Fixed trip from its output when no same-region work lies inside. With Fixed trips taken from
  the sheet that deletes many of them (408 in the run above). For the weekly flow set `allow_cut_fixed_trip: false` in `SolverParams.yaml`.
* **Red FI/FO cells** (about 630 in the sample sheet): the tool reads their text as FI/FO like any other FI/FO cell, and the decoder still
  turns the same day into an unavailable date. Nobody has decided what the red colour means, so nothing special is done.
* **`移動` cells** stay misc tasks; no trip is built from them.
* **Trips without a determinable region** (820 above, many have no bar at all between FI and FO) are skipped; `--default-trip-region r_other`
  keeps them as "away, place unknown" (the days inside are then closed for module work).
* The comparison is only as good as last week's file: with the decoder output as stand-in some modules look changed because the decoder had cut
  cells. With a real FlexibleSchedule written back to Excel by 3-3 this noise disappears.
* 2-4 (adding new modules) is done in between by the planner; this tool does not add new modules, and a module that is not in last
  week's flexible plan always becomes misc.
