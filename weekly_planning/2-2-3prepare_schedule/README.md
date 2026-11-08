# 2-2-3 prepare_schedule  (weekly planning, steps 2-2 + 2-3 in one tool)

Folder: `tools/weekly_planning/2-2-3prepare_schedule/`. The other tools of the weekly phase are next to it in `tools/weekly_planning/`
(2-5 `2-5set_plan_period`, 3-2 `3-2extract_flexible_schedule`, 3-3 `3-3write_su_others`).
Japanese explanation (how FI/FO is read, repaired and compared, step by step): [README_ja.md](README_ja.md).

## What it does

* **2-2** looks at last week's flexible plan and at `SU_Others.xlsm` and decides what the planner has changed:
  * a **module** is judged only by its work and its text: if the (worker, date) cells of the module in the sheet differ from last week's
    assignments, **or the text of its bar was changed** (the planner only renamed the cell), it becomes **Fixed** (it turns into misc tasks, with the
    new text as the name). Hours are not compared;
  * a module whose work all ended before `--today` becomes **Fixed**;
  * a **trip** (FI/FO) is judged only by its FI and FO days: a flexible trip whose FI and FO are no longer on the same days in the sheet is no
    longer flexible, the sheet's FI/FO become a **Fixed trip**;
  * **FI/FO never decide a module.** If a trip was moved but the work of the modules inside did not move, those modules stay flexible;
  * every other module and trip stays **flexible**.
* The result of 2-2 is `<today>_FlexibleWork.yaml`: the list of flexible **modules** and the list of flexible **trips** (the FI/FO of last week's
  Timefold result that the planner did not change).
* **2-3** builds the Timefold input:
  * flexible modules and their assignments are copied from last week unchanged (only the worker ids are re-mapped);
  * flexible trips are copied from last week unchanged;
  * every other bar of the sheet becomes a misc task with Fixed rows (region from the 製番 sheet when the bar carries a module code, else `r_other`);
  * every FI/FO pair of the sheet that is not an unchanged flexible trip becomes a Fixed `trip_list` entry;
  * `EnvConfig`: workflows, fabs, regions, customers and companies of last week are kept (ids never change), new names from
    `初期データ追加情報` are added; `worker_list` is rebuilt from the sheet (ids of known workers are kept by name, skills and
    regular/spot are computed like decoder7).

**`cell_name`.** Every assignment row of a flexible module that goes into `01_Schedule.yaml` gets two keys with the same text: `cell_name` (the text of the
bar in SU_Others; the most common text of that worker's cells on the row's days) and, if it is empty, `description`. The solver keeps the `description` of a
(worker, task) in its result but drops other new keys of the rows it rebuilds, so `description` is how the text reaches `04_Schedule.yaml`;
tool 3-2 turns it into `cell_name` of `FlexibleSchedule.yaml`, tool 3-3 writes it into the cells, and next week this tool checks it. An existing non-empty
`description` is never overwritten (then that row has no text to carry).

`<today>_FlexibleWork.yaml` looks like this:

```yaml
flexible_work:
  modules:
  - 530N03359A
  - 530N03361A
  trips:
  - worker: w007
    start_date: 2026/10/19
    end_date: 2026/10/23
    region: r1
```

## Run

There are two kinds of SU_Others workbook, and the tool takes both:

| Option | File | Used for |
|---|---|---|
| `--su-others` | the dated, planner-maintained file, e.g. `20260726 SU_Others.xlsm` | ongoing tasks (bars, FI/FO, personal business) and the worker list |
| `--skill-file` | `SU_Others_skill level有配布禁止_.xlsm` | skills, skill level and Primary/Secondary status only |

Generic form (`N` = this week's folder, `(N-1)` = last week's folder):

```
python prepare_schedule.py ^
  --prev-flexible  (N-1)/FlexibleSchedule.yaml ^
  --prev-env       (N-1)/04_EnvConfig.yaml ^
  --su-others      "<dated SU_Others>.xlsm" ^
  --skill-file     "<SU_Others_skill level>.xlsm" ^
  --seiban-info    "<初期データ追加情報>.xlsx" ^
  --out-dir        N ^
  --today          YYYY/MM/DD ^
  --report
```

**Example** (PowerShell, in this folder; last week's files in `20261030\`, this week's output goes to `20261106\`, the Excel files are the ones in `tools\decorder`):

```powershell
python prepare_schedule.py --prev-flexible 20261030\FlexibleSchedule.yaml --prev-env 20261030\04_EnvConfig.yaml --su-others "..\..\decorder\20260726 SU_Others.xlsm" --skill-file "..\..\decorder\20260915 SU_Others_skill level有配布禁止_.xlsm" --seiban-info "..\..\decorder\初期データ追加情報 _20260930.rev1.xlsx" --out-dir 20261106 --today 2026/11/06 --report
```

Output in `20261106\`: `20261106_FlexibleWork.yaml`, `01_EnvConfig.yaml`, `01_Schedule.yaml`, and with `--report` `20261106_prepare_report.txt`.
If Python prints Japanese file names wrongly on Windows, run it as `$env:PYTHONUTF8="1"; python -X utf8 prepare_schedule.py ...`.

* Workers are matched to the skill file by ID, else by name. A worker that is **not** in the skill file keeps the skills of last week's
  EnvConfig (listed in the report); a brand-new worker not in the skill file gets no skills.
* `--skill-file` is optional: without it the skill sheet is looked for inside `--su-others`, like decoder7 does.
* Sheet names: `--su-sheets` is the task sheet in the dated file (default `予定表_2026`), `--skill-sheet` is the skill sheet in the skill file
  (default `予定表_2026 (2)`). If the name is wrong the tool stops and lists the sheets that exist.

Other options: `--seiban-start-row` (default 3 = whole list), `--default-trip-region <id>` (region of a Fixed trip whose region cannot be found; default `r_other`,
`none` skips and lists such trips), `--no-fifo-repair` (do not repair FI/FO without a partner), `--max-gap-days` (default 14) and `--chain-days` (default 3), see "FI/FO patterns",
`--decoder-dir` (folder of `decoder7.py`, default `tools/decorder`). Needs `pyyaml`, `openpyxl`, `pandas` (the same as decoder7).
Reading the 7 MB workbook takes about 25 seconds.

`--prev-flexible` is normally the `FlexibleSchedule.yaml` of tool 3-2. A Timefold result `Schedule.yaml` also works: everything with a module is
treated as flexible, misc tasks in it are ignored.

## Rules in detail

| Topic | Rule |
|---|---|
| Module identity | the workflow task `name` (the module code such as `530N03359A`) is matched with the code in the sheet's bar text |
| "Changed" module | the set of (worker, date) in the sheet differs from last week's rows (hours are not compared). A worker that is renamed or gone counts as a change. FI/FO are not looked at |
| Changed text of a bar | every assignment row of `FlexibleSchedule.yaml` has a `cell_name` (see below). If the most common text of the worker's cells of that row in the sheet is not the `cell_name` (compared without case, width and spaces), the module is changed: reason `cell text changed in SU_Others (<worker>: 'old' -> 'new')`. One odd cell in a bar does not count. Rows without a `cell_name` are not checked |
| Complete module | last assignment date `< --today` |
| FI/FO pairing | per worker, every `FI` is paired with the next `FO`; marks without a partner are repaired, see "FI/FO patterns" below |
| Flexible trip kept | last week's Flexible trip whose worker, FI day and FO day are all still in the sheet: kept as Flexible, copied unchanged (region included), listed in `FlexibleWork.trips` |
| Flexible trip changed | last week's Flexible trip that is not in the sheet any more (FI or FO moved or removed): reported as `CHANGED flexible trip`; the sheet's FI/FO pair is a Fixed trip |
| Module "changed" and decoder7 | a sheet cell of a module that last week's file already holds as a misc task named after that module (decoder7 does this for outlier cells and for low-data modules) is not a change |
| Region of a Fixed trip | 1. the same trip of last week; 2. module codes between FI and FO (through 製番); 3. a fab name (3+ characters, one region) in a bar label between FI and FO; otherwise `--default-trip-region` (`r_other`) |
| Misc tasks | one misc task per distinct cell text; Fixed rows, 10 hours per day, colour of the cells; grey cells without a module code become personal-business misc tasks |
| Misc region | module code known in 製番: its region; everything else `r_other` (Timefold treats it as "no region") |
| Worker ids | known workers keep their id (by name); new workers get the next free `w###` |
| Skills | only operations that exist in the workflow of last week's EnvConfig are kept (others are dropped and reported) |

## FI/FO patterns

Counted on `20260927 SU_Others`: 2,172 `FI` and 2,391 `FO` cells (629 of them red), plus 8 `FI(仮)` and 2 `Fo`; no `F1` / `Fl` / `F0` in this file, but they are read anyway.
Marks without a partner: 97 workers start with an FO, 322 FO follow an FO, 103 FI follow an FI, 95 workers end with an FI.
"Busy day" = any day with text in the sheet for that worker (module, other text, personal business).

| Pattern | What the tool does |
|---|---|
| typos / case / full width / `FI(仮)` | read as FI / FO (`F1`, `Fl`, `F0`, `Fi`, `ＦＩ` ...); the report lists every text that was not exactly FI / FO |
| FI without FO (next mark is an FI, or the sheet ends) | the trip ends on the last day of the busy run after the FI (gaps up to `--max-gap-days`, default 14); nothing busy: a one-day trip |
| FO without FI (first mark, or after an FO) | the trip starts on the first day of the busy run before the FO, after the previous mark; if the run reaches the start of the sheet the trip starts at the sheet start |
| `FI \| TaskA ... \| FO \| TaskB ... \| FO` (FO after an FO, work in between) | **same date**: the second trip starts on the date of the first FO (FI = that FO date), not on the first TaskB day. Applies when the busy run starts within `--chain-days` (default 3, covers a weekend) after the first FO; otherwise the trip starts on the first busy day |
| `FI \| TaskA ... \| FI \| TaskB ... \| FO` (FI before an FI) | same date: the first trip ends on the date of the second FI, if that FI is within `--chain-days` of the end of the busy run |
| FI, FI / FO, FO on neighbouring days (<= 2 days, nothing busy between) | one mark: the first FI, the last FO |
| FI, FI far apart / FO, FO far apart | the two rules above |
| empty days inside FI..FO | kept: the trip is just FI to FO |
| a very long gap inside a repaired trip | the busy run stops at a gap larger than `--max-gap-days` |
| `移動` | not a mark: 173 of the 183 cells are more than 3 days from any FI/FO (inside a trip), so they stay misc tasks |
| trip without any region | kept with region `r_other` (away, place unknown) instead of being dropped; `--default-trip-region none` skips them |

Every repaired trip is a `NOTE: ... INFERRED ...` line in the report (292 in the run on `20260927`). `--no-fifo-repair` turns the repair off.
Red FI/FO cells are read like any other FI/FO.

## Tests

```powershell
python -m unittest discover -s tests -v
```

37 tests on plain data (decisions, FI/FO pairing and repair, region rules, id stability, master lists, misc tasks, YAML round trip). No Excel needed.

## What was tried on real data

* 2026-11-05, `20260726 SU_Others` (dated file) + the skill-level file + `初期データ追加情報 _20260930.rev1`, `decorder/decoder7_out` as "last week", `--today 2026/07/27`:
  runs through (default decoder path); 8 modules flexible, 18 Fixed (that decoder output was made from a different SU_Others, so most modules differ), 1,424 misc tasks, 1,834 Fixed trips, 221 repaired.
* 2026-10-28 / 11-02, the decoder7 output of `20260927 SU_Others` as "last week" and the same file as the sheet: 26 modules flexible, 0 Fixed.
  The generated files load in Timefold; Timefold needed `JVM_MAX_HEAP=12g` for the 2026-10-28 input (it ran out of memory with 6g).

## Open points (not decided by the tool)

* **`allow_cut_fixed_trip`**: Timefold removes a Fixed trip from its output when no same-region work lies inside. With Fixed trips taken from
  the sheet that deletes many of them. For the weekly flow set `allow_cut_fixed_trip: false` in `SolverParams.yaml`.
* **Red FI/FO cells** (about 630 in the sample sheet): read like any other FI/FO; the decoder still turns the same day into an unavailable date.
  Nobody has decided what the red colour means, so nothing special is done.
* **`移動` cells** stay misc tasks; no trip is built from them.
* The repairs of FI/FO without a partner are guesses from the busy days; every one is listed in the report.
* The comparison is only as good as last week's file: with the decoder output as stand-in some modules look changed because the decoder had cut
  cells. With a real FlexibleSchedule (tool 3-2) written back to Excel by 3-3 this noise disappears.
* 2-4 (adding new modules) is done in between by the planner; this tool does not add new modules, and a module that is not in last
  week's flexible plan always becomes misc.

## Example with the `20261105Weeklyplanning` test folders (run from the `tools` folder, PowerShell)

Reads `20261105Weeklyplanning/20261030Initialplan/2-2-3/input`, writes `20261105Weeklyplanning/20261030Initialplan/2-2-3/output` (`20261030_FlexibleWork.yaml`, `01_EnvConfig.yaml`, `01_Schedule.yaml`, `20261030_prepare_report.txt`).

```powershell
python -X utf8 weekly_planning/2-2-3prepare_schedule/prepare_schedule.py --prev-flexible 20261105Weeklyplanning/20261030Initialplan/2-2-3/input/20261023_FlexibleSchedule.yaml --prev-env 20261105Weeklyplanning/20261030Initialplan/2-2-3/input/20261023_04EnvConfig.yaml --su-others "20261105Weeklyplanning/20261030Initialplan/2-2-3/input/20260927 SU_Others.xlsm" --skill-file "20261105Weeklyplanning/20261030Initialplan/2-2-3/input/20260915 SU_Others_skill level有配布禁止_.xlsm" --seiban-info "20261105Weeklyplanning/20261030Initialplan/2-2-3/input/初期データ追加情報 _20260930.rev1.xlsx" --out-dir 20261105Weeklyplanning/20261030Initialplan/2-2-3/output --today 2026/10/30 --report
```

(`python -X utf8` is enough for Japanese file names; do not put `PYTHONUTF8=1` in front in PowerShell, that is Git Bash syntax.)
