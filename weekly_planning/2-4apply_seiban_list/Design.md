# 2-4 apply_seiban_list — Design

## 1. Purpose

2-4 adds the **new seiban (modules)** of the week to the plan. It replaces the "add new modules" part of the planner step (until now done by hand in the planner tool).

It reads the seiban list workbook, then changes **both** yaml files:

- **EnvConfig**: customers, fabs, regions and workflows (process templates) that are new, plus the skills of the workers for new operations.
- **Schedule**: one workflow task per new seiban; the workflow task of a seiban that is marked for deletion is removed.

Position in the flow (output numbers may change later):

| Tool | In | Out |
|---|---|---|
| 2-2 + 2-3 | previous plan, SU_Others, ... | `01_EnvConfig.yaml`, `01_Schedule.yaml` |
| **2-4** | `01_*`, the seiban list workbook | `02_EnvConfig.yaml`, `02_Schedule.yaml`, report (`--report`) |
| 2-5 | `02_*` | `03_*` |

2-4 only reads the workbook. Recolouring rows afterwards (white to grey, removing pink rows) is done by the workbook's own macro.

## 2. What 2-4 does not do

- **Grey rows** (seiban that were already planned) are not touched. If something about them changed, 2-2-3 handles it (it turns changed modules into misc tasks). 2-2-3 may need a change for this later.
- **Changing a template** (different hours, added or removed operation, swapping the template of a module) and **renaming** existing ids are not handled. A changed template simply does not match any workflow, so a new workflow is created and the old one stays (see 5.2). The real handling will be decided with the team.
- Existing workflows, customers, fabs and regions are never deleted or renamed.
- `plan_range` is not changed (2-5 sets the period).
- The column for the device name is ignored for now.

## 3. Input: the seiban list workbook

Sheets that are read:

| Sheet | Used for |
|---|---|
| `計画対象` | the rows to handle. Columns: registration time, seiban, device name (ignored), customer, Fab, Region, available-from date (ignored), desired due date (ignored), template choice (工程選択), and start-available / end-deadline dates for processes 1 to 4 |
| `搬入先テーブル` | delivery sites: site, customer, Fab, Region. Used to fill a Region that is empty in `計画対象` |
| `工程テンプレート<N>` | one sheet per template: process number, work number, work name, man-hours (工数), minimum and maximum number of workers |

`製番リスト` (the master list) and `データ抽出` are not read.

### Row colour (of the seiban cell)

| Colour | Meaning | 2-4 |
|---|---|---|
| none / white | new seiban | add |
| grey `D9D9D9` | already planned | ignore |
| pink `FFDCE6` | to be deleted | delete |
| any other colour | unknown | skip, list in the report |

### Normalisation

- Seiban: normalised with the module-code rule of `common.text.tool_code`. A cell that is not a module code: row skipped, listed in the report.
- Names of customer / fab / region: compared by a key (NFKC, no spaces, case-insensitive), so stray spaces and full-width letters do not create duplicates. A new entry keeps the text of the sheet (trimmed).
- Work name: trimmed and NFKC; `Q/C` is read as `QC` (the EnvConfig name).

## 4. Master data (EnvConfig)

Only what the valid white rows need is created or changed.

| Item | Rule |
|---|---|
| Customer | matched by name key; new: next free `c<n>`, no unavailable dates |
| Region | matched by name key; new: next free `r<n>` with the same default values the decoder used (maximum stay and yearly stay 10000, stay-off interval 3, no unavailable dates), and transit rows (1 day) between the new region and every existing region in both directions |
| Fab | matched by **customer + fab name**. The same fab name under another customer is a different fab. New: next free `f<n>` with its customer and region |
| Customer without a fab name | a fab named after the customer is used (the Schedule always needs a fab id) |
| Region empty | taken from `搬入先テーブル` (same customer and fab); still empty: `r_other` and a report line |
| Customer empty | `c_other` (both empty: fab `f_other`) |
| Existing fab with another region | the region of the fab is **updated** to the row's region, and a report line says so |

Differences between a row and `搬入先テーブル` are listed in the report; the row wins.

## 5. Workflows (process templates)

### 5.1 What a template defines

A template sheet gives, per process number P and work number K: work name, man-hours, minimum and maximum workers. A template may have no process 1 (or any other number); only the processes that exist in the sheet are used. Dates of a process the template does not have are ignored.

### 5.2 Match or create

For every template used by a valid white row:

1. Look for an **existing workflow** with the same operations: same process numbers, work numbers, work names, and minimum / maximum workers (ids, names and `work_hours` are not compared).
2. Found: use it. Existing workflows (including legacy ones such as `wf_tool`) are kept as they are.
3. Not found: **create a new workflow**.

A template that changed therefore never edits an old workflow; it creates a new one. The legacy `wf_tool` has different minimum / maximum workers, so on the first run it will probably not match template 2, and a new workflow is created for it while old modules stay on `wf_tool`.

### 5.3 Ids of a new workflow

`n` = highest `wf<number>` in the EnvConfig + 1 (legacy ids without a number are ignored; the first one is `wf1`).

| Item | Id | Name |
|---|---|---|
| Workflow | `wf<n>` | `wf<n>` |
| Phase (process P) | `wf<n>p<P>` | `p<P>` |
| Operation (work K) | `wf<n>p<P>o<K>` | the work name (`Mech`, `Elec`, `QC`) |

Operation fields: `work_hours: [10]` (as now), `min_worker_num`, `max_worker_num` from the template (not 1 and 8 any more). The solver finds the phase number as the last `p<number>` in a phase id, so `wf1p1` is phase 1.

### 5.4 Skills of the workers

When a workflow is created, for each worker and each **new** operation:

- take the existing operations (of any workflow) with the **same name** that the worker has in `skill_map`;
- the new operation gets the **highest** level among them, and the `worker_type_by_operation` entry of that operation;
- no same-name operation: the worker gets no skill for it.

A work name that no existing operation has means nobody can do it; this is only a report line. Entries such as `other_op` are not touched.

## 6. Schedule

### 6.1 White row → new module

Rows are checked first: seiban valid, template sheet found and valid, every process of the template has a start and an end date, end is not before start. An invalid row is skipped and listed in the report; it creates nothing (no master data, no workflow). A seiban that already is a workflow task in the Schedule is skipped with a report line.

A valid row creates one workflow task:

| Field | Value |
|---|---|
| module id | next free `e<n>` |
| `name` | the seiban |
| `workflow`, `fab` | the ids of 5 and 4 |
| phase task | id `<module><pP>` (for example `e57p1`), `phase` = the phase id, name `p<P>`, start / end = process P start-available / end-deadline |
| operation task | id `<module><pP>o<K>`, `operation` = the operation id, `workload_hours` = man-hours of the template |
| not written | `recommends_worker_min/max` (the solver treats an empty value as no recommendation) |

The task ids deliberately stay in the short `e57p1o1` form: the solver splits operation task ids with the pattern `...p<number>o<number>`, which would cut a long id such as `e57wf1p1o1` at the wrong place when it has to read a task it does not know.

### 6.2 Pink row → delete

If a workflow task with that seiban exists: remove its phase and operation tasks and **every assignment row of its operation tasks**. These assignments are flexible by construction (a seiban the planner touched, or one that is already past, is a misc task and has no workflow task), so Timefold would plan them again anyway. If no workflow task exists (fixed, past, or never planned): nothing to do, one report line.

A seiban that is both white and pink: both rows are skipped and listed.

### 6.3 Order

1. read the workbook and check every row; 2. delete (pink); 3. create masters, workflows and skills that valid white rows need; 4. add modules (white); 5. write the files and the report.

## 7. Output and options

- `02_EnvConfig.yaml` and `02_Schedule.yaml` in `--out-dir` (the number may change). Inputs are never overwritten.
- The terminal shows only counts (rows added / deleted / skipped, masters and workflows created, number of warnings). **Every warning goes to the report**, written only with `--report` (`<name>_apply_report.txt`), as in 2-2-3.
- Options: `--schedule`, `--env`, `--seiban-list`, `--out-dir`, `--sheet` (default `計画対象`), `--report`. A `config.txt` like 2-6 / 3-1 can be added later.

## 8. Code layout and review steps

- `common/` gets the shared reading code the other tools can also use (the Excel reading of the seiban list and templates). Code used only by 2-4 stays in the tool folder. Tools never import from each other.
- Step A: read the workbook (rows with colours, templates, delivery sites) into plain data, with tests on small generated workbooks.
- Step B: EnvConfig changes (masters, workflow matching and creation, skills), pure functions on plain data.
- Step C: Schedule changes (add / delete), command line, report, README.

## 9. Effects on other tools and open points

- **2-2-3 (needed later):** `common/worker_skills.py` maps the role letters to the fixed ids `p2o1 ... p4o1`. After 2-4 creates `wf<n>...` operations, next week's 2-2-3 would drop the skills of those operations. It has to map by operation **name** using the EnvConfig.
- **Cell name:** a fab named after the customer would repeat the name in the cell name of 3-2 / 3-3 (`<customer><fab>_...`); the template can skip the fab when it equals the customer.
- **Matching includes min / max workers** (5.2): confirm with the team that this is wanted.
- **Rows with a missing date are skipped** (6.1): confirm.
- **Sites:** when a row and `搬入先テーブル` disagree, the row wins (4): confirm.
- **Template changes and swaps:** to be discussed with the team.
