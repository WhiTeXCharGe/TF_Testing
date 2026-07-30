# decoder6.py — what it does and how

This documents `decoder6.py`, generated from/replacing `decoder5.py`. It covers
what changed, exactly which cells/modules get cut and why, and how workload
hours get computed.

## Inputs

Three files, all via CLI flags (no more hardcoded filenames):

| Flag | File | Role |
|---|---|---|
| `--su-others` | `*.xlsm` (e.g. `20260726 SU_Others.xlsm`) | Actual day-by-day worker assignments (ground truth) |
| `--seiban-info` | `初期データ追加情報.xlsx` | Base planned-module data (mostly a stub in practice) |
| `--seiban-info-r` | `初期データ追加情報 _r.xlsx` | Revised planned-module data — **authoritative**, used first |
| `--plan-start` / `--plan-end` | *(optional)* | Force the plan range instead of deriving it from SU_Others |

`--seiban-info` and `--seiban-info-r` share the same sheet names (`製番`,
`作業者`); every value is read from `_r` first, and only falls back to the
base file when `_r`'s cell is blank.

---

## What's different from decoder5

1. **Input files.** decoder5 read `新規製番リスト` (planned dates) and
   `スキル集計` (numeric skill levels) — both gone. decoder6 reads `製番` and
   `作業者` from the two 初期データ追加情報 files instead.

2. **SU_Others column layout is auto-detected**, not hardcoded. The old sheet
   (`予定表_2025`) has company/name/role in columns A/B/E; the new sheet
   (`予定表_2026`) moved things around and added an ID column, so name is now
   column C and role is column G (`担当職種`). decoder6 reads the header row
   and locates each column by its Japanese label, so both sheet generations
   work through the same code path. Both sheets are read by default
   (`--su-sheets 予定表_2025,予定表_2026`).

3. **Phase 2 (Hardware Setup) now has two operations** instead of one: Mech
   (`p2o1`, from 作業種別=M) and Elec (`p2o2`, from 作業種別=E), each with its
   own headcount/workload — matching the new 製番 sheet, which has separate
   M and E columns where decoder5's source only had one combined phase-2
   field.

4. **Worker regular/spot classification is new.** 作業者 has one R/S column
   per worker (not per operation). decoder6 applies that single value across
   *all* of that worker's real task-skill operations
   (`worker_type_by_operation: {p2o1: regular, p3o1: regular, ...}`) —
   whichever of p2o1/p2o2/p3o1/p4o1 the worker actually has a nonzero skill
   for.

5. **Missing dates default to the plan range**, per-field. decoder5 either
   had a fully-specified date row or dropped the module. decoder6's 製番 rows
   are often partially blank (only 26/60 rows had a phase-2 start date in the
   sample data); each missing field defaults independently (see "Defaulting
   rules" below) instead of nuking the whole row if one field is blank.

6. **`SKIP_MODULE_IF_NO_SU_MATCH` defaults to `False`** (decoder5 defaulted
   to `True`). A module that never appears in SU_Others still shows up in
   Schedule.yaml using its planned/defaulted dates, instead of being omitted
   entirely.

7. **Output schema matches the current GanttChartEditor types**
   (`src/types/schedule.ts`, `envConfig.ts`, `yamlService.ts`), not
   decoder5's older shape:
   - `workload_hours` instead of `workload_days`.
   - IDs are `e{n}p{phase}o{k}` (no underscores) instead of `e{n}_p{phase}`.
   - "Other work" and "Personal Business" are now flat `misc_task_list`
     entries (no phase/operation wrapper) — assignments reference the misc
     task's own `id` directly as `operation_task`.
   - Workers carry `worker_type_by_operation` (new).
   - No numeric skill levels (0-5) — decoder6 has no numeric-skill source
     anymore, so `skill_map` is binary: `1` if the worker's SU_Others role
     text indicates that operation, else `0` (see below).

8. **Two heuristics tuned differently after actually running it** (see
   "Bugs found by running it" at the bottom) — `CUT_MODULE_IF_PHASE_ZERO_WORKLOAD`
   is off by default, and phase-4's *length* (for ratio/proportion purposes
   only) is capped.

---

## The pipeline, in order

1. Parse SU_Others → `worker_date_map` (cell text per worker/day),
   `worker_personal_map` (grey PB cells), `worker_roles` (per-worker role
   text), red cells → `unavailable_dates`.
2. Parse 製番 (merged base + `_r`) → `planned_meta` per module code.
3. Parse 作業者 (merged base + `_r`) → R/S per worker name.
4. **Cut/cleanup pipeline** (below) mutates `worker_date_map` in place,
   breaking tool-codes it doesn't trust into non-matching text (so they fall
   through to "other work" instead of being deleted).
5. **Shift**: for each surviving module code, take its real worked days from
   SU_Others and split them into p2/p3/p4 windows (`build_shifted_meta`).
6. **Assign**: re-walk `worker_date_map`; anything landing inside a shifted
   phase window becomes a real `Flexible` assignment; everything else becomes
   a `Fixed` misc/personal-business assignment.
7. Compute `workload_hours` / `recommends_worker_min/max` from the real
   assignment counts.
8. Write `EnvConfig.yaml`, `Schedule.yaml`, `TransformationLog.txt`.

---

## Cell/module cut rules (why something becomes "dummy other" instead of a real task)

A tool-code cell is never deleted — it's "broken" (last character flipped so
the regex `\d{3}[A-Z0-9]\d{5}A` no longer matches it), so it silently falls
through to become a generic "other work" misc task instead of a scheduled
tool-install task. Every cut is logged in `TransformationLog.txt` with the
exact reason. In application order:

### A. `cut_su_outlier_cells` — per-module and per-worker noise filtering

1. **Head-of-range cut**: if a module's *first* occurrence falls within the
   first `DUMMY_HEAD_DAYS_FROM_PLAN_START = 10` days of the SU_Others plan
   range, the **entire module** is cut. (Rationale: entries right at the very
   start of the tracked calendar window are often carryover/phantom data from
   before tracking began.) ⚠️ Now that decoder6 reads two combined sheets
   (2025+2026), "first 10 days of the range" means 2025/01/01–01/10 for the
   *whole* multi-year dataset — this caught one heavily-worked real module
   (`840700002A`) in testing. Left as-is; worth reviewing (see bottom).
2. **Too little evidence**: if a module has fewer than `4` unique worked days
   (`cut_module_if_unique_days_lt`), or fewer than `4` total cells
   (`cut_module_if_total_cells_lt`), or only `1` cell total → cut entirely.
3. **Per-worker cleanup**: for each worker on the module —
   - only 1 day worked on this module → that worker's cells are cut.
   - cluster the worker's dates with a `7`-day gap tolerance
     (`cluster_gap_days`); keep only clusters containing a run of ≥2
     *consecutive* days; everything else (isolated single days, non-adjacent
     clusters) is cut.
4. **Re-check after cleanup**: if what's left is still under the unique-days
   or total-cells thresholds, cut the rest of the module too.
5. **Far-from-planned-window cut**: cluster the *cleaned* dates (1-day gap);
   keep clusters up to `planned_end + 90 days` (`cut_if_far_from_planned_days`).
   Beyond that, a cluster is still kept if it's within `ONGOING_TAIL_KEEP_GAP_DAYS = 30`
   days of the last kept cluster (an "ongoing work, still close enough"
   allowance); the first cluster that fails both checks, and everything
   after it, is cut.
6. **Final safety pass**: per (worker, module) pair, if the worker has only 1
   day total, or no two of their days are adjacent, those cells are cut too.

### B. `cut_su_short_span_modules_to_dummy`

If a module's *cleaned* unique worked days are still `< MIN_WORKED_DAYS_FOR_TOOL = 4`,
the whole module is cut and removed from `planned_meta` (won't become a tool
task at all, not even with zero workload).

### C. `cut_module_if_remaining_dates_too_small_vs_planned`

If remaining unique worked days `< ceil(planned_total_days × MIN_LEFT_DATE_SPAN_RATIO)`
= `ceil(planned_total_days × 0.20)`, cut the module. `planned_total_days` is
the module's planned p2+p3+p4 length — see the phase-4 capping note below,
since without it this rule over-fires badly on the new data.

### D. `cut_module_if_phase_zero_workload` — **disabled by default in decoder6**

decoder5 pre-checked, before shifting, whether a naive calendar-proportional
split of the actual span would leave any of p2/p3/p4 with zero worked days,
and cut the module if so. On the new dataset this produced false positives
(good real data discarded because work isn't evenly spread across the
calendar span) — see "Bugs found" below. `CUT_MODULE_IF_PHASE_ZERO_WORKLOAD = True`
in the file to re-enable it.

### E. `cut_modules_with_no_qc_to_dummy`

After shifting, if a module has *no* worker whose role text contains "QC"
among everyone actually assigned to it, the whole module is cut (a
tool-install without any QC step is treated as untrustworthy data).

### F. `cut_final_zero_workload_modules_to_dummy`

After shifting, if any of p2/p3/p4's *real* allocated worked-days come out to
zero, the module is cut. This is the "safety net" that still runs even with
rule D disabled — it checks the actual shifted result, not a synthetic
pre-split.

### G. Distance cut (`CUT_DISTANCE_DAYS = 365`)

If the actual worked span is more than 365 days outside the planned p2–p4
window (measured from either edge), the module is cut and treated as if it
were never in 製番 at all.

### H. `SKIP_MODULE_IF_NO_SU_MATCH` (default `False` in decoder6)

If `True`, any module with zero real SU_Others matches is omitted from
Schedule.yaml entirely, instead of appearing with its planned/defaulted dates
and zero workload.

---

## Defaulting rules (missing 製番 dates)

For each module, p2/p3/p4 start dates and 希望納期 (delivery) are resolved
independently, in order:

- `p2_start` missing → defaults to `plan_start`.
- `p3_start` missing → defaults to `p2_start`; if present but `< p2_start`,
  clamped up to `p2_start`.
- `p4_start` missing → defaults to `p3_start`; same clamp rule.
- `delivery` missing → defaults to `max(p4_start, plan_end)`; same clamp
  rule if present but earlier than `p4_start`.

Every default/clamp is logged per-field in `TransformationLog.txt`
("DEFAULTED field(s) [...]"), so you can tell a module with one blank field
apart from one with none of its planned dates at all.

Phase lengths: `phase2_len = p3_start − p2_start`, `phase3_len = p4_start − p3_start`,
`phase4_len = delivery − p4_start`. **For proportion/ratio purposes only**
(splitting the actual span, and the ratio cut above), `phase4_len` is capped
at `phase2_len + phase3_len` — see "Bugs found" below for why. The real
delivery date is still used as the module's actual planned end date in the
output; only the *weight* it gets in ratio math is capped.

If a module ends up with zero total planned length even after defaulting, it
collapses to a single day at `plan_start`.

---

## How workload_hours / recommends_worker gets computed

Nothing here is estimated from 工数 (it's blank in essentially every real row
seen so far) — it's all counted directly from real SU_Others assignments,
per **operation**, after shifting:

1. For each module code that survives the cut pipeline, its real worked days
   (from SU_Others) are split into p2/p3/p4 date windows (`build_shifted_meta`,
   unchanged from decoder5 — see phase-3 trigger logic below).
2. Walking `worker_date_map` again, any (worker, day) cell whose module code
   and day fall inside a phase window becomes a match. **Which operation** it
   counts toward:
   - Phase 3 / Phase 4 → always the sole QC operation (`p3o1`/`p4o1`).
   - Phase 2 → based on that worker's role text: contains "M" → counts toward
     `p2o1` (Mech); contains "E" → counts toward `p2o2` (Elec); a role like
     "M/E" counts toward **both** (the worker did both that day); a role with
     neither (e.g. "搬送"/"溶接"/blank) defaults to Mech (`p2o1`) so the day
     isn't silently dropped.
3. `workload_hours` for an operation = `(unique worker-days assigned to it) × HOURS_PER_WORKDAY (10)`.
4. `recommends_worker_min/max`: `avg = workload_hours/10 ÷ (unique assigned
   dates)`, floored/ceiled respectively; floored up to 1 if there's any real
   assignment at all. If 製番 gave a 推奨人数 (recommended headcount) for that
   phase/operation, `recommends_worker_max` is raised to at least that value.
5. If an operation ends up with zero worker-days after all of the above, it's
   logged under "WORKLOAD WARNING" in the transformation log (still written
   to Schedule.yaml with `workload_hours: 0`, not omitted).

**Phase-3 start trigger** (unchanged from decoder5, `_find_qc_phase3_start_day`):
the boundary between "phase 2" and "phase 3+4" in the *compressed* worked-day
timeline (gaps stitched out) is chosen by looking for the first worker whose
role is *purely* QC (contains "QC", not "M"/"E"), or failing that, where the
last purely-M/E worker's run stops. The p3/p4 tail is then split roughly
evenly, with the phase that had the larger *planned* length getting the odd
extra day.

**skill_map** is binary, not leveled: a worker gets `1` for `p2o1`/`p2o2` if
their role text (from SU_Others) contains "M"/"E" respectively, and `1` for
`p3o1`+`p4o1` if it contains "QC" — anywhere across all their rows, not
per-module. `other_op` and `personal_business_op` are always `1`.
`worker_type_by_operation` is layered on top from 作業者's R/S column, applied
uniformly to whichever of those four operations the worker has a nonzero
skill for.

---

## Bugs found by actually running it against real data

1. **Corrupted plan range.** One SU_Others header cell (`予定表_2026`, column
   264) carried date-number formatting on a near-zero/leftover value,
   evaluating to `1900-01-09`. Since plan range is `min()`/`max()` over every
   header date, this silently set the whole plan range's start to 1900.
   Fixed: header dates more than 3 years from the header row's median are
   dropped as formatting artifacts (logged as a `WARNING:` on stderr).

2. **希望納期 skewing ratio cuts.** Unlike decoder5's source (which had a
   tight, real p4-end date), 希望納期 is a customer delivery deadline that can
   sit 800+ days past phase-4's start. Used raw, phase-4's length dominated
   every proportion calculation, which broke rule C above — e.g.
   `530N02566A` has 79 genuine worked days but was being cut because the
   ratio check (based on an 822-day "planned total") wanted 165+. Fixed by
   capping phase-4's *length* to `phase2_len + phase3_len` for ratio purposes
   only (see "Defaulting rules" above).

3. **Phase-zero pre-check false positives.** Rule D's naive calendar-
   proportional split doesn't hold up against the new data's actual work
   patterns (gaps, uneven density) — it was discarding hundreds of good
   assignment cells across 5 modules (e.g. 451 cells on `840400015A` alone).
   Disabled by default; `cut_final_zero_workload_modules_to_dummy` (rule F,
   which checks the real shifted result, not a synthetic split) still guards
   against genuinely-empty phases.

## Still open / worth your judgment

- Rule A's "first 10 days of plan range" head-cut caught one real,
  heavily-worked module (`840700002A`) once the plan range spans two years.
  Not changed — I don't have enough context to know whether that's real
  carryover noise (as decoder5 assumed for a single-year sheet) or a
  legitimate early start. Check `TransformationLog.txt` for it.
- 工数 (workload man-hours) in 製番 is essentially always blank in the real
  data seen so far, so it's currently unused entirely (only 推奨人数 headcount
  feeds `recommends_worker_max`). If it starts getting filled in, it isn't
  wired in yet.
