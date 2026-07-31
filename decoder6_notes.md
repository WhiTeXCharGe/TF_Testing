# decoder6.py — what it does and how

This documents `decoder6.py`, generated from/replacing `decoder5.py`. It covers
what changed, exactly which cells/modules get cut and why, and how workload
hours get computed. Updated after a third pass based on real test runs
(current test config: `--plan-start 2026/08/01`, no `--plan-end`, `予定表_2025`
excluded).

## Inputs

Three files, all via CLI flags (no more hardcoded filenames):

| Flag | File | Role |
|---|---|---|
| `--su-others` | `*.xlsm` (e.g. `20260726 SU_Others.xlsm`) | Actual day-by-day worker assignments (ground truth) |
| `--seiban-info` | `初期データ追加情報.xlsx` | Base planned-module data (mostly a stub in practice) |
| `--seiban-info-r` | `初期データ追加情報 _r.xlsx` | Revised planned-module data — **authoritative**, used first |
| `--plan-start` / `--plan-end` | *(optional)* | Force the plan range instead of deriving it from SU_Others |
| `--su-sheets` | *(optional, default `予定表_2026`)* | Which SU_Others sheet(s) to read |

`--seiban-info` and `--seiban-info-r` share the same sheet names (`製番`,
`作業者`); every value is read from `_r` first, and only falls back to the
base file when `_r`'s cell is blank.

---

## What's different from decoder5

1. **Input files.** decoder5 read `新規製番リスト` (planned dates) and
   `スキル集計` (numeric skill levels) — both gone. decoder6 reads `製番` and
   `作業者` from the two 初期データ追加情報 files instead.

2. **`予定表_2025` is excluded by default.** decoder6 reads only
   `予定表_2026` unless `--su-sheets` explicitly lists more. (Column
   auto-detection still supports the old 2025 layout if it's ever re-added.)

3. **SU_Others column layout is auto-detected**, not hardcoded. The old sheet
   (`予定表_2025`) has company/name/role in columns A/B/E; the new sheet
   (`予定表_2026`) moved things around and added an ID column, so name is now
   column C and role is column G (`担当職種`). decoder6 reads the header row
   and locates each column by its Japanese label, so both sheet generations
   work through the same code path.

4. **Phase 2 (Hardware Setup) now has two operations** instead of one: Mech
   (`p2o1`, from 作業種別=M) and Elec (`p2o2`, from 作業種別=E), each with its
   own headcount/workload — matching the new 製番 sheet, which has separate
   M and E columns where decoder5's source only had one combined phase-2
   field.

5. **Worker regular/spot classification is new.** 作業者 has one R/S column
   per worker (not per operation). decoder6 applies that single value across
   *all* of that worker's real task-skill operations
   (`worker_type_by_operation: {p2o1: regular, p3o1: regular, ...}`) —
   whichever of p2o1/p2o2/p3o1/p4o1 the worker actually has a nonzero skill
   for.

6. **SU_Others is the main source of truth for whether a module is real —
   同 decoder5's original design.** A module missing usable p2/p3/p4 dates in
   *both* 初期データ追加情報 files is only dummy if it *also* has no real
   occurrences in SU_Others. If SU_Others does have real data for it, the
   module is kept and a nominal planned window is built from SU_Others' own
   actual span instead — 製番 is a planning *reference*, not a gate. See
   "Dummy vs. rescued rules for 製番" below.

7. **Plan range no longer dummies a module for starting early — it splits
   Fixed vs. Flexible instead.** A module/phase that already started before
   `plan_start` isn't dropped; its assignments are marked
   `plan_flexibility: Fixed` (already happened, not up for the scheduler to
   touch), while phases at/after `plan_start` stay `Flexible`. See
   "Fixed vs. Flexible" below.

8. **`SKIP_MODULE_IF_NO_SU_MATCH` defaults to `False`** (decoder5 defaulted
   to `True`). A module with complete planned dates but zero actual
   SU_Others matches still shows up in Schedule.yaml using its planned dates,
   instead of being omitted entirely.

9. **Worker `description` block, sourced from SU_Others (not 作業者).** See
   "Worker description fields" below.

10. **Output schema matches the current GanttChartEditor types**
    (`src/types/schedule.ts`, `envConfig.ts`, `yamlService.ts`), not
    decoder5's older shape:
    - `workload_hours` instead of `workload_days`.
    - IDs are `e{n}p{phase}o{k}` (no underscores) instead of `e{n}_p{phase}`.
    - `environment.workflow_list` only has `wf_tool` — no `wf_other`/
      `wf_personal_business` catalog entries; those only exist as plain
      string tags on flat `misc_task_list` entries in Schedule.yaml, matching
      the reference schema (which has no catalog entry for them either).
    - "Other work" and "Personal Business" are flat `misc_task_list` entries
      (no phase/operation wrapper) — assignments reference the misc task's
      own `id` directly as `operation_task`.
    - Workers carry `worker_type_by_operation` and `description` (new).
    - No numeric skill levels (0-5) — decoder6 has no numeric-skill source
      anymore, so `skill_map` is binary: `1` if the worker's SU_Others role
      text indicates that operation, else `0` (see below).

11. **Two heuristics tuned differently after actually running it** (see
    "Bugs found by running it" at the bottom) — `CUT_MODULE_IF_PHASE_ZERO_WORKLOAD`
    and the "head of plan range" rule are both off by default, and phase-4's
    *length* (for ratio/proportion purposes only) is capped.

---

## The pipeline, in order

1. Parse SU_Others → `worker_date_map` (cell text per worker/day),
   `worker_personal_map` (grey PB cells), `worker_roles` (per-worker role
   text), `worker_description` (業務形態/VISA/海外運転/OJT per worker), red
   cells → `unavailable_dates`.
2. Parse 製番 (merged base + `_r`) → `planned_meta` per module code, applying
   the dummy/default rules below.
3. Parse 作業者 (merged base + `_r`) → R/S per worker name.
4. **Cut/cleanup pipeline** (below) mutates `worker_date_map` in place,
   breaking tool-codes it doesn't trust into non-matching text (so they fall
   through to "other work" instead of being deleted).
5. **Shift**: for each surviving module code, take its real worked days from
   SU_Others and split them into p2/p3/p4 windows (`build_shifted_meta`).
6. **Assign**: re-walk `worker_date_map`; anything landing inside a shifted
   phase window becomes a real tool-task assignment, marked `Fixed` if that
   phase's shifted start is before `plan_start` or `Flexible` otherwise (see
   "Fixed vs. Flexible" below); everything else becomes a `Fixed`
   misc/personal-business assignment.
7. Compute `workload_hours` / `recommends_worker_min/max` from the real
   assignment counts.
8. Write `EnvConfig.yaml`, `Schedule.yaml`, `TransformationLog.txt`.

---

## Dummy vs. rescued rules for 製番 (module-level, in `parse_seiban_merged`)

This runs first and decides whether a module code gets a `planned_meta` entry
at all. A code that fails here **never becomes a tool task** — any SU_Others
cells bearing that code fall through to "other work" misc tasks, exactly like
decoder5 treated codes missing from `新規製番リスト`. Each is logged per-code
in `TransformationLog.txt` (`DUMMY:` = dropped, `NOTE:` = kept with a caveat).

**SU_Others is the main source of truth** (decoder5's original philosophy:
"SU_Others provides actual execution span"). 製番's p2/p3/p4 dates are only a
*planned reference* used for proportions/ratio checks — real work always
wins. Concretely:

1. **p2/p3/p4 missing or out of order in both files:**
   - If the module code has **no occurrences anywhere in SU_Others** either
     → `DUMMY: missing phase start date(s)... and no SU_Others data found
     either`. Genuinely no data anywhere, nothing to schedule.
   - If the module code **does** have real SU_Others occurrences → **not
     dummy.** A nominal planned window is built directly from SU_Others'
     own actual first/last occurrence dates, split evenly across p2/p3/p4
     (`_allocate_phase_lengths_v5`), logged as `NOTE: ...SU_Others is the
     main source, so using its actual span ... as the planned reference
     instead`. The real shift step (below) then re-derives the true p2/p3/p4
     boundaries from role data anyway, so this nominal split mostly just
     feeds the ratio-cut threshold (self-consistent by construction, since
     "planned" ≈ "actual" here) and the tail tie-break.
   - Real data: 34/60 codes have none of p2/p3/p4 in 製番, and most of those
     still have real SU_Others data and get rescued this way — only a
     handful have no SU_Others match either and are truly dummy.
2. **希望納期 (delivery) is handled separately, always defaults, never dummies**
   — see below.

### Why delivery always defaults instead of ever triggering dummy

Delivery is blank in **100%** of real rows in both files (0/60 filled),
unlike p2/p3/p4 start (26/60 filled directly, more via the SU_Others rescue
above). Treating a missing delivery the same as missing start dates would
dummy every module. Instead: missing (or earlier-than-p4_start) delivery
defaults to `max(p4_start, plan_end)`, logged as `NOTE:`, and the module
proceeds normally otherwise.

Phase lengths: `phase2_len = p3_start − p2_start`, `phase3_len = p4_start − p3_start`,
`phase4_len = delivery − p4_start`. **For proportion/ratio purposes only**
(splitting the actual SU_Others span, and rule C below), `phase4_len` is
capped at `phase2_len + phase3_len` — see "Bugs found" for why. The real
delivery date is still used as the module's actual planned end date in the
output; only the *weight* it gets in ratio math is capped.

## Fixed vs. Flexible (plan range no longer dummies early-starting modules)

Earlier behavior dummied a module entirely if any part of it fell before
`plan_start`. Changed: nothing gets dropped for this anymore. Instead, in
`build_assignments_v6`, **per phase** (not per module, not per day): if that
phase's real shifted start date is before `plan_start`, every assignment on
that phase is written with `plan_flexibility: Fixed` (it already happened —
don't let the scheduler move it); phases starting at/after `plan_start` stay
`Flexible`.

Real example (`--plan-start 2026/08/01`), module `530N02566A`: 製番 said
p2=`08/10`, p3=`08/31`, but the real SU_Others data shows work from `07/20`
onward. Since SU_Others wins, the shifted result is p2 = `2026/01/21 –
2026/08/17`, p3 = `2026/08/18`, p4 = `2026/08/19 – ...`. p2 started long
before the plan range → every p2 assignment (both Mech and Elec) is `Fixed`.
p3 and p4 start after `plan_start` → their assignments are `Flexible`, i.e.
still open for the scheduler. This matches exactly: "phase 2 already started,
only p3/p4 are left to schedule."

Note this is decided **per phase as a whole** by its start date, not
per-day — a phase that straddles `plan_start` (starts before it, continues
past it) is entirely `Fixed`, on the reasoning that work already
substantially underway shouldn't be reshuffled by the optimizer.

---

## Cell/module cut rules (SU_Others side — why something becomes "dummy other" instead of a real task)

These run after the 製番-level dummy rules above, only on module codes that
survived. A tool-code cell is never deleted — it's "broken" (last character
flipped so the regex `\d{3}[A-Z0-9]\d{5}A` no longer matches it), so it
silently falls through to become a generic "other work" misc task instead of
a scheduled tool-install task. In application order:

### A. `cut_su_outlier_cells` — per-module and per-worker noise filtering

1. **Head-of-range cut — disabled by default (`ENABLE_HEAD_OF_RANGE_CUT = False`).**
   decoder5's rule: if a module's *first* occurrence falls within the first
   `DUMMY_HEAD_DAYS_FROM_PLAN_START = 10` days of the SU_Others plan range,
   cut the entire module (rationale: entries right at the very start of the
   tracked window are often carryover/phantom data). This wrongly caught a
   real, heavily-worked module (`840700002A`, 1456 cells) in testing, so it's
   off. Set the flag back to `True` in the file to re-enable.
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
the module's planned p2+p3+p4 length, using the phase-4-capped length above.

### D. `cut_module_if_phase_zero_workload` — **disabled by default**

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
were never in 製番 at all. This is now the main guard against wildly
implausible actual-vs-planned mismatches, since the plan range itself no
longer dummies anything (see "Fixed vs. Flexible" above).

### H. `SKIP_MODULE_IF_NO_SU_MATCH` (default `False` in decoder6)

If `True`, any module with zero real SU_Others matches is omitted from
Schedule.yaml entirely, instead of appearing with its planned dates and zero
workload.

---

## Worker description fields

Source is **SU_Others (`予定表_2026`), not 作業者** — 作業者 only has 4 columns
(company/ID/name/R-S), it doesn't carry these. The relevant SU_Others header
columns are read by label, same auto-detection approach as company/name/role:

| EnvConfig `description` key | SU_Others column(s) |
|---|---|
| `業務形態` | `業務形態` column, as-is |
| `VISA` | `VISA1` + `" "` + `VISA2`, joined and trimmed |
| `海外運転` | `海外運転` column, as-is |
| `備考` | `"OJT"` if the `OJT` column has any mark (e.g. `〇`), otherwise the key is omitted entirely |

A worker only gets a `description` block at all if at least one of these four
source cells is non-empty. Workers merged in only from 作業者 (never seen in
SU_Others) never get one. Since these columns only exist on `予定表_2026`'s
layout, description stays empty if `予定表_2025` is ever re-added without a
matching column mapping.

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
   only.

3. **Phase-zero pre-check false positives.** Rule D's naive calendar-
   proportional split doesn't hold up against the new data's actual work
   patterns (gaps, uneven density) — it was discarding hundreds of good
   assignment cells across 5 modules (e.g. 451 cells on `840400015A` alone).
   Disabled by default; `cut_final_zero_workload_modules_to_dummy` (rule F,
   which checks the real shifted result, not a synthetic split) still guards
   against genuinely-empty phases.

4. **"All-or-nothing" dummy rule over-fired on delivery.** First attempt at
   "missing info in both files → dummy" applied the same rule to 希望納期 as
   to p2/p3/p4, which dummied all 60 modules (delivery is blank everywhere).
   Fixed by special-casing delivery to always default instead of ever
   triggering dummy.

5. **Same over-eager dummy rule also ignored SU_Others.** The first version
   of the "missing 製番 info → dummy" rule dropped a module purely on 製番
   being incomplete, even when the module had substantial real SU_Others
   data. Since SU_Others is supposed to be the primary source (matching
   decoder5's philosophy), this was backwards. Fixed by only dummying when
   the code is missing from *both* 製番 (incomplete/out-of-order) *and*
   SU_Others (no occurrences at all) — see "Dummy vs. rescued rules" above.

6. **Hand-rolled YAML string quoting didn't escape special characters.**
   `_ys()` (the helper writing worker/task/misc-task `name`/`description`
   values) wrote free-text strings unquoted. Real SU_Others "other work"
   labels include values like a bare `,` or `-` (junk/placeholder cells),
   which broke YAML's plain-scalar syntax outright (`name: ,` fails to
   parse). Fixed: quote+escape (via `json.dumps`, which produces valid
   YAML double-quoted scalars) whenever a value contains YAML-risky
   characters (`: , # [ ] { } & * ! | > ' " % @` `, or a leading `-`/`?`),
   has leading/trailing whitespace, is a YAML keyword (`true`/`null`/...),
   or looks like a bare number. Verified by round-tripping both output
   files through a real YAML parser, not just visual inspection.

## Still open / worth your judgment

- 工数 (workload man-hours) in 製番 is essentially always blank in the real
  data seen so far, so it's currently unused entirely (only 推奨人数 headcount
  feeds `recommends_worker_max`). If it starts getting filled in, it isn't
  wired in yet.
- The Fixed/Flexible split (see above) is decided per-phase, not per-day —
  a phase straddling `plan_start` is entirely `Fixed`. If finer-grained
  per-day splitting inside a straddling phase is wanted instead, that's a
  bigger change (would need to split a single phase's assignments, and
  possibly the phase itself, at the plan_start boundary).
