# Guide 06 — Why did job `20261013_01` fail with `S1_INVALID`, and how to find the input to fix

Folder analysed: `sigmaI/…/screenas-workerassign-api/20261013_01/` (inputs, logs, SCIP log, result files).
Everything below is read from those files. **What is proven and what is not is marked.**

## 1. What happened (proven by the logs)

| # | Fact | Where |
|---|---|---|
| 1 | The API says `OPTIMIZATION_FAILED`; the real core status is **`S1_INVALID`** (Stage 1 invalid). Stage 2 never started. | `status.json`, `result.json → meta.core_status` |
| 2 | **It was not a lack of solving time.** SCIP found its final answer after **~101 s** (gap 0.00 %, 211 solutions) and then idled until the 7200 s limit. The `timelimit` label is misleading. | `scip_S1.log`: final primal first seen at 101 s; `Gap : 0.00 %` |
| 3 | After solving, the **structural check failed**: *15 phases were left unassigned*. | `optimization.log`: `[Integrity/S1-C1] 未割り当てのphaseがあります (計15件)` |
| 4 | Those same 15 phases were flagged **before** solving, while the model was built: *"no valid Y variable"* (= no possible start time could be created for them). Because they had no start variable, the "every phase must start once" rule was skipped for them, so SCIP "optimally" solved a model that simply left them out. | `optimization.log`: `S1-C1: phase … に有効な Y 変数がありません` |
| 5 | The objective value is huge (**2.1 × 10⁷**), i.e. soft penalties (weights 100 000 and 10 000) are dominating — the plan is heavily short of capacity somewhere. Secondary hint, not the cause. | `optimization.log` weights; `scip_S1.log` |

**The 15 phases:** `e1p2, e9p2, e9p3, e10p3, e11p2, e11p3, e14p3, e16p2, e19p2, e20p2, e20p3, e20p4, e24p2, e24p3, e26p4`.

Other warnings in the same log (not proven to matter): 1057 "existing plan input diagnostics" (`work_date_out_of_calendar`, mostly
assignments on weekends); about 46 % of the existing assignments use a worker whose `skill_map` lacks that operation.

## 2. What I checked in the inputs and ruled out

I compared the 15 failing phases with the other 63 on the input data. None of these separates them cleanly, so **I am not claiming any of them as the cause**:

| Hypothesis | Result |
|---|---|
| Phase window too short for its workload | 55 phases are "too tight" in the same way, only 6 of them fail |
| Minimum duration (8 workers × 10 h) longer than the window | 15 of 15 failing phases are *fine* by this test |
| Worker lacks the skill (existing assignments) | equally common in failing (46 %) and passing (47 %) phases |
| Fixed vs Flexible flag | **not clean, but the best lead — see below** |
| Hours/day, weekend work, double booking | identical in failing and passing phases |
| Fab / region / customer unavailable dates | all empty |

### The best lead (a strong correlation, **not** a proven cause)

Looking at the existing assignment rows (`assignment_list`) of the 78 phases:

| Phases with… | count | of which fail |
|---|---|---|
| no existing assignment rows | 38 | **0** |
| only `Flexible` rows | 19 | **0** |
| at least one **`Fixed`** row | 21 | **15** (the other 6 pass: `e10p2, e14p2, e16p3, e19p3, e26p2, e26p3`) |

So a phase can only fail if it contains a `Fixed` (already committed) assignment, and 15 of the 21 such phases fail. Plausible reading — **my inference, not confirmed**: a `Fixed` row pins part of a phase, and for these 15 phases the pinned part leaves no consistent way to place the remaining work, so no start time can be created. The 6 survivors are mostly the first/last phase next to a failing neighbour, which fits "the pinned part must line up with its neighbours".

Test it directly: variant **`v1_phase_fixed_to_flexible`** (below) changes only those `Fixed` rows to `Flexible`. If the 15 phases then get a start variable, this is confirmed.

## 3. Sigma-i's explanation (`readme.txt`) and what it means for your data

Sigma-i's note (translated): each start-date candidate **Y** = *start date × work mode (people, hours/day, days needed)*. The days are
**consecutive working days** from the start date (weekends not counted). A candidate is kept only if it **contains every existing `Fixed` day**
of that phase, **and the `Fixed` worker must also work all the additional days of the run** — if that worker is on leave or already
busy with other work on one of those days, the candidate is dropped. Another worker **cannot** cover just that one day. If *all*
candidates are dropped, the phase gets no Y and is left unassigned. (The log prints only the first 5 such phases; the full list of 15 is in section 1.)

So the question for each failing phase is: *can its `Fixed` worker(s) work a run of consecutive working days that is long enough for one of the
allowed modes, including all their fixed days?*

### Verified against your files (6 phases — exact reasons)

Checked per phase from `Schedule.yaml` + `EnvConfig.yaml` (worker leave days and **all** other assignment rows, including `misc_other_*`):

| Phase | Fixed worker(s) | Fixed working days | Why the run cannot be extended |
|---|---|---|---|
| `e9p2` | `w189` | 10/13 – 11/03 (16 d) | day before (10/09): **on leave**; next day (11/04): **busy with `e9p3`** (also fixed). Needs 22–26 days (1520 h with 6–7 workers); only 17 free |
| `e9p3` | `w189` | 11/04 – 11/10 (5 d) | before: busy with `e9p2`; after (11/11): busy with **`misc_other_1006`**. Needs 18–24 days; only 5 free |
| `e10p3` | `w168` | 11/16 – 11/19 (4 d) | before: busy with `e10p2`; after (11/20): busy with **`misc_other_1006`**. Needs ~10 days (600 h, 6 workers); only 4 |
| `e20p3` | `w039` | 11/26 – 12/10 (11 d) | before: busy with `e20p2`; after (12/11): busy with `e20p4`. Run length is stuck at exactly 11 days and no allowed mode gives 11 |
| `e26p4` | `w256`, `w270` | 09/21 – 09/22 (2 d) | both busy before with `e26p3`; `w270` busy again 09/30 (`misc_other_1093`). Needs ~30 days (600 h, 1–2 workers) |
| `e1p2` | `w263`, `w271`, `w274` | 09/01 – 10/01 (23 d, from `w274`) | **all three** must work the whole run, but `w263`/`w271` are busy with `misc_other_1093` on 09/07 – 09/09 … inside it |

Pattern: **the same person is `Fixed` on neighbouring phases back-to-back** (`w189`: `e9p2 → e9p3`, `e16p2 → e16p3`; `w039`: `e20p2 → p3 → p4 → e19p2 → p3`;
`w031`: `e24p2 → p3`; `w236`: `e11p2 → p3`; `w142/446/496`: `e14p2 → p3`) and often also fixed on `misc_other_*` work. Each phase's run is then
boxed in by the neighbouring phase or other work, so its length cannot change, and no allowed mode matches it.

### Not explained yet (9 phases)

`e11p2, e11p3, e14p3, e16p2, e19p2, e20p2, e20p4, e24p2, e24p3` have the same *kind* of neighbours (same worker fixed in the next/previous phase), but my
check finds room to extend their run on at least one side, so I **cannot prove** their reason. They may depend on how Sigma-i computes the allowed period (CPM:
predecessor/successor chain and day counts) or on the exact mode list, which I cannot see. The experiment below separates the two groups.

## 3b. What to change in the input (options, in order of effort)

For the phases above, in `Schedule.yaml` → `assignment_list`:

1. **Make the rows `Flexible` instead of `Fixed`** for those phases (simplest; the optimiser then picks the workers). This is exactly variant `v1` / `v6`.
2. **Fix a whole crew, not one person for the whole run:** keep `Fixed` only if that worker is genuinely free on every day of the run, and add the missing `Fixed` days so the fixed span equals a possible run length.
3. **Remove the conflict:** move or shorten the `misc_other_*` task (e.g. `misc_other_1006` for `w189`/`w168`, `misc_other_1093` for `w263`/`w271`/`w254`) or the leave day (`w189` on 10/09) that sits right next to the phase.
4. **Don't pin the same worker to back-to-back phases** unless you pin the full phase length of each.

Which one is right is a business decision (are these assignments really committed?). Because Sigma-i cannot swap one person for a single day, partial pins that end where another assignment begins always produce this problem.

## 4. The experiment kit (already built and tested here against a fake Sigma-I)

`tools/sigmai_diagnose/`

| File | What it does |
|---|---|
| `summarize_run.py` | prints a 10-line summary of one job folder (status, SCIP result, the phases without a start variable, integrity result) |
| `make_variants.py` | makes edited copies of `Schedule.yaml` (only `assignment_list` is changed; `EnvConfig.yaml` stays identical) |
| `variants_20261013_01/` | **already generated** from your inputs (7 variants, ~17 MB) |
| `run_variants.sh` | runs the variants on the machine where Sigma-I runs and collects the logs + one summary file |

The seven variants:

| Variant | Change | If the 15 phases are then fine, it means… |
|---|---|---|
| `v0_baseline` | none | (control) must show the same 15 phases — if not, the failure is not reproducible |
| `v1_phase_fixed_to_flexible` | the 47 `Fixed` rows inside workflow phases become `Flexible` | the *Fixed* status of those rows is the trigger (**most likely — run this one right after v0**) |
| `v2_no_phase_assignments` | delete the 269 existing assignment rows of workflow phases (keep "other work") | the existing assignments of phases are the trigger |
| `v3_no_misc_assignments` | delete the 5424 "other work" (`misc_*`) rows | other work blocking the workers is the trigger |
| `v4_drop_failing_phase_rows` | delete only the rows of the 15 failing phases | the problem is limited to those phases' rows |
| `v6_flexible_only_blocked_phases` | `Fixed` → `Flexible` only in the 6 verified phases (`e1p2, e9p2, e9p3, e10p3, e20p3, e26p4`) | those 6 lose their "no valid Y" warning, while the other 9 failing phases still warn → the 9 have a different reason |
| `v5_no_assignments_at_all` | delete every assignment row | sanity check: with no existing plan it should solve; if it does not, the problem is not the existing plan at all |

## 5. Steps (on the other PC, where Sigma-I runs)

Each run takes about **1 min to build the model + the Stage 1 limit (300 s) + a little** — a few minutes, not 2 hours.

1. **Copy** the folder `tools/sigmai_diagnose/` (with `variants_20261013_01/`) to the other PC (USB / OneDrive).
2. **Start Sigma-I** as in the delivery README: image loaded, the one `.whl` in `wheels/`, then
   ```bash
   cd screenas-workerassign-api
   docker compose --env-file .env.server up -d
   source .env.client && curl "$API_BASE_URL/health"      # → {"status":"ok",...}
   ```
   Wait until the worker log shows `ready` (first start installs the wheel).
3. **Run the control first** to confirm the failure reproduces:
   ```bash
   cd sigmai_diagnose
   bash run_variants.sh ./variants_20261013_01 /path/to/screenas-workerassign-api v0_baseline
   ```
   (`/path/to/screenas-workerassign-api` = the folder that contains `docker-compose.yml`, `.env.client` and `data/`.)
   Expect: `15 phases with NO valid start variable` — the same list as in section 1.
4. **Run the rest:**
   ```bash
   bash run_variants.sh ./variants_20261013_01 /path/to/screenas-workerassign-api
   ```
   Options: `STAGE1_TIMEOUT=600 bash run_variants.sh …` to give SCIP more time; `MAX_WAIT=3600` if a run is slow.
5. **Collect:** `variants_20261013_01/results/ALL_SUMMARIES.txt` (small text file). Paste its content into the chat, and keep the `results/` folder (logs of each run, no large files).

### What to look at in each summary

```
phases with NO valid start variable (S1-C1) : 15 -> ...      ← the number that matters
integrity check: unassigned phases after solve = 15
SCIP S1: ... final primal first seen at 101 s
```
`0` phases and `status=finished` = that variant works.

### Reading the combination of results

| v0 | v1 | v2 | v3 | v4 | v5 | Conclusion |
|---|---|---|---|---|---|---|
| 15 | **0** | | | | | the *Fixed* flag on phase rows causes it → set those rows to `Flexible` (or complete/remove them) in the real input |
| 15 | 15 | **0** | 15 | 0 | 0 | the existing assignments of workflow phases cause it → next, test subsets of the 15 phases (see 6) |
| 15 | 15 | 15 | **0** | | 0 | the "other work" rows (misc) cause it |
| 15 | 15 | 15 | 15 | 15 | **0** | the failure needs the combination of both kinds of rows |
| 15 | 15 | 15 | 15 | 15 | **15** | **not the existing plan** — the cause is in the phases/dates/workers themselves; tell me, I will build the next variants (plan range, removing workflows, skill coverage) |
| ≠15 | | | | | | not reproducible → differences in the environment/version; send the summary |

**Reading `v6`:** it should show exactly the 9 not-yet-explained phases (`e11p2, e11p3, e14p3, e16p2, e19p2, e20p2, e20p4, e24p2, e24p3`) and none of the 6 verified ones. If more than 9 remain, one of my 6 is wrong; if fewer, some of the 9 were only knock-on effects of the 6 — send me the summary either way.

## 6. Narrowing down once a variant works

To find **which phase / rows** matter, regenerate with a subset (pick a new output folder):
```bash
python make_variants.py <inputs_dir> variants_sub1 --bad-phases e20p2,e20p3,e20p4
```
and run only `v4_drop_failing_phase_rows`. Halve the list each time (like a binary search) until one phase or one workflow remains. Then
look at that phase's rows in `assignment_list` (worker, dates, Fixed/Flexible) and compare them with a working phase of the same shape.

## 7. Quick wins that don't need the root cause

- **Don't use 7200 s for Stage 1.** In this run the answer existed at 101 s. For diagnosis use 300 s; for real runs 600–1800 s is a safer start (the right value depends on the data; check `final primal first seen at … s` in the summary).
- Add this to the Sigma-i report: *"Phases listed in `S1-C1 … 有効な Y 変数がありません` are left unassigned and the job then ends as `S1_INVALID` after burning the whole time limit. Could Stage 1 stop early (fail fast) when a phase has no start variable?"* — that alone would have saved the 2 hours.
- Ask Sigma-i directly what makes a phase get no valid Y (the 15 phases above are the concrete example) — I can attach the table from section 2.

## 8. What to send Sigma-i

`ALL_SUMMARIES.txt`, the `v0_baseline` log folder, and the list of 15 phases. Do **not** send `result_context.pkl` (19 MB, internal) unless they ask.
