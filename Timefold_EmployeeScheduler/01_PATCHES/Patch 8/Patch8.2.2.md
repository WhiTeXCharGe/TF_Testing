# v8.2.2 — Selective Ramp, Fixed Assignments, and Snapshots (Java / Timefold)

> Command:
> 
> `mvn -q exec:java -D"exec.args=src/main/resource/EnvConfig.yaml src/main/resource/Schedule.yaml"`

## 1) Purpose and High-Level Behavior

This version keeps the 2-pass architecture:

- **Pass 1 (Blocks):** Choose _when_ each operation runs (start day, duration in days, heads per day, and hours per worker) **inside its time window**, while respecting hard feasibility (no underfill, phase order, head capacity per operation per day, etc.). It uses a **selective ramp loop**: only blocks that violate hard constraints “tier-up” their allowed hours; non-violators keep their solved choice (frozen).
    
- **Snapshots:** After each Pass 1 iteration, write `Schedule{N}.yaml`. These snapshots **expand both flexible and pinned seats** and, for visualization, assign **employee id=1** to all flexible seats so Gantt views are visible without running Pass 2. Pinned seats keep their real worker.
    
- **Pass 2 (Crew seats):** Assign real workers to the seats created by Pass 1 **plus pinned seats**. Hard rules: eligible skill, one-factory/day, 12h/day cap, ≥1 manager per block, worker personal OFF, and **respect pinned**. Soft rules: company pairing, skill diversity, block average skill near org average, and total hours balance.
    

If Pass 1 ends **with hard=0**, Pass 2 runs (and polishes). If Pass 1 still has hard violations at the loop end, Pass 2 is **skipped** and snapshots remain for diagnosis.

---

## 2) Inputs and Outputs

**Inputs**

- `EnvConfig.yaml`
    
    - Workers (skills, is_manager, worker_company, unavailable dates)
        
    - Workflow (phases → operations; min/max workers; allowed `work_hours` list)
        
    - Fab / Region / Customer blackout calendars
        
    - Worker company off calendars
        
- `Schedule.yaml`
    
    - Global plan range (start_date, end_date)
        
    - Workflow task windows (module/fab/phase start/end dates, per-operation `workload_days`)
        
    - **Fixed assignments** (`assignment_list` with `plan_flexibility: fixed`)
        

**Outputs**

- **Snapshots** `Schedule{1..N}.yaml` after each Pass 1 loop iteration (seats expanded; pinned seats retain real worker; flexible seats use employee id=1 just for viewing).
    
- **Final** overwrite of the original `Schedule.yaml` with **real** Pass 2 assignments (only if Pass 1 reached hard=0).
    

---

## 3) Domain Model (Key Classes)

### Common

- **`DaySlot { id, date }`**: planning horizon discrete day.
    
- **`EmployeeFact { id, wid, name, skills{op->level}, isManager, workerCompany }`**
    
- **`TaskWindow`**: derived from `Schedule.workflow_task_list` + `EnvConfig.workflow_list`.
    
    - `module, factory, phaseId/phaseNum, opId, startDayId, endDayId, allowed (hours), minHeads, maxHeads, workloadDays`
        

### Pass 1 — Block planning

- **`BlockDecision`** _(PlanningEntity)_
    
    - Window: `windowStart..windowEnd`
        
    - Variables: `startDay`, `heads`, `days`
        
    - Parameters: `requiredHours` (already subtracts fixed hours), `allowed` (candidate hours values), `minHeads`, `maxHeads`, `seedHours`, `phaseId/phaseNum`, `opId`, `module`, `factory`.
        
- **`Pass1Plan`** _(PlanningSolution)_: value ranges (`dayIds`, `headOptions`, `dayCountOptions`), facts (`daySlots`), entities (`blocks`), and `score`.
    

### Pass 2 — Crew assignment

- **`CrewSeat`** _(PlanningEntity)_: the “seat” a person will take.
    
    - Derived from a block: `module, factory, phaseId/Num, opId, startDayId, days, hours, seatIndex, seatKey, blockId`
        
    - Planning variable: `employee: EmployeeFact`
        
    - **Pinned** flags for fixed rows: `pinned=true`, `pinnedWid=...`
        
- **`SeatDay`**: an occurrence of a seat on a specific day with `hours` and `factory`.
    
- **`Pass2Plan`** _(PlanningSolution)_: value ranges (`days, employees`), facts (`seatDays`), entities (`seats`), and `score`.
    

---

## 4) Calendars and “Working Day” Semantics

`buildCalendars(...)` composes blackout sets aligned to the current plan range:

- Global **weekends** (Sat/Sun)
    
- **Fab off** + **Region off** + **Customer off** (mapped from fab→region and fab→customer)
    
- **Worker personal off** (by worker)
    
- **Worker company off** (available; currently not enforced as Pass 2 hard rule)
    

`isWorkingDay(dayId, fabId)` returns false on any blackout above.  
This drives two important places:

1. **Pass 1 production math**:  
    `workingDaysCount(start, days, factory)` counts only working days → `produced(b) = heads × autoHours(b) × workingDaysCount(...)`.
    
2. **Pass 1 head capacity by operation per day**:  
    The group-by day only includes **working days**.
    

---

## 5) Fixed Assignments → Pinned Seats

From `Schedule.assignment_list` where `plan_flexibility: fixed`:

- Parse `operation_task` into `module|opId` and `worker` (wid).
    
- Read `work_date_list` per-day hours.
    
- Construct **`CrewSeat` with `pinned=true`**, `pinnedWid=wid`.
    
- Expand **exact** `SeatDay`s using the given daily hours (skip blackout days).
    
- Attach real `EmployeeFact` by `wid`.
    
- These **do not** come from Pass 1 and **must** be respected in Pass 2 (`respectPinnedAssignments` hard constraint).
    

Also, for each `(module|opId)` we **sum fixed hours** and subtract from the Pass 1 demand:

`requiredHours = workloadDays * baselineHours - sumFixedHours(module|opId)`

Baseline hours = 8 except “4-only operations” use 4.

---

## 6) Pass 1 Objective, Hours Auto-Selection, and Constraints

### Auto hours selection

`autoHours(b)` chooses the best hours value from `b.allowed` given:

- Heads `H`, working days `D`, and residual demand `R = requiredHours`.
    
- Prefer the **smallest** hours that **meet or barely exceed** `R` without overshooting by more than **one day worth** of work (`H*h`).
    
- Otherwise, minimize a key tuple that penalizes underfill or deep overfill and favors hours close to 8.
    

This is what lets the loop **freeze** an hours choice per block when it behaves.

### Pass 1 Hard constraints (class `Pass1Constraints`)

- `p1-within-window`: `startDay..end` inside `[windowStart..windowEnd]` and `days≥1`.
    
- `p1-days-within-window-length`: `days` cannot exceed the window length.
    
- `p1-hours-in-allowed`: hours must be in `allowed` (checked via `autoHours`).
    
- `p1-heads-in-minmax`: `minHeads ≤ heads ≤ maxHeads`.
    
- `p1-no-underfill`: `produced(b) ≥ requiredHours`.
    
- `p1-overfill-at-most-one-day`: `(produced - required) ≤ H*h`.
    
- `p1-phase-order`: phase N must end **before** phase N+1 starts for the same module.
    
- `p1-daily-head-capacity-by-op`: sum of heads per operation per working day ≤ **OP_CAPACITY(op)** (count of skilled workers for that op).
    

### Pass 1 Medium/Soft preferences

- MED: `p1-med-penalize-stack-by-op`: discourage stacking multiple blocks of the same op on a day.
    
- SOFT: hours near 8, prefer smaller hours, minimize heads, minimize days, prefer earlier start.
    

> Constants: `PREF_HOURS_WEIGHT=1000`, `SMALLER_HOURS_W=100`, `SMALLER_HEADS_W=10`, `FEWER_DAYS_W=1`, `EARLIER_START_W=1`, `STACK_PAIR_WEIGHT=2`.

---

## 7) Selective Ramp Loop (Pass 1)

**Idea:** Only widen `allowed` hours **for violators**; freeze non-violators to reduce churn and keep good shapes.

### Steps

1. **Seed** (`seedBlocksForTier`) using current **tier** per block:
    
    - `allowed = first k values` of the operation’s `work_hours` (k = tier).
        
    - Initial `heads = minHeads`, `startDay = windowStart`, `days` = minimal safe days so `heads*hours*workingDays` approximates `requiredHours`.
        
    - `requiredHours` already subtracts fixed hours.
        
2. **Solve** Pass 1 to target `0hard/*/*` (30 min spent, 60s unimproved cap).
    
3. **Snapshot** this iteration (`writeScheduleSnapshot`): expand flexible+pinned seats, assign employee id=1 to flexible seats only.
    
4. If **hard=0**:
    
    - **Polish** (20 min / 60s unimproved), **write snapshot** again for the same iteration index, and **STOP**.
        
5. **Detect violators** (`detectHardViolators`):
    
    - Recompute all Pass 1 hard checks + capacity and flag offending block IDs.
        
6. **Next seed**:
    
    - For **non-violators**: **freeze** hours to the chosen `autoHours`, and also carry over `startDay/heads/days`. Set `allowed` to **exactly one value** (the chosen hours).
        
    - For **violators**: **tier-up** (`allowed = first (tier+1) values`). Recalculate seed `days`.
        
7. If **no tier changed** in step 6, stop the loop (keep the best known). Otherwise, repeat from step 2.
    

`perBlockMaxTier` equals the count of allowed hours for the operation (so the loop is bounded).

---

## 8) From Blocks to Seats (and Pinned Seats)

### Flexible seats (`expandToSeats`)

For each solved block:

- Create `headCount` seats (`seatIndex = 0..heads-1`), each with `hours = autoHours(b)`, `startDayId`, and `days`.
    
- For each day, if `isWorkingDay`, add a `SeatDay` with that seat’s hours.
    

### Pinned seats (`expandPinnedSeats`)

- Build `CrewSeat` per fixed row, using explicit per-day hours and real `EmployeeFact` `(pinned=true, pinnedWid)`.
    
- Skip blackout days.
    

Finally, **merge** pinned + flexible seats and seat-days.

---

## 9) Pass 2 — Worker Assignment

**Hard**

- `p2-assigned+eligible-skill`: no unassigned seats; skill level ≥1 for the op.
    
- `p2-one-factory-per-emp-day`: same employee cannot be in 2 fabs on the same day.
    
- `p2-daily-cap-12h`: sum of seat-day hours per employee per day ≤12.
    
- `p2-at-least-one-manager-per-block`: each block must have a manager assigned somewhere.
    
- `p2-worker-unavailable-day`: no assignment on worker **personal off** (company off available but commented as hard).
    
- `p2-hard-respect-pinned-assignments`: pinned seat must keep its `pinnedWid`.
    

**Soft**

- `p2-soft-same-company-pairs`: reward pairs from the same company in a block.
    
- `p2-soft-encourage-skill-variety`: penalize same skill level duplicates within the same block/op (encourage variety).
    
- `p2-soft-balance-block-avg-skill`: keep block average skill near **OP_AVG_SKILL(op)** from the workforce.
    
- `p2-soft-balance-total-hours`: balance total hours per employee around **TARGET_HOURS_PER_EMP** (computed from total demand / number of real employees).
    

> Constants: `COMPANY_PAIR_W=5`, `SKILL_DIVERSITY_W=3`, `SKILL_AVG_W=50`, `DAILY_CAP=12`.

Polish runs if Pass 2 reaches hard=0.

---

## 10) Score and Termination

`buildSolver(...)` creates a solver per pass with:

- **Score director** from the constraint provider class.
    
- **Termination**: `bestScoreLimit` (for Pass 1/2 initial run), `spentLimit` (minutes), and `unimprovedSpentLimit` (seconds).
    
- Pass 1/2 target: `"0hard/*medium/*soft"` then **polish** with no bestScoreLimit (still time-boxed).
    

Helper `hardZero(Score)` checks if string starts with `"0hard"`.

---

## 11) Export and Snapshots

- **Snapshots**: `writeScheduleSnapshot(iter, blocks, cfg)`
    
    - Expand flexible+pinned, force assign employee id=1 only for **flexible** seats, then call:
        
    - `ExportSchedule.overwriteScheduleWithAssignments(snapPlan, planStart, snapshotPath, envPath)`
        
- **Final export** (only if Pass 1 reached hard=0 and Pass 2 ran):  
    `ExportSchedule.overwriteScheduleWithAssignments(finalPass2, planStart, originalSchedulePath, envPath)`
    

---

## 12) Configuration Knobs (Where to Tune)

- **Hours preferences (Pass 1 soft):** `PREF_HOURS_WEIGHT`, `SMALLER_HOURS_W`.
    
- **Resource frugality (Pass 1 soft):** `SMALLER_HEADS_W`, `FEWER_DAYS_W`, `EARLIER_START_W`.
    
- **Stack penalty (medium):** `STACK_PAIR_WEIGHT`.
    
- **Pass 2 soft weights:** `COMPANY_PAIR_W`, `SKILL_DIVERSITY_W`, `SKILL_AVG_W`.
    
- **Time limits:** Pass 1/2 `spentMinutes` and `unimprovedSeconds`.
    
- **Daily cap:** `DAILY_CAP=12`.
    
- **Average skill target:** built from workforce skills per operation (`OP_AVG_SKILL`).
    
- **Operation capacity per day:** derived from how many employees have `skill(op) > 0` (`OP_CAPACITY`).
    

---

## 13) Edge Cases and Notes

- **Operations with only 4h allowed:** baseline demand uses 4 instead of 8 when computing `workloadDays * baseline`.
    
- **Zero demand after fixed subtraction:** the corresponding block is **not created** in Pass 1.
    
- **Blackout semantics:** working day logic affects both production (Pass 1) and seat-day expansion, and head capacity checks skip non-working days.
    
- **Snapshots vs reality:** snapshots assign **employee id=1** to flexible seats _only to visualize shapes_. Real assignments are produced only by Pass 2.
    
- **Pinned seats conflict:** if fixed rows demand violates hard rules (e.g., worker personal off, >12h/day), Pass 2 hard constraints will expose the violation.
    
- **Phase order**: computed by comparing `phaseNum` and dates; if phase N overlaps into phase N+1 start, it’s penalized.
    

---

## 14) Data Flow (Narrative)

1. **Parse Env** → build `opdef`, `employees`, `OP_CAPACITY`, `OP_AVG_SKILL`.
    
2. **Parse Schedule** → plan range, `daySlots`, `windows`, **fixed hours** map, **fixed rows**.
    
3. **Build calendars** for the plan range (weekends + per-entity OFF).
    
4. **Compute total demand** and `TARGET_HOURS_PER_EMP`.
    
5. **Pass 1 selective ramp**:
    
    - Seed by tier → solve → snapshot → if hard=0 then polish+snapshot and stop
        
    - Else, detect violators → freeze good blocks, tier-up violators → next loop
        
6. **Expand seats** from solved blocks and **merge** with **pinned seats**.
    
7. **Pass 2** (only if Pass 1 hard=0) → assign workers → polish.
    
8. **Export** final assignments into original `Schedule.yaml`.
    

---

## 15) Minimal Pseudocode (for the loop)

`perBlockTier := 1 for each window best := null for iter in 1..maxTier:   blocks := seedBlocksForTier(windows, perBlockTier, fixedHoursByKey)   solved := solvePass1(blocks)   snapshot(iter, solved.blocks, pinned + flexible)   best = betterOf(best, solved)   if hardZero(solved.score):     solved = polishPass1(solved)     snapshot(iter, solved.blocks)     break   violators := detectHardViolators(solved.blocks)   changed := false   next := []   for each blockId in order:     if blockId ∉ violators:       freeze hours/start/heads/days from solved; allowed := [chosenHours]     else:       perBlockTier[blockId] := min(perBlockTier[blockId]+1, maxTier[blockId]); changed := changed or tier increased       recompute seed with wider allowed     next.add(block)   if not changed: break # if hardZero(best): expand seats (flex + pinned) → Pass 2 → polish → export; else skip Pass 2`

---

## 16) Testing / Acceptance

- **Pass 1** reaches hard=0 in fewer loops than global ramp; snapshots appear as `Schedule{N}.yaml`.
    
- **Pinned rows**: in snapshots they retain `pinnedWid`; flexible seats are employee id=1 for visibility.
    
- **Fixed hours subtraction** reduces demand as expected; blocks may disappear when demand is fully fixed.
    
- **Working-day skips** (weekends, fab/region/customer off) are visible in production counts and seat-day expansion.
    
- **Pass 2** respects daily 12h cap, one-factory/day, eligibility, manager presence, and pinned workers; softs improve after polish.
    

---

## 17) Glossary

- **Tiered hours**: a prefix of the allowed hours list per operation (e.g., `[4]`, then `[4,8]`, then `[4,8,10]`).
    
- **Freeze/pin (Pass 1)**: keep a solved block’s hours/start/heads/days fixed by restricting `allowed` to the chosen hours.
    
- **Pinned seat (Pass 2)**: a seat created from a fixed assignment row that **must** be assigned to `pinnedWid`.
    
- **OP_CAPACITY(op)**: number of employees with `skill(op)>0` (upper bound on heads per day per op).
    
- **TARGET_HOURS_PER_EMP**: total demand ÷ number of real employees (used for balancing).