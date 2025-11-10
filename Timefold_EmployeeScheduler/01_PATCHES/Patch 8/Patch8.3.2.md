# v8.3.2 — Single-Pass Scheduler (Transit & Max-Stay, Value Ranges, Strength Comparators)

> Build & run
> 
> `mvn -DskipTests clean package mvn -q exec:java -D"exec.args=src/main/resource/EnvConfig.yaml src/main/resource/Schedule.yaml"`

## 1) Why we moved to Single-Pass (from 8.3.1)

- **2-pass limitation:** Transit gaps / region max-stay were enforced only in Pass2 (people), while Pass1 (blocks) could not insert travel buffers or shift windows. This often forced awkward reassignments or hard failures when only time-shaping would fix it.
    
- **Single pass benefit:** Blocks (start, days, hours) and seats (who works) are optimized **together** under one score director. Hard rules (phase order, underfill/overfill bounds, daily head capacity, region transit, max-stay) now pressure both time and assignment variables simultaneously.
    

## 2) Planning Model (entities & value ranges)

### 2.1 `BlockDecision` (entity)

- **Vars**
    
    - `startDay` — range `[windowStart, windowEnd]` via **`ValueRangeFactory.createIntValueRange`** (`vrStartWithinWindow`)
        
    - `days` — range `[1, windowLen]` via **`ValueRangeFactory.createIntValueRange`** (`vrDaysWithinWindow`)
        
    - `hours` — **discrete** allowed list (`8,10,12,...`) via **`ListValueRange`** (`vrAllowedHours`)
        
- **Strength comparators**
    
    - `StartDayStrength`: earlier days stronger → solver tends to anchor blocks early.
        
    - `HoursStrength`: smaller hours stronger → solver gravitates to 8h unless forced otherwise.
        
- **Production math** uses `b.chosenHours()` (the current `hours` if set, else the smallest allowed) and `workingDaysCount(b.startDay, b.days, b.factory)`.
    

### 2.2 `CrewSeat` (entity)

- One seat per potential head; `seatIndex=0` is **manager seat** (`needManager=true`).
    
- **Pinned seats** keep real dates/hours in `pinnedStart/pinnedDays/pinnedHours`, and are **hard-pinned** (`eligibleEmployeesForSeat` collapses to the pinned worker).
    
- **Entity-dependent employee ranges:** `eligibleEmployeesForSeat` returns a **CountableValueRange<EmployeeFact>** built from a **pre-filtered candidate list** (see §4). For manager seats, the range is manager-only (no UNASSIGNED fallback).
    

### 2.3 `SinglePassPlan` (solution)

- Problem facts: `days`, `employees`.
    
- Entities: `blocks`, `seats`.
    
- Score: `HardMediumSoftScore`.
    

## 3) Calendars & YAML Inputs

Loaded in `buildCalendars()` (same keys as 8.3.1, required for transit/max-stay):

- `fab_list[].region`, `customer_company`, `unavailable_dates`
    
- `region_list[].max_stay_on`, `stay_off_interval`, `unavailable_dates`
    
- `transite_day_map[]: {from, to, days}`
    
- `worker_list[].unavailable_dates`
    
- Weekends auto-OFF.  
    **Tip:** If a fab has no `region`, transit/max-stay checks for those seats are skipped.
    

## 4) Candidate Generation (ValueRange inputs)

`fillSeatCandidatesSinglePass(seats, blocks, employees)` computes **per-seat** candidates before solving:

1. **Skill gate:** `skill(e, opId) ≥ 1`.
    
2. **Personal OFF gate:** employee must be available on the **estimated** seat span (`estStart/estDays`, using chosen values if already set, otherwise the whole window).
    
3. **Manager gate:** if `needManager`, keep only managers; throw if empty (explicit visibility of infeasible inputs).
    
4. **Pinned seats:** range collapses to the pinned worker.  
    The **eligible range** is exposed via `@ValueRangeProvider("eligibleEmployeesForSeat")`. Non-manager seats may allow UNASSIGNED as last resort (feasibility pressure remains in hard constraints).
    

## 5) Constraints (high-level)

### 5.1 Block feasibility & flow

- **End within window (H):** `startDay + days − 1 ≤ windowEnd` (`block-end-within-window`).
    
- **Hours in allowed (H):** `chosenHours ∈ allowed`.
    
- **Phase order (H):** per module, `phase k` must finish before `phase k+1` starts.
    
- **No underfill by block (H):** staffedHeads × hours × workingDays ≥ requiredHours (penalize deficit).
    
- **Overfill ≤ 1 day (H):** excess over `staffedHeads × hours` is penalized (prevents wild overstaffing).
    
- **Daily head capacity by op (H):** per (day, op) heads ≤ `OP_CAPACITY[op]` (built from skill>0 workers).
    

### 5.2 Seat / employee hard rules

- **Availability (H):** employee cannot work on personal OFF days (fab OFF is skipped).
    
- **Pinned respected (H):** seat must be filled by pinned worker.
    
- **One factory per employee per day (H):** distinct factories for same (emp, day) must be 1.
    
- **Daily 12h cap (H):** sum of hours per (emp, day) ≤ 12 (pinned seats use pinnedHours).
    
- **Region transit gap (H):** for the same employee, if moving from region R1 day **d1** to R2 day **d2**, require `d2 − d1 > transite_day_map[R1→R2]`; otherwise penalize proportional to missing gap.
    
- **Region max-stay (H):** longest consecutive ON-span per (emp, region), where OFF gaps `< stay_off_interval` **do not** reset the streak, must be ≤ `max_stay_on`.
    

### 5.3 Soft preferences

- **Hours near 8 (M):** pull hours toward 8.
    
- **Smaller hours (S):** mild bias to smaller hour choices.
    
- **Earlier start (S).**
    
- **Same-company synergy (S):** reward pairs from same worker company within a block.
    
- **Skill diversity per op/block (S):** gentle penalty for identical skill levels.
    
- **Balance total hours (S):** per employee toward `TARGET_HOURS_PER_EMP` (computed from total required ÷ real employees).
    

## 6) Build Pipeline and Termination

Two stages, both with the same constraint set:

- **Stage 1:** bestScoreLimit `0hard/*/*`, `spent=90m`, `unimproved=300s`.
    
- **Stage 2 (polish):** `spent=60m`, `unimproved=300s`.  
    You can shorten times for smoke tests (e.g., `spent=5m` / `10m`). Hard-zero early exit is allowed by the bestScoreLimit in stage 1.
    

## 7) Input Parsing & Entity Build

### 7.1 Env parsing

`parseEnv()` builds:

- `opdef` (allowed hours, min/max heads, phase numbers),
    
- `employees` (with UNASSIGNED at id=0),
    
- `OP_CAPACITY` from count of employees with `skill>0` per op,
    
- `OP_AVG_SKILL` across skilled workers (used by soft avg balancing).
    

### 7.2 Schedule parsing

`parseSchedule()` builds:

- Plan horizon → `DaySlot[id,date]`.
    
- `TaskWindow` per (module, phase, op) with window `[startDayId..endDayId]`, `workloadDays`.
    
- **Fixed assignments** (`assignment_list` with `plan_flexibility: fixed`): converted to **independent pinned CrewSeats** with exact dates & hours. The fixed hours also reduce each op’s flexible `requiredHours`.
    
- **Phase push:** windows for phase k+1 are nudged to start **after** the latest fixed end of phase k (even outside horizon if present).
    

### 7.3 Entities for single pass

`buildEntitiesSinglePass()` creates:

- `BlockDecision` per `TaskWindow` with `requiredHours = workloadDays × baseline (8 or 4) − fixedHoursOnSameKey`.
    
- `CrewSeat` per potential head (0..maxHeads−1) with `needManager=true` only for `seatIndex=0`.
    
- Plus **pinned seats** created from fixed assignments (blockId = −1).
    

## 8) Testing Scenarios (quick checklist)

1. **Transit basic:** r1→r2 needs 2 days; assign same employee r1 at day 5 and r2 at day 6 → **hard** violation; move r2 to ≥ day 8 → OK.
    
2. **Transit asymmetry:** r1→r2=2, r2→r1=1 behaves independently by direction.
    
3. **Max-stay (K=5, B=2):** 6 consecutive ON days in r1 → **hard** 1; `3ON,1OFF,3ON` (OFF<B) counts continuous 7 → **hard** 2; `3ON,2OFF,3ON` resets → OK.
    
4. **One factory/day:** create two seats same day for same employee in different fabs → **hard**.
    
5. **Daily 12h cap:** two seats of 8h on same day → **hard** 4.
    
6. **Pinned respected:** change pinned worker in input and confirm **hard**.
    
7. **Manager seat:** ensure at least one manager candidate exists; otherwise you get an explicit `IllegalStateException`.
    

## 9) Performance Notes

- **Entity-dependent ranges** slash search space: skill/availability filtering happens **before** solving.
    
- **Strength comparators** guide construction heuristics: earlier `startDay` and smaller `hours` chosen earlier.
    
- **Capacity guard** keeps head counts realistic per op/day (prevents solver from inflating seats just to hit production).
    
- If candidates are too sparse (many UNASSIGNED), consider relaxing manager seat creation (e.g., only when op requires) or adding more managers in env.
    

## 10) Known Behaviors / Design Choices

- **No synthetic “travel tasks”:** Transit is enforced as a **gap** rule; we do not allocate explicit travel entities.
    
- **Fab OFF vs Personal OFF:** Availability rule flags **personal OFF** only; fab/region/customer OFF already removes those days from `workingDaysCount` and coverage checks.
    
- **Hours choice:** `hours` is a real PlanningVariable; softs bias toward 8, but underfill / overfill constraints will push hours up or down as needed.
    
- **Manager exactly one per block:** Implemented by construction (seatIndex 0 is manager) + hard availability; you can add `atLeastOneManagerPerBlock` if you later make manager seats optional.
    

## 11) Code Landmarks

- **Value ranges:** `BlockDecision.vrStartWithinWindow`, `vrDaysWithinWindow`, `vrAllowedHours` (ListValueRange).
    
- **Strength comparators:** `BlockDecision.StartDayStrength`, `BlockDecision.HoursStrength`.
    
- **Seat candidate range:** `CrewSeat.eligibleEmployeesForSeat()` + `fillSeatCandidatesSinglePass(...)`.
    
- **Transit rule:** `SinglePassConstraints#regionTransitGap`.
    
- **Max-stay rule:** `SinglePassConstraints#regionStayMaxOn` (+ helper `maxSegmentSpanWithBreak`).
    
- **Capacity guard:** `dailyHeadCapacityByOp`.
    
- **Production:** `noUnderfillByBlock`, `overfillAtMostOneDayByBlock`.
    
- **One-factory/day & 12h cap:** `oneFactoryPerEmpPerDay`, `dailyCap12h`.
    
- **Parsing / build:** `parseEnv`, `parseSchedule`, `buildEntitiesSinglePass`.
    

## 12) Migration Notes (8.3.1 → 8.3.2)

- Remove Pass1/Pass2 classes; use `SinglePassPlan` and `SinglePassConstraints`.
    
- Keep the same YAML schema; **add** `transite_day_map`, `region_list.max_stay_on`, `stay_off_interval` if you want the new rules.
    
- Exporter API updated to accept `SinglePassPlan` (`ExportSchedule.overwriteScheduleWithAssignments(plan, planStart, schedPath, envPath)`).
    

## 13) Acceptance Criteria

- Hard score reaches `0hard` when feasible (production ≥ required; phase order; capacity; availability; pinned; one-factory/day; 12h cap; transit gaps; max-stay) and softs reflect preferences (hours near 8, earlier, balanced hours, etc.).
    
- When transit/max-stay forces reshaping, the solver is **allowed** to change both time (start/days/hours) and staffing to satisfy constraints.
    

## 14) Troubleshooting

- **“No manager candidates …”** → env lacks a manager with `skill≥1` & availability; add a manager or relax the requirement.
    
- **Hard underfill persists** → raise hours (allowed list), widen windows, add seats (maxHeads), or add skilled workers.
    
- **Transit violations remain** → extend windows or add OFF days to create legal gaps, or reduce `transite_day_map` days.
    
- **Max-stay violations** → lower `requiredHours` or increase `max_stay_on`, or ensure OFF breaks meet `stay_off_interval`.
    
- **Slow solve** → reduce horizon, reduce candidates (skills), or shorten termination for iteration.