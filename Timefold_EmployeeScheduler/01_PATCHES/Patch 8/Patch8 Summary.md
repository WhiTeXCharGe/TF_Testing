# Employee Scheduler (Timefold) – README

> A single‑pass workforce scheduling demo built on **Timefold 1.27.x** that reads **EnvConfig.yaml** and **Schedule.yaml**, creates a feasible plan with rich constraints (windows, staffing, skills, calendars, travel, hours caps, etc.), and writes assignments back. Includes a **Python score mirror** to re‑compute the Timefold score offline.

    

---

## What is Timefold?

[Timefold](https://timefold.ai/) is a constraint solving toolkit (the successor of OptaPlanner ideas) that turns hard optimization problems—like employee rostering or production planning—into an easy‑to‑use, java‑first modeling problem. You describe a **PlanningSolution**, add **PlanningEntities** with **PlanningVariables**, encode business rules as **constraints**, and let the solver search for a high‑quality plan.

This repo demonstrates a **single‑pass** scheduler that assigns workers to operational blocks across days while respecting windows, skills, capacity, travel rules, and more.

---

## Project goals

- Translate a structured plan from YAML (**EnvConfig.yaml** + **Schedule.yaml**) into a solvable model.
    
- Enforce **feasibility**: window end, phase ordering, staffing, head‑capacity per operation/day, personal/factory/region/customer off days, travel gaps, max consecutive days in a region, per‑day hour caps, pinning, etc.
    
- Encourage **quality**: prefer earlier starts, 8‑hour days, smaller hours, balanced skills, balanced hours across employees, and company pair affinity.
    
- Be **transparent**: the same constraints used by the solver are re‑implemented in a **Python score mirror** so you can inspect scores without running the solver.
    

---

## Domain model

### PlanningSolution

`EmployeeSchedule.SinglePassPlan`

- `days: List<DaySlot>` – horizon days.
    
- `employees: List<EmployeeFact>` – includes a sentinel UNASSIGNED id=0.
    
- `blocks: List<BlockDecision>` – one block per operation window after subtracting fixed hours.
    
- `seats: List<CrewSeat>` – crew slots per block (max heads), plus pinned seats for fixed assignments.
    
- `score: HardMediumSoftScore` – Timefold score.
    

### PlanningEntities

1. **BlockDecision**
    
    - **Variables**: `startDay`, `days`, `hours`.
        
    - **Value ranges**:
        
        - `vrStartWithinWindow`: `[windowStart, windowEnd]` via `ValueRangeFactory.createIntValueRange`.
            
        - `vrDaysWithinWindow`: `1..(windowEnd-windowStart+1)`.
            
        - `vrAllowedHours`: discrete set (e.g., {8,10,12}) via `ListValueRange`.
            
    - **Strength comparators**:
        
        - `StartDayStrength` → earlier dates first.
            
        - `HoursStrength` → smaller hours first.
            
    - **Helpers**: `chosenHours()` picks the selected hours (or smallest allowed as default bias).
        
2. **CrewSeat**
    
    - Represents a seat inside a block; `seatIndex=0` typically requires a **manager**.
        
    - **Variable**: `employee` (from `eligibleEmployeesForSeat()`), strict manager gating for the first seat.
        
    - **Pinned seats**: created from fixed assignments; may lock `wid`/date span/hours.
        

### Problem facts

- **DaySlot** – day id + date.
    
- **EmployeeFact** – worker skills, company, isManager flag, etc.
    

### Helpers

- OP capacities/skill averages computed from the environment.
    
- Calendar resolver for weekends, fab/region/customer/company off, personal off, region travel, and stay rules.
    

---

## Core Timefold concepts used

- **PlanningSolution** – the aggregate of all facts/entities & the score.
    
- **PlanningEntity** – objects whose fields (variables) the solver changes to improve the plan.
    
- **PlanningVariable + ValueRangeProvider** – the search space definition for each variable.
    
- **ConstraintProvider** – functional definition of constraints using **Constraint Streams**.
    
- **Joiners** – correlate stream items (e.g., block A must end before block B starts).
    
- **Score** – `HardMediumSoftScore` used here as `hard/medium/soft`. (We only use **hard** and **soft**.)
    
- **Termination** – staged solve with a best‑score limit in stage 1 and time/unimproved limits in stage 2.
    

---

## Constraints (hard & soft)

Below are the constraints implemented in `SinglePassConstraints`.

### Hard constraints (feasibility)

- **block-end-within-window** (`endWithinWindow`) – `startDay + days - 1 ≤ windowEnd`.
    
- **block-hours-in-allowed** (`hoursValueAllowed`) – chosen hours must be in allowed set.
    
- **phase-order** (`phaseOrder`) – `phase N` must finish before `phase N+1` starts.
    
- **block-no-underfill** (`noUnderfillByBlock`) – staffed×hours×workingDays ≥ requiredHours.
    
- **block-overfill-at-most-one-day** (`overfillAtMostOneDayByBlock`) – at most one extra day of production beyond required.
    
- **daily-head-capacity-by-op** – per `(day, op)` heads cannot exceed available skilled workers.
    
- **seat-worker-available-all-days** – assigned worker must not be on personal off; factory/weekend off is accounted separately.
    
- **seat-pinned-respected** – pinned seat must keep its worker.
    
- **seat-one-factory-per-emp-day** – an employee cannot be in two factories on the same day.
    
- **seat-daily-cap-12h** – an employee’s total hours per day ≤ 12.
    
- **emp-region-transit-gap** – require travel gap days between different regions, based on `transite_day_map`.
    
- **emp-region-stay-max-on** – cap consecutive on‑days in a region; requires breaks of `stay_off_interval`.
    

> Also available (switchable) guards:
> 
> - **block-within-window** (`withinWindow`) – full start/length guard (we commonly rely on `endWithinWindow`).
>     
> - **block-days-window-length** (`daysWithinWindowLen`) – prevent `days` from exceeding window length.
>     

### Soft constraints (quality)

- **soft-hours-near-8** – weight 3000 × |hours − 8|.
    
- **soft-smaller-hours** – weight 40 × hours.
    
- **soft-earlier-start** – weight 1 × startDay.
    
- **soft-same-company-pairs** – reward small bonus when co‑workers in a block share company.
    
- **soft-encourage-skill-variety** – penalize equal skill levels on the same op within a block.
    
- **soft-balance-block-avg-skill** – draw block average toward global op average.
    
- **soft-balance-total-hours** – pull per‑employee daily totals toward target average.
    

> **Weights** are easy to tune; see constants in `SinglePassConstraints`.

---

## Calendars & availability

`Calendars` composes off‑day logic from multiple layers:

- **Weekends** (Saturday/Sunday).
    
- **Factory off** (`fab_list.unavailable_dates`).
    
- **Region off** (`region_list.unavailable_dates`).
    
- **Customer off** (`customer_company_list.unavailable_dates`).
    
- **Worker‑company off** (`worker_company_list.unavailable_dates`).
    
- **Personal off** (`worker_list[].unavailable_dates`).
    
- **Region travel** (`transite_day_map`), **stay caps** (`max_stay_on`) and **breaks** (`stay_off_interval`).
    

Working‑day checks are used by production and seat/day joins so that production only counts on true working days.

---

## Parsing YAML inputs

- **EnvConfig.yaml** (environment): workflow phases/operations (allowed hours, min/max heads), workers (manager flag, skills, company, personal off), factories/regions/customers/off days, region travel rules and stay limits.
    
- **Schedule.yaml** (plan): horizon dates, per‑module phase windows and operation workloads (days), and **assignment_list** for pinned/fixed work (with `plan_flexibility: fixed`). Fixed hours are subtracted before creating a `BlockDecision`, and also materialized as **pinned CrewSeat** rows to preserve who/when.
    

> Baseline production per workload day uses 8h by default (or 4h if the op’s only allowed value is 4h).

---

## Build & run

### Prereqs

- Java 17+
    
- Maven 3.9+
    

### Build

```bash
mvn -DskipTests clean package
```

### Run the solver (two stages)

```bash
# Windows PowerShell
mvn -q exec:java -D"exec.args=src\main\resource\EnvConfig.yaml src\main\resource\Schedule.yaml"

# or (Linux/macOS)
mvn -q exec:java -Dexec.args="src/main/resource/EnvConfig.yaml src/main/resource/Schedule.yaml"
```

The app will print Stage 1 and Stage 2 durations and final score. After solving, it calls:

```java
ExportSchedule.overwriteScheduleWithAssignments(plan, planStart, schedPath, envPath);
```

so your **Schedule.yaml** is updated with assignments.

### CLI arguments

```
<EnvConfig.yaml> <Schedule.yaml>
```

If omitted, it defaults to `EnvConfig.yaml`, `Schedule.yaml` in working dir.

---

## Python score mirror

A small utility to **recompute the score from YAML only**, mirroring Timefold constraints. Useful for CI checks, diffs, or quick validation without running the solver.

### Run

**PowerShell (same folder):**

```powershell
python .\score_schedule.py .\EnvConfig.yaml .\Schedule.yaml `
  --on endWithinWindow,hoursValueAllowed,phaseOrder,noUnderfillByBlock,overfillAtMostOneDayByBlock, `
      dailyHeadCapacityByOp,employeeAvailableAllDays,pinnedRespected,oneFactoryPerEmpPerDay,dailyCap12h, `
      regionTransitGap,regionStayMaxOn,preferHoursNear8,preferSmallerHours,preferEarlierStart, `
      softSameCompanyPairs,softEncourageSkillVariety,softBalanceBlockAvgSkill,softBalanceTotalHours `
  --also-available withinWindow,daysWithinWindowLen
```

**Bash/WSL:**

```bash
python ./score_schedule.py ./EnvConfig.yaml ./Schedule.yaml \
  --on endWithinWindow,hoursValueAllowed,phaseOrder,noUnderfillByBlock,overfillAtMostOneDayByBlock,\
      dailyHeadCapacityByOp,employeeAvailableAllDays,pinnedRespected,oneFactoryPerEmpPerDay,dailyCap12h,\
      regionTransitGap,regionStayMaxOn,preferHoursNear8,preferSmallerHours,preferEarlierStart,\
      softSameCompanyPairs,softEncourageSkillVariety,softBalanceBlockAvgSkill,softBalanceTotalHours \
  --also-available withinWindow,daysWithinWindowLen
```

### Notes

- `--on` toggles which constraints are applied. You can switch guards like `withinWindow` on/off even if Java enforces them via value ranges.
    
- Output format: `HARD=<int> SOFT=<int>` plus per‑constraint breakdown if run with `--verbose` (if you add it).
    

---

## File layout

```
├─ src/main/java/com/yourorg/scheduler/
│  ├─ EmployeeSchedule.java        # main model, constraints, run stages
│  ├─ ExportSchedule.java          # writes assignments back (called by main)
│  └─ ...
├─ src/main/resource/
│  ├─ EnvConfig.yaml
│  └─ Schedule.yaml
└─ tools/
   └─ score/score_schedule.py      # Python score mirror (optional)
```

---

## Tuning, tips, and FAQs

**Why single pass?**

- It’s fast and simpler to reason about. If you later need 2‑pass (e.g., coarse then polish per‑op), the entities split naturally.
    

**Where to change weights?**

- In `SinglePassConstraints` constants: `PREF_HOURS_WEIGHT`, `SMALLER_HOURS_W`, etc.
    

**How do hours get chosen?**

- The `hours` variable uses a discrete range and a strength comparator to bias smaller values; a soft prefers 8h.
    

**What if there’s no manager candidate?**

- The value range for a manager seat is strictly manager‑only (no UNASSIGNED fallback). You’ll get a clear exception.
    

**How are fixed assignments handled?**

- Their hours are subtracted from requiredHours; they also create a pinned `CrewSeat` so constraints (daily cap, factory uniqueness, travel, etc.) still apply consistently.
    

**Travel/stay rules?**

- `transite_day_map` defines minimal day gaps between regions. `max_stay_on` + `stay_off_interval` cap long continuous stays per region.
    

**Common pitfalls**

- Phase IDs must be numeric like `p1`, `p2` to compute ordering.
    
- If an operation lacks allowed hours in EnvConfig, parsing fails.
    
- Personal off dates in `assignment_list` are still honored via seat validations.
    

---

## License

MIT (or your preferred license).