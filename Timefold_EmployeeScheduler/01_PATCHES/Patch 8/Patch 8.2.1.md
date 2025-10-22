# Patch 8.2.1 — **Weekend & Unavailable-Date Aware Scheduling** (Java/Timefold)

> This patch extends **8.1.1** by adding **calendar blackouts** (weekends + per-entity unavailable dates) to both Pass 1 (block sizing) and Pass 2 (people assignment). On any blacked-out day, a block’s **effective hours = 0**. Assignments are also prevented on days when a worker is unavailable.

---

## 🎯 What’s new (high level)

- **Calendar model**: Weekends and multiple “unavailable” sources (fab, region, customer company, worker company, individual worker).
    
- **Pass 1 production math respects calendars**: `produced()` and `autoHours()` now multiply by **working-day count**, not raw day count.
    
- **Pass 2 feasibility respects calendars**:
    
    - Seat-days are not created for blacked-out dates (so they cannot be staffed).
        
    - Hard constraint blocks assigning a worker on a day they’re marked **unavailable**.
        
- **Capacity-by-op only on working days**: Daily operator head-count check ignores blacked-out dates.
    
- **Exporter signature**: `ExportSchedule.overwriteScheduleWithAssignments(..., envPath)` (to keep calendar context if needed).
    

---

## 🔧 Key code additions (by component)

### 1) Calendar primitives

- **Types & maps**
    
    - `Calendars CAL` with: `weekends`, `fabOff`, `regionOff`, `customerOff`, `workerCompanyOff`, `workerOffByWid`, and crosswalks: `fabToRegion`, `fabToCustomer`.
        
- **Builders**
    
    - `buildCalendars(envPath, planStart, planEnd)` populates all sets from **EnvConfig.yaml** plus computed **weekends** inside plan range.
        
    - `dayIdFromDate(...)` converts `yyyy/MM/dd` (or `yyyy-MM-dd`) into day indices.
        
- **Queries**
    
    - `isWorkingDay(dayId, fabId)` checks weekend + fab → region → customer blackouts.
        
    - `workingDaysCount(startDay, dayCount, fabId)` counts only **working** days in the interval.
        

### 2) Pass 1: block math with blackouts

- **Produced hours**
    
    - `produced(b)` now:  
        `H * autoHours(b) * workingDaysCount(b.startDay, b.days, b.factory)`  
        (if `startDay`/`days` unset → 0)
        
- **Hours selection**
    
    - `autoHours(b)` computes feasibility against **working day count**, thus:
        
        - honors **underfill** relative to real working capacity,
            
        - still caps **overfill ≤ one extra day** using the same rule, but with _effective_ days.
            
- **Capacity constraint**  
    `dailyHeadCapacityByOp`: only counts heads on **working days**.
    

### 3) Pass 2: seat-days and worker availability

- **Seat-day generation**
    
    - `expandToSeats(...)`: skips creating seat-days for blacked-out dates (weekends/unavailable).
        
- **New hard constraint**
    
    - `employeeAvailableOnSeatDays`: prevents assignment if worker’s **personal unavailable** set contains that day.
        
- **Unchanged hard rules** still apply (skill eligibility, 12h/day cap, one factory/person/day, ≥1 manager/block).
    
- **Unchanged soft rules** (same-company cohesion, skill variety, block avg skill, total hour balance).
    

---

## 🧭 YAML fields used (no contract change)

From `EnvConfig.yaml` (under `environment`):

- `fab_list[].unavailable_dates: [ "2025/10/21", ... ]`
    
- `region_list[].unavailable_dates: [...]`
    
- `customer_company_list[].unavailable_dates: [...]`
    
- `worker_company_list[].unavailable_dates: [...]`
    
- `worker_list[].unavailable_dates: [...]`
    
    - Also used for **worker-specific** availability in Pass 2.
        

> Dates may be `"yyyy/MM/dd"` or `"yyyy-MM-dd"`; both are accepted.

No change to `Schedule.yaml` schema. The blackout logic is derived from **EnvConfig** + plan range.

---

## ⚙️ Constraint deltas (summary)

|Area|Before (8.1.1)|Now (8.2.1)|
|---|---|---|
|Pass 1 – production|Days = `b.days`|Days = **working days only** (weekends/unavailable → 0 contribution)|
|Pass 1 – over/underfill|vs. `H * h * D` (raw)|vs. `H * h * D_working`|
|Pass 1 – daily capacity by op|All active days|**Working days only**|
|Pass 2 – seat-days|All calendar days in block|**No seat-days on blackouts**|
|Pass 2 – worker availability|N/A|**Hard**: cannot assign if worker marked unavailable on that day|

---

## 📈 Expected impact

- **More realistic block sizing**: blocks spanning weekends/holidays will auto-increase `days`/`heads`/**or** pick larger `hours` tier to hit required effort without cheating through blacked-out days.
    
- **Fewer infeasible assignments**: Pass 2 never tries to staff on blacked-out days; worker private time off is respected.
    
- **Stable performance**: Complexity increase is minimal (set lookups). Typical runtime near 8.1.1.
    

---

## 🧪 Quick sanity checklist

-  Weekend dates inside the plan range produce **no seat-days** and **no capacity accumulation**.
    
-  `unavailable_dates` at fab/region/customer cascade as expected.
    
-  Workers with `unavailable_dates` cannot be assigned on those days (hard violation if attempted).
    
-  Underfill/overfill constraints reflect **working** days, not raw span.
    
-  Existing soft-score behavior unchanged aside from ripple effects of true working capacity.
    

---

## 🧰 How to run

`# Example (Windows path form kept from earlier notes) mvn -q exec:java -D"exec.args=EnvConfig.yaml Schedule.yaml"`

- The solver will:
    
    1. Build calendars from **EnvConfig**,
        
    2. Solve Pass 1 with blackout-aware production,
        
    3. Expand seats (skipping blackouts),
        
    4. Solve Pass 2 with worker-availability enforcement,
        
    5. Export assignments back to `Schedule.yaml` (exporter now takes `envPath` too).
        

---

## 🗂️ Notable code touchpoints

- `buildCalendars`, `isWorkingDay`, `workingDaysCount`, `dayIdFromDate`
    
- `produced`, `autoHours` (Pass 1 math)
    
- `Pass1Constraints.dailyHeadCapacityByOp` (working-day filter)
    
- `expandToSeats` (skip blackouts)
    
- `Pass2Constraints.employeeAvailableOnSeatDays` (new hard rule)
    
- `ExportSchedule.overwriteScheduleWithAssignments(finalPlan, planStart, schedPath, envPath)`
    

---

## ⚠️ Notes & pitfalls

- If **no** `unavailable_dates` are configured, weekends are still treated as **non-working** by default.
    
- Make sure `fab_list[].id`, `region_list[].id`, `customer_company_list[].id`, and `worker_company_list[].id` match the references used in your schedule/env so cascade mapping works (`fabToRegion`, `fabToCustomer`).
    
- Use consistent calendars for multi-fab modules; capacity checks are per-op **per working day**.
    

---

## 🔮 Next ideas

- Optional **“work Saturday”** toggle per fab/region/customer to treat Saturdays as working.
    
- **Per-operation** blackout overrides (e.g., test ops allowed on weekends).
    
- Snapshot diagnostics (like 8.1.2) with blackout overlays for Gantt visualization.