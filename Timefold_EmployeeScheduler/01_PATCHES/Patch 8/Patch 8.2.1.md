# Patch 8.2.1 — **Weekend & Unavailable-Date Aware Scheduling + Recalculate Flexible & Fixed** (Java/Timefold)

> This patch extends **8.1.1** by adding **calendar blackouts** (weekends + per-entity unavailable dates) to both Pass 1 (block sizing) and Pass 2 (people assignment), and adds a unified **recalculate** flow for **Fixed** (pinned) and **Flexible** schedules. On any blacked-out day, a block’s **effective hours = 0**. Assignments are also prevented on days when a worker is unavailable. Existing fixed rows in `Schedule.yaml` are re-read, pinned, and **subtracted** from flexible demand before planning.

---

## 🎯 What’s new (high level)

- **Calendar model**: Weekends and multiple “unavailable” sources (fab, region, customer company, worker company, individual worker).
    
- **Pass 1 math respects calendars**: `produced()` and `autoHours()` use **working-day count** (not raw span).
    
- **Pass 2 feasibility respects calendars**: seat-days on blacked-out dates are **not created**; workers cannot be assigned on their **unavailable** dates (hard).
    
- **Capacity-by-op only on working days**; also **includes fixed heads** so flexible seats don’t overbook.
    
- **Recalculate Fixed**: parse `assignment_list` with `plan_flexibility: fixed` → create **pinned** seats & exact seat-days (hours per day).
    
- **Recalculate Flexible**: remaining hours after fixed subtraction are planned by Pass 1/2.
    
- **Exporter signature change** (keep calendars available):  
    `ExportSchedule.overwriteScheduleWithAssignments(Pass2Plan finalPlan, LocalDate planStart, String schedPath, String envPath)`.
    

---

## 🔧 Key code additions (by component)

### 1) Calendar primitives

- **Types & maps**: `Calendars CAL` with `weekends`, `fabOff`, `regionOff`, `customerOff`, `workerCompanyOff`, `workerOffByWid`, and crosswalks `fabToRegion`, `fabToCustomer`.
    
- **Builders**: `buildCalendars(envPath, planStart, planEnd)` loads **EnvConfig.yaml** and generates **weekends** across the horizon.
    
- **Queries**:
    
    - `isWorkingDay(dayId, fabId)` → weekend + fab→region→customer blackouts.
        
    - `workingDaysCount(startDay, dayCount, fabId)` → only **working** days.
        
    - `dayIdFromDate(...)` → `"yyyy/MM/dd"` or `"yyyy-MM-dd"` → day index.
        

### 2) Pass 1: block math with blackouts

- **Produced hours**:  
    `produced(b) = H * autoHours(b) * workingDaysCount(b.startDay, b.days, b.factory)` (0 if start/days unset).
    
- **Hours selection** (`autoHours`) uses **effective working days**, still capping **overfill ≤ one extra day** (vs. `H*h`).
    
- **Capacity** (`dailyHeadCapacityByOp`) counts **flex heads on working days** and **adds fixed heads** (`FixedHeadDay`) to avoid overbooking.
    
- **Stacking penalty** only considers **working** days.
    

### 3) Pass 2: seat-days & availability

- **Seat-day generation**: `expandToSeats(...)` **skips** blacked-out days (weekends/unavailable).
    
- **Hard constraints**:
    
    - `employeeAvailableOnSeatDays` (worker personal unavailable → **hard**).
        
    - `assignedAndSkill`, `dailyCap12h`, `oneFactoryPerEmpDay`, `atLeastOneManagerPerBlock`.
        
    - `respectPinnedAssignments` (for fixed seats).
        
- **Soft constraints** unchanged: company cohesion, skill variety, block average skill, total hour balance.
    

### 4) Recalculate Fixed & Flexible (unified)

- **Parse Fixed**: from `schedule.assignment_list` where `plan_flexibility: fixed`. Supports both `work_date_list` and the typo `work_date_lsit`.
    
- **Pin & seat-days**: `expandPinnedSeats(...)` creates **pinned** `CrewSeat`s (`pinned=true`, `pinnedWid`), with **exact** seat-days/hours (also filtered by `isWorkingDay`).
    
- **Subtract Fixed from demand**: `fixedHoursByKey` (sum of per-day fixed hours) is subtracted from `(module|op)` workload in Pass 1.
    
- **Fixed heads → capacity**: `buildFixedHeadDays(...)` contributes heads to daily op capacity checks so flexible planning respects real headroom.
    
- **Merge**: final seat set = **Pinned + Flexible**; both exported.
    

---

## 🧭 YAML fields used (no contract change)

From `EnvConfig.yaml` (under `environment`):

- `fab_list[].unavailable_dates: [ "2025/10/21", ... ]`
    
- `region_list[].unavailable_dates: [...]`
    
- `customer_company_list[].unavailable_dates: [...]`
    
- `worker_company_list[].unavailable_dates: [...]`
    
- `worker_list[].unavailable_dates: [...]` (used as **worker-specific** availability)
    

From `Schedule.yaml` (for Fixed):

`- worker: w17   operation_task: e16p4o1   start_date: 2025/10/21   end_date: 2025/10/24   plan_flexibility: fixed   work_date_list:     - {date: 2025/10/21, hour: 8}     - {date: 2025/10/23, hour: 6}`

> Dates may be `"yyyy/MM/dd"` or `"yyyy-MM-dd"`. No schema changes required.

---

## ⚙️ Constraint deltas (summary)

|Area|Before (8.1.1)|Now (8.2.1)|
|---|---|---|
|Pass 1 – production|`H*h*D_raw`|`H*h*D_working` (weekends/unavailable → 0)|
|Pass 1 – under/overfill|vs raw days|vs **working** days|
|Pass 1 – daily capacity|Flex only, all active days|**Working days only**, **+ Fixed heads**|
|Pass 2 – seat-days|All calendar days|**No seat-days on blackouts**|
|Pass 2 – worker availability|N/A|**Hard**: cannot assign on worker unavailable day|
|Fixed handling|N/A|Parse, **pin**, **subtract** hours, include heads in capacity|

---

## 📈 Expected impact

- **Realistic block sizing** across weekends/holidays; blocks may lengthen or select larger `hours`.
    
- **Fewer infeasible assignments**: no seat-days on blackouts; worker personal time off respected.
    
- **Stable performance**: set lookups only; similar runtime to 8.1.1.
    

---

## 🧪 Quick sanity checklist

- Weekend/holiday dates ⇒ **no seat-days** and **no capacity accumulation**.
    
- Fab/region/customer blackouts cascade; company/worker unavailable respected.
    
- Fixed rows **reduce** flexible demand and **consume capacity**.
    
- Phase windows shift right if a previous phase has fixed work that ends later.
    
- Exported `assignment_list` only contains **actual working days** (runs grouped), with `work_date_list` exact hours.
    

---

## 🧰 How to run

`mvn -q exec:java -Dexec.args="EnvConfig.yaml Schedule.yaml"`

Flow:

1. Build calendars from **EnvConfig** (weekends + all blackouts).
    
2. **Recalculate Fixed**: parse fixed rows → pin seats & seat-days; compute `fixedHoursByKey` & `FixedHeadDay`.
    
3. **Recalculate Flexible**: Pass 1 plans the remainder (working-day aware) with fixed heads in capacity.
    
4. Expand flexible seats (skip blackouts) and merge with pinned seats.
    
5. Pass 2 assigns (respecting availability, capacity, managers, etc.).
    
6. Export assignments back to `Schedule.yaml` via  
    `ExportSchedule.overwriteScheduleWithAssignments(finalPlan, planStart, schedPath, envPath)`.
    

---

## 🗂️ Notable code touchpoints

- Calendars: `buildCalendars`, `isWorkingDay`, `workingDaysCount`, `dayIdFromDate`
    
- Pass 1 math: `produced`, `autoHours`
    
- Pass 1 capacity: `Pass1Constraints.dailyHeadCapacityByOp` (**working-day filter + fixed heads**)
    
- Seats: `expandToSeats` (**skip blackouts**)
    
- Fixed path: `parseSchedule` → `fixedRows` & `fixedHoursByKey`, `expandPinnedSeats`, `buildFixedHeadDays`
    
- Pass 2 hard rules: `employeeAvailableOnSeatDays`, `respectPinnedAssignments`, `dailyCap12h`, `oneFactoryPerEmpDay`, `assignedAndSkill`, `atLeastOneManagerPerBlock`
    
- Exporter: `ExportSchedule.overwriteScheduleWithAssignments(finalPlan, planStart, schedPath, envPath)`
    

---

## ⚠️ Notes & pitfalls

- If **no** `unavailable_dates` configured, weekends are still **non-working** by default.
    
- Ensure IDs line up across env & schedule: `fab_list[].id`, `region_list[].id`, `customer_company_list[].id`, `worker_company_list[].id`, and worker `id` (wid).
    
- Fixed on a blackout date ⇒ that day is **skipped** (no seat-day); adjust either the fixed row or blackout sets.
    
- Update the call site to the **4-arg exporter**; otherwise calendars may be empty at export time.
    

---

## 🔮 Next ideas

- Per-fab/region/customer **“work Saturday”** toggle.
    
- Per-operation blackout overrides (e.g., allow test ops on weekends).
    
- Snapshot diagnostics (8.1.2 style) with blackout overlays for Gantt.