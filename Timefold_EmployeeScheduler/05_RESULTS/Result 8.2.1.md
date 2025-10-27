## Scope

Patch **8.2.1** adds (A) **unavailable-date awareness** (weekends + YAML blackouts) and (B) a **reschedule-in-range** feature for **flexible** work only. Fixed rows are respected/pinned and subtracted from demand.

---

## A) Unavailable Dates (Weekend & Blackouts)

### Expected behavior

- **Calendars**: Weekends are non-working by default. Blackouts from `EnvConfig.yaml` are honored for:
    
    - `fab_list[].unavailable_dates`
        
    - `region_list[].unavailable_dates`
        
    - `customer_company_list[].unavailable_dates`
        
    - `worker_company_list[].unavailable_dates`
        
    - `worker_list[].unavailable_dates` (personal)
        
- **Pass 1**:
    
    - `produced()` and `autoHours()` use **workingDaysCount** only.
        
    - `dailyHeadCapacityByOp` counts **only working days** and **adds Fixed heads**.
        
- **Pass 2**:
    
    - Seat-days are **not created** on blacked-out days.
        
    - **Hard**: worker cannot be assigned on their personal unavailable day.
        
- **Export**: Only actual working seat-days are written back.
    

### Acceptance checks

- Weekend/holiday dates yield **0** seat-days and no capacity accumulation.
    
- Workers listed unavailable on a given date are **never** assigned that date.
    
- Pass 1 never meets demand by “counting” blacked-out days.
    

---

## B) Reschedule (Flexible Only) in a Date Range

### Goal

Recalculate **only flexible** workload within `[reschedStart, reschedEnd]`, while:

- Preserving all **Fixed** rows (pinned seats & capacity).
    
- Keeping phase/order feasibility and blackout handling.
    

### Expected behavior

- **Inputs**: `reschedStart`, `reschedEnd`.
    
- For each (module, op) window intersecting the range:
    
    - Subtract fixed hours (`fixedHoursByKey`) first.
        
    - Constrain **flexible** `startDay` to **not move earlier than**:
        
        - `max(window.originalStart, reschedStart, prevPhaseFixedEnd+1)`.
            
- **Capacity**: `dailyHeadCapacityByOp` must include **FixedHeadDay** every working day so flex cannot overlap capacity already consumed by fixed.
    
- **Outputs**:
    
    - Only seats inside the range may shift; fixed seats remain identical.
        
    - No double-stacking with fixed on the same day/op beyond capacity.
        

### Acceptance checks

- No flexible block starts **before** `reschedStart`.
    
- No capacity exceedance where fixed already consumes heads.
    
- Blocks outside the range are unchanged.
    

---

## Known Bug (current)

### Symptom

During reschedule, some **flexible** blocks shift **left of `reschedStart`** and end up **stacked with Fixed** on the same (day, op), causing effective head shortage downstream.

### Likely causes

1. **Missing lower bound** on `startDay` for reschedule windows (flex allowed to begin before `reschedStart`).
    
2. **Capacity not binding** strongly enough when Fixed heads are present (e.g., join gaps or missing rows when flex=0).
    

### Fix options

- **Option A — Lock start day (recommended)**
    
    - At build time for the reschedule run:
        
        - For every affected `BlockDecision`, set  
            `b.windowStart = max(b.windowStart, reschedStart, latestFixedEndOfPrevPhase+1)`.
            
        - Add/keep **hard** constraint (or input filtering) so `b.startDay ≥ b.windowStart`.
            
- **Option B — Strengthen Pass-1 capacity**
    
    - Ensure `dailyHeadCapacityByOp` **always** joins with `FixedHeadDay` and penalizes `(flexHeads + fixedHeads) > OP_CAPACITY[op]`, even if flexHeads=0.
        
    - Keep the **isWorkingDay** filter on both sides of the join.
        

> Either A or B solves the overlap; using **both** gives stricter guarantees (no early shift + no over-capacity).

---

## Quick test matrix

- **T1**: Weekend across phase window → workingDaysCount correct; `produced()` matches only weekdays.
    
- **T2**: Worker personal off on a Tuesday → no assignment on that Tuesday.
    
- **T3**: Fixed seat on Wed/Thu, reschedule range Mon–Fri → flex blocks do **not** move to Mon/Tue if `reschedStart=Wed`.
    
- **T4**: Fixed heads saturate daily capacity → flex produces **0** heads that day for the same op.
    

---

## Definition of Done

- All acceptance checks pass on sample data.
    
- No flexible start earlier than `reschedStart`.
    
- No day/op exceeds capacity once Fixed heads are included.
    
- Exported YAML shows only working-day seat-days; Fixed rows unchanged.