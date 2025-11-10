# v8.3.1 — Transit & Region Max-Stay (Prototype / Ideal)

> Run:
> 
> `mvn -q exec:java -D"exec.args=src/main/resource/EnvConfig.yaml src/main/resource/Schedule.yaml"`

## 1) What’s new in 8.3.1 (compared to 8.2.x)

- **Region Transit Gap (Pass 2 hard):** If the same employee appears on two different regions on nearby days, they must have enough gap days between assignments according to `environment.transite_day_map` (e.g., RegionA→RegionB requires 2 days).
    
- **Region Max-Stay-On (Pass 2 hard):** Limits the maximum **consecutive** ON-days an employee can work in a region, with a configurable **OFF break** (`stay_off_interval`) that resets the streak. Values come from `environment.region_list` (`max_stay_on`, `stay_off_interval`).
    
- **Entity-dependent value ranges (Pass 2):** Each **CrewSeat** precomputes a **candidate employee list** (skill ≥1, no personal OFF clash, no cross-factory same-day clash with pinned work) to prune search space **before** solving.
    
- **Calendar extensions:** `buildCalendars()` now also loads:
    
    - `transite_day_map` → `CAL.transitDays(fromRegion→toRegion)`.
        
    - Per-region `max_stay_on` and `stay_off_interval`.
        

> Note: These are **Pass 2** constraints only. Pass 1 does not yet insert “travel buffer days” or shape blocks around transit—this is the main reason this version is marked **prototype/ideal**.

---

## 2) Inputs the code expects (YAML)

### 2.1 `environment.region_list[*]`

`region_list:   - id: r1     unavailable_dates: [2025/10/01, 2025/10/14]     max_stay_on: 10            # max consecutive working days allowed in this region     stay_off_interval: 2       # OFF gap (>=2 days) resets the ON streak`

### 2.2 Transit day map (region→region)

`transite_day_map:   - from: r1     to:   r2     days: 2                     # need ≥ 2 days between last r1 workday and first r2 workday   - from: r2     to:   r1     days: 1`

### 2.3 Fab→Region / Fab→Customer mapping

Ensure each fab has `region` and (optionally) `customer_company`:

`fab_list:   - id: fA     region: r1     customer_company: cX     unavailable_dates: []`

> Other existing sections (`worker_list`, `workflow_list`, blackout calendars, etc.) remain unchanged from 8.2.x.

---

## 3) Data Model Recap (only the deltas)

- **`Calendars` (CAL)** now includes:
    
    - `Map<String, Map<String,Integer>> transitDays`
        
    - `Map<String,Integer> regionStayMaxOn`
        
    - `Map<String,Integer> regionStayOffInterval`
        
- **`CrewSeat`**: unchanged shape, but Pass 2 sets **candidate employees** per seat via `setCandidateEmployees(...)`.
    
- **`SeatDay`**: unchanged; used to aggregate per-emp-day constraints and region transitions.
    
- **`FixedHeadDay`**: unchanged; still used by Pass 1 for per-day op capacity.
    

---

## 4) Pass 2 — New Hard Constraints

### 4.1 Region Transit Gap (`p2-region-transit-gap`)

**Intent:** An employee cannot move from region **R1** to **R2** without at least `CAL.transitDays(R1,R2)` **whole days** in between their last day in R1 and first day in R2.

**Implementation pattern:**  
For any employee who appears on `(SeatDay sd1 @ region r1)` and later `(SeatDay sd2 @ region r2)`:

- `need = transitDays(r1, r2)`
    
- If `need > 0` and `delta = sd2.dayId - sd1.dayId ≤ need`, **penalize hard** by `(need - delta + 1)`.
    

**Notes & limits:**

- Works even if `r1 == r2` (need=0 → no penalty).
    
- Looks across all seats; pinned + flexible both count.
    
- **Travel days themselves are not created** in Pass 1; enforcement is entirely in Pass 2, so solver may be forced to reassign employees to avoid violations (ideal direction), but cannot push blocks to add buffers (limitation of 2-pass).
    

### 4.2 Region Max-Stay-On (`p2-visa-presence-max-stay-on`)

**Intent:** An employee must not exceed **K consecutive ON-days** in a region unless they observe an OFF break of at least **B** days; `K = max_stay_on`, `B = stay_off_interval`.

**Implementation pattern:**

- Group **SeatDay** by `(employee, region)` and collect the day IDs they work.
    
- Compute `maxSegmentSpanWithBreak(dayList, B)` → the longest run of **calendar days** where consecutive ON segments are only broken by OFF gaps **≥ B** (small gaps < B are treated as **continuous**).
    
- If `maxSpan > K`, **penalize hard** by `(maxSpan - K)`.
    

**Interpretation examples (K=10, B=2):**

- 10 straight ON days in r1 → OK.
    
- 12 straight ON days in r1 → hard penalty 2.
    
- 7 ON, **1 day OFF**, 5 ON → because OFF gap (1) < B (2), treated as a **continuous** 13-day streak → hard penalty 3.
    
- 7 ON, **2 days OFF**, 5 ON → OFF gap resets → two segments 7 and 5 → **OK**.
    

---

## 5) Pass 2 — Pre-filtering Candidates (Entity-Dependent Value Ranges)

Each **CrewSeat** computes its own candidate employees list **before** solving:

- **Skill filter:** `skill(e, opId) ≥ 1`.
    
- **Personal OFF filter:** `e` must be available on **all** `SeatDay`s of this seat.
    
- **Pinned cross-factory clash:** If `e` is **already pinned** to a different factory on any of these days, exclude `e`.
    
- **Pinned seats:** If the seat is pinned, restrict the value range to the pinned worker only.
    

This drastically reduces assignment branching, making room for the added region logic.

---

## 6) What remains “ideal” (not fully realized yet)

- **No travel buffers in Pass 1:** Blocks are not shifted/reshaped to create explicit transit days. The solver enforces feasibility by **choosing different employees** rather than moving production windows.
    
- **No multi-leg travel modeling:** Only a single gap check between two regions; it doesn’t create intermediate “travel tasks”.
    
- **Max-stay interacts only with assigned days:** It uses **SeatDay** presence; it doesn’t consider “on-region but idle” days, because such days don’t exist in the current model.
    
- **No manager-specific travel rules:** Transit applies equally to managers and non-managers (could be extended).
    

---

## 7) Testing Checklist

1. **Transit basic:**
    
    - Regions r1, r2 with `transite_day_map: r1→r2: 2`.
        
    - Assign the **same** employee on r1 (day 5) and r2 (day 6–7) → expect **hard** violation.
        
    - Assign r2 starting day 8 or later → OK.
        
2. **Transit symmetry:**
    
    - Define asymmetric entries (r1→r2=2, r2→r1=1). Check each direction independently.
        
3. **Max-stay basic (K=5, B=2):**
    
    - 6 straight ON days in r1 → **hard** violation size 1.
        
    - 3 ON, 1 OFF, 3 ON → OFF gap <2 → treated continuous 7 → **hard** size 2.
        
    - 3 ON, 2 OFF, 3 ON → two segments 3 and 3 → OK.
        
4. **Pinned cross-factory candidate trim:**
    
    - Pin worker W to factory fA on day 10; build a flexible seat for fB on day 10. W should **not** appear as a candidate for that seat.
        
5. **Personal OFF candidate trim:**
    
    - Put OFF on day 12 for worker W; any seat covering day 12 must **not** list W as a candidate.
        
6. **Performance sanity:**
    
    - Confirm total candidates per seat drastically smaller than total employees.
        

---

## 8) Known Edge Cases & Behaviors

- **Same-day multi-factory via pinned + flexible:** Hard rule `one-factory-per-emp-day` still forbids this even if candidate pruning missed it (it shouldn’t for pinned).
    
- **Transit within same region:** `transitDays(r,r)` is treated as `0`. No gap needed.
    
- **Region resolution:** Region is derived from **seat.factory → CAL.fabToRegion**. If a seat’s factory is missing the region mapping, transit/max-stay checks **don’t trigger** for that seat.
    
- **Max-stay with non-working days:** Only days with **SeatDay** count as ON; weekends or blackout days don’t add to the streak (unless you assign seats there, which you can’t as `isWorkingDay` filters them out).
    
- **Pinned seats violating transit/max-stay:** Solver can’t change pinned assignments; expect hard violations if impossible.
    

---

## 9) Suggested Roadmap to “Complete” Transit/Stay

- **3-pass design:**
    
    - **Pass 1A (Blocks)** → insert **transit buffer stubs** between regions when seats would otherwise require illegal jumps.
        
    - **Pass 1B (Shaping)** → re-shape start/days to absorb the buffers while staying within windows.
        
    - **Pass 2 (People)** → then assign with transit/max-stay already satisfied structurally.
        
- **Travel entities:** Model **TravelSeat** per employee with duration `transitDays(r1,r2)` to make gaps explicit and visible in exports.
    
- **Manager-specific rules:** Add options like “manager must arrive 1 day earlier” or “manager gets longer rest”.
    
- **Costed transit (soft):** Penalize unnecessary region hops; reward consolidating work within a region to reduce travel.
    

---

## 10) Code Landmarks (for maintainers)

- **Calendar loads:** `buildCalendars()` → `CAL.transitDays`, `CAL.regionStayMaxOn`, `CAL.regionStayOffInterval`.
    
- **Transit constraint:** `Pass2Constraints#regionTransitGap(...)`.
    
- **Max-stay constraint:** `Pass2Constraints#regionStayMaxOn(...)` with helper `maxSegmentSpanWithBreak(...)`.
    
- **Candidate pruning:** `fillSeatCandidateEmployees(...)` and `CrewSeat.empRangePerSeat()`.
    
- **Where region is computed:** `CAL.regionOfFab(cs.factory)`.
    

---

## 11) Acceptance Criteria (for this prototype)

- When **Transit** or **Max-Stay** rules are violated, Pass 2 returns a **non-zero hard** score and tries to reassign employees to satisfy rules **without** moving blocks.
    
- If all assignments that satisfy skill/availability also satisfy transit/max-stay, score becomes **`0hard/*/*`**.
    
- Candidate lists per seat **exclude** employees that are obviously infeasible (OFF or pinned elsewhere) and include those with correct skills.
    

---

## 12) Glossary

- **Transit gap:** Minimum blank days between last workday in region R1 and first workday in region R2 for the **same employee**.
    
- **Max-stay-on (K):** Maximum allowed consecutive working days inside a region; OFF gaps shorter than **B** do not reset the streak.
    
- **Stay-off interval (B):** Minimum OFF gap to **reset** the consecutive ON counter.