# Excel Export Tool — EnvConfig/Schedule → Analytics Workbook

> Purpose: Read `EnvConfig.yaml` and `Schedule.yaml`, then generate a four-sheet Excel workbook for planning, auditing, and presentation.

---

## 1) What it reads

- **EnvConfig.yaml**
    
    - Workers (name, company, skills, optional `is_manager`)
        
    - Workflow/phases/operations (allowed hours, min/max heads)
        
    - Fab/Region/Customer meta (for labeling)
        
- **Schedule.yaml**
    
    - Plan range (`start_date` → `end_date`)
        
    - Module → Phase windows (start/end) → Operation tasks (workload days)
        
    - `assignment_list` (worker, operation_task, date/hour) — used to reconstruct who worked when
        

---

## 2) What it writes

- A single Excel file with **four sheets**:
    
    1. **Tasks x Dates** – a day-by-day timeline for each `(module, op_id)`
        
    2. **Employees x Dates** – per-employee calendar, with skill flags
        
    3. **Dashboard** – KPIs, progress/throughput/utilization, team-quality analytics
        
    4. **Breaches** – tidy tables of all violations for quick audits
        
- Consistent **color legend** across sheets for fast scanning.
    

---

## 3) Sheet Details

### 3.1 Sheet: “Tasks x Dates”

**Row = one task block** (a module’s operation).  
**Columns A–G (meta):**

- `module`, `module_name`
    
- `fab_id`, `fab_name`
    
- `region`, `customer`
    
- `task` = `op_id name`
    

**Column H (manager):**

- Lists **all managers** who ever worked on the task (by name).
    
- **Blank = highlighted** (“no manager ever assigned”).
    

**Columns I–J (hours summary):**

- `required_hours` = `workload_days × 8` (from `Schedule.yaml`)
    
- `assigned_hours` = sum of actual assigned hours for that task
    
    - **Yellow fill** if `assigned_hours < required_hours` (under-assigned).
        

**Date columns (one per plan day):**

- **Cell text:** `WorkerName(8H) | WorkerName(10H) | …`
    
- **Decorative markers (not violations):**
    
    - Light blue = **module start** date
        
    - Red = **phase end** (deadline) for this task’s phase
        
- **Violation colors (overrides decorative markers):**
    
    - **Purple** = **Phase window breach** (work logged before phase start or after end)
        
    - **Blue** = **Phase ordering breach** (work on phase _k+1_ on/before the **last day** of phase _k_)
        
    - **Pink** = **Staffing min/max breach** (heads that day outside `min_worker_num…max_worker_num` for the op)
        

**Intended use:**

- Visual Gantt-like view with **who actually staffed** each day.
    
- Quickly spot **unmanaged tasks**, **under-assignment**, and **calendar misalignments**.
    

---

### 3.2 Sheet: “Employees x Dates”

**Columns A–D (roster):**

- `company`, `employee`
    
- `skills` (e.g., `p1o1:3, p2o2:2`)
    
- `Manager` = `True` if the person is a manager
    

**Per-day columns:**

- **Cell text:** `module+op (8H)` entries for that employee on that day
    
- **Orange fill** = **Skill mismatch** (employee worked an `op_id` with no skill level)
    

**Trailing summary columns:**

- `Workdays` = count of days with any assigned hours
    
- `WorkHours` = total hours across the plan horizon
    

**Intended use:**

- Audit **who worked when** and whether **skills matched** the assignment.
    
- Check **total load** and rough **utilization** by person.
    

---

### 3.3 Sheet: “Dashboard”

#### KPIs (top-left)

- **Unique workers**: headcount with any work
    
- **Avg hours/worker-day**: fairness/overload signal
    
- **Overtime hours (>8)**: sum of per-person daily hours above 8
    
- **Cap breach (>12h) days / %**: overload risk metric
    
- **Completion %**: `Σ assigned_hours ÷ Σ required_hours` (across all modules)
    

#### Progress by module

- `module`, `required_hours`, `assigned_hours` (**yellow** if under), `%complete`
    
- `planned_end` (latest phase end), `last_assigned` (latest actual work date), `delay (days) = last_assigned − planned_end`
    

#### Assignment Utilization (per employee)

- `employee`, `start_date`, `end_date` (their earliest/latest assignment)
    
- `utilization% = worked_days ÷ planned_days in that span`
    

#### Overtime & capacity

- **Table 1:** `employee`, `total_ot_hours` (sum above 8/day)
    
- **Table 2:** `date`, `total_hours` (all people summed per day)
    
- **Line chart:** _Total hours per day_ → staffing demand trend
    

#### Workload balance (per employee)

- `employee`, `workdays`, `total_hours`, `avg/day`, `stdev/day`, `CoV` (coefficient of variation; lower = steadier load)
    

#### Top 20 total hours

- `employee`, `total_hours`
    
- **Bar chart:** _Total hours by employee (Top 20)_ → workload concentration
    

#### Team Quality by Block

**What is a “block”?**  
A grouped stint where a set of workers did the same `(module, op_id)` over a **continuous** date range with the **same daily hours per person**.

**Columns:**

- **Identity/Sizing:** `module`, `op_id`, `phase`, `start_date`, `end_date`, `heads`, `hours`, `days`
    
- **Leadership:** `managers` (all names across the block)
    
- **Company mix:** e.g., `Alpha×3 | Beta×2`
    
- **Cohesion:** `max_pairs`, `same_company_pairs`, `cohesion_% = same_company_pairs ÷ all_pairs × 100`
    
- **Skills:**
    
    - `op_avg_skill` = mean skill level for this `op_id` across the org
        
    - `team_avg_skill` = mean level among the block’s members
        
    - `balance_dev = |team_avg_skill − op_avg_skill|` (lower is better)
        
    - `balance_score` (0–100) penalizing large `balance_dev`
        
    - `variety_score` (0–100) from a Shannon index over the levels present (higher = more diverse)
        
- **Scatter chart:** _Balance vs Variety (by block)_
    
    - **X =** `balance_dev` (left is better)
        
    - **Y =** `variety_score` (higher is more diverse)
        
    - **Top-left quadrant** = well-balanced **and** diverse teams
        

**Intended use:**

- Macro view for stakeholders: **are modules on track**, **who is overworked**, where **teams lack leadership or diversity**.
    

---

### 3.4 Sheet: “Breaches”

Each table is **sorted** for quick auditing:

1. **Phase window breaches**
    
    - Columns: `date`, `module`, `phase`, `op_id`, `worker`, `reason (early/late)`, `phase_start`, `phase_end`
        
2. **Phase ordering breaches**
    
    - Columns: `date`, `module`, `phase(later)`, `op_id`, `worker`, `required_prev_phase_last_date`
        
3. **Skill mismatches**
    
    - Columns: `date`, `worker`, `company`, `module`, `op_id`
        
4. **Staffing min/max breaches**
    
    - Columns: `date`, `module`, `op_id`, `heads`, `min`, `max`, `status (below/above)`
        
5. **Tasks with no manager**
    
    - Columns: `module`, `op_id`
        

**Intended use:**

- Hand a single page to reviewers for **all exceptions**; filter/sort as needed to drive fixes.
    

---

## 4) Color Legend (quick)

- **Light blue**: module start (marker)
    
- **Red**: phase end / deadline (marker)
    
- **Purple**: phase **window** breach
    
- **Blue**: phase **ordering** breach
    
- **Pink**: **staffing min/max** breach
    
- **Yellow**: **under-assigned** task (`assigned_hours < required_hours`)
    
- **Orange**: **skill mismatch** (employee worked op without skill)
    
- **Manager column (Tasks x Dates)**: highlighted when **no manager** ever assigned
    

---

## 5) Definitions & Calculations

- **Required hours (task)** = `workload_days × 8`  
    _(Export uses 8h as the baseline requirement; the daily hour actually assigned can be 8/10/12 depending on the data.)_
    
- **Assigned hours (task)** = Sum of all hours logged to that `(module, op)`.
    
- **Overtime** per person, per day = `max(0, daily_hours − 8)`;  
    **Cap breach** if `daily_hours > 12`.
    
- **Utilization% (per employee)** = `worked_days ÷ planned_days`  
    where `planned_days` = days between the employee’s earliest and latest assigned dates (inclusive).
    
- **Team balance/variety** calculated from per-block **skill levels** (integers).  
    `variety_score` uses a normalized Shannon index; `balance_score` is a 0–100 scale that penalizes deviation from the global average for that op.
    

---

## 6) Typical Workflow

1. Run the solver to produce/update `assignment_list` in `Schedule.yaml`.
    
2. Execute the export tool → writes the Excel workbook.
    
3. Inspect **Tasks x Dates** for coverage and timing; fix breaches via the planner.
    
4. Use **Employees x Dates** to check skill fit and individual load.
    
5. Review **Dashboard** KPIs and the **Team Quality** scatter to adjust staffing strategy.
    
6. Use **Breaches** as the single source of truth for exceptions to resolve.
    

---

## 7) Assumptions & Notes

- Dates are treated as whole days (no partial-day calendars).
    
- `required_hours` uses an 8-hour baseline for comparability against actuals.
    
- Manager presence is inferred from any manager assignment on the task across the horizon.
    
- Skill mismatch = **no positive skill level** for the specific `op_id`.
    
- The “block” heuristic groups continuous days with the **same per-person daily hours**; minor hour fluctuations split blocks.
    

---

## 8) Extensibility Ideas

- Add **per-phase** required vs assigned charts.
    
- Introduce **per-factory** views (tabs or pivots).
    
- Export **CSV** alongside Excel for pipelines.
    
- Optional: compute **per-day headroom** against min/max for proactive warnings.