# 🧩 Patch 7.4.2 – Pass 1 Overtime Loop + Manager Constraint

## 📅 Sprint Link
Related Sprint: [[Sprint_20251007]]

---

## 🎯 Overview
Patch 7.4.3 introduces a new **Pass 1 overtime loop logic** and integrates the **manager assignment constraint** from the previous sprint plan.

The main idea of this patch is to allow **adaptive block recalculation** — where Pass 1 iteratively expands operation hours based on which blocks fail hard constraints (underfill, window overflow, or phase violation).

---

## ⚙️ Key Additions

### 🌀 Pass 1 Overtime Loop
- Introduced `_solve_pass1_overtime_loop_and_export()`  
- Pass 1 now runs iteratively:
  1. Starts with smallest allowed work hours.
  2. Solves blocks; checks for “negative” blocks (underfilled, window overflow, or invalid phase).
  3. In next loop, only widens hours for affected operations.
  4. Stops when **hard score = 0** or all operations reach their max allowed hours.
- Each iteration exports a separate YAML file:
Schedule_1.yaml  
Schedule_2.yaml  
- Includes optional “polish” step once a feasible solution (0 hard) is found.

**Purpose:**  
Improves Pass 1 convergence by handling underfill/overtime systematically instead of one-shot solving.

---

### 👨‍🏭 Manager Constraint Integration
- Integrated from previous sprint (`Sprint_20251007` task).  
- Manager logic now available for Pass 2 (enforced in constraint factory).  
- Ensures at least one manager per block but **kept isolated from Pass 1** logic to avoid breaking group_by patterns.

---

## 🔧 Supporting Changes
- Added `_capacity_for_day_op()` and `_OP_CAPACITY_BY_DAY_OP` to prepare for future day-specific capacity tuning.  
- Unified function `_build_solver()` with safe handling of `Duration(minutes=int(...))`.  
- Improved `_auto_hours()` selection logic with overfill ≤ one extra day rule.  
- Added tiered ramp `_solve_pass1_with_hours_ramp()` helper for testing short runs.

---

## 🧪 Behavior Summary
| Item | Result |
|------|---------|
| Pass 1 | Iterative loop, converges faster with fewer infeasible starts |
| Pass 2 | Manager constraint active, verified working |
| Export | Multi-file export per iteration |
| Hard score | Reaches 0 after several iterations in most datasets |
| Soft score | Improved slightly with polishing step |

---

## 🗂️ Related Files
- `employee_schedule.py` → updated (main logic)  
- `export_schedule.py` → unchanged  
- YAML I/O: `EnvConfig.yaml`, `Schedule.yaml`  

---

## 🔮 Next Steps
- [ ] Add Pass 1 summary export (produced vs required per operation).  
- [ ] Prepare comparison Excel showing each iteration’s overfill and hard score.  
- [ ] Evaluate per-day manager rule for possible future enhancement.  


