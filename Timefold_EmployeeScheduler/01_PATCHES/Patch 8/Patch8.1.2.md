## Discussion — Patch 8.1.2 (Java / Timefold)

> `mvn -q exec:java -D"exec.args=src\main\resource\EnvConfig.yaml src\main\resource\Schedule.yaml"`

### 🎯 What 8.1.2 tries to solve

Patch 8.1.2 is a **targeted Pass 1 looping algorithm** that:

- **Freezes (pins) “good” blocks** that don’t violate hard constraints, and
    
- **Recalculates only the problematic blocks** by **ramping their allowed hours tier-by-tier**,
    
- While emitting **per-iteration snapshots** (`Schedule1.yaml`, `Schedule2.yaml`, …) so planners can **see convergence**.
    

This keeps the successful two-pass architecture (Pass 1 create blocks → Pass 2 assign people) from 7.x/8.1.1, but adds **selective ramping + pinning** to reduce churn and make diagnostics first-class.

---

## 🆚 What’s new vs 8.1.1 / 7.4.3

|Area|7.4.3 (Py)|8.1.1 (Java)|**8.1.2 (Java)**|
|---|---|---|---|
|Pass 1 ramping|Overtime loop (global)|Tiered ramp (global)|**Selective ramp** (only violators)|
|Pinning/freeze|N/A|N/A|**Pin non-violators (start/head/day/hours)**|
|Snapshots|Optional|N/A|**Automatic per-iteration YAML snapshots**|
|Snapshot content|N/A|N/A|Seats expanded; **all seats assigned to employee id=1** (for visualization only)|
|Final export|Yes|Yes|**Yes** (after Pass 2, overwrites original `Schedule.yaml`)|
|Constraints|Same|Same|Same + **Respect pinned** in Pass 1|

---

## 🧠 Core idea (Pass 1 Selective Ramp)

1. **Seed** every block with tier 1 of its allowed hours (e.g., `work_hours: [4,8,10]` → use `4`).
    
2. **Solve** Pass 1 with standard hard rules (window, phase order, min/max heads, underfill, one-day overfill cap, daily op capacity).
    
3. **Detect violators** (blocks that still break hard constraints).
    
4. **Pin non-violators** (record their start/heads/days and selected hours; lock them).
    
5. **Tier-up only the violators** (widen their allowed hours list by one tier).
    
6. **Write snapshot** (`Schedule{iter}.yaml`) that expands seats and force-assigns **employee id=1** to every seat (so you can see the block shape without running Pass 2).
    
7. **Repeat** until either **0 hard** or all blocks hit their **max tier**.
    
8. If **0 hard**, do a **polish** pass for soft score, snapshot again, then go to **Pass 2**.
    

**Why this helps:** instead of re-optimizing everything on each loop, we **freeze stability** and **focus computation** on the minimum set of “bad” blocks. That generally accelerates convergence and preserves earlier good structure.

---

## 🧩 New entities & constraints (highlights)

### Pinning on `BlockDecision`

- `pinned` + `pinStart`, `pinHeads`, `pinDays`, `pinHours`
    
- **New hard constraint** `respectPins` (Pass 1): if a block is pinned, the solver must keep those values.
    

### Snapshot pipeline

- During each loop, generate `Schedule{N}.yaml` **alongside** the original file.
    
- Snapshot is **block → seat** expansion, but **does not** run Pass 2.
    
- For clarity, snapshots **assign everyone to employee id=1** (or first non-zero) so Gantt views are visible.
    

---

## 🔁 Algorithm at a glance

`perBlockTier := 1 for all blocks repeat until all blocks hit max tier:   blocks := seed(windows, perBlockTier)   solve Pass1 with hard=0 target   write snapshot Schedule{iter}.yaml   if hard=0:      polish Pass1; snapshot again      break   violators := detectHardViolators(solved.blocks)   for each block:      if not in violators:         pin(block) with its solved values and hours         allowed := [autoHours(block)]   // reduce degrees of freedom      else:         perBlockTier[block]++           // widen only here end if hard=0:   expand seats; run Pass2; polish if hard=0; overwrite original Schedule.yaml else:   skip Pass2 (diagnostic only)`

---

## ⚙️ Pass 1 constraints (unchanged + pinning)

- **Hard:** window bounds, window length, allowed-hours membership, min/max heads, **no underfill**, **overfill ≤ one extra day**, phase order, **daily head capacity by op**, **respect pinned**.
    
- **Soft:** prefer 8h, prefer smaller hours, minimize heads/days, earlier start, avoid stacking same op.
    

## ⚙️ Pass 2 constraints (unchanged)

- **Hard:** eligible skill, 12h/day cap, one factory per person per day, ≥1 manager/block.
    
- **Soft:** reward same-company pairs, encourage skill diversity, keep block avg skill near org average, balance total hours vs target.
    

---

## 🗂️ File I/O contract

- **Inputs**: `EnvConfig.yaml` (workers, ops, phases) + `Schedule.yaml` (windows/tasks).
    
- **Snapshots**: `Schedule1.yaml`, `Schedule2.yaml`, … (diagnostic; **fake assignment to employee id=1**).
    
- **Final output**: original `Schedule.yaml` **overwritten** with real Pass 2 assignments once Pass 1 reaches hard=0.
    

---

## 🏗️ Code touchpoints

- Pass 1 loop: `solvePass1SelectiveRamp(...)`
    
- Violator detection: `detectHardViolators(...)` (mirrors hard rules without reusing the engine)
    
- Pinning logic & hours ramp: `seedBlocksForTier(...)` + loop body
    
- Snapshots: `writeScheduleSnapshot(...)` → uses `ExportSchedule.overwriteScheduleWithAssignments(...)`
    

---

## 📈 Expected impact

- **Runtime**: fewer full restarts and less churn → typically faster than a global ramp on large instances.
    
- **Stability**: early good choices are preserved via pinning → **less oscillation**.
    
- **Debuggability**: snapshot trail lets you **see exactly when/why** blocks escalate hours.
    

---

## 🧪 Acceptance checklist

-  On a known dataset, Pass 1 **converges** with fewer loops than global ramp.
    
-  Snapshots appear as `Schedule{N}.yaml` and render expected block shapes (all seats set to employee id=1).
    
-  Final `Schedule.yaml` is overwritten **only after** Pass 2.
    
-  Pinned blocks stay fixed across subsequent iterations (verified by logs / diff).
    
-  Hard=0, and soft score **not worse** than 8.1.1 after polish.
    

---

## 🔮 Next steps / ideas

- **Snapshot options**: CLI flag to disable snapshots or change the placeholder employee ID.
    
- **Granular tiering**: allow **per-operation** step sizes (e.g., 8→9→10 rather than 8→[8,10]).
    
- **Heuristic pin duration**: unpin soft-only blocks if they block global soft improvements after polish.
    
- **Report**: per-iteration CSV with counts of violators, changed tiers, produced vs required, and score deltas.
    

---

## 🧰 Run command (Windows path sample)

`mvn -q exec:java -D"exec.args=src\main\resource\EnvConfig.yaml src\main\resource\Schedule.yaml"`

This executes Pass 1 selective ramp with snapshots, then Pass 2 (if hard=0), and finally overwrites the original `Schedule.yaml` with real assignments.