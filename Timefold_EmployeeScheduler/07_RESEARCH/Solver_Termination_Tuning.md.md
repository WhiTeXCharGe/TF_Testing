> Practical guidelines to tune solver termination for the two-pass employee scheduling project (Pass 1 = block sizing, Pass 2 = employee assignment). Focus: reliability (hard=0), predictable runtimes, and stable soft scores.

---

## 1) Core Termination Knobs

- **`best_score_limit`**  
    Stop as soon as a target score is reached. Typical pattern: `"0hard/*medium/*soft"` to prioritize feasibility.
    
- **`spent_limit`**  
    Wall-clock time budget (e.g., 30s, 2m). Guarantees predictable runtime.
    
- **`unimproved_spent_limit`**  
    “Early stop” when no improvement for a while (e.g., 60s without better best score).
    
- **(Optional) `step_count_limit`**  
    If available, caps the number of move evaluations (less stable than time-based in mixed hardware).
    
- **Phase-specific vs Global**  
    Termination can be set per phase (construction vs local search) or at top level to bound the whole run.
    

---

## 2) Project Defaults (Good Starting Points)

### Pass 1 (Block sizing)

- **Primary goal:** Hit **0 hard** quickly; medium/soft polishing is secondary.
    
- **Recommended:**
    
    - `best_score_limit="0hard/*medium/*soft"`
        
    - `spent_limit=60–120s` per iteration (or per tier if using hours-tier loop)
        
    - `unimproved_spent_limit=30–60s` (enables early exit when stuck)
        
    - (Optional) **Polish pass**: +30–60s only if hard=0
        

### Pass 2 (Employee assignment)

- **Primary goal:** Maintain **0 hard**, then improve soft balance.
    
- **Recommended:**
    
    - `best_score_limit="0hard/*medium/*soft"`
        
    - `spent_limit=60–180s` (scale with employee×seat size)
        
    - `unimproved_spent_limit=45–90s`
        
    - **Short polish** when hard=0: +30–60s
        

---

## 3) Budgeting Patterns

- **Two-phase budget:**
    
    1. **Feasibility phase** (short, aggressive): stop as soon as hard=0.
        
    2. **Polish phase** (soft improvement): short extra time only when hard=0.
        
- **Tiered restarts (Pass 1 hours ramp / overtime loop):**  
    Allocate a small, fixed budget per tier/iteration (e.g., 60s), export the best, widen hours or adjust capacity, then continue. Prevents runaway time on impossible tiers.
    
- **Adaptive early-stop:**  
    If `best_score.hard == 0` and the last N% of time yields <X% soft improvement, end early.
    

---

## 4) Practical Targets & Health Checks

- **Feasibility SLA:**
    
    - Pass 1: reach `hard=0` within 1–2 tiers or ≤5 minutes total.
        
    - Pass 2: reach `hard=0` within ≤3 minutes on typical instances.
        
- **Soft-score stability:**  
    Record median and best soft across seeds; if variance is high, increase polish time slightly or reduce randomness.
    
- **Time predictability:**  
    Prefer `spent_limit` + small `unimproved_spent_limit` over step limits for heterogeneous machines.
    

---

## 5) Instrumentation & Logging

- Log at each solve:
    
    - Start/end timestamps, **`spent`** time, **`best_score`**, **`init_score`**
        
    - Time to first feasible (when `hard` transitions to 0)
        
    - Count of unimproved intervals
        
- For Pass 1 loops:
    
    - Tier/iteration index, allowed-hours tier, any caps applied
        
    - “Negative blocks” count used to decide widening
        
- Persist a small **run summary JSON/CSV** for trend inspection.
    

---

## 6) Example: Python Snippets

### 6.1 Build a solver with feasibility-first + polish

`from timefold.solver.config import SolverConfig, TerminationConfig, Duration from timefold.solver.score import HardMediumSoftScore  def build_feasible_first_solver(solution_cls, entity_cls_list, constraint_fn,                                 main_minutes=1, polish_minutes=1, do_polish=True):     # Main phase: try to reach 0 hard quickly     main_term = TerminationConfig(         best_score_limit="0hard/*medium/*soft",         spent_limit=Duration(minutes=int(main_minutes)),         unimproved_spent_limit=Duration(seconds=0)     )     cfg = SolverConfig(         solution_class=solution_cls,         entity_class_list=entity_cls_list,         score_director_factory_config={"constraint_provider_function": constraint_fn},         termination_config=main_term     )     solver = SolverFactory.create(cfg).build_solver()     result = solver.solve(...)  # provide initial solution     # Optional polish if 0 hard reached     if "hard=0" in str(result.score) and do_polish and polish_minutes > 0:         polish_term = TerminationConfig(             spent_limit=Duration(minutes=int(polish_minutes)),             unimproved_spent_limit=Duration(seconds=60)         )         cfg.termination_config = polish_term         solver2 = SolverFactory.create(cfg).build_solver()         result = solver2.solve(result)     return result`

### 6.2 Per-iteration budget in a Pass-1 tier loop

`TIER_MINUTES = 1.0 POLISH_MINUTES = 1.0  solver = _build_solver(     solution_cls=Pass1Plan,     entity_cls_list=[BlockDecision],     constraint_fn=pass1_constraints,     best_limit="0hard/*medium/*soft",     spent_minutes=TIER_MINUTES,     unimproved_seconds=0 ) solved = solver.solve(pass1)  if solved.score.hard == 0 and POLISH_MINUTES > 0:     polish = _build_solver(         Pass1Plan, [BlockDecision], pass1_constraints,         best_limit=None, spent_minutes=POLISH_MINUTES, unimproved_seconds=60     )     solved = polish.solve(solved)`

---

## 7) Tuning Checklist

1. **Set a feasibility stop:** `best_score_limit="0hard/*medium/*soft"`.
    
2. **Choose a wall-clock budget:** `spent_limit` per phase (short for feasibility, shorter for polish).
    
3. **Add stagnation guard:** `unimproved_spent_limit` (30–90s typical).
    
4. **Measure time-to-feasible:** if too long, increase CH quality (better seeding) or relax Pass-1 options (e.g., allow one more hour tier earlier).
    
5. **Stabilize soft variance:** small polish time + consistent seeds when benchmarking.
    
6. **Automate summaries:** CSV/JSON per run; compare across seeds and sizes.
    
7. **Scale with size:** For Pass 2, increase budgets with (#seats × #employees); for Pass 1, with (#blocks × horizon).
    

---

## 8) Troubleshooting

- **Never reaches hard=0 (Pass 1):**
    
    - Window too tight or heads too small: permit higher hours tiers earlier.
        
    - Phase-order clashes: bias earlier starts less (reduce “earlier start” weight) to unlock feasibility.
        
    - Daily head cap too strict: verify capacity counts; allow small overfill where acceptable.
        
- **Feasible but slow:**
    
    - Lower `spent_limit` and rely on `unimproved_spent_limit`.
        
    - Improve CH seeding (closer to feasible shape).
        
    - In Pass 1 loops, reduce iterations by widening multiple ops at once when many negatives appear.
        
- **Soft score unstable across runs:**
    
    - Increase polish time slightly (e.g., +30s).
        
    - Reduce randomness (fixed seed) for demos; re-enable randomness for final benchmarks.
        

---

## 9) Suggested Baselines (for this project)

|Phase|Main Budget|Early-Stop|Polish (if hard=0)|Notes|
|---|---|---|---|---|
|Pass 1|60–120 s|30–60 s|30–60 s|Use per-tier budget; export after each iteration.|
|Pass 2|60–180 s|45–90 s|30–60 s|Scale with instance size; keep 0-hard priority.|

---

## 10) Policy for Demos vs Nightly

- **Demo mode:** small `spent_limit`, strict early-stop, polish only if hard=0, fixed seed.
    
- **Nightly batch:** longer `spent_limit`, generous unimproved window, multiple seeds (aggregate medians), keep best artifact per instance.