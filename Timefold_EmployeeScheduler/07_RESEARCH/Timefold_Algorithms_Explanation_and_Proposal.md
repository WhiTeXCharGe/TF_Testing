# TIMEFOLD Algorithms — Explanation & Proposal (for Employee Scheduling)

## Summary

Timefold supports multiple optimization approaches:

- **Construction Heuristics (CH)**
    
- **Local Search / Metaheuristics**
    
    - **Late Acceptance (LA)**
        
    - **Tabu Search (TS)**
        
    - **Simulated Annealing (SA)**
        
- **Exhaustive Search**
    

These can be **composed into phases** (e.g., `CH → Local Search`), and their behavior is shaped by **Acceptor / Forager / Move Selector** settings.

- The docs’ **Benchmarker** examples compare configurations such as **TS / SA / LA** side-by-side.
    
- The **Move Selector** catalog (Change / Swap / Pillar / Ruin & Recreate / list- & chain-specific moves) is the “neighborhood” toolbox for Local Search.
    
- **Exhaustive Search** (Brute Force / Branch & Bound) is documented but only practical for very small instances.
    

> References (official docs):
> 
> - Optimization algorithms — overview: [https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/overview](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/overview?utm_source=chatgpt.com)
>     
> - Local Search: [https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/local-search](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/local-search?utm_source=chatgpt.com)
>     
> - Move Selector reference: [https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference?utm_source=chatgpt.com)
>     
> - Construction Heuristics: [https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/construction-heuristics](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/construction-heuristics?utm_source=chatgpt.com)
>     
> - Exhaustive Search: [https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/exhaustive-search](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/exhaustive-search?utm_source=chatgpt.com)
>     
> - Benchmarking & tweaking: [https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/benchmarking-and-tweaking](https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/benchmarking-and-tweaking?utm_source=chatgpt.com)
>     
> - Running the solver: [https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/running-the-solver](https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/running-the-solver?utm_source=chatgpt.com)
>     
> - Enterprise/default move selectors: [https://docs.timefold.ai/timefold-solver/latest/enterprise-edition/enterprise-edition](https://docs.timefold.ai/timefold-solver/latest/enterprise-edition/enterprise-edition?utm_source=chatgpt.com)
>     

---

## Short Explanations of Timefold Algorithms

### A) Construction Heuristics (CH)

**Purpose:** Quickly build a **complete initial solution** (not necessarily optimal).  
**Examples:** _First Fit / First Fit Decreasing_, _Cheapest / Regret Insertion_.  
**Notes:** A CH phase typically terminates once a full solution is built; you then refine with **metaheuristics**.

### B) Local Search (Metaheuristics)

**Purpose:** Iteratively improve a solution by applying small edits (**Moves**).  
**Components:**

- **MoveSelector** (generates candidate moves)
    
- **Acceptor** (decides which moves are acceptable—even some worse ones)
    
- **Forager** (chooses which accepted move to take)
    

**Common variants:**

- **Late Acceptance (LA):** Accept if the current score is better than the score from `L` steps ago (“aging baseline”). Simple, robust default.
    
- **Tabu Search (TS):** Maintain a short-term **tabu list** to forbid recently used moves/solutions; **Aspiration** allows tabu moves that beat the best-so-far. Strong anti-cycling.
    
- **Simulated Annealing (SA):** Accept worse moves with a temperature-based probability; gradually cool to favor improvement-only moves, combining broad early exploration with late-stage polishing.
    

### C) Move Selectors (Neighborhood)

- **Change** (change one variable on one entity)
    
- **Swap** (swap values between two entities)
    
- **Pillar** (change/swap a group of entities sharing the same value)
    
- **Ruin & Recreate** (unassign a subset, then rebuild with a CH)
    
- **List / Chain** specific moves (k-opt, SubList, SubChain, etc.)  
    → Combine these in a **union** (mix of fine and coarse moves).
    

### D) Exhaustive Search (Brute Force / Branch & Bound)

Try all (or pruned) possibilities. **Optimal** if it finishes, but **impractical** for realistic scheduling sizes; useful for toy cases and verification.

### E) Practical Phase Composition & Benchmarking

A standard pattern is **CH to reach feasibility → LS (LA/TS/SA) to improve**.  
Use the **Benchmarker** to compare **FFD + LA vs FFD + TS vs FFD + SA**, etc.

---

## Proposal: Experiments to Try

**Goal: improve feasibility speed and soft-score quality**

### Context

- **Two-pass scheduling**
    
    - **Pass 1:** block sizing (`start_day / heads / days`)
        
    - **Pass 2:** seat/employee assignment
        
- **Baseline:** **Late Acceptance (LA)** as the default Local Search
    
- **Alternatives to evaluate:** **Tabu Search (TS)** and **Simulated Annealing (SA)**
    

### What I Want to Try

1. **Replace LA with TS** in **Pass 2** (optionally test TS in Pass 1).
    
2. **Replace LA with SA** in **Pass 1** (optionally test SA in Pass 2).
    
3. Keep the **CH phase fixed** (e.g., First Fit / FFD) and vary only the **LS algorithm & Move Selector union** for clean, apples-to-apples comparisons (use Benchmarker).
    

### Why (differences vs LA & expected benefits)

**TS vs LA**

- **TS** uses short-term memory (tabu list); **LA** uses an aging reference window.
    
- **TS** is stronger at preventing cycling and offers intuitive shape control via **tabu size**, which can stabilize search near tight constraints.
    

**SA vs LA**

- **SA** probabilistically accepts worse moves based on **temperature**; **LA** uses a fixed step window comparison.
    
- **SA** naturally shifts from **wide exploration** to **exploitation** as it cools, helping cross “ridges” between solution basins.
    

### Where Each Helps in This Project

**Pass 1 (block sizing)**

- **SA:** Good when the landscape is rugged due to windows, auto-derived hours, and per-day head caps—early forgiveness supports big layout jumps.
    
- **TS:** If you see back-and-forth between near-equivalent block settings near capacity/phase boundaries, TS’s memory reduces oscillation.
    

**Pass 2 (employee assignment; manager-per-block hard, skill/overtime softs)**

- **TS:** Classic fit for assignment problems; reduces swap ping-pong and converges to stable feasible rosters.
    
- **SA:** If you have many near-ties among soft preferences (balance, variety, clustering), SA’s stochastic acceptance explores richer mixes before cooling.
    

### Recommended Move Selector Union

- **Core:** `Change` (single-seat reassignment), `Swap` (swap two seats)
    
- **Coarser:** `Pillar` (move/swap small crews), `Ruin & Recreate` (rare escape from deep traps)
    
- **Guideline:** throttle expensive moves (low probability), and **combine** multiple selectors.
    

### Starter Parameters (tune with Benchmarker)

- **Tabu Search (TS):** `entity/value tabu size = 7–11` (or ≈ √ of candidate moves per step), **Aspiration = ON**, `acceptedCountLimit ≈ 4–8`
    
- **Simulated Annealing (SA):** **Initial temperature** so ~**50–70%** of typical uphill moves are accepted at the start; **cooling rate** `α ≈ 0.95–0.99` every N steps; **stop** when temperature floor is reached or no improvement for some time
    
- **CH:** keep **First Fit / FFD** fixed to isolate LS effects
    

### What to Measure

- **Feasibility rate:** % of runs reaching `hard = 0` in each pass
    
- **Time to first feasible:** seconds to hit `0-hard`
    
- **Medium/Soft scores:** median & best (e.g., Pass 1: stacking/phase-gap; Pass 2: overtime, skill balance, manager coverage, clustering)
    
- **Stability:** variance across random seeds
    
- Use **Benchmarker** to automate comparisons: **(CH + LA) vs (CH + TS) vs (CH + SA)**
    

### Concrete “Recipes” (conceptual wiring)

**Pass 1 × Simulated Annealing (SA)**

- **CH:** keep your seeded blocks
    
- **LS:** SA with a union of moves
    
    - Example weights: `start_day ±1: 60%`, `heads ±1: 25%`, `days ±1: 15%`
        
    - Always respect **min/max** and **window**
        
- **Cooling:** `α ≈ 0.97`
    
- **Initial temperature:** estimate from typical **Soft/Medium** deltas (see _Running the solver_ notes)
    

**Pass 2 × Tabu Search (TS)**

- **CH:** Greedy / First-fit assignment
    
- **LS:** TS with move union
    
    - Example weights: **Swap within same `(op_id, day)`: 50%**, **Change (reassign): 30%**, **short chain-swap: 20%**
        
    - **tabu size ≈ 7–11**, **Aspiration = ON**
        

### Quick Guide (when to prefer which)

- **Tabu Search (TS):** choose when you observe cycling/oscillation and want memory-guided, steady progress.
    
- **Simulated Annealing (SA):** choose when early feasibility is hard or the landscape is spiky; it explores widely first, then settles.
    

---

## References (Official Docs & Trusted Sources)

- Optimization algorithms — overview  
    [https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/overview](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/overview?utm_source=chatgpt.com)
    
- Local Search (Acceptor / Forager; LA / TS / SA)  
    [https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/local-search](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/local-search?utm_source=chatgpt.com)
    
- Move Selector reference (Change / Swap / Pillar / Ruin & Recreate / list & chain moves)  
    [https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference?utm_source=chatgpt.com)
    
- Construction Heuristics (First Fit / FFD)  
    [https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/construction-heuristics](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/construction-heuristics?utm_source=chatgpt.com)
    
- Exhaustive Search (Brute Force / Branch & Bound + config example)  
    [https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/exhaustive-search](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/exhaustive-search?utm_source=chatgpt.com)
    
- Benchmarking & tweaking (compare TS / SA / LA)  
    [https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/benchmarking-and-tweaking](https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/benchmarking-and-tweaking?utm_source=chatgpt.com)
    
- Running the solver (time-gradient algorithms like SA; termination notes)  
    [https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/running-the-solver](https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/running-the-solver?utm_source=chatgpt.com)
    
- Enterprise/default move selectors (defaults, nearby selection)  
    [https://docs.timefold.ai/timefold-solver/latest/enterprise-edition/enterprise-edition](https://docs.timefold.ai/timefold-solver/latest/enterprise-edition/enterprise-edition?utm_source=chatgpt.com)