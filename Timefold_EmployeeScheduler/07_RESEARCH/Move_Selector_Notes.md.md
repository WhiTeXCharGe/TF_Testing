# Timefold Move Selector Notes

This note summarizes **Timefold’s Move Selectors** (the building blocks of Local Search / Metaheuristics). It explains what each move does, when to use it, and how it maps to your two-pass Timefold scheduler.

> Source: Timefold “Move Selector reference”. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)

---

## 0) What’s a Move Selector?

A **Move** is a small edit to the current solution (e.g., change one variable, swap two assignments).  
A **Move Selector** is the generator that picks candidate moves every step of Local Search.

Timefold provides many built-ins; you typically **combine** them (often in a _union_) so the solver can both explore broadly and make fine adjustments. A `changeMoveSelector` is almost always included so every solution remains reachable in principle. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)

---

## 1) Move selectors for basic variables

Basic variables are planning variables that are **not** lists or chains. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)

### 1.1 `ChangeMoveSelector` — “Change one value”

- **What it does:** Selects one entity and assigns a new value to one variable. Simplest, finest-grained move. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    
- **Why include it:** Ensures reachability of the entire solution space; usually present in every setup. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    
- **Obsidian tip:** Use for tiny adjustments in Pass 2 (e.g., move one seat to a different employee).
    

### 1.2 `SwapMoveSelector` — “Swap two entities’ values”

- **What it does:** Picks two entities and swaps the values of their variables. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    
- **Why useful:** Sometimes two single changes would break a hard constraint mid-step, but a **swap** keeps feasibility (classic trick). [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    
- **Project fit:** Swap two `CrewSeat` assignees to fix skill/manager coverage without causing intermediate violations.
    

### 1.3 Pillar moves — “Move or swap groups with the same value”

A **pillar** is a set of entities sharing the same value(s).

- **`PillarChangeMoveSelector`**: Move a whole pillar (or a sub-pillar) to a new value. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    
- **`PillarSwapMoveSelector`**: Swap the values of two pillars. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    
- **Sub-pillars:** You can operate on _subsets_ of a pillar; beware combinatorics—JIT random selection is used because the count explodes for large pillars. Sequential sub-pillars require ordering (comparator or `Comparable`). [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    
- **Project fit:** Move or swap multiple seats that currently share the same `op_id`/hours/factory.
    

### 1.4 `RuinRecreateMoveSelector` — “Break and rebuild”

- **What it does:** Unassign a subset of entities, then run a **Construction Heuristic** to re-assign them. Useful to escape local optima. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    
- **Frequency control:** Because it’s expensive, run it with **probabilistic selection** (e.g., 1 out of 100 steps) to keep the solver fast. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    
- **Project fit:** If Pass 2 gets stuck, occasionally “ruin + recreate” a troublesome module or block.
    

---

## 2) Move selectors for list variables

For problems modeled with **list variables** (e.g., routes, ordered sequences). The reference lists:

- **List change move** — move one element to a new index (or other entity’s list). [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    
- **List swap move** — swap two elements. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    
- **SubList change move** — move a sublist to a new position. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    
- **SubList swap move** — swap two sublists. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    
- **k-opt move** — remove `k` edges and reconnect (routing-style optimization). [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    
- **List Ruin & Recreate** — remove a subset of values from lists and rebuild with a CH (again, use probabilistic frequency). [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    

**Project note:** Your current employee scheduler uses **basic variables** (not list variables), so treat these as future options if you later model sequences (e.g., shift ordering or chained training tasks).

---

## 3) Move selectors for chained variables

For a **chained planning variable** (think vehicle routing or predecessor links). [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)

### 3.1 `TailChainSwapMoveSelector` (2-opt)

- **What it does:** Swap tail chains; if within the same chain, it acts like a partial reverse—classic 2-opt. Often faster than more general subchain moves. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    
- **Why a value selector (not secondary entity):** Needed for proper coverage and **nearby selection** behavior. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    

### 3.2 `SubChainChangeMoveSelector`

- **What it does:** Cut a subchain and paste it elsewhere (possibly reversing). `minimumSubChainSize` can avoid trivial moves; reversing is toggleable. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    

### 3.3 `SubChainSwapMoveSelector`

- **What it does:** Select two subchains and swap them. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    

**Project note:** Not used now, but relevant if you introduce chained tasks (e.g., training sequences or machine routes).

---

## 4) Selection & configuration patterns

### 4.1 Union of moves (mix coarse + fine)

- You often **combine** move selectors in a `<unionMoveSelector>`. The docs recommend keeping a `changeMoveSelector` present and adding coarser moves (swap, pillar, ruin&recreate) for bigger jumps. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    

### 4.2 Probabilistic selection

- For expensive moves (e.g., `ListRuinRecreateMove`), use **fixed probability weights** so they run rarely (e.g., 1% of the time). [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    

### 4.3 Caching

- Some move selectors support **phase/solver caching** (precomputing candidates) if they don’t touch **chained** variables; otherwise caching may be disabled or memory-heavy. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    

### 4.4 Entity/Value selectors

- Every move selector wraps **entitySelector** / **valueSelector** configs to target the right class/variable(s) or to restrict to specific variables. Examples are shown throughout the docs (e.g., room scheduling). [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    

---

## 5) How to apply this to your project (quick recipes)

### Pass 2 (employee assignment) — recommended union

- `changeMoveSelector` on `CrewSeat.employee`
    
- `swapMoveSelector` across `CrewSeat` (limit to variable `employee`)
    
- Optional: `pillarChangeMoveSelector` where a pillar = seats with same `(module, op_id, factory, start_day_id)` to move crews together
    
- Optional: `ruinRecreateMoveSelector` with low probability to refresh one block’s assignments when stuck
    

**Why this mix:**

- `change` = fine adjustments for skill/availability.
    
- `swap` = feasibility-preserving jumps (avoid mid-step constraint breaks).
    
- `pillar` = move small batches for bigger gains.
    
- `ruin&recreate` = escape deep local minima.
    

### Pass 1 (block sizing)

- If you ever run Local Search here (you currently solve with your tiered CH/loop), a simple set is:
    
    - `changeMoveSelector` on `(start_day | heads | days)` individually.
        
    - `swapMoveSelector` between blocks of the same `(module, phase)` to reshuffle starts.
        

---

## 6) Short summaries (cheat sheet)

- **Change**: tweak one variable of one entity. Always include. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    
- **Swap**: swap two entities’ variable values; preserves feasibility mid-step. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    
- **Pillar Change / Swap**: move or swap _groups_ that share a value; supports sub-pillars & sequences. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    
- **Ruin & Recreate**: unassign a subset and rebuild via CH to jump out of local minima; use with low probability. [docs.timefold.ai+1](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    
- **List/SubList moves, k-opt**: for list variables (routes/sequences). [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    
- **TailChain / SubChain**: for chained variables; includes 2-opt and subchain moves. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    
- **Union + probabilities**: combine moves; throttle expensive ones. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    
- **Caching**: sometimes supported; chained variables limit caching. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)
    

---

### Snippet: throttling a coarse move by probability

If/when you configure XML/JSON/YAML solver configs, this pattern (from docs) keeps ruin&recreate rare: [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference)

`<unionMoveSelector>   <unionMoveSelector>     <fixedProbabilityWeight>100.0</fixedProbabilityWeight>     <changeMoveSelector/>     <swapMoveSelector/>   </unionMoveSelector>   <ruinRecreateMoveSelector>     <fixedProbabilityWeight>1.0</fixedProbabilityWeight>   </ruinRecreateMoveSelector> </unionMoveSelector>`

---

If you want, I can also tailor a **ready-to-use solver phase config** (Python API) that matches your Pass 2 needs and mirrors the move selector union above.