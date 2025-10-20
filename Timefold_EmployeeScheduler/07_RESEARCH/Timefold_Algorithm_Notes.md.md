# Timefold Algorithm Notes

This document is a deep dive into the optimization algorithms supported by **Timefold Solver**, how they work, when to use them, and their trade-offs. It’s intended as a reference while you design and tune your two-pass scheduling system.

Based on Timefold documentation: *“Optimization Algorithms: Overview”* :contentReference[oaicite:0]{index=0}  
Also on *Construction Heuristics* and *Local Search* pages. :contentReference[oaicite:1]{index=1}

---

## 1. Algorithm Families in Timefold

Timefold supports three **families** of optimization algorithms:  
1. **Exhaustive Search**  
2. **Construction Heuristics**  
3. **Metaheuristics** (a.k.a. Local Search & its variants)  
:contentReference[oaicite:2]{index=2}

Here’s a high-level comparison:

| Family | Scalable? | Guarantees optimum? | Setup difficulty | Typical use |
|---|---|---|---|---|
| Exhaustive Search | ❌ (fails for large scale) | ✅ | Simple (but often impractical) | Smaller toy instances; correctness check |
| Construction Heuristics | ✅ | ❌ | Low | Produce an initial feasible solution |
| Metaheuristics | ✅ (given good design) | ❌ | Medium to high | Refinement after a CH, or standalone if CH not enough |

Timefold is designed to combine them: often a **Construction Heuristic** to get a starting solution, then a **Metaheuristic / Local Search** to improve further. :contentReference[oaicite:3]{index=3}

> *“In practice, Metaheuristics (in combination with Construction Heuristics to initialize) are the recommended choice.”* :contentReference[oaicite:4]{index=4}

---

## 2. Exhaustive Search

These are brute-force search methods that explore the entire solution space (or use branch-and-bound pruning).

### 2.1 Brute Force

- Try every possible assignment of planning variables.
- Guarantees the optimal solution if you can explore the full space.
- **Not feasible** for nontrivial scheduling problems with many combinations.

### 2.2 Branch and Bound

- A smarter exhaustive method: it prunes parts of the search tree once it detects they can’t produce better solutions.
- Still usually infeasible for large problems, but helpful for small instances or verification.

**When to use**: only as a baseline on very small test cases, or to validate correctness of your constraints.

---

## 3. Construction Heuristics (CH)

Construction Heuristics build a complete solution from scratch by assigning each planning entity (or variable) in turn.

They are fast and guarantee feasibility (if constraints and domain allow), but they **do not guarantee optimality**.

### 3.1 Common CH Types (in Timefold)

Timefold supports several CH strategies:

- **First Fit / First Fit Decreasing**  
  Assign each entity to the first feasible value found (or use sorting to assign more “difficult” entities first). :contentReference[oaicite:5]{index=5}  
- **Weakest Fit / Strongest Fit**  
  Choose the least or most “capacity-consuming” slot, depending on a heuristic. :contentReference[oaicite:6]{index=6}  
- **Cheapest Insertion**  
  Insert entities in the position that yields the smallest cost increase. :contentReference[oaicite:7]{index=7}  
- **Regret Insertion**  
  Consider not only best placement but also the “regret” of skipping the second-best; choose placements to minimize future regret. :contentReference[oaicite:8]{index=8}  

### 3.2 Cartesian vs Sequential Assignment

- **Cartesian product** style: assign all planning variables of an entity at once (i.e. try all combinations). This gives better quality but is much more computationally expensive. :contentReference[oaicite:9]{index=9}  
- **Sequential**: assign one variable at a time per entity, which scales better but might yield lower-quality initial solutions. :contentReference[oaicite:10]{index=10}  
- Hybrid: combine cartesian for some key variables and sequential for others. :contentReference[oaicite:11]{index=11}  

### 3.3 Partitioned Search & Filtering

- Partitioned Search: break the problem into partitions, run CH in parallel. :contentReference[oaicite:12]{index=12}  
- Move selectors can use filters or limited selections to reduce the number of candidate moves. (E.g. only consider “nearby” values). :contentReference[oaicite:13]{index=13}  

### 3.4 Pros & Cons

**Pros:**
- Very fast compared to metaheuristics.
- Guarantees a feasible solution if constraints allow.
- Good starting point for further improvement.

**Cons:**
- Quality may be poor (local greedy).
- No guarantee of optimal.
- Sensitive to entity ordering, variable order, and heuristics.

---

## 4. Local Search & Metaheuristics

Once a full solution is built (e.g. by CH), Local Search or Metaheuristics iteratively improve it by applying **moves** (small changes) to get better solutions.

### 4.1 Local Search – Basics

- Start from an initial solution.
- At each step, evaluate a set of candidate moves (e.g. swap assignments, change a variable).
- Use an **Acceptor** and **Forager** to decide which move to accept and apply. :contentReference[oaicite:14]{index=14}  
- Keep track of the **best solution seen so far**, even if current solution goes down temporarily. :contentReference[oaicite:15]{index=15}  

**Key components:**
- **MoveSelector**: enumerates possible moves.
- **Acceptor**: decides which moves are acceptable (maybe some worsen score).  
- **Forager**: picks among accepted moves.  
:contentReference[oaicite:16]{index=16}  

### 4.2 Common Local Search Variants in Timefold

Timefold supports several local search (metaheuristic) types:

- **Hill Climbing**  
  Always take improving moves. Very simple.  
  *Risk:* easily stuck in local optimum. :contentReference[oaicite:17]{index=17}  

- **Tabu Search**  
  Maintain a tabu list of recently moved entities to avoid cycling.  
  Good choice when you need to escape local minima. :contentReference[oaicite:18]{index=18}  

- **Simulated Annealing**  
  Accept sometimes worse moves with certain probability (depends on “temperature”) to escape local minima. Good for explorative search. :contentReference[oaicite:19]{index=19}  

- **Late Acceptance**  
  Compare current score vs score from several steps ago; allow worse moves if still better than “old” solutions. :contentReference[oaicite:20]{index=20}  

- **Great Deluge**  
  Maintain a threshold (water level); allow moves as long as score does not exceed current threshold. :contentReference[oaicite:21]{index=21}  

- **Step Counting Hill Climbing**  
  A variant of hill climbing with extra control to avoid cycling. :contentReference[oaicite:22]{index=22}  

- **Variable Neighborhood Descent**  
  Change neighborhoods dynamically (start small, then explore larger move sets). :contentReference[oaicite:23]{index=23}  

- **Evolutionary Algorithms (EA)**  
  Genetic algorithms or evolutionary strategies—less commonly used internally, but supported in some configurations. :contentReference[oaicite:24]{index=24}  

### 4.3 When to Use Which

- Start with **Hill Climbing** or **Tabu Search**; they are straightforward and often effective.
- Use **Simulated Annealing** or **Late Acceptance** when your landscape has many local peaks and you want more exploration.
- Use **EAs** or **Neighborhood methods** in large combinatorial problems or when domain structure suggests crossover operations.

### 4.4 Parameter Tuning & Trade-offs

- Each algorithm has parameters (tabu size, temperature schedule, acceptance thresholds). The **benchmarker** tool is recommended to test and tune them. :contentReference[oaicite:25]{index=25}  
- Time or iteration limits (spent limit, unimproved steps limit) are key in preventing runaway solving. :contentReference[oaicite:26]{index=26}  
- A solver phase chain may use multiple local search strategies sequentially (e.g. late acceptance → tabu) or nested. :contentReference[oaicite:27]{index=27}  

---

## 5. Phases & Solver Composition

Timefold’s solver is composed of **phases** (each using one algorithm) executed in order:

1. **Construction Heuristic Phase**  
2. **One or more Local Search / Metaheuristic Phases**  
3. Optionally, further polishing phases

You can configure phases via `SolverConfig`, sequencing, time limits, acceptors, etc. :contentReference[oaicite:28]{index=28}  

If no explicit phases are configured, Timefold defaults to a Construction Heuristic + Local Search setup. :contentReference[oaicite:29]{index=29}  

---

## 6. Strategy Advice (for Your Scheduling Project)

- **Start simple**: Use a fast CH (First Fit / First Fit Decreasing) to get a baseline solution.
- **Then refine**: Add a metaheuristic (Tabu or Late Acceptance) to improve both hard and soft score.
- Tune termination parameters (time limits, unimproved thresholds) based on your instance size.
- Use the **benchmarker** mode to compare multiple CH + LS combinations systematically. :contentReference[oaicite:30]{index=30}  
- Watch out for local optima; if your solver stagnates, switch acceptor strategies or restart.
- Use **difficulty comparators** for planning entities so CH assigns tougher blocks earlier. :contentReference[oaicite:31]{index=31}  

---

Feel free to integrate this into your vault and refine it as you discover more by experimentation.  
If you like, I can also generate a **template version** of this note with placeholders for your own empirical results and parameter settings.
::contentReference[oaicite:32]{index=32}
