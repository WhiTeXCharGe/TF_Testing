# Timefold (Python): Algorithm/Move-Selector Configuration Not Available — Halted Support + Repro Evidence

## Executive Summary

- Official maintainers announced that **active development of the Python Solver (Beta) is halted**, noting the effort required to match the JVM solver’s **performance and features**. This indicates feature gaps compared to Java/Kotlin. [GitHub+1](https://github.com/TimefoldAI/timefold-solver/discussions/1698?utm_source=chatgpt.com)
    
- The **ability to switch algorithms and configure Local Search phases, move selectors, acceptors, and foragers** is documented for the **JVM solver** (XML / Java `*Config` classes). These configuration entry points are **not exposed** in the Python package. [docs.timefold.ai+2docs.timefold.ai+2](https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/configuration?utm_source=chatgpt.com)
    
- A **reproducible test on Python 1.24.0b** shows that passing a `phase_list` (Local Search + Tabu + union move selectors) to `SolverConfig(...)` raises `TypeError: unexpected keyword argument 'phase_list'`. This demonstrates that **algorithm and move-selector customization cannot be done** through the Python API used in this project. _(Trace and code below.)_
    
- Conclusion: constraints and termination can be tuned in Python, but **changing the algorithm (e.g., to Tabu Search or Simulated Annealing) and customizing move selectors requires the JVM stack**. [docs.timefold.ai+1](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/overview?utm_source=chatgpt.com)
    

---

## Background & Web Evidence

- **Python Solver status (halted):**  
    Announcement from a Timefold maintainer: _“halting active development on the Beta version of our Python Solver … effort to match its performance and features with Open Source Java Solver is substantial.”_ [GitHub](https://github.com/TimefoldAI/timefold-solver/discussions/1698?utm_source=chatgpt.com)
    
- **Where algorithm/phase configuration lives (JVM):**  
    The configuration guides state that it’s easy to **switch optimization algorithms by changing configuration** and show phase configuration, **Local Search components** (MoveSelector / Acceptor / Forager), and **Move Selector** catalogs for the JVM solver. [docs.timefold.ai+2docs.timefold.ai+2](https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/configuration?utm_source=chatgpt.com)
    
- **Algorithms supported (context):**  
    Official overview lists **Exhaustive Search**, **Construction Heuristics (CH)**, and **Metaheuristics** (e.g., Late Acceptance, Tabu Search, Simulated Annealing), typically composed as **CH → Local Search**. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/overview?utm_source=chatgpt.com)
    
- **Python as a secondary target (performance note):**  
    PyPI and Timefold posts note Python is **significantly slower** than Java/Kotlin, reinforcing that Python trails in performance and features. [PyPI+2PyPI+2](https://pypi.org/project/timefold/?utm_source=chatgpt.com)
    

---

## Reproduction (Python 1.24.0b): Attempting to Configure Local Search + Tabu + Move Selectors

**Intent:** Enable **Local Search** with **Tabu Search** and a **union of move selectors** (`Change` on variables, plus `Swap`) via `SolverConfig(...)`.

`print("[EVIDENCE] Attempting Local Search + Tabu + move selectors (expected to FAIL on 1.24.0b).") cfg = SolverConfig(     solution_class=Pass1Plan,     entity_class_list=[BlockDecision],     score_director_factory_config=ScoreDirectorFactoryConfig(         constraint_provider_function=pass1_constraints     ),     # This key ('phase_list') + the LS section below is what should trigger the TypeError.     phase_list=[         {             "phaseType": "CONSTRUCTION_HEURISTIC",             "constructionHeuristicType": "FIRST_FIT"         },         {             "phaseType": "LOCAL_SEARCH",             "moveSelectorConfig": {                 "unionMoveSelectorConfig": {                     "moveSelectorConfigList": [                         { "changeMoveSelectorConfig": {                             "entityClass": BlockDecision, "variableName": "start_day"                         }},                         { "changeMoveSelectorConfig": {                             "entityClass": BlockDecision, "variableName": "heads"                         }},                         { "changeMoveSelectorConfig": {                             "entityClass": BlockDecision, "variableName": "days"                         }},                         { "swapMoveSelectorConfig": { "entityClass": BlockDecision } }                     ]                 }             },             "acceptorConfig": { "entityTabuSize": 7, "valueTabuSize": 7 },             "foragerConfig": { "acceptedCountLimit": 4 }         }     ],     termination_config=TerminationConfig(         spent_limit=Duration(seconds=10)     ) )`

**Observed error (abbreviated):**

`TypeError: SolverConfig.__init__() got an unexpected keyword argument 'phase_list'`

**Interpretation:** The Python `SolverConfig` constructor **does not accept** `phase_list` (and thus no nested `Local Search`, `moveSelectorConfig`, `acceptorConfig`, or `foragerConfig`). This confirms the **absence of JVM-style algorithm/phase configuration in Python** used for this project.  
Related manuals that describe these knobs target the **JVM** (XML / Java `*Config`), not Python. [docs.timefold.ai+1](https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/configuration?utm_source=chatgpt.com)

---

## Implications for the Project

- **Not feasible in Python (current package):**  
    Switching from the default behavior to **Tabu Search** or **Simulated Annealing**, or customizing **move selectors**, **cannot be configured** via the Python API employed here. [GitHub](https://github.com/TimefoldAI/timefold-solver/discussions/1698?utm_source=chatgpt.com)
    
- **Available levers in Python:**  
    Maintain the current two-pass architecture, keep constraints robust, and continue using **termination** and **external orchestration** (e.g., pass-1 tiered hours loop) to steer search.
    
- **If algorithm experiments are required now:**  
    Implement solver configuration and phases on **Java/Kotlin**, where **Local Search**, **move selectors**, **acceptors**, and **foragers** are first-class and well-documented; keep YAML I/O/orchestration around it as needed. [docs.timefold.ai+1](https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/configuration?utm_source=chatgpt.com)
    

---

## Key References

- Announcement halting Python Solver (maintainer post). [GitHub](https://github.com/TimefoldAI/timefold-solver/discussions/1698?utm_source=chatgpt.com)
    
- Optimization algorithms — overview (CH / Metaheuristics / composition). [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/overview?utm_source=chatgpt.com)
    
- Local Search (MoveSelector / Acceptor / Forager; phase configuration patterns). [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/local-search?utm_source=chatgpt.com)
    
- Move Selector reference (Change / Swap / Pillar / Ruin & Recreate / list & chain moves). [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference?utm_source=chatgpt.com)
    
- Configuration guide — switching algorithms via configuration; Benchmarker. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/configuration?utm_source=chatgpt.com)
    
- PyPI notes on Python performance vs JVM. [PyPI+1](https://pypi.org/project/timefold/?utm_source=chatgpt.com)