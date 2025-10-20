# Timefold: Python Algorithm/Move-Selector Limitations vs Java (Research + Evidence)

## TL;DR

- **Python solver (beta) is halted**; it lacks feature parity with Java—specifically the ability to configure **phases**, **local-search algorithms**, and **move selectors**. [GitHub+1](https://github.com/TimefoldAI/timefold-solver/discussions/1698?utm_source=chatgpt.com)
    
- Timefold docs showing configurable **algorithms** (CH/LS/Metaheuristics) and **move selectors** target the **Java/Kotlin** solver configuration (XML/Java API). [docs.timefold.ai+3docs.timefold.ai+3docs.timefold.ai+3](https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/configuration?utm_source=chatgpt.com)
    
- **Real test (Python 1.24.0b):** attempting to specify `phase_list` / move selectors in `SolverConfig(...)` throws `TypeError: unexpected keyword argument 'phase_list'`. (See reproduction below.)
    

---

## 1) Official statements & docs

### 1.1 Python solver status

- **Announcement:** “Timefold is halting active development on the Beta version of our Python Solver … effort to match its performance and features with Open Source Java Solver is substantial.” (posted by a maintainer). [GitHub](https://github.com/TimefoldAI/timefold-solver/discussions/1698?utm_source=chatgpt.com)
    
- Discussions page lists the same announcement thread. [GitHub](https://github.com/timefoldai/timefold-solver/discussions?utm_source=chatgpt.com)
    

### 1.2 Where algorithms & move selectors are configurable

- Docs say you can **switch optimization algorithm(s)** by changing solver configuration—this is in the **Solver configuration** manual (Java/Kotlin). [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/configuration?utm_source=chatgpt.com)
    
- The **Optimization Algorithms** chapters detail **Exhaustive Search / Construction Heuristics / Local Search**, including how Local Search is composed of **MoveSelector, Acceptor, Forager** and shows configuration structure. (Java solver). [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/local-search?utm_source=chatgpt.com)
    
- **Move Selector reference** lists the built-ins (Change, Swap, Pillar, Ruin&Recreate, list/chained variants, etc.)—these are the components you configure in LS phases. (Java). [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference?utm_source=chatgpt.com)
    
- Example XML snippets for choosing algorithms (e.g., exhaustive search type) are shown in the docs—again, **Java XML config**. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/exhaustive-search?utm_source=chatgpt.com)
    
- Config classes in the docs are the **Java `*Config` classes** (representation of XML). [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/0.8.x/configuration/configuration?utm_source=chatgpt.com)
    

> Conclusion: The public documentation that demonstrates picking algorithms/phases/moves refers to the **Java/Kotlin** solver. The Python port does not expose equivalent configuration hooks today.

---

## 2) Real test (Python 1.24.0b): cannot set phases/move selectors

### 2.1 Reproduction code (abridged)

`cfg = SolverConfig(     solution_class=Pass1Plan,     entity_class_list=[BlockDecision],     score_director_factory_config=ScoreDirectorFactoryConfig(         constraint_provider_function=pass1_constraints     ),     # Expectation: define CH + LS (Tabu) + union of move selectors     phase_list=[         {             "phaseType": "CONSTRUCTION_HEURISTIC",             "constructionHeuristicType": "FIRST_FIT"         },         {             "phaseType": "LOCAL_SEARCH",             "moveSelectorConfig": {                 "unionMoveSelectorConfig": {                     "moveSelectorConfigList": [                         {"changeMoveSelectorConfig": {"entityClass": BlockDecision, "variableName": "start_day"}},                         {"changeMoveSelectorConfig": {"entityClass": BlockDecision, "variableName": "heads"}},                         {"changeMoveSelectorConfig": {"entityClass": BlockDecision, "variableName": "days"}},                         {"swapMoveSelectorConfig":   {"entityClass": BlockDecision}}                     ]                 }             },             "acceptorConfig": {"entityTabuSize": 7, "valueTabuSize": 7},             "foragerConfig":  {"acceptedCountLimit": 4}         }     ],     termination_config=TerminationConfig(spent_limit=Duration(seconds=10)) )`

### 2.2 Observed error

`TypeError: SolverConfig.__init__() got an unexpected keyword argument 'phase_list'`

**Interpretation:** The Python `SolverConfig` constructor does **not** accept `phase_list` (and therefore no nested `moveSelectorConfig`, `acceptorConfig`, or `foragerConfig`). This confirms that, in Python 1.24.0b, **you cannot programmatically select Local Search algorithms or move selectors**.

> Related signal: The docs’ algorithm & move-selector configuration are shown for Java (XML/`*Config`), not Python; the halt notice further explains missing parity. [GitHub+2docs.timefold.ai+2](https://github.com/TimefoldAI/timefold-solver/discussions/1698?utm_source=chatgpt.com)

---

## 3) What the docs _do_ show (for Java)

- **Algorithm families** supported and recommended phase chaining (CH → LS) are thoroughly documented. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/0.8.x/optimization-algorithms/optimization-algorithms?utm_source=chatgpt.com)
    
- **Local Search** components (MoveSelector, Acceptor, Forager) and their configuration are described, including phase config structure. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/local-search?utm_source=chatgpt.com)
    
- The **Move Selector reference** enumerates change/swap/pillar/ruin-recreate/list/chained moves you can configure. [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/move-selector-reference?utm_source=chatgpt.com)
    
- Config is presented via **XML** and **Java `*Config`** classes (the Java representation of XML). [docs.timefold.ai+1](https://docs.timefold.ai/timefold-solver/latest/optimization-algorithms/exhaustive-search?utm_source=chatgpt.com)
    

---

## 4) Practical implications for this project

- **Short term (staying in Python):**
    
    - Keep your current approach: drive **Pass 1** search externally (your hours-tier loop that widens allowed hours), and use Pass 2 with constraints—since internal LS/move-selector config isn’t available.
        
    - Tune **termination** and **constraints**—these are available in Python and already effective in your 2-pass design.
        
- **If you need Tabu / SA / custom move unions now:**
    
    - Implement the solver core in **Java/Kotlin**, where you can set phases and move selectors; keep YAML I/O + orchestration in Python (CLI/REST bridge).
        
    - Optionally benchmark different LS setups (docs suggest Benchmarker for Java). [docs.timefold.ai](https://docs.timefold.ai/timefold-solver/latest/using-timefold-solver/configuration?utm_source=chatgpt.com)
        

---

## 5) One-liner conclusion

Timefold’s **Java/Kotlin** solver fully supports **configurable phases and move selectors** (via XML/`*Config` classes). The **Python** solver (beta, now **halted**) **does not expose** those configuration hooks—hence the `SolverConfig(..., phase_list=...)` **TypeError** and inability to switch algorithms or move selectors programmatically. [GitHub+2docs.timefold.ai+2](https://github.com/TimefoldAI/timefold-solver/discussions/1698?utm_source=chatgpt.com)