# Patch 8 — Overall

## Purpose

Move the scheduler implementation from Python to Java while preserving the same two-pass planning approach introduced in Patch 7:

1. Pass 1: build production blocks (start day, heads, days; hours auto-derived from allowed values).
    
2. Pass 2: assign employees to generated seats with hard/soft constraints.
    

## Why this change

- Full access to Timefold’s latest algorithms, configuration surface, and ecosystem on the JVM.
    
- Better long-term maintainability in an enterprise stack (Maven build, typed domain, IDE/tooling).
    
- Performance and stability improvements from running natively on the Timefold Java engine.
    

## Scope of Patch 8

- Create a Maven project structure (apps/v812) with standard `src/main/java`, `resources`, and `pom.xml`.
    
- Port the entire domain model and constraints to Java:
    
    - Pass 1 entities: `BlockDecision`, problem facts (`DaySlot`), and `Pass1Plan`.
        
    - Pass 2 entities: `CrewSeat`, supporting facts (`SeatDay`, `EmployeeFact`), and `Pass2Plan`.
        
    - Constraint providers: `Pass1Constraints` and `Pass2Constraints` using Constraint Streams.
        
- Keep YAML I/O parity: read `EnvConfig.yaml` and `Schedule.yaml`, write the final assignments back to `Schedule.yaml`.
    
- Maintain the two-phase pipeline and termination/tuning strategy from Patch 7:
    
    - Pass 1 “hours ramp” search to find a 0-hard block layout, then optional polish.
        
    - Expand blocks to seats, run Pass 2 assignment, then optional polish.
        
- Keep the Excel export path unchanged (external `export_schedule_excel.py` remains compatible with the produced YAML).
    

## What stays the same (functional parity)

- Two-pass design and objective: Pass 1 creates feasible block plans; Pass 2 fills seats with employees.
    
- Hard rules: phase windows/order, min/max heads, no underfill/limited overfill, daily 12h cap, one factory per person per day, at least one manager per block, eligibility by skill.
    
- Soft rules: prefer 8h, fewer heads/days, earlier starts; team quality (same-company cohesion, skill variety), balance block average skill toward org average, balance total hours across people.
    
- YAML schema and Excel outputs.
    

## What changes (platform & structure)

- JVM build/run: Maven manages dependencies and packaging.
    
- Strong typing and annotations (`@PlanningSolution`, `@PlanningEntity`, `@PlanningVariable`, etc.).
    
- Constraint Streams expressed directly in Java, enabling richer refactors and reuse.
    
- Solver construction via Java `SolverConfig`, preserving termination targets used before.
    

## Deliverables

- Java sources for the full pipeline (`EmployeeSchedule.java`, `ExportSchedule.java`).
    
- Maven project that compiles and runs the two-pass solver from CLI.
    
- YAML in/out compatible with downstream reporting.
    

## Risks & mitigations

- Risk: Behavioral drift during port.
    
    - Mitigation: Keep constraint names/weights and test with the same example YAMLs from Patch 7.
        
- Risk: Performance regressions.
    
    - Mitigation: Benchmark Pass 1/Pass 2 durations and scores vs. Patch 7; use Maven profiles to tune JVM options if needed.
        

## Next steps

- Run side-by-side comparisons on representative schedules to confirm parity of hard feasibility and soft-score quality.
    
- Enable Benchmarker configs in Java to compare alternative Local Search metaheuristics.
    
- Package a runnable JAR and update project docs with build/run instructions.