# Generate Schedule — How-to

## Overview

`generate_schedule.py` creates a `Schedule.yaml` from your `EnvConfig.yaml`. It builds any number of “modules” (e1, e2, …), each containing phase tasks (p1..pN) and operation tasks (p?o?). All phase/operation structure and names come from `EnvConfig` → `environment.workflow_list`.

## What it reads / writes

- **Reads:** `EnvConfig.yaml` (for workflows, phases, operations, and fab IDs)
    
- **Writes:** `Schedule.yaml` with:
    
    - `plan_range: { start_date, end_date }`
        
    - `workflow_task_list: [...]`
        
    - `assignment_list: []` (left untouched for solver)
        

## Key behavior

- **Per-module dates:** All phases in a module share the same **start_date**. Each phase gets its **end_date** by adding a random offset range (per phase).
    
- **Inter-module spacing:** Module e2 starts a random number of days after e1 (configurable), e3 after e2, and so on.
    
- **Workload days:** Each operation (e.g., `p2o1`) gets a single constant `workload_days` used for all modules (default or per-op override).
    
- **Fab selection:** Each module’s `fab` is randomly chosen from `EnvConfig.environment.fab_list`.
    
- **Plan range overrun:** If a phase end date goes past `plan_range.end_date`, it’s still written. A console warning is printed.
    

## Defaults you can edit in code

At the top of the file:

- `DEFAULT_PLAN_START`, `DEFAULT_PLAN_END`
    
- `DEFAULT_NUM_MODULES`, `DEFAULT_WORKFLOW_ID`
    
- Module naming: `DEFAULT_NAME_BASE`, `DEFAULT_NAME_PREFIX`, `DEFAULT_NAME_SUFFIX`
    
- Inter-module shift: `DEFAULT_START_SHIFT_MIN`, `DEFAULT_START_SHIFT_MAX`
    
- Phase offsets: `DEFAULT_PHASE_OFFSETS = {"p1":[20,25], "p2":[30,35], ...}`
    
- Workload days: `DEFAULT_WORKLOAD_DAYS_DEFAULT`, `DEFAULT_WORKLOAD_DAYS_MAP`
    
- Randomness: `DEFAULT_RANDOM_SEED`
    

## CLI overrides (examples)

`python generate_schedule.py --env EnvConfig.yaml --out Schedule.yaml python generate_schedule.py --env EnvConfig.yaml --out Schedule.yaml --modules 12 python generate_schedule.py --env EnvConfig.yaml --out Schedule.yaml \   --plan-start 2025/09/01 --plan-end 2025/12/10 \   --workflow-id workflow \   --name-base 1000 --name-prefix "SU " --name-suffix "A" \   --start-shift-min 0 --start-shift-max 4 \   --phase-offsets "{p1:[20,25], p2:[30,35], p3:[40,42], p4:[51,55]}" \   --workload-default 10 \   --workload-map "{p1o1:9, p1o2:15, p2o1:10, p2o2:13, p3o1:11, p4o1:8}" \   --seed 42`

## Formats for inline maps

- **Phase offsets:** `"{p1:[20,25], p2:[30,35]}"`  
    (keys = phase IDs; values = `[min,max]` days from module start)
    
- **Workload map:** `"{p1o1:9, p2o2:13, p3o1:11}"`  
    (keys = operation IDs; values = `workload_days`)
    

## IDs and names

- Modules: `e1`, `e2`, … (auto) with names like `SU 1001A`, `SU 1002A` (configurable).
    
- Phase task IDs: `e{idx}{phaseId}` e.g., `e1p2`.
    
- Operation task IDs: `e{idx}{phaseId}o{suffix}` e.g., `e1p2o1`.  
    (`suffix` is taken after the first “o” in the operation ID so `p2o1` → `o1`)
    

## Common errors & fixes

- **Missing phases/ops:** Ensure your `EnvConfig.environment.workflow_list[?].phase_list[*].operation_list[*].id` are set.
    
- **Unknown phase in offsets:** Add that phase to `DEFAULT_PHASE_OFFSETS` or pass via `--phase-offsets`.
    
- **Non-positive workload:** Ensure `--workload-default` and any `--workload-map` overrides are > 0.
    

## Tips

- Keep `--seed` fixed to reproduce the same schedule; change it to get a different random layout.
    
- Use `--workflow-id` if `EnvConfig` has multiple workflows; otherwise the first is used.