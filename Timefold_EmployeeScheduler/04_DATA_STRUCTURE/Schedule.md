# Schedule.yaml – Work Plan & Assignments (Overview + Field Guide)

## What this file is

Schedule.yaml is the variable plan for a specific horizon.  
It defines:

- the planning window (start/end dates),
    
- which modules/jobs run at which fabs,
    
- each phase’s time window and workload per operation,
    
- the concrete day-by-day worker assignments.
    

All worker, skill, company, and workflow definitions come from EnvConfig.yaml.  
Schedule.yaml just says “what to do when” and “who is assigned”.

---

## Top-level layout

`schedule:   plan_range:     start_date: YYYY/MM/DD     end_date:   YYYY/MM/DD    workflow_task_list:       # modules/jobs to execute in this plan     - id: e1                # module id       name: Module A        # optional       workflow: workflow    # optional, links to EnvConfig workflow       fab: fab01            # where it runs       phase_task_list:         - phase: p1           start_date: YYYY/MM/DD           end_date:   YYYY/MM/DD           operation_task_list:             - operation: p1o1               name: Mech    # optional               workload_days: 3             - operation: p1o2               workload_days: 2         - phase: p2           start_date: ...           end_date:   ...           operation_task_list:             - operation: p2o1               workload_days: 4             ...    assignment_list:          # concrete worker scheduling     - worker: w12       operation_task: e1p1o1       # module id + op id       start_date: YYYY/MM/DD       # optional (block-level)       end_date:   YYYY/MM/DD       # optional       work_date_list:              # per-day records (sometimes spelled work_date_lsit)         - date: YYYY/MM/DD           hour: 8         - date: YYYY/MM/DD           hour: 10     - worker: w07       operation_task: e1p2o1       work_date_list:         - date: YYYY/MM/DD           hour: 8`

---

## Field-by-field (brief)

### schedule.plan_range

- start_date, end_date  
    Planning horizon. All phase windows and assignments must fit inside this range.
    

### schedule.workflow_task_list[*]

Each entry is a “module” (a job instance) that will follow the standard workflow from EnvConfig.

Module fields

- id: module identifier (e.g., e1, e2).
    
- name: display label (optional).
    
- workflow: which workflow definition to use (optional; defaults to first).
    
- fab: fab id (must exist in EnvConfig).
    
- phase_task_list: the timeline for each phase in this module.
    

Phase fields

- phase: phase id (p1, p2, …). Order is implied by the number.
    
- start_date, end_date: allowed window to execute this phase.
    
- operation_task_list: list of operations to perform within this phase.
    

Operation task fields

- operation: op id (p#o#) defined in EnvConfig’s workflow.
    
- name: label (optional).
    
- workload_days: planned effort measured in “baseline days”.  
    The system converts this into required hours:
    
    - If the op’s allowed hours in EnvConfig is only [4], baseline = 4.
        
    - Otherwise baseline = 8.  
        Required hours = workload_days × baseline.  
        Pass 1 then chooses heads × hours/day × days to meet or exceed this.
        

### schedule.assignment_list[*]

Concrete, day-by-day staffing.

Fields

- worker: worker id from EnvConfig.worker_list.
    
- operation_task: concatenation of module id + operation id (e.g., e1p3o2).
    
- start_date, end_date: optional “block” span (used for summaries/quality tables).
    
- work_date_list (or work_date_lsit): array of day records:
    
    - date: work date (YYYY/MM/DD).
        
    - hour: assigned hours that day (integer).
        

How it’s used

- Builds the two calendars:
    
    - Tasks × Dates: shows who worked on each module/operation per day.
        
    - Employees × Dates: shows what each person did per day.
        
- Powers KPI calculations (unique workers, overtime >8h, capacity >12h breach).
    
- Drives “Breaches”:
    
    - Window breach: assignment outside phase date window.
        
    - Ordering breach: later phase overlaps or starts before previous phase ends.
        
    - Min/Max staffing breach: heads/day outside op min/max from EnvConfig.
        
    - Skill mismatch: worker lacks skill>0 for the op.
        

---

## Relationships & rules (quick)

- operation in Schedule must exist in EnvConfig workflow (p#o#).
    
- Phase dates constrain assignments for ops in that phase.
    
- Required hours per (module, op) come from workload_days × baseline; under-assignment is flagged.
    
- One factory per employee per day: enforced during solving and validated in outputs.
    
- Manager presence: at least one manager per block (checked in solver and reported).
    

---

## Minimal example (skeleton)

`schedule:   plan_range: { start_date: 2025/04/01, end_date: 2025/04/14 }    workflow_task_list:     - id: e1       fab: fab01       phase_task_list:         - phase: p1           start_date: 2025/04/01           end_date:   2025/04/05           operation_task_list:             - operation: p1o1               workload_days: 3         - phase: p2           start_date: 2025/04/06           end_date:   2025/04/10           operation_task_list:             - operation: p2o1               workload_days: 4    assignment_list:     - worker: w1       operation_task: e1p1o1       work_date_list:         - { date: 2025/04/01, hour: 8 }         - { date: 2025/04/02, hour: 8 }     - worker: w2       operation_task: e1p2o1       work_date_list:         - { date: 2025/04/06, hour: 8 }`

---

## Common authoring tips

- Keep dates in YYYY/MM/DD (the tools normalize “-” to “/”).
    
- Use the correct op ids (p#o#) and phase ids (p#).
    
- If you provide start_date/end_date on an assignment, also include work_date_list; summaries use both.
    
- Typos: work_date_list is preferred; the system tolerates work_date_lsit but it’s better to fix it.
    
- When changing workload_days, re-export to refresh required vs assigned hours and breach tables.