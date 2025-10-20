# EnvConfig.yaml – Environment Data Model (Overview & Field Guide)

## What this file is

EnvConfig.yaml is the fixed, shared “environment” dataset used by the scheduler. It describes:

- the standard workflow (phases and operations),
    
- geography and customer/factory context,
    
- worker companies,
    
- workers and their skills/availability,
    
- travel/transition rules between fabs (optional).
    

Schedulers read this once to understand what work can exist, who can do it, where they can do it, and the constraints such as allowed hours or min/max crew sizes.

---

## Top-level layout

`environment:   workflow_list:            # canonical process: phases → operations   region_list:              # e.g., US, JP, TW   customer_company_list:    # customers/owners of fabs   fab_list:                 # factories, each tied to a region + customer   worker_company_list:      # staffing companies/vendors   worker_list:              # people (skills, manager flag, company, OOO)   transite_day_map:         # optional travel days between fabs   workflow_transition_hour_map: # optional per-step setup/transition hours`

Each item below explains what it’s for and the important columns/fields you’ll see.

---

## workflow_list (the standard workflow)

Defines the canonical process as an ordered list of phases; each phase has operations (“ops”).

Key fields

- id: workflow identifier (usually “workflow”).
    
- name: friendly name.
    
- phase_list: array of phases.
    

Phase fields

- id: phase id like p1, p2, p3 … (the number is the sequence).
    
- name: phase name.
    
- operation_list: array of operations inside this phase.
    

Operation fields

- id: operation id like p1o1, p2o3. The prefix p# tells which phase it belongs to.
    
- name: display name (e.g., Mech, Elec).
    
- work_hours: allowed daily hour choices for this op (e.g., [8,10,12]). The solver can pick one.
    
- min_worker_num / max_worker_num: valid headcount range per day for this op.
    

How it’s used

- Workload from Schedule.yaml references these op ids (p#o#).
    
- Pass 1 uses work_hours and min/max to choose block size (heads × hours × days).
    
- Phase order comes from the p# numbers.
    

---

## region_list

Catalog of regions.

Fields

- id: region key.
    
- name: display name.
    

How it’s used

- Fabs link to a region for reporting and potential rules by region.
    

---

## customer_company_list

Catalog of end customers.

Fields

- id: customer id.
    
- name: display name.
    

How it’s used

- Fabs refer to a customer; useful for grouping and reporting.
    

---

## fab_list

Factories (execution locations).

Fields

- id: fab id.
    
- name: display name.
    
- region: region id (from region_list).
    
- customer_company: customer id (from customer_company_list).
    

How it’s used

- Schedule tasks are tied to a fab via the module.
    
- Pass 2 enforces “one factory per employee per day”.
    

---

## worker_company_list

Staffing companies/vendors.

Fields

- id: vendor id.
    
- name: display name.
    

How it’s used

- Workers reference one worker_company.
    
- Dashboard can show team company mix and cohesion.
    

---

## worker_list

People available to schedule.

Core fields

- id: worker id (e.g., w1).
    
- name: initials or full name (used in Excel output).
    
- worker_company: vendor id (links to worker_company_list).
    
- is_manager: true/false; used to satisfy “at least one manager per block”.
    
- skill_map: map of op id → level (integer). Level > 0 means qualified.
    
- unavailable_dates: list of dates they cannot work (optional; ISO date strings).
    
- fab_suitability_map: optional per-fab preference/permission list.
    

How it’s used

- Pass 2 only assigns a worker to an operation if skill_map[op] > 0.
    
- Manager presence is checked per block.
    
- Company is used for team mix KPIs.
    
- Unavailability can be used to exclude specific dates (if enabled).
    
- Skills also drive capacity summaries (how many people can do each op).
    

Skill levels

- Integers (e.g., 1–5). Higher means more experienced.
    
- The system also computes average skill per op across all workers for balance KPIs.
    

---

## transite_day_map (optional travel time)

Describes travel days between fabs.

Typical structure

- A list of entries, e.g., { from: <fab_id>, to: <fab_id>, days: <int> }.
    

How it’s used

- Can be used to prevent the same person from being scheduled in two distant fabs on consecutive days if travel time > 0.
    

---

## workflow_transition_hour_map (optional setup/transition hours)

Defines extra hours or buffers when moving between operations or phases.

Typical structure

- A map keyed by transition (e.g., p1o2→p1o3) to an hour value.
    

How it’s used

- For future extensions: to account for setup time when sequencing blocks.
    

---

## ID and naming conventions (quick reference)

- Phases: p1, p2, p3 … (the number is the phase order).
    
- Operations: p#o# (e.g., p2o3 means operation 3 in phase 2).
    
- Workers: w1, w2, … (unique per person).
    
- Fabs, regions, customers, companies: free-form ids defined in their lists.
    

---

## How the scheduler consumes this file

1. Loads workflow_list to know which operations exist, their allowed hours, and min/max crew sizes.
    
2. Loads workers and builds:
    
    - eligible ops per worker (skill_map),
        
    - per-op capacity (how many people can do it),
        
    - average skill per op (used for balance scoring).
        
3. Loads fabs/regions/customers for grouping and the “one factory per day” rule.
    
4. Optionally reads travel days and transition hours for future constraints.
    

This environment is stable across runs; the variable plan (dates, modules, and workload_days) comes from Schedule.yaml.