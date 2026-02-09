## 1. Overview

This report summarizes the preparation, execution, and results of the Timefold-based scheduling experiment using Excel-driven inputs and the **Decoder** pipeline.

The goal of this run is to generate `EnvConfig.yaml` and `Schedule.yaml` from multiple Excel sources, then validate feasibility and optimization behavior under the current constraint set.

---

## 2. Planning Range

- **Plan range:** `2025/07/01 – 2026/05/05`
    
- All tasks, assignments, and availability are interpreted within this range.
    

---

## 3. Input Data (Preparation)

The following three Excel files are used as inputs to Decoder:

1. **SU_Others_予定表_2025_新規製番リスト_20260127**
    
    - Source of all target 製番 (modules)
        
    - Provides:
        
        - 製番 code
            
        - Customer company
            
        - Region (country)
            
        - Fab
            
        - Start dates for each 工程 (p1–p4)
            
        - End date for 工程4
            
2. **20260105 SU_Others**
    
    - Source of historical worker data
        
    - Provides:
        
        - Worker list and所属会社
            
        - Daily work records
            
        - Red cells → unavailable dates
            
        - Yellow cells → personal business
            
        - Historical assignments to 製番
            
3. **スキル集計_20260127**
    
    - Main source of formal skill evaluation
        
    - Provides:
        
        - Skill totals and minimum levels per工程
            
        - Used to compute normalized skill levels (0–5)
            

Decoder2 integrates these three sources to produce a unified environment and schedule, following the same structure as previous Timefold experiments.

---

## 4. 製番 (Module) and Initial Assignment Handling

### 4.1 製番 selection and filtering

- 製番 are taken from **新規製番リスト**.
    
- 製番 entries with insufficient information (e.g. `N/A`) are excluded.
    
- All valid 製番 are treated as **flexible / adjustable tasks** by default.
    
- Additional metadata (fab, region, customer company) is derived from the same list.
    

### 4.2 Phase date construction 

For each 製番:

- 工程1〜工程4 start dates are taken directly from the 新規製番リスト.
    
- 工程1〜工程3 end dates are automatically set to **the day before the next 工程 starts**.
    
- 工程4 end date is taken from the provided 工程4終了予定日.
    
- If some start dates are missing, Decoder fills them conservatively to ensure a valid continuous window.
    

### 4.3 Initial assignments from SU_Others (重要)

If a 製番 from 新規製番リスト **also appears in 20260105 SU_Others**:

- Decoder2 checks historical work records.
    
- When a worker worked on that 製番 **within the date range of the corresponding 工程**, an **initial assignment** is created.
    
- The 工程 is inferred from the work date relative to the 工程 start/end window.
    

However:

- If the same 製番 appears in SU_Others **outside the date window defined in 新規製番リスト**,
    
    - That work is **not treated as a tool-install 工程**.
        
    - Instead, it is classified as a fixed **“other” task**.
        
    - This prevents invalid back-propagation of old work into the new schedule window.
        

### 4.4 Other work handling

- All SU_Others records that do not match a valid 製番＋工程 window are converted into:
    
    - Fixed **“other” tasks**
        
- These tasks:
    
    - Cannot be moved by the solver
        
    - Still count for workload and reporting
        
        

---

## 5. Availability and Personal Business

- **Red cells (SU_Others)**  
    → Converted to worker unavailable dates (hard constraint)
    
- **Yellow cells (SU_Others)**  
    → Converted to fixed “personal business” tasks  
    → These block availability but are not optimized
    

---

## 6. Workflow Structure

- All 製番 use the workflow `wf_tool`
    
- 工程 mapping:
    
    - p1: Module Setup
        
    - p2: Hardware Setup
        
    - p3: Function Setup
        
    - p4: Utility
        

Other work and personal business use dedicated fixed workflows.


---

## 6.2 Workload Calculation Method 

The workload of each 工程 is determined by taking the **maximum** of two different calculations.

### Option A — Window-based workload

This is the traditional method.

`workload = end_date - start_date + 1`

Meaning: If a phase lasts 5 calendar days, the required workload is 5.  

### Option B — Actual worker assignment workload  This method uses historical SU_Others data.  Rule:

1 worker working 1 day = 1 workload

`Example:  - p1: 01–06   - Worker A works 03–06 → contributes 4   - Total = 4`
### Final workload rule

- final_workload = max(Option A, Option B)

`Purpose:  - Prevent underestimation of manpower when historical data shows heavier effort. - Keep minimum duration requirement from planning window.`

---

## 7. Skill Map Construction (Detailed)

Employee skill maps are built by **merging two independent sources** and taking the **higher value per 工程**.

### 7.1 Source A: SU_Others (historical assignments)

- If a worker is observed working in SU_Others during a 工程 window:
    
    - That 工程 skill is set to **at least level 1**
        
- If never observed:
    
    - Skill remains 0
        

This represents a **minimum guaranteed capability** based on actual past work.

### 7.2 Source B: スキル集計_20260127 (formal evaluation)

For each worker and each 工程 group, two values are provided:

- 合計 / Level
    
- 最小 / Level
    

Skill is computed as:

1. Convert 合計 / Level into buckets
    
    - ≤20 → 0
        
    - > 20 → floor(total / 20)
        
2. Compare with 最小 / Level
    
    - `level = max(bucket, minimum)`
        
3. Clamp to range 0–5
    

工程4 (p4) uses the same value as 工程3 (p3).

### 7.3 Final merge rule

For each worker and each 工程:

`final_skill = max(skill_from_SU_Others, skill_from_skill_sheet)`
    

---

## 8. Timefold Constraints Used (Summary with Meaning)

### Hard constraints (must always hold)

- **End within window**  
    Each 工程 must stay inside its allowed date range.
    
- **Allowed daily work hours**  
    Only valid daily hour patterns are permitted.
    
- **Phase order (工程順序)**  
    Later 工程 cannot start before earlier 工程 is completed.
    
- **No underfill per block**  
    Required minimum staffing must be satisfied.
    
- **Overfill limited to one day**  
    Exceeding max staffing is allowed only in a very limited manner.
    
- **Employee availability**  
    No assignment on unavailable dates.
    
- **One factory per employee per day**  
    Prevents multi-fab assignments on the same day.
    
- **Daily work limit (12h)**  
    Caps daily workload per worker.
    
- **Region transit gap**  
    Enforces rest/transition days when moving between regions.
    

### Soft constraints (optimization goals)

- **Prefer smaller hours**  
    Avoids unnecessary long workloads.
    
- **Prefer earlier start**  
    Encourages earlier scheduling within allowed windows.
    
- **Prefer same-company pairing**  
    Improves coordination by grouping same-company workers.
    
- **Encourage skill variety**  
    Avoids overly uniform staffing.
    
- **Balance average skill per block**  
    Prevents skill concentration.
    
- **Balance total working hours**  
    Avoids extreme workload imbalance.

---

## 8.1 Solver Calculation Time

The total solver runtime for this experiment was **8 hours**, divided into two stages:

- **Phase 1 – Fix flexible assignments:** 4 hours  
  The solver focuses on stabilizing the base plan and resolving feasibility.

- **Phase 2 – Optimization:** 4 hours  
  The solver improves balance, allocation quality, and soft constraint satisfaction.

---

## 9. Solver Result Summary

- Phase order breaches: **111**
- Tasks with no manager: **397**
- Region transit gap: **1680**
- Block no underfill: **20**
        

---

## 10. Result Interpretation and Known Issues

### 10.1 Phase order

Most violations are caused by inconsistencies in 新規製番リスト,  
for example later 工程 having earlier start dates than previous ones.

---
### 10.2 Tasks with no manager

These are primarily from fixed **other** tasks.

Because they are fixed, the solver cannot add managers.

They are therefore expected in Excel output.

---

### 10.3 Region transit gap

This occurs when workers are scheduled across regions without enough buffer days.

The solver respects the rule, but historical fixed assignments may still produce visible counts.

---

### 10.4 Block no underfill

A small number of blocks remain under the required staffing level.

This usually happens when:

- available workers are limited  
- or skill requirements are tight